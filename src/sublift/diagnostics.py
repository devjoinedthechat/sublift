"""Checks that run before anyone reads the effect.

Every estimator in sublift assumes the arms were assigned at random. If that
assumption fails, nothing downstream is worth reading, and the failure is
usually silent: a targeting rule that quietly excluded a segment, a feature flag
that defaulted on for iOS, an ETL job that dropped a partition. The result still
prints a number with a tight interval.

So the sample ratio mismatch test is not an optional extra here. It runs
automatically whenever an effect is estimated, and it warns loudly, because the
cost of ignoring a real SRM is a wrong decision and the cost of a false alarm is
five minutes of checking.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import PanelError
from .panel import SubscriberPanel

__all__ = [
    "check_randomization",
    "RandomizationCheck",
    "check_censoring",
    "CensoringCheck",
]

# The community-standard SRM threshold. Deliberately far stricter than 0.05:
# with a genuinely random assignment this fires once in a thousand experiments,
# so when it does fire it is almost always a real pipeline bug.
SRM_ALPHA = 1e-3
SMD_THRESHOLD = 0.10


@dataclass
class RandomizationCheck:
    counts: dict[str, int]
    expected_ratio: float
    srm_p_value: float
    balance: pd.DataFrame | None
    srm_alpha: float = SRM_ALPHA
    smd_threshold: float = SMD_THRESHOLD

    @property
    def srm_flagged(self) -> bool:
        return self.srm_p_value < self.srm_alpha

    @property
    def imbalanced(self) -> list[str]:
        if self.balance is None:
            return []
        return self.balance.loc[self.balance["std_diff"].abs() > self.smd_threshold, "covariate"].tolist()

    @property
    def ok(self) -> bool:
        return not self.srm_flagged and not self.imbalanced

    def __str__(self) -> str:
        total = sum(self.counts.values())
        split = "   ".join(f"{label}: {n:,}" for label, n in self.counts.items())
        shares = " / ".join(f"{n / total:.4f}" for n in self.counts.values())
        lines = [
            "Randomization check",
            "===================",
            f"  {split}",
            f"  observed split {shares}   SRM p = {self.srm_p_value:.2e}",
        ]
        if self.srm_flagged:
            lines += [
                "",
                "  SAMPLE RATIO MISMATCH. The arms are not the sizes randomization should",
                "  have produced. Something upstream is filtering subscribers differently by",
                "  arm -- assignment, logging or the extract. Fix that before reading any",
                "  effect below; the estimate is not salvageable by adjusting for it.",
            ]
        if self.balance is not None:
            lines += ["", "  baseline balance (standardized mean difference):"]
            for _, r in self.balance.iterrows():
                mark = "  <-- imbalanced" if abs(r["std_diff"]) > self.smd_threshold else ""
                lines.append(f"    {r['covariate']:<28s} {r['std_diff']:+.4f}{mark}")
        if self.ok:
            lines += ["", "  No problems found."]
        return "\n".join(lines)


def check_randomization(
    panel: SubscriberPanel,
    *,
    expected_ratio: float = 0.5,
    expected_shares: dict[str, float] | None = None,
    srm_alpha: float = SRM_ALPHA,
    smd_threshold: float = SMD_THRESHOLD,
) -> RandomizationCheck:
    """Sample ratio mismatch, plus baseline covariate balance if the panel has covariates.

    Parameters
    ----------
    expected_ratio
        The share of subscribers the assignment was *supposed* to send to the
        treatment arm. Pass the designed ratio, not the observed one -- checking
        the data against itself proves nothing.
    """
    if expected_shares is None and not 0 < expected_ratio < 1:
        raise ValueError(f"expected_ratio must be in (0, 1), got {expected_ratio}.")

    counts = {label: int((panel.arm == a).sum()) for a, label in enumerate(panel.arm_labels)}
    n = sum(counts.values())
    observed = [counts[label] for label in panel.arm_labels]

    if panel.n_arms == 2:
        expected = [n * (1 - expected_ratio), n * expected_ratio]
    else:
        # With more than two arms an equal split is the only sane default; a designed
        # imbalance has to be passed explicitly, because guessing it from the data is
        # exactly the check being performed.
        if expected_ratio not in (None, 0.5):
            raise ValueError(
                "expected_ratio describes a two-arm split. For a multi-arm panel pass "
                "expected_shares={'arm': share, ...} or leave it unset for an equal split."
            )
        expected = [n / panel.n_arms] * panel.n_arms
    if expected_shares is not None:
        missing = set(panel.arm_labels) - set(expected_shares)
        if missing:
            raise ValueError(f"expected_shares is missing arm(s) {sorted(missing)}.")
        total = float(sum(expected_shares[label] for label in panel.arm_labels))
        expected = [n * expected_shares[label] / total for label in panel.arm_labels]
    p_value = float(stats.chisquare(observed, f_exp=expected).pvalue)

    balance = _balance(panel) if panel.covariates is not None else None
    return RandomizationCheck(
        counts=counts,
        expected_ratio=expected_ratio,
        srm_p_value=p_value,
        balance=balance,
        srm_alpha=srm_alpha,
        smd_threshold=smd_threshold,
    )


def _balance(panel: SubscriberPanel) -> pd.DataFrame:
    """Standardized mean differences on baseline covariates.

    Standardized differences rather than t-tests on purpose: with a large
    experiment a trivially small imbalance is 'significant', and with a small one
    a serious imbalance is not. The standardized difference measures how big the
    imbalance is, which is the question. Above about 0.1 is worth investigating.
    """
    rows = []
    # Balance is reported against the control arm, which is the comparison every
    # contrast is made against.
    treat, ctrl = panel.arm > 0, panel.arm == 0
    for col in panel.covariates.columns:
        s = panel.covariates[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            series = {col: pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)}
        else:
            as_str = s.astype(str).to_numpy()
            series = {f"{col}={lv}": (as_str == lv).astype(float) for lv in sorted(pd.unique(as_str))}
        for name, values in series.items():
            a, b = values[treat], values[ctrl]
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            smd = float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0
            rows.append(
                {
                    "covariate": name,
                    "control_mean": float(b.mean()),
                    "treatment_mean": float(a.mean()),
                    "std_diff": smd,
                }
            )
    return pd.DataFrame(rows)


def warn_on_srm(panel: SubscriberPanel, expected_ratio: float = 0.5) -> RandomizationCheck | None:
    """Run the SRM check and warn if it fires. Called automatically by the estimators."""
    try:
        check = check_randomization(panel, expected_ratio=expected_ratio)
    except Exception:  # a diagnostic must never be the reason an analysis fails
        return None
    if check.srm_flagged:
        labels = list(check.counts)
        warnings.warn(
            f"Sample ratio mismatch: {check.counts[labels[0]]:,} {labels[0]} vs "
            f"{check.counts[labels[1]]:,} {labels[1]} (p={check.srm_p_value:.2e}, expected split "
            f"{expected_ratio:.2f}). Randomization or logging is filtering subscribers by arm; "
            "the effect below is not trustworthy until that is explained. Pass "
            "expected_ratio=... if the design was not 50/50.",
            stacklevel=3,
        )
    return check


@dataclass
class CensoringCheck:
    """Whether censoring looks administrative, or looks like it depends on the subscriber."""

    known_exactly: bool
    n_censored: int
    lr_statistic: float
    lr_p_value: float
    df: int
    coefficients: pd.DataFrame | None
    alpha: float = 0.01

    @property
    def depends_on_covariates(self) -> bool:
        return (not self.known_exactly) and self.lr_p_value < self.alpha

    @property
    def ok(self) -> bool:
        return self.known_exactly or not self.depends_on_covariates

    def __str__(self) -> str:
        lines = ["Censoring check", "==============="]
        if self.known_exactly:
            lines += [
                f"  {self.n_censored:,} subscribers censored, and potential follow-up is recorded.",
                "",
                "  Censoring is administrative: every subscriber's follow-up was fixed by their",
                "  assignment date and the data cut, so it cannot depend on anything they did.",
                "  This is the good case and nothing further is needed.",
            ]
            return "\n".join(lines)

        lines.append(f"  {self.n_censored:,} subscribers censored; potential follow-up not recorded.")
        lines.append(
            f"  Does censoring depend on baseline covariates?  LR chi2({self.df}) = "
            f"{self.lr_statistic:.1f}, p = {self.lr_p_value:.2e}"
        )
        if self.coefficients is not None and len(self.coefficients):
            lines.append("")
            lines.append("  censoring log-odds per unit of covariate:")
            ordered = self.coefficients.reindex(
                self.coefficients["coefficient"].abs().sort_values(ascending=False).index
            )
            for _, r in ordered.head(8).iterrows():
                lines.append(f"    {r['covariate']:<28s} {r['coefficient']:+.3f}")

        if self.depends_on_covariates:
            lines += [
                "",
                "  CENSORING IS NOT INDEPENDENT. Subscribers are leaving the data for reasons",
                "  related to who they are, so the product-limit estimator -- and therefore",
                "  estimator='unadjusted' and estimator='stratified' -- is biased.",
                "",
                "  What to do: use estimator='adjusted' with these covariates. Censoring that",
                "  depends only on X is independent *given* X, so a hazard model containing X",
                "  removes most of the bias. In simulation that holds the bias roughly flat",
                "  while the nonparametric estimators drift by 4x.",
            ]
        else:
            lines += ["", "  No evidence that censoring depends on these covariates."]
        return "\n".join(lines)


def check_censoring(
    panel: SubscriberPanel,
    covariates: Sequence[str] | None = None,
    *,
    alpha: float = 0.01,
) -> CensoringCheck:
    """Test the assumption that censoring is independent of the subscriber.

    Every estimator here assumes subscribers are censored for reasons unrelated
    to their propensity to churn -- normally true, because the data cut is a
    calendar fact. It stops being true when subscribers are *lost* rather than
    merely not-yet-observed: accounts deleted, a cohort dropped by a migration,
    a plan the extract stopped covering.

    That failure is silent. The estimate still prints a tight interval; it is
    just measuring a population that quietly selected itself. This is the one
    assumption in sublift that used to be untestable, and it is testable whenever
    the panel carries baseline covariates: fit the censoring hazard with and
    without them and compare the fits.

    If ``potential_followup`` is recorded, censoring is administrative by
    construction and the question does not arise.
    """
    from scipy import stats as _stats

    from .censoring import censoring_person_period
    from .logistic import design_matrix, fit_logistic

    n_censored = int((~panel.event).sum())
    if panel.potential_followup is not None:
        return CensoringCheck(True, n_censored, 0.0, 1.0, 0, None, alpha)

    if panel.covariates is None:
        raise PanelError(
            "Censoring cannot be checked without baseline covariates. Rebuild the panel with "
            "covariates=[...], or record potential_followup, which settles the question outright."
        )
    cols = list(covariates) if covariates else list(panel.covariates.columns)
    missing = [c for c in cols if c not in panel.covariates.columns]
    if missing:
        raise PanelError(f"Covariate(s) {missing} not in the panel's covariates.")

    horizon = panel.followup
    X, names, _ = design_matrix(panel.covariates[cols])
    rows, period, censored = censoring_person_period(panel, horizon)
    if censored.sum() == 0 or X.shape[1] == 0:
        return CensoringCheck(False, n_censored, 0.0, 1.0, 0, None, alpha)

    dummies = np.zeros((period.size, horizon))
    dummies[np.arange(period.size), period - 1] = 1.0
    full = np.hstack([dummies, X[rows]])

    fit_null = fit_logistic(dummies, censored)
    fit_full = fit_logistic(full, censored)
    statistic = 2.0 * (_loglik(full, censored, fit_full.coef) - _loglik(dummies, censored, fit_null.coef))
    df = X.shape[1]
    p_value = float(_stats.chi2.sf(max(statistic, 0.0), df))

    coefficients = pd.DataFrame({"covariate": names, "coefficient": fit_full.coef[horizon:]})
    return CensoringCheck(False, n_censored, float(statistic), p_value, df, coefficients, alpha)


def _loglik(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    eta = X @ coef
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))
