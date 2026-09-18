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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .panel import SubscriberPanel

__all__ = ["check_randomization", "RandomizationCheck"]

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
        labels = list(self.counts)
        total = sum(self.counts.values())
        observed = self.counts[labels[1]] / total
        lines = [
            "Randomization check",
            "===================",
            f"  {labels[0]}: {self.counts[labels[0]]:,}   {labels[1]}: {self.counts[labels[1]]:,}",
            f"  observed split {observed:.4f} vs expected {self.expected_ratio:.4f}"
            f"   SRM p = {self.srm_p_value:.2e}",
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
    if not 0 < expected_ratio < 1:
        raise ValueError(f"expected_ratio must be in (0, 1), got {expected_ratio}.")

    counts = {label: int((panel.arm == a).sum()) for a, label in enumerate(panel.arm_labels)}
    n = sum(counts.values())
    expected = [n * (1 - expected_ratio), n * expected_ratio]
    observed = [counts[panel.arm_labels[0]], counts[panel.arm_labels[1]]]
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
    treat, ctrl = panel.arm == 1, panel.arm == 0
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
