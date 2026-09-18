"""Several treatment arms against one control, without inflating the error rate.

Testing three save offers against a holdout and reporting whichever looked best
is not three experiments; it is one experiment with three chances to be wrong.
Run as separate pairwise tests at 5% each, the probability of declaring at least
one winner when none of them works is close to 15%, and it grows with the number
of arms. That is the same failure as peeking, in a different direction: there
the extra chances come from looking repeatedly, here from looking widely.

The correction used by default is **single-step max-t**, and it is worth
explaining why rather than reaching for Bonferroni. Every contrast here shares
the same control arm, so the contrasts are *positively correlated* -- if the
control arm happens to look bad, every treatment arm looks good together. A
Bonferroni correction assumes the worst about that dependence and pays for
independence it does not have. max-t estimates the correlation from the
influence functions and calibrates the critical value against it, which is
strictly less conservative while controlling the same family-wise error rate.

With a shared control and equal arm sizes the correlation is 0.5, exactly as
theory says, and the measured saving is honest but modest: the critical value
comes in 1.1% below Bonferroni at two treatment arms, 2.4% at five, 3.1% at
eight. Because required sample size scales with the square of the critical
value, that is roughly 2% to 6% fewer subscribers for the same power -- worth
having and free, but not the difference between a conclusive test and an
inconclusive one. The reason to prefer it is that it is the correct calibration,
not that it rescues an underpowered experiment.

Bonferroni and Holm are available for when a reviewer wants the familiar thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .estimators import _fit_arm, _resolve_horizon
from .exceptions import NotIdentifiedError, PanelError
from .family import calibrate as _calibrate
from .family import correlation as _correlation
from .panel import SubscriberPanel, stratum_codes

__all__ = ["multi_arm_lift", "MultiArmResult", "ArmContrast"]

from .family import CORRECTIONS as _CORRECTIONS


@dataclass(frozen=True)
class ArmContrast:
    label: str
    n: int
    value: float
    estimate: float
    se: float
    ci: tuple[float, float]
    marginal_ci: tuple[float, float]
    p_value: float
    adjusted_p_value: float

    @property
    def significant(self) -> bool:
        """Whether this arm beats the control after correcting for the family."""
        return self.ci[0] > 0 or self.ci[1] < 0


@dataclass
class MultiArmResult:
    horizon: int
    metric: str
    alpha: float
    correction: str
    estimator: str
    control_label: str
    control_value: float
    control_n: int
    contrasts: list[ArmContrast]
    critical_value: float
    bonferroni_critical_value: float
    correlation: np.ndarray
    n_subjects: int
    influence: np.ndarray | None = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)

    @property
    def winners(self) -> list[ArmContrast]:
        """Arms that beat the control after the correction, best first."""
        return sorted(
            (c for c in self.contrasts if c.significant),
            key=lambda c: c.estimate,
            reverse=True,
        )

    def best(self) -> ArmContrast | None:
        """The best arm that survives the correction, or ``None`` if nothing does.

        Deliberately returns ``None`` rather than the largest point estimate. In
        a null experiment some arm always has the largest estimate, and reporting
        it is the exact mistake this module exists to prevent.
        """
        found = self.winners
        return found[0] if found else None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "arm": c.label,
                    "n": c.n,
                    "value": c.value,
                    "estimate": c.estimate,
                    "se": c.se,
                    "ci_low": c.ci[0],
                    "ci_high": c.ci[1],
                    "p_value": c.p_value,
                    "adjusted_p": c.adjusted_p_value,
                    "significant": c.significant,
                }
                for c in self.contrasts
            ]
        )

    def confidence_sequences(self, *, n_target: int | None = None, calibration: str = "bonferroni") -> dict:
        """Anytime-valid intervals, valid over arms *and* over every interim look.

        A confidence sequence has no max-t form -- its boundary is derived for a
        scalar process and does not see the correlation between arms -- so the
        family has to be handled by splitting ``alpha``. Splitting it ``k`` ways
        is Bonferroni and always valid. Splitting it by the *effective*
        multiplicity instead charges for the number of independent comparisons the
        family actually behaves like, which for arms sharing a control is
        meaningfully fewer than ``k``.

        Bonferroni is the default, and that is a deliberate retreat from the
        fixed-sample case. Measured under a global null monitored at eight interim
        looks: no correction at all lands at 2.5% against a nominal 5%, Bonferroni
        and the effective-multiplicity calibration both at 1.0%, and the latter's
        intervals are about 1% narrower. The reason the gain is so small is that
        ``alpha`` enters the sequence boundary inside a logarithm, so dividing it
        by 3.5 rather than 4 barely moves anything -- and the reason the rates are
        all well under 5% is that a confidence sequence is already conservative
        relative to its nominal level.

        So ``"max-t"`` is available and validated, and buys almost nothing.
        Defaulting to an approximation for a one percent interval is not a trade
        worth making, and the provable union bound is.
        """
        from .family import effective_multiplicity
        from .sequential import confidence_sequence

        if self.influence is None:
            raise NotIdentifiedError(
                f"The {self.estimator!r} estimator does not expose per-subject influence values "
                "here. Use estimator='unadjusted' or 'stratified' to monitor a multi-arm test."
            )
        if calibration not in ("max-t", "bonferroni"):
            raise ValueError("calibration must be 'max-t' or 'bonferroni'.")

        k = len(self.contrasts)
        divisor = effective_multiplicity(self.correlation, alpha=self.alpha) if calibration == "max-t" else k
        split = self.alpha / divisor
        return {
            c.label: confidence_sequence(
                self.influence[i], estimate=c.estimate, alpha=split, n_target=n_target
            )
            for i, c in enumerate(self.contrasts)
        }

    def summary(self) -> str:
        unit = "revenue" if self.metric == "ltv" else "periods"
        head = f"{len(self.contrasts)} arms vs {self.control_label}, over {self.horizon} billing periods"
        width = max(len(c.label) for c in self.contrasts)
        lines = [
            head,
            "=" * len(head),
            f"  {self.control_label} (n={self.control_n:,}): {self.control_value:,.4f} {unit}",
            "",
        ]
        for c in sorted(self.contrasts, key=lambda c: c.estimate, reverse=True):
            mark = "*" if c.significant else " "
            lines.append(
                f" {mark}{c.label:<{width}s}  n={c.n:>7,}  {c.estimate:+.4f}  "
                f"[{c.ci[0]:+.4f}, {c.ci[1]:+.4f}]   p={c.adjusted_p_value:.4f}"
            )
        best = self.best()
        lines += [
            "",
            f"  {1 - self.alpha:.0%} simultaneous intervals, {self.correction} correction "
            f"(critical value {self.critical_value:.3f}"
            + (
                f" vs Bonferroni {self.bonferroni_critical_value:.3f})" if self.correction == "max-t" else ")"
            ),
        ]
        if best is not None:
            lines.append(f"  Best arm that survives the correction: {best.label}")
        else:
            lines.append(
                "  No arm beats the control once the family is accounted for. The largest "
                "point estimate is not a winner."
            )
        if self.notes:
            lines += [""] + [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def multi_arm_lift(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    metric: str = "retained_periods",
    estimator: str = "stratified",
    strata: list[str] | None = None,
    price=None,
    alpha: float = 0.05,
    correction: str = "max-t",
    comparisons: str = "vs-control",
    allow_extrapolation: bool = False,
    seed: int = 0,
) -> MultiArmResult:
    """Every treatment arm against the control, with family-wise error control.

    Parameters
    ----------
    comparisons
        ``"vs-control"`` (default) tests each treatment arm against the control.
        ``"all-pairs"`` tests every pair of arms, which is what you want when the
        arms are alternatives rather than variations on a holdout -- three
        pricing tiers, say -- and which grows the family from ``k`` comparisons
        to ``k(k+1)/2``, so it should be asked for rather than assumed.
    correction
        ``"max-t"`` (default) calibrates the critical value against the estimated
        correlation between contrasts, which share a control arm and are
        therefore positively correlated. ``"bonferroni"`` and ``"holm"`` ignore
        that correlation and are more conservative; ``"holm"`` is more powerful
        than Bonferroni for testing but does not yield simultaneous intervals,
        so its intervals fall back to Bonferroni. ``"none"`` reports
        per-comparison intervals and is there for when the arms are genuinely
        separate experiments that happen to share a panel.
    """
    if correction not in _CORRECTIONS:
        raise ValueError(f"correction must be one of {_CORRECTIONS}, got {correction!r}.")
    pairs = _pairs(panel.n_arms, comparisons)
    if estimator not in ("unadjusted", "stratified"):
        raise ValueError(
            "multi_arm_lift supports estimator='unadjusted' and 'stratified'. The adjusted "
            "estimator's influence function is per-comparison; use panel.contrast('<arm>') "
            "with it and correct the p-values yourself."
        )
    if panel.n_arms < 3:
        raise PanelError(
            f"This panel has {panel.n_arms} arms. With a single treatment there is no family "
            "to correct for -- use incremental_ltv() or retained_periods_lift()."
        )

    horizon = _resolve_horizon(panel, horizon, allow_extrapolation)
    weights = _multi_arm_weights(panel, horizon, metric, price)
    notes: list[str] = []

    if estimator == "stratified":
        if not strata:
            raise ValueError("estimator='stratified' needs strata=[...].")
        values, arm_psi, stratified = _stratified_arms(
            panel, horizon, weights, list(strata), allow_extrapolation, notes
        )
    else:
        values, arm_psi, stratified = _unadjusted_arms(panel, horizon, weights, allow_extrapolation)

    estimates, psi = _contrast_rows(pairs, values, arm_psi, stratified)
    k = len(pairs)
    n = panel.n_subjects
    cov = (psi @ psi.T) / (n**2)
    se = np.sqrt(np.diag(cov))
    z_scores = np.divide(estimates, se, out=np.zeros_like(estimates), where=se > 0)
    raw_p = 2 * stats.norm.sf(np.abs(z_scores))

    corr = _correlation(cov)
    critical, adjusted_p = _calibrate(correction, corr, z_scores, raw_p, alpha, k, seed)
    bonferroni = float(stats.norm.ppf(1 - alpha / (2 * k)))
    marginal = float(stats.norm.ppf(1 - alpha / 2))

    contrasts = [
        ArmContrast(
            label=(panel.arm_labels[a] if b == 0 else f"{panel.arm_labels[a]} vs {panel.arm_labels[b]}"),
            n=int((panel.arm == a).sum()),
            value=values[a],
            estimate=float(estimates[i]),
            se=float(se[i]),
            ci=(float(estimates[i] - critical * se[i]), float(estimates[i] + critical * se[i])),
            marginal_ci=(
                float(estimates[i] - marginal * se[i]),
                float(estimates[i] + marginal * se[i]),
            ),
            p_value=float(raw_p[i]),
            adjusted_p_value=float(adjusted_p[i]),
        )
        for i, (a, b) in enumerate(pairs)
    ]

    return MultiArmResult(
        horizon=horizon,
        metric=metric,
        alpha=alpha,
        correction=correction,
        estimator=estimator,
        control_label=panel.arm_labels[0],
        control_value=values[0],
        control_n=int((panel.arm == 0).sum()),
        contrasts=contrasts,
        critical_value=float(critical),
        bonferroni_critical_value=bonferroni,
        correlation=corr,
        n_subjects=n,
        influence=psi,
        notes=notes,
    )


# ----------------------------------------------------------------- internals


def _multi_arm_weights(panel, horizon, metric, price):
    """Per-arm revenue weights, generalizing the two-arm helper to K arms."""
    from .estimators import _weights_for

    if metric == "retained_periods":
        return {a: (np.ones(horizon), False) for a in range(panel.n_arms)}
    if metric != "ltv":
        raise ValueError("metric must be 'ltv' or 'retained_periods'.")

    if price is not None:
        if isinstance(price, dict):
            missing = set(panel.arm_labels) - set(price)
            if missing:
                raise ValueError(f"price dict is missing arm(s) {sorted(missing)}.")
            return {a: (_schedule(price[panel.arm_labels[a]], horizon), False) for a in range(panel.n_arms)}
        sched = _schedule(price, horizon)
        return dict.fromkeys(range(panel.n_arms), (sched, False))

    if not panel.has_revenue:
        raise ValueError(
            "No revenue in the panel and no price= given, so LTV cannot be formed. Pass "
            "price=..., build the panel with revenue=..., or use metric='retained_periods'."
        )
    out = {}
    for a in range(panel.n_arms):
        mask = panel.arm == a
        out[a] = (_weights_for(panel, mask, horizon), True)
    return out


def _schedule(price, horizon):
    if np.isscalar(price):
        return np.full(horizon, float(price))
    arr = np.asarray(price, dtype=float)
    if arr.size < horizon:
        raise ValueError(f"price schedule covers {arr.size} periods but the horizon is {horizon}.")
    return arr[:horizon]


def _pairs(n_arms: int, comparisons: str) -> list[tuple[int, int]]:
    """Which arm pairs make up the family."""
    if comparisons == "vs-control":
        return [(a, 0) for a in range(1, n_arms)]
    if comparisons == "all-pairs":
        return [(b, a) for a in range(n_arms) for b in range(a + 1, n_arms)]
    raise ValueError(f"comparisons must be 'vs-control' or 'all-pairs', got {comparisons!r}.")


def _unadjusted_arms(panel, horizon, weights, allow_extrapolation):
    """Per-arm values and per-arm influence, on the whole-sample scale."""
    n = panel.n_subjects
    values, arm_psi = {}, np.zeros((panel.n_arms, n))
    for a in range(panel.n_arms):
        mask = panel.arm == a
        _, value, inf, _ = _fit_arm(panel, mask, horizon, weights[a], allow_extrapolation)
        values[a] = value
        arm_psi[a, mask] = inf / (mask.sum() / n)
    return values, arm_psi, None


def _stratified_arms(panel, horizon, weights, strata, allow_extrapolation, notes):
    """The same, post-stratified, plus what the stratum-share term needs.

    The within-stratum influence is per arm, but the term for the error in the
    stratum shares depends on the *contrast*, so it is left to be added once the
    pairs are known -- which is what lets all-pairs reuse this unchanged.
    """
    if panel.covariates is None:
        raise PanelError("Panel carries no covariates; rebuild it with covariates=[...].")
    missing = [c for c in strata if c not in panel.covariates.columns]
    if missing:
        raise PanelError(f"Strata column(s) {missing} not in the panel's covariates.")

    codes, levels = stratum_codes(panel.covariates, strata)
    n = panel.n_subjects
    n_arms = panel.n_arms
    arm_psi = np.zeros((n_arms, n))
    stratum_value = np.zeros((n_arms, len(levels)))
    shares = np.zeros(len(levels))
    kept = np.zeros(n, dtype=bool)

    for s_index in range(len(levels)):
        in_s = codes == s_index
        sizes = {a: int((in_s & (panel.arm == a)).sum()) for a in range(n_arms)}
        if min(sizes.values()) < 2:
            notes.append(f"stratum {levels[s_index]!r} dropped: smallest arm had {min(sizes.values())}")
            continue
        kept |= in_s
        shares[s_index] = in_s.sum()
        for a in range(n_arms):
            mask = in_s & (panel.arm == a)
            _, value, inf, _ = _fit_arm(panel, mask, horizon, weights[a], allow_extrapolation)
            stratum_value[a, s_index] = value
            arm_psi[a, mask] = inf / (sizes[a] / in_s.sum())

    total = shares.sum()
    if total == 0:
        raise NotIdentifiedError("Every stratum was too small to estimate; use coarser strata.")
    if kept.sum() < n:
        notes.append(f"{n - int(kept.sum())} of {n} subscribers fell in dropped strata")

    pi = shares / total
    values = {a: float(stratum_value[a] @ pi) for a in range(n_arms)}
    return values, arm_psi, (stratum_value, pi, codes, kept)


def _contrast_rows(pairs, values, arm_psi, stratified):
    """Estimates and influence rows for each pair, adding the stratum-share term."""
    estimates = np.array([values[a] - values[b] for a, b in pairs])
    psi = np.vstack([arm_psi[a] - arm_psi[b] for a, b in pairs])
    if stratified is None:
        return estimates, psi

    stratum_value, pi, codes, kept = stratified
    deltas = np.vstack([stratum_value[a] - stratum_value[b] for a, b in pairs])
    combined = deltas @ pi
    psi = psi[:, kept] + (deltas[:, codes[kept]] - combined[:, None])
    return combined, psi
