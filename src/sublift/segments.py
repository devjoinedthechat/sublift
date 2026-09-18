"""Effects by segment, without the segment story being an artefact of looking.

The most common way a retention experiment produces a false finding is not
peeking and is not many arms. It is this: the offer did not move the headline
number, so someone slices the base and finds that it worked beautifully for
annual subscribers on iOS in their second year. Slice a null experiment twenty
ways and something always looks significant -- that is arithmetic, not insight.

What this module does about it is force the questions into the right order.

**First: is there any real variation at all?** Cochran's Q tests whether the
segment effects differ by more than their own standard errors can explain. If it
does not reject, the segments are one effect seen through noise, and no
individual segment result should be believed no matter how large it looks. That
test comes first in the output because it comes first in the reasoning.

**Second: does it work here?** Per-segment effects with intervals that are
simultaneous across every segment examined, not per-comparison.

**Third: does it work *differently* here?** Usually the claim actually being
made. "It works better for annual" is a statement about the difference between
the segment effect and the pooled effect, and that difference has its own,
wider, uncertainty -- wider because both terms are estimated. Reporting the
segment effect and letting a reader compare it by eye to the headline is how a
segment that is entirely consistent with the average gets written up as a
discovery.

Segments from different columns overlap -- a subscriber is both `plan=monthly`
and `tenure=new` -- so the contrasts are correlated, and the correction is
calibrated against the correlation estimated from the influence functions rather
than assuming independence.

One more trap, and it is the subtle one
---------------------------------------
Heterogeneity in *retained periods* is expected even when the treatment does
exactly the same thing to everybody. A constant odds ratio on churn buys more
absolute periods in a segment that was churning faster to begin with, because
the map from hazard to retained periods is not linear. On this library's own
simulator a perfectly uniform odds ratio of 0.88 produces absolute heterogeneity
that Cochran's Q detects 24% to 31% of the time at 120k subscribers -- rising
with sample size, because the variation is real.

Real, and almost always misread. "The offer works better for disengaged
subscribers" is a claim about mechanism; what the data shows is that disengaged
subscribers had more room to improve. The targeting implication can still hold --
you really do save more periods there -- but the causal story does not.

The scale on which a uniform treatment actually *is* uniform is the odds ratio
it acts on -- not the proportional change in retained periods, which is no more
collapsible than the absolute one. So heterogeneity is tested twice: once on
retained periods, which is what the business cares about, and once on the
per-period churn odds ratio, which is what the treatment does. Absolute
heterogeneity with no odds-ratio heterogeneity is the signature of this artefact,
and the summary says so in those words rather than leaving it to be noticed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .estimators import _arm_weights, _fit_arm, _require_two_arms, _resolve_horizon
from .exceptions import NotIdentifiedError, PanelError
from .family import calibrate as _calibrate
from .family import correlation as _correlation
from .panel import SubscriberPanel, stratum_codes

__all__ = ["segment_scan", "SegmentScan", "SegmentEffect", "Heterogeneity"]


@dataclass(frozen=True)
class SegmentEffect:
    dimension: str
    label: str
    n: int
    n_control: int
    n_treatment: int
    estimate: float
    se: float
    ci: tuple[float, float]
    marginal_ci: tuple[float, float]
    p_value: float
    adjusted_p_value: float
    interaction: float
    interaction_se: float
    interaction_ci: tuple[float, float]
    control_value: float = float("nan")
    relative: float = float("nan")
    log_odds_ratio: float = float("nan")
    log_odds_ratio_se: float = float("nan")

    @property
    def odds_ratio(self) -> float:
        """Effect on the per-period odds of churning -- the scale the treatment acts on."""
        return float(np.exp(self.log_odds_ratio))

    @property
    def name(self) -> str:
        return f"{self.dimension}={self.label}"

    @property
    def significant(self) -> bool:
        """Effect differs from zero in this segment, after correcting for the family."""
        return self.ci[0] > 0 or self.ci[1] < 0

    @property
    def differs_from_overall(self) -> bool:
        """Effect differs from the *pooled* effect -- the claim a segment story makes."""
        return self.interaction_ci[0] > 0 or self.interaction_ci[1] < 0


@dataclass(frozen=True)
class Heterogeneity:
    dimension: str
    scale: str  # "absolute" (retained periods / revenue) or "odds ratio" (what the treatment does)
    q_statistic: float
    df: int
    p_value: float
    adjusted_p_value: float = float("nan")

    @property
    def detected(self) -> bool:
        """Uses the p-value corrected across every dimension tested on this scale."""
        reference = self.adjusted_p_value
        if reference != reference:  # NaN
            reference = self.p_value
        return reference < 0.05


@dataclass
class SegmentScan:
    horizon: int
    metric: str
    alpha: float
    correction: str
    overall: float
    overall_se: float
    overall_ci: tuple[float, float]
    segments: list[SegmentEffect]
    heterogeneity: list[Heterogeneity]
    critical_value: float
    n_subjects: int
    n_comparisons: int
    notes: list[str] = field(default_factory=list)

    @property
    def any_heterogeneity(self) -> bool:
        """Real variation in the absolute effect, on any dimension."""
        return any(h.detected for h in self.heterogeneity if h.scale == "absolute")

    @property
    def mechanism_heterogeneity(self) -> bool:
        """Real variation in the churn odds ratio -- evidence the treatment acts differently."""
        return any(h.detected for h in self.heterogeneity if h.scale == "odds ratio")

    @property
    def scale_artefact(self) -> bool:
        """Variation in retained periods with none in the odds ratio.

        The treatment does the same thing to everybody's churn odds, and the
        differences in retained periods are the segments' different starting
        points showing through. Worth targeting on, not worth explaining.
        """
        return self.any_heterogeneity and not self.mechanism_heterogeneity

    def credible_segments(self) -> list[SegmentEffect]:
        """Segments whose effect genuinely differs from the pooled effect.

        Empty unless some dimension shows real heterogeneity first. A segment
        that differs from the average by more than the correction allows, in a
        scan where the variation as a whole is indistinguishable from noise, is
        the thing this module exists to not report.
        """
        if not self.any_heterogeneity:
            return []
        detected = {h.dimension for h in self.heterogeneity if h.detected and h.scale == "absolute"}
        return sorted(
            (s for s in self.segments if s.dimension in detected and s.differs_from_overall),
            key=lambda s: abs(s.interaction),
            reverse=True,
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dimension": s.dimension,
                    "segment": s.label,
                    "n": s.n,
                    "estimate": s.estimate,
                    "se": s.se,
                    "ci_low": s.ci[0],
                    "ci_high": s.ci[1],
                    "adjusted_p": s.adjusted_p_value,
                    "vs_overall": s.interaction,
                    "vs_overall_low": s.interaction_ci[0],
                    "vs_overall_high": s.interaction_ci[1],
                    "differs_from_overall": s.differs_from_overall,
                }
                for s in self.segments
            ]
        )

    def summary(self) -> str:
        unit = "revenue" if self.metric == "ltv" else "periods"
        head = f"Effect by segment, over {self.horizon} billing periods"
        lines = [
            head,
            "=" * len(head),
            f"  overall  {self.overall:+.4f} {unit} [{self.overall_ci[0]:+.4f}, {self.overall_ci[1]:+.4f}]",
            "",
            "  Is there real variation between segments?",
            f"    {'dimension':<22s} {'periods':>10s} {'odds ratio':>12s}",
        ]
        absolute = {h.dimension: h for h in self.heterogeneity if h.scale == "absolute"}
        mechanism = {h.dimension: h for h in self.heterogeneity if h.scale == "odds ratio"}
        for dimension, h in absolute.items():
            rel = mechanism.get(dimension)
            mark_a = "YES" if h.detected else "no"
            mark_r = ("YES" if rel.detected else "no") if rel else "-"
            lines.append(f"    {dimension:<22s} {mark_a:>10s} {mark_r:>12s}")

        if self.scale_artefact:
            lines += [
                "",
                "  Retained periods vary between segments; the churn odds ratio does not. The",
                "  treatment is doing the same thing to everyone, and the segments differ because",
                "  they were churning at different rates to begin with -- the same odds ratio buys",
                "  more periods where there were more periods to lose. Worth targeting on. Not",
                "  evidence that the offer works differently for these people.",
            ]
        elif not self.any_heterogeneity:
            lines += [
                "",
                "  No dimension shows variation beyond what its own standard errors explain.",
                "  The segments below are one effect seen through noise. Read them as a",
                "  description of this sample, not as a finding about who the offer works for.",
            ]

        width = max(len(s.name) for s in self.segments)
        lines += [
            "",
            f"  {'segment':<{width}s}  {'n':>8s}  {'effect':>9s}  "
            f"{'simultaneous CI':>22s}  {'odds ratio':>10s}  {'vs overall':>11s}",
        ]
        for segment in self.segments:
            mark = "*" if segment.differs_from_overall else " "
            ratio = (
                f"{segment.odds_ratio:>10.3f}"
                if segment.log_odds_ratio == segment.log_odds_ratio
                else f"{'-':>10s}"
            )
            lines.append(
                f" {mark}{segment.name:<{width}s}  {segment.n:>8,}  {segment.estimate:>+9.4f}  "
                f"[{segment.ci[0]:>+9.4f}, {segment.ci[1]:>+9.4f}]  {ratio:>10s}  "
                f"{segment.interaction:>+11.4f}"
            )

        credible = self.credible_segments()
        lines += [
            "",
            f"  {1 - self.alpha:.0%} intervals, simultaneous over {self.n_comparisons} "
            f"comparisons ({self.correction}, critical value {self.critical_value:.3f}).",
        ]
        if credible:
            names = ", ".join(s.name for s in credible)
            lines.append(f"  Segments whose absolute effect differs from the average: {names}")
            if self.scale_artefact:
                lines.append("  -- but see above: proportionally the treatment is doing the same thing.")
        else:
            lines.append(
                "  No segment differs from the average by more than this scan can distinguish from noise."
            )
        if self.notes:
            lines += [""] + [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def segment_scan(
    panel: SubscriberPanel,
    *,
    by: Sequence[str],
    horizon: int | None = None,
    metric: str = "retained_periods",
    price=None,
    alpha: float = 0.05,
    correction: str = "max-t",
    cross: bool = False,
    min_per_arm: int = 100,
    allow_extrapolation: bool = False,
    seed: int = 0,
) -> SegmentScan:
    """Estimate the effect within each segment, correcting across every segment looked at.

    Parameters
    ----------
    by
        Columns to slice on. By default each column is scanned separately, so
        ``by=["plan", "tenure_bucket"]`` gives the effect for every plan *and*
        for every tenure bucket -- the way these questions are actually asked.
        All of them count towards one multiplicity family.
    cross
        Slice on the cross-product instead (``plan x tenure_bucket``). More
        segments, each smaller, and a correction that has to cover all of them.
    min_per_arm
        Segments with fewer subscribers than this in either arm are dropped and
        reported, rather than contributing a comparison so noisy it can only
        dilute the family.
    """
    _require_two_arms(panel)
    if panel.covariates is None:
        raise PanelError("Panel carries no covariates; rebuild it with covariates=[...].")
    missing = [c for c in by if c not in panel.covariates.columns]
    if missing:
        raise PanelError(f"Segment column(s) {missing} not in the panel's covariates.")
    if not by:
        raise ValueError("by=[...] needs at least one column to slice on.")

    horizon = _resolve_horizon(panel, horizon, allow_extrapolation)
    weights = _arm_weights(panel, horizon, metric, price)
    notes: list[str] = []
    n = panel.n_subjects

    overall_value, overall_psi, _, _ = _contrast(
        panel, np.ones(n, dtype=bool), horizon, weights, allow_extrapolation
    )

    definitions = _definitions(panel, list(by), cross)
    # Preallocated and filled in place: collecting rows in a list and vstacking them
    # holds the whole influence matrix twice, which at ten million subscribers is a
    # gigabyte spent on a copy.
    psi = np.empty((len(definitions), n))
    rows, control_rows = [], []
    for dimension, label, mask in definitions:
        sizes = [int((mask & (panel.arm == a)).sum()) for a in (0, 1)]
        if min(sizes) < min_per_arm:
            notes.append(f"{dimension}={label} dropped: {sizes[0]} control / {sizes[1]} treatment")
            continue
        value, influence, control_value, control_psi = _contrast(
            panel, mask, horizon, weights, allow_extrapolation
        )
        psi[len(rows)] = influence
        rows.append((dimension, label, mask, sizes, value, control_value))
        control_rows.append((control_value, control_psi))

    if not rows:
        raise NotIdentifiedError(
            f"Every segment had fewer than {min_per_arm} subscribers per arm. Use coarser "
            "segments, or lower min_per_arm and read the result as exploratory."
        )

    psi = psi[: len(rows)]
    estimates = np.array([r[4] for r in rows])
    control_values = np.array([r[5] for r in rows])
    cov = (psi @ psi.T) / (n**2)
    se = np.sqrt(np.diag(cov))
    z_scores = np.divide(estimates, se, out=np.zeros_like(estimates), where=se > 0)
    raw_p = 2 * stats.norm.sf(np.abs(z_scores))

    k = len(rows)
    critical, adjusted_p = _calibrate(correction, _correlation(cov), z_scores, raw_p, alpha, k, seed)
    marginal = float(stats.norm.ppf(1 - alpha / 2))

    # The interaction contrast carries the uncertainty of both terms, including
    # their covariance -- the segment is part of the pooled estimate it is being
    # compared against, so the two are far from independent.
    # Var(psi_j - overall) expands into terms already computed, so the differenced
    # influence matrix never has to exist.
    cross = psi @ overall_psi
    overall_var = float(overall_psi @ overall_psi)
    interaction_se = np.sqrt(np.maximum(np.diag(cov) - 2 * cross / (n**2) + overall_var / (n**2), 0.0))
    interactions = estimates - overall_value

    relatives = np.divide(
        estimates, control_values, out=np.full_like(estimates, np.nan), where=control_values != 0
    )
    # The churn odds ratio inside each segment: the scale the treatment acts on, and
    # the one a uniform treatment is actually uniform on.
    odds = np.array([_segment_odds_ratio(panel, r[2], horizon) for r in rows])

    segments = [
        SegmentEffect(
            dimension=dimension,
            label=label,
            n=sizes[0] + sizes[1],
            n_control=sizes[0],
            n_treatment=sizes[1],
            estimate=float(estimates[i]),
            se=float(se[i]),
            ci=(float(estimates[i] - critical * se[i]), float(estimates[i] + critical * se[i])),
            marginal_ci=(
                float(estimates[i] - marginal * se[i]),
                float(estimates[i] + marginal * se[i]),
            ),
            p_value=float(raw_p[i]),
            adjusted_p_value=float(adjusted_p[i]),
            interaction=float(interactions[i]),
            interaction_se=float(interaction_se[i]),
            interaction_ci=(
                float(interactions[i] - critical * interaction_se[i]),
                float(interactions[i] + critical * interaction_se[i]),
            ),
            control_value=float(control_values[i]),
            relative=float(relatives[i]),
            log_odds_ratio=float(odds[i, 0]),
            log_odds_ratio_se=float(odds[i, 1]),
        )
        for i, (dimension, label, mask, sizes, _, _) in enumerate(rows)
    ]

    overall_se = float(np.sqrt((overall_psi**2).sum()) / n)
    return SegmentScan(
        horizon=horizon,
        metric=metric,
        alpha=alpha,
        correction=correction,
        overall=overall_value,
        overall_se=overall_se,
        overall_ci=(overall_value - marginal * overall_se, overall_value + marginal * overall_se),
        segments=segments,
        heterogeneity=_heterogeneity(segments),
        critical_value=float(critical),
        n_subjects=n,
        n_comparisons=k,
        notes=notes,
    )


# ----------------------------------------------------------------- internals


def _definitions(panel, by, cross):
    """(dimension, label, mask) for every segment to be examined."""
    covariates = panel.covariates
    if cross:
        codes, labels = stratum_codes(covariates[by], by, sep=" x ")
        name = " x ".join(by)
        return [(name, label, codes == index) for index, label in enumerate(labels)]

    out = []
    for column in by:
        codes, labels = stratum_codes(covariates[[column]], [column])
        out.extend((column, label, codes == index) for index, label in enumerate(labels))
    return out


def _contrast(panel, mask, horizon, weights, allow_extrapolation):
    """Effect inside ``mask``, with influence functions on the whole-sample scale.

    Returns the contrast, its influence function, the control arm's own value and
    the control arm's influence function -- the last two so the proportional
    effect can be given a delta-method interval rather than being read off the
    absolute one.
    """
    n = panel.n_subjects
    share = mask.sum() / n
    values, psi = {}, np.zeros(n)
    control_psi = np.zeros(n)
    for a in (0, 1):
        arm_mask = mask & (panel.arm == a)
        _, value, inf, _ = _fit_arm(panel, arm_mask, horizon, weights[a], allow_extrapolation)
        values[a] = value
        conditional = arm_mask.sum() / mask.sum()
        scaled = inf / (share * conditional)
        psi[arm_mask] = (1 if a == 1 else -1) * scaled
        if a == 0:
            control_psi[arm_mask] = scaled
    return values[1] - values[0], psi, values[0], control_psi


def _heterogeneity(segments: list[SegmentEffect]) -> list[Heterogeneity]:
    """Cochran's Q per dimension and per scale, corrected across dimensions.

    Valid within a dimension because a subscriber falls in exactly one of its
    levels, so the level estimates are independent. Across dimensions they are
    not, so there is no single pooled Q -- instead each dimension is tested
    separately and the p-values are Holm-corrected within a scale, because asking
    "is there heterogeneity?" of five dimensions is five chances to find some.
    """
    out = []
    for scale, value_of, se_of in (
        ("absolute", lambda s: s.estimate, lambda s: s.se),
        ("odds ratio", lambda s: s.log_odds_ratio, lambda s: s.log_odds_ratio_se),
    ):
        raw = []
        for dimension in dict.fromkeys(s.dimension for s in segments):
            levels = [s for s in segments if s.dimension == dimension]
            if len(levels) < 2:
                continue
            values = np.array([value_of(s) for s in levels], dtype=float)
            errors = np.array([se_of(s) for s in levels], dtype=float)
            usable = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
            if usable.sum() < 2:
                continue
            values, errors = values[usable], errors[usable]
            weights = 1.0 / errors**2
            pooled = float(weights @ values / weights.sum())
            q = float(np.sum(weights * (values - pooled) ** 2))
            df = int(usable.sum()) - 1
            raw.append((dimension, q, df, float(stats.chi2.sf(q, df))))

        for dimension, q, df, p_value in raw:
            out.append(Heterogeneity(dimension, scale, q, df, p_value, _holm(raw, p_value)))
    return out


def _segment_odds_ratio(panel, mask, horizon) -> tuple[float, float]:
    """Treatment effect on the per-period churn odds inside one segment.

    A discrete-time survival likelihood factorises into independent Bernoulli
    terms over the person-periods at risk, so this is an ordinary logistic fit
    and its model-based standard error is the right one -- no clustering needed,
    despite the repeated rows per subscriber.

    Built from the three arrays it needs rather than from a sub-panel: taking a
    sub-panel would copy the covariate frame and the revenue grid for every
    segment, which on a large base costs more than the fit.
    """
    from .logistic import fit_logistic

    n_periods = panel.n_periods[mask]
    event = panel.event[mask]
    arm = panel.arm[mask]
    if n_periods.size == 0:
        return float("nan"), float("nan")

    # The design here is time dummies plus an arm indicator -- entirely categorical, so
    # however many million person-periods a segment has, there are only 2 x horizon
    # distinct rows. Counting them and fitting the aggregate is the same likelihood.
    capped = np.minimum(n_periods, horizon)
    churned_in = event & (n_periods <= horizon)

    at_risk = np.zeros((2, horizon))
    churned = np.zeros((2, horizon))
    for a in (0, 1):
        side = arm == a
        counts = np.bincount(capped[side], minlength=horizon + 1)[1:]
        at_risk[a] = np.cumsum(counts[::-1])[::-1]
        churned[a] = np.bincount(n_periods[side & churned_in], minlength=horizon + 1)[1 : horizon + 1]
    if churned.sum() == 0:
        return float("nan"), float("nan")

    cells = 2 * horizon
    design = np.zeros((2 * cells, horizon + 1))
    outcome = np.zeros(2 * cells)
    weight = np.zeros(2 * cells)
    row = 0
    for a in (0, 1):
        for t in range(horizon):
            for success in (1.0, 0.0):
                design[row, t] = 1.0
                design[row, horizon] = a
                outcome[row] = success
                weight[row] = churned[a, t] if success else at_risk[a, t] - churned[a, t]
                row += 1

    fit = fit_logistic(design, outcome, sample_weight=weight)
    try:
        variance = np.linalg.inv(fit.hessian)[horizon, horizon]
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return float(fit.coef[horizon]), float(np.sqrt(max(variance, 0.0)))


def _holm(raw, p_value: float) -> float:
    """Holm-adjusted p-value for one test within its family of dimensions."""
    ordered = sorted(p for _, _, _, p in raw)
    k = len(ordered)
    running, adjusted = 0.0, {}
    for rank, p in enumerate(ordered):
        running = max(running, (k - rank) * p)
        adjusted[p] = min(1.0, running)
    return adjusted[p_value]
