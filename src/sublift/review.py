"""One entry point that runs the checks in the order that matters.

By now sublift has a lot of separate pieces, and knowing which to call when is
most of the skill. That ordering should not live only in the documentation,
because the failure mode it guards against is precisely someone calling
``incremental_ltv`` first, getting a tight interval, and shipping a decision
on an experiment whose randomization was broken.

So :func:`review` runs the diagnostics before the estimate, grades what it finds,
and leads with the grade. A **blocker** means the number below it is not
measuring what it says; the number is still computed, because hiding it invites
someone to go and compute it a worse way, but it is labelled. A **warning** means
the result stands with a caveat that changes how far it should be pushed. A
**note** is a finding in its own right -- most usefully, the case where retention
went up and lifetime value went down.

Nothing here is new statistics. It is the existing estimators in the order an
experienced analyst would run them, with the verdict written down instead of
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .competing import ChurnDecomposition, churn_decomposition
from .diagnostics import CensoringCheck, RandomizationCheck, check_censoring, check_randomization
from .estimators import LiftResult, incremental_ltv, retained_periods_lift
from .exceptions import PanelError, SubliftError
from .occupancy import occupancy_lift
from .panel import SubscriberPanel
from .segments import SegmentScan, segment_scan

__all__ = ["review", "ExperimentReview", "Finding"]

Level = Literal["blocker", "warning", "note"]
_ORDER = {"blocker": 0, "warning": 1, "note": 2}
_MARK = {"blocker": "BLOCKER", "warning": "warning", "note": "note"}


@dataclass(frozen=True)
class Finding:
    level: Level
    title: str
    detail: str


@dataclass
class ExperimentReview:
    horizon: int
    findings: list[Finding]
    randomization: RandomizationCheck
    censoring: CensoringCheck | None = None
    retention: LiftResult | None = None
    value: LiftResult | None = None
    decomposition: ChurnDecomposition | None = None
    segments: SegmentScan | None = None
    monitoring: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """Something makes the estimate untrustworthy, not merely uncertain."""
        return any(f.level == "blocker" for f in self.findings)

    @property
    def headline(self) -> LiftResult | None:
        """Lifetime value if it could be computed, otherwise retained periods."""
        return self.value or self.retention

    def summary(self) -> str:
        lines = ["Experiment review", "=================", ""]

        if self.blocked:
            lines += [
                "  VERDICT: do not act on this. See the blockers below.",
                "",
            ]
        elif any(f.level == "warning" for f in self.findings):
            lines += ["  VERDICT: usable, with the caveats below.", ""]
        else:
            lines += ["  VERDICT: no problems found in the checks sublift can run.", ""]

        for finding in sorted(self.findings, key=lambda f: _ORDER[f.level]):
            lines.append(f"  [{_MARK[finding.level]}] {finding.title}")
            for line in finding.detail.splitlines():
                lines.append(f"      {line}")
            lines.append("")

        if self.retention is not None:
            lines += ["  " + line for line in self.retention.summary().splitlines()]
            lines.append("")
        if self.value is not None:
            lines += ["  " + line for line in self.value.summary().splitlines()]
            lines.append("")
        if self.decomposition is not None:
            lines += ["  " + line for line in self.decomposition.summary().splitlines()]
            lines.append("")
        if self.segments is not None:
            lines += ["  " + line for line in self.segments.summary().splitlines()]
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def __str__(self) -> str:
        return self.summary()


def review(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    price=None,
    strata: list[str] | None = None,
    segments: list[str] | None = None,
    alpha: float = 0.05,
    expected_ratio: float = 0.5,
    monitoring: bool = False,
    allow_extrapolation: bool = False,
) -> ExperimentReview:
    """Run the checks, then the estimate, and say what to make of both.

    Parameters
    ----------
    price
        A revenue schedule. Given one -- or a panel carrying revenue -- the
        review reports lifetime value *and* retained periods, because the case
        where they disagree in sign is the finding most worth surfacing.
    segments
        Columns to scan. Optional, and gated behind a heterogeneity test; see
        :func:`sublift.segment_scan`.
    monitoring
        Set it if anyone has been watching this experiment. The review then
        leads with the anytime-valid interval instead of the p-value, which is
        the honest number in that case.
    """
    findings: list[Finding] = []
    estimator = "stratified" if strata else "unadjusted"

    randomization = _randomization(panel, expected_ratio, findings)
    censoring = _censoring(panel, findings)
    _sizes(panel, findings)

    retention, value = _effects(
        panel,
        horizon,
        price,
        estimator,
        strata,
        alpha,
        expected_ratio,
        allow_extrapolation,
        findings,
    )
    resolved = (retention or value).horizon if (retention or value) else (horizon or 0)

    _margin(retention, value, findings)
    decomposition = _decomposition(panel, resolved, alpha, allow_extrapolation, findings)
    scan = _segments(panel, segments, resolved, alpha, allow_extrapolation, findings)
    _monitoring(retention or value, monitoring, findings)

    return ExperimentReview(
        horizon=resolved,
        findings=findings,
        randomization=randomization,
        censoring=censoring,
        retention=retention,
        value=value,
        decomposition=decomposition,
        segments=scan,
        monitoring=monitoring,
    )


# ------------------------------------------------------------------- checks


def _randomization(panel, expected_ratio, findings):
    check = check_randomization(panel, expected_ratio=expected_ratio)
    if check.srm_flagged:
        findings.append(
            Finding(
                "blocker",
                f"Sample ratio mismatch (p = {check.srm_p_value:.1e})",
                "The arms are not the sizes randomization should have produced, so something\n"
                "upstream is filtering subscribers differently by arm. No statistical\n"
                "adjustment rescues this: the estimate is measuring the pipeline. Find the\n"
                "filter before reading anything below.",
            )
        )
    if check.imbalanced:
        findings.append(
            Finding(
                "warning",
                f"Baseline imbalance on {', '.join(check.imbalanced)}",
                "Standardized differences above 0.1. With a clean randomization this happens\n"
                "by chance about as often as you would expect, but it is worth stratifying or\n"
                "adjusting on these columns rather than hoping.",
            )
        )
    return check


def _censoring(panel, findings):
    try:
        check = check_censoring(panel)
    except (PanelError, SubliftError):
        return None
    if check.depends_on_covariates:
        drivers = ", ".join(
            check.coefficients.reindex(
                check.coefficients["coefficient"].abs().sort_values(ascending=False).index
            )["covariate"].head(3)
        )
        findings.append(
            Finding(
                "warning",
                "Censoring is not independent of the subscriber",
                f"Subscribers are leaving the data for reasons related to who they are ({drivers}).\n"
                "The product-limit estimator assumes that cannot happen. Use\n"
                "estimator='adjusted' with those covariates, which more than halves the bias in\n"
                "simulation, and read the result as directional rather than precise.",
            )
        )
    return check


def _sizes(panel, findings):
    smallest = min(int((panel.arm == a).sum()) for a in range(panel.n_arms))
    if smallest < 500:
        findings.append(
            Finding(
                "warning",
                f"Smallest arm has {smallest:,} subscribers",
                "Intervals here lean on asymptotics and will read tighter than they are.\n"
                "Prefer estimator='unadjusted' or 'stratified', both of which have exact\n"
                "influence functions, over covariate adjustment.",
            )
        )


def _effects(panel, horizon, price, estimator, strata, alpha, expected_ratio, extrapolate, findings):
    shared = {
        "horizon": horizon,
        "estimator": estimator,
        "strata": strata,
        "alpha": alpha,
        "expected_ratio": expected_ratio,
        "allow_extrapolation": extrapolate,
    }
    if panel.has_spells:
        findings.append(
            Finding(
                "note",
                "Subscribers return, so periods paid for is the estimand reported",
                "This panel records more than one paying spell per subscriber. Measuring time\n"
                "to first cancellation would overstate the effect, because the subscribers it\n"
                "writes off as lost are disproportionately in the control arm.",
            )
        )
        retention = occupancy_lift(
            panel,
            horizon=horizon,
            strata=strata,
            alpha=alpha,
            expected_ratio=expected_ratio,
            allow_extrapolation=extrapolate,
        )
        value = None
        if price is not None or panel.has_revenue:
            value = occupancy_lift(
                panel,
                horizon=horizon,
                metric="ltv",
                price=price,
                strata=strata,
                alpha=alpha,
                expected_ratio=expected_ratio,
                allow_extrapolation=extrapolate,
            )
        return retention, value

    retention = retained_periods_lift(panel, **shared)
    value = None
    if price is not None or panel.has_revenue:
        value = incremental_ltv(panel, price=price, **shared)
    return retention, value


def _margin(retention, value, findings):
    """The finding most worth surfacing: retention bought with margin."""
    if retention is None or value is None:
        return
    if retention.estimate > 0 and value.estimate < 0:
        findings.append(
            Finding(
                "note",
                "Retention went up and lifetime value went down",
                f"The intervention bought {retention.estimate:+.3f} billing periods and "
                f"{value.estimate:+.3f} in value, per subscriber.\n"
                "Whatever it spent to hold those subscribers cost more than holding them was\n"
                "worth. A retention-only readout would have called this a win.",
            )
        )
    elif retention.estimate < 0 and value.estimate > 0:
        findings.append(
            Finding(
                "note",
                "Lifetime value went up while retention went down",
                "Fewer periods, more value per period. Usually a pricing or mix effect rather\n"
                "than a retention one; check what changed about who stayed.",
            )
        )


def _decomposition(panel, horizon, alpha, extrapolate, findings):
    if panel.cause is None or panel.n_arms > 2:
        return None
    try:
        result = churn_decomposition(panel, horizon=horizon, alpha=alpha, allow_extrapolation=extrapolate)
    except SubliftError:
        return None

    involuntary = [c for c in result.causes if "involuntary" in c.label.lower()]
    if involuntary and result.total:
        share = result.share(involuntary[0])
        if share > 0.4:
            findings.append(
                Finding(
                    "note",
                    "Most of the effect runs through involuntary churn",
                    f"{share:.0%} of the retention gain is subscribers whose payments stopped\n"
                    "failing, not subscribers who decided to stay. That is a payments result\n"
                    "rather than a retention one, and it belongs to whoever changed dunning.",
                )
            )
    return result


def _segments(panel, segments, horizon, alpha, extrapolate, findings):
    if not segments:
        return None
    try:
        scan = segment_scan(panel, by=segments, horizon=horizon, alpha=alpha, allow_extrapolation=extrapolate)
    except SubliftError:
        return None

    if not scan.any_heterogeneity:
        findings.append(
            Finding(
                "note",
                "No segment differs from the average",
                "The segment table below describes this sample; it is not a finding about who\n"
                "the intervention works for. Segments always look different from each other.",
            )
        )
    elif scan.scale_artefact:
        findings.append(
            Finding(
                "note",
                "Segment differences are a scale effect, not a different mechanism",
                "Retained periods vary between segments while the churn odds ratio does not.\n"
                "The intervention is doing the same thing to everyone; the segments differ\n"
                "because they were churning at different rates. Target on it if you like, but\n"
                "it is not evidence the offer lands differently for these people.",
            )
        )
    return scan


def _monitoring(result, monitoring, findings):
    if result is None:
        return
    if monitoring and result.influence is not None:
        sequence = result.confidence_sequence()
        verdict = "excludes zero" if sequence.excludes_zero else "includes zero"
        findings.append(
            Finding(
                "note",
                f"Monitored test: the anytime-valid interval {verdict}",
                f"{sequence}\n"
                "Read this, not the p-value below. A fixed-sample interval is only a 95%\n"
                "interval if you looked once; checked at sixteen interim points its false\n"
                "positive rate is 26%.",
            )
        )
        return
    if monitoring and result.influence is None:
        findings.append(
            Finding(
                "warning",
                "This test has been monitored, but the estimator has no confidence sequence",
                "The p-value reported assumes a single look. Use estimator='stratified' or\n"
                "'unadjusted' to get an anytime-valid interval.",
            )
        )
    elif not monitoring and result.influence is not None:
        findings.append(
            Finding(
                "note",
                "If anyone watched this experiment, read the anytime-valid interval",
                "The p-value is only valid for a single look. Pass monitoring=True and the\n"
                "review will lead with the interval that survives interim checks.",
            )
        )
