"""Human-readable summaries.

The failure mode this guards against is a number reported without the context
that decides whether it means anything: the horizon, the censoring, and whether
anyone has been watching.
"""

from __future__ import annotations

__all__ = ["format_result"]

_UNITS = {"ltv": "revenue", "retained_periods": "periods"}


def format_result(result) -> str:
    unit = _UNITS.get(result.metric, "")
    conf = f"{1 - result.alpha:.0%}"
    ctrl, treat = result.control, result.treatment

    lo, hi = result.ci
    sign = "+" if result.estimate >= 0 else ""
    head = f"{'Incremental LTV' if result.metric == 'ltv' else 'Incremental retained periods'} over {result.horizon} billing periods"

    lines = [
        head,
        "=" * len(head),
        f"  {sign}{result.estimate:,.4f} {unit} per subscriber "
        f"({result.relative:+.1%} vs {ctrl.label})",
        f"  {conf} CI [{lo:+,.4f}, {hi:+,.4f}]   se {result.se:,.4f}   p = {result.p_value:.4f}",
        "",
        f"  {ctrl.label:<12s} n={ctrl.n:>8,}   value {ctrl.value:>12,.4f}",
        f"  {treat.label:<12s} n={treat.n:>8,}   value {treat.value:>12,.4f}",
        "",
        f"  estimator: {result.estimator}   inference: {result.inference}",
    ]
    if result.strata_used:
        lines.append(f"  strata: {', '.join(result.strata_used)}")
    if result.covariates_used:
        lines.append(f"  covariates: {', '.join(result.covariates_used)}")

    if result.influence is not None:
        cs = result.confidence_sequence()
        lines += [
            "",
            "  if you have been monitoring this test, read this line instead:",
            f"  {cs}",
        ]
    else:
        lines += [
            "",
            "  no anytime-valid interval: the 'adjusted' estimator uses the bootstrap.",
            "  the p-value above assumes this is the only look you have taken.",
        ]

    if result.notes:
        lines += [""] + [f"  note: {n}" for n in result.notes]
    return "\n".join(lines)
