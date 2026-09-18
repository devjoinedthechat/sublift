"""Human-readable summaries.

The failure mode this guards against is a number reported without the context
that decides whether it means anything: the horizon, the censoring, and whether
anyone has been watching.
"""

from __future__ import annotations

__all__ = ["format_result", "html_result", "html_decomposition"]

_UNITS = {"ltv": "revenue", "retained_periods": "periods"}


def format_result(result) -> str:
    unit = _UNITS.get(result.metric, "")
    conf = f"{1 - result.alpha:.0%}"
    ctrl, treat = result.control, result.treatment

    lo, hi = result.ci
    sign = "+" if result.estimate >= 0 else ""
    head = f"{'Incremental LTV' if result.metric == 'ltv' else 'Incremental retained periods'} over {result.horizon} billing periods"

    rel_lo, rel_hi = result.relative_ci
    rel = (
        f"{result.relative:+.2%} [{rel_lo:+.2%}, {rel_hi:+.2%}]"
        if rel_lo == rel_lo
        else f"{result.relative:+.2%}"
    )
    lines = [
        head,
        "=" * len(head),
        f"  {sign}{result.estimate:,.4f} {unit} per subscriber",
        f"  {conf} CI [{lo:+,.4f}, {hi:+,.4f}]   se {result.se:,.4f}   p = {result.p_value:.4f}",
        f"  relative to {ctrl.label}: {rel}",
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

    check = getattr(result, "randomization", None)
    if check is not None and check.srm_flagged:
        lines += [
            "",
            f"  !! SAMPLE RATIO MISMATCH (p = {check.srm_p_value:.1e}). The arms are not the",
            "     sizes randomization should have produced, so nothing above is trustworthy.",
            "     Run sublift.check_randomization() and fix the pipeline first.",
        ]
    if result.notes:
        lines += [""] + [f"  note: {n}" for n in result.notes]
    return "\n".join(lines)


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# Notebook themes vary, so the palette is built from currentColor and transparency
# rather than fixed greys: it stays legible in light and dark without a media query.
_CELL = "padding:4px 10px 4px 0;border:0;text-align:left;vertical-align:baseline"
_MUTED = "opacity:.62"
_RULE = "border:0;border-top:1px solid currentColor;opacity:.18;margin:10px 0"


def html_result(result) -> str:
    """Compact HTML rendering of a :class:`LiftResult`, for notebooks."""
    unit = _UNITS.get(result.metric, "")
    title = "Incremental LTV" if result.metric == "ltv" else "Incremental retained periods"
    lo, hi = result.ci
    rel_lo, rel_hi = result.relative_ci
    conf = f"{1 - result.alpha:.0%}"
    sign = "+" if result.estimate >= 0 else "&minus;"

    rel = (
        f"{result.relative:+.2%} <span style='{_MUTED}'>[{rel_lo:+.2%}, {rel_hi:+.2%}]</span>"
        if rel_lo == rel_lo
        else f"{result.relative:+.2%}"
    )
    rows = [
        (f"{conf} interval", f"[{lo:+,.4f}, {hi:+,.4f}]"),
        ("standard error", f"{result.se:,.4f}"),
        (f"relative to {_esc(result.control.label)}", rel),
    ]
    for arm in result.arms.values():
        rows.append((f"{_esc(arm.label)} (n={arm.n:,})", f"{arm.value:,.4f}"))
    rows.append(("estimator", f"{_esc(result.estimator)} &middot; {_esc(result.inference)} inference"))

    body = "".join(
        f"<tr><td style='{_CELL};{_MUTED}'>{label}</td><td style='{_CELL}'>{value}</td></tr>"
        for label, value in rows
    )

    parts = [
        "<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;"
        "line-height:1.5;max-width:40rem'>",
        f"<div style='{_MUTED};font-size:12px'>{_esc(title)} over {result.horizon} billing periods</div>",
        f"<div style='font-size:22px;font-weight:600;margin:2px 0 8px'>"
        f"{sign}{abs(result.estimate):,.4f} <span style='{_MUTED};font-size:13px;font-weight:400'>"
        f"{_esc(unit)} per subscriber</span></div>",
        f"<table style='border-collapse:collapse'>{body}</table>",
    ]

    if result.influence is not None:
        cs = result.confidence_sequence()
        verdict = "excludes 0" if cs.excludes_zero else "includes 0"
        parts.append(
            f"<hr style='{_RULE}'><div style='font-size:12px'>"
            f"<span style='{_MUTED}'>if you have been monitoring this test &mdash; "
            f"anytime-valid {conf} CS:</span><br>"
            f"[{cs.lower:+,.4f}, {cs.upper:+,.4f}] ({verdict}), "
            f"{cs.peeking_cost:.2f}&times; the fixed-sample width</div>"
        )

    check = getattr(result, "randomization", None)
    if check is not None and check.srm_flagged:
        parts.append(
            f"<hr style='{_RULE}'><div style='font-size:12px;font-weight:600'>"
            f"&#9888; Sample ratio mismatch (p = {check.srm_p_value:.1e})</div>"
            f"<div style='font-size:12px;{_MUTED}'>The arms are not the sizes randomization "
            "should have produced, so nothing above is trustworthy.</div>"
        )
    if result.notes:
        notes = "<br>".join(_esc(n) for n in result.notes)
        parts.append(f"<div style='font-size:11px;{_MUTED};margin-top:8px'>{notes}</div>")

    parts.append("</div>")
    return "".join(parts)


def html_decomposition(decomposition) -> str:
    """Compact HTML rendering of a :class:`ChurnDecomposition`, for notebooks."""
    d = decomposition
    rows = "".join(
        f"<tr><td style='{_CELL}'>{_esc(c.label)}</td>"
        f"<td style='{_CELL};text-align:right'>{c.estimate:+.4f}</td>"
        f"<td style='{_CELL};{_MUTED}'>[{c.ci[0]:+.4f}, {c.ci[1]:+.4f}]</td>"
        f"<td style='{_CELL};text-align:right'>{d.share(c):.0%}</td></tr>"
        for c in d.causes
    )
    return (
        "<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;"
        "line-height:1.5;max-width:40rem'>"
        f"<div style='{_MUTED};font-size:12px'>Retention effect by cause of churn, over "
        f"{d.horizon} billing periods</div>"
        f"<div style='font-size:22px;font-weight:600;margin:2px 0 8px'>{d.total:+.4f} "
        f"<span style='{_MUTED};font-size:13px;font-weight:400'>periods per subscriber</span></div>"
        f"<table style='border-collapse:collapse'>{rows}</table>"
        f"<div style='font-size:11px;{_MUTED};margin-top:8px'>Causes sum to the total exactly; "
        "the split is an identity, not an attribution.</div></div>"
    )
