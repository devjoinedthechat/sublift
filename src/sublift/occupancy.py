"""Expected periods *paid for*, when subscriptions come back.

The survival estimators answer "how long until they cancel". For a large part of
consumer subscriptions that is the wrong question, because a meaningful share of
cancellations are followed by a return: people leave for the summer, resubscribe
for a season, pause and come back. Reduced to a first cancellation, a subscriber
who paid for periods 1-3 and 6-12 is recorded as churning at period 3 -- nine
paid periods thrown away.

The direction of the resulting error is the opposite of what it looks like.
Ignoring returns does not understate the treatment; it **overstates** it, because
the subscribers it writes off are disproportionately in the control arm. A
control subscriber who cancelled in March and resubscribed in May was never
really lost, and counting them as lost inflates the gap. On this library's
simulator, holding everything else fixed, time-to-first-cancellation reports the
same number regardless of how many subscribers come back -- it cannot see them --
while the truth falls away beneath it::

    win-back hazard     true effect    time-to-first-cancellation
              0%           +0.2585        +0.2609   (  1% high)
              5%           +0.2352        +0.2609   ( 11% high)
             10%           +0.2149        +0.2609   ( 21% high)
             20%           +0.1814        +0.2609   ( 44% high)

The estimand that survives win-backs is the one the finance team already uses::

    A(H) = sum_{t=1}^{H} E[ w_t * 1{paying in period t} ]

expected billing periods paid for within the horizon, or expected revenue when
``w_t`` is a price. For a subscription that never returns this is the same
quantity the restricted mean survival time estimates, and on single-spell data
the two land within a fraction of a standard error of each other -- they are
different consistent estimators of one estimand, not the same arithmetic. The
product-limit is somewhat the more efficient of the two there, because it lets a
subscriber censored in period nine inform the hazard in period three, which a
period-by-period proportion cannot. That efficiency is what you give up in
exchange for an estimand that survives a subscriber coming back.

The estimator is a plain proportion per period, which makes it both simpler and
more robust than the product-limit -- no hazards, no cumulative product, and no
assumption that ending is terminal. What it does need is to know, for every
subscriber, how many periods they *could* have been observed for: without that,
"not paying in period nine" and "we have no idea about period nine" are the same
row. That is exactly what ``potential_followup`` records, so this estimator
requires it rather than guessing.
"""

from __future__ import annotations

import numpy as np

from .diagnostics import warn_on_srm
from .estimators import ArmSummary, LiftResult, _interval, _resolve_horizon
from .exceptions import NotIdentifiedError, PanelError
from .panel import SubscriberPanel

__all__ = ["occupancy_lift"]


def occupancy_lift(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    metric: str = "retained_periods",
    price=None,
    strata: list[str] | None = None,
    alpha: float = 0.05,
    expected_ratio: float = 0.5,
    allow_extrapolation: bool = False,
) -> LiftResult:
    """Incremental billing periods paid for, counting spells after the first.

    Use this instead of :func:`retained_periods_lift` when subscribers return.
    Build the panel with :meth:`SubscriberPanel.from_spells`.

    Parameters
    ----------
    strata
        Pre-assignment columns to stratify on, as elsewhere. Omit for the
        unstratified estimate.
    price
        A scalar, per-period array, or ``{"control": ..., "treatment": ...}``.
        With ``metric="ltv"`` and no price, per-subscriber revenue recorded on
        the panel is used, which is the natural choice here: a paused period
        earns nothing, and the grid already knows which periods those are.
    """
    if panel.active is None:
        raise PanelError(
            "This panel has no activity grid, so returning subscribers cannot be counted. "
            "Build it with SubscriberPanel.from_spells(...). If the first cancellation really "
            "is the outcome, retained_periods_lift() is the right function."
        )
    if panel.potential_followup is None:
        raise NotIdentifiedError(
            "occupancy_lift needs potential_followup: without it, a subscriber who is not "
            "paying in period nine is indistinguishable from one nobody has observed that far. "
            "from_spells() records it."
        )
    if panel.n_arms > 2:
        raise NotIdentifiedError(
            f"This panel has {panel.n_arms} arms; occupancy_lift compares two. Use panel.contrast('<arm>')."
        )
    if metric not in ("ltv", "retained_periods"):
        raise ValueError("metric must be 'ltv' or 'retained_periods'.")

    horizon = _resolve_horizon(panel, horizon, allow_extrapolation)
    randomization = warn_on_srm(panel, expected_ratio)
    notes: list[str] = []

    outcome = _outcome(panel, horizon, metric, price)
    observable = np.arange(1, horizon + 1)[None, :] <= panel.potential_followup[:, None]

    if strata:
        values, psi, control_psi = _stratified(panel, outcome, observable, horizon, strata, notes)
    else:
        values, psi, control_psi = _unstratified(panel, outcome, observable, horizon)

    estimate = values[1] - values[0]
    n = psi.size
    se = float(np.sqrt((psi**2).sum()) / n)
    ci = _interval(estimate, se, alpha, None)

    arms = {}
    for a in (0, 1):
        mask = panel.arm == a
        arms[panel.arm_labels[a]] = ArmSummary(
            label=panel.arm_labels[a],
            n=int(mask.sum()),
            value=values[a],
            survival=_curve(outcome[mask], observable[mask]),
            at_risk=observable[mask].sum(axis=0).astype(np.int64),
            weights=np.ones(horizon),
        )

    notes.append("occupancy estimand: periods paid for, including spells after the first")
    return LiftResult(
        estimator="occupancy" + (" (stratified)" if strata else ""),
        metric=metric,
        horizon=horizon,
        estimate=estimate,
        se=se,
        ci=ci,
        alpha=alpha,
        arms=arms,
        n_subjects=panel.n_subjects,
        inference="influence",
        influence=psi,
        control_influence=control_psi,
        strata_used=list(strata) if strata else None,
        randomization=randomization,
        notes=notes,
    )


# ----------------------------------------------------------------- internals


def _outcome(panel: SubscriberPanel, horizon: int, metric: str, price) -> np.ndarray:
    """What each subscriber earns in each period: one if paying, or the revenue."""
    active = panel.active[:, :horizon].astype(float)
    if active.shape[1] < horizon:
        active = np.pad(active, ((0, 0), (0, horizon - active.shape[1])))
    if metric == "retained_periods":
        return active

    if price is None:
        if panel.revenue is None:
            raise ValueError("No revenue on the panel and no price= given, so LTV cannot be formed.")
        revenue = np.nan_to_num(panel.revenue[:, :horizon], nan=0.0)
        if revenue.shape[1] < horizon:
            revenue = np.pad(revenue, ((0, 0), (0, horizon - revenue.shape[1])))
        return revenue * active

    schedule = np.zeros((panel.n_subjects, horizon))
    for a in (0, 1):
        per_arm = price[panel.arm_labels[a]] if isinstance(price, dict) else price
        row = (
            np.full(horizon, float(per_arm))
            if np.isscalar(per_arm)
            else np.asarray(per_arm, dtype=float)[:horizon]
        )
        if row.size < horizon:
            raise ValueError(f"price schedule covers {row.size} periods, horizon is {horizon}.")
        schedule[panel.arm == a] = row
    return schedule * active


def _curve(outcome: np.ndarray, observable: np.ndarray) -> np.ndarray:
    """Mean outcome per period among subscribers who could have been observed in it."""
    counts = observable.sum(axis=0)
    totals = (outcome * observable).sum(axis=0)
    return np.divide(totals, counts, out=np.zeros_like(totals, dtype=float), where=counts > 0)


def _arm_value(outcome, observable, mask, n):
    """Value for one arm, and its per-subscriber influence on the whole-sample scale.

    Each period is a simple mean over the subscribers observable in it, so the
    influence function is the ordinary one for a mean -- no product-limit, and
    exact in finite samples rather than first-order.
    """
    horizon = outcome.shape[1]
    curve = _curve(outcome[mask], observable[mask])
    share = mask.sum() / n
    pi = observable[mask].sum(axis=0) / mask.sum()
    scale = np.divide(1.0, pi, out=np.zeros(horizon), where=pi > 0)

    residual = (outcome[mask] - curve[None, :]) * observable[mask]
    influence = (residual * scale[None, :]).sum(axis=1) / share
    return float(curve.sum()), influence


def _unstratified(panel, outcome, observable, horizon):
    n = panel.n_subjects
    values, psi, control_psi = {}, np.zeros(n), np.zeros(n)
    for a in (0, 1):
        mask = panel.arm == a
        value, influence = _arm_value(outcome, observable, mask, n)
        values[a] = value
        psi[mask] = (1 if a == 1 else -1) * influence
        if a == 0:
            control_psi[mask] = influence
    return values, psi, control_psi


def _stratified(panel, outcome, observable, horizon, strata, notes):
    if panel.covariates is None:
        raise PanelError("Panel carries no covariates; rebuild it with covariates=[...].")
    missing = [c for c in strata if c not in panel.covariates.columns]
    if missing:
        raise PanelError(f"Strata column(s) {missing} not in the panel's covariates.")

    as_str = panel.covariates[strata].astype(str)
    first = as_str[strata[0]]
    key = (
        first.str.cat([as_str[c] for c in strata[1:]], sep="|").to_numpy()
        if len(strata) > 1
        else first.to_numpy()
    )
    levels, codes = np.unique(key, return_inverse=True)

    n = panel.n_subjects
    psi, control_psi = np.zeros(n), np.zeros(n)
    deltas = np.zeros(len(levels))
    control_values = np.zeros(len(levels))
    stratum_values = np.zeros((2, len(levels)))
    shares = np.zeros(len(levels))
    kept = np.zeros(n, dtype=bool)

    for s in range(len(levels)):
        in_s = codes == s
        sizes = [int((in_s & (panel.arm == a)).sum()) for a in (0, 1)]
        if min(sizes) < 2:
            notes.append(f"stratum {levels[s]!r} dropped: {sizes[0]} control / {sizes[1]} treatment")
            continue
        kept |= in_s
        shares[s] = in_s.sum()
        for a in (0, 1):
            mask = in_s & (panel.arm == a)
            value, influence = _arm_value(outcome, observable, mask, int(in_s.sum()))
            stratum_values[a, s] = value
            psi[mask] = (1 if a == 1 else -1) * influence
            if a == 0:
                control_psi[mask] = influence
                control_values[s] = value
        deltas[s] = stratum_values[1, s] - stratum_values[0, s]

    total = shares.sum()
    if total == 0:
        raise NotIdentifiedError("Every stratum was too small to estimate; use coarser strata.")
    if kept.sum() < n:
        notes.append(f"{n - int(kept.sum())} of {n} subscribers fell in dropped strata")

    pi = shares / total
    values = {a: float(stratum_values[a] @ pi) for a in (0, 1)}
    estimate = float(deltas @ pi)
    control_total = float(control_values @ pi)

    psi = psi[kept] + (deltas[codes[kept]] - estimate)
    control_psi = control_psi[kept] + (control_values[codes[kept]] - control_total)
    return values, psi, control_psi
