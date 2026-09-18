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

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .clustering import influence_se
from .diagnostics import warn_on_srm
from .estimators import ArmSummary, LiftResult, _interval, _resolve_horizon
from .exceptions import NotIdentifiedError, PanelError
from .panel import SubscriberPanel, stratum_codes

__all__ = ["occupancy_lift", "occupancy_decomposition", "OccupancyDecomposition", "LapseCause"]


def occupancy_lift(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    metric: str = "retained_periods",
    price=None,
    strata: list[str] | None = None,
    covariates: list[str] | None = None,
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
    covariates
        Pre-assignment columns to adjust for instead. Fits an outcome model per
        period and augments it with the observed residuals -- the same one-step
        idea as the covariate-adjusted survival estimator, but far simpler here
        because each period is an ordinary mean rather than a product-limit. The
        augmentation keeps it consistent under randomization whether or not the
        outcome model is any good, so the model only ever costs variance, never
        correctness. Prefer it to ``strata`` when the useful covariates are
        continuous.
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

    if strata and covariates:
        raise ValueError(
            "Give strata= or covariates=, not both. Stratifying and adjusting on the same "
            "information twice does not reduce variance twice."
        )
    if covariates:
        values, psi, control_psi = _adjusted(panel, outcome, observable, horizon, covariates)
    elif strata:
        values, psi, control_psi = _stratified(panel, outcome, observable, horizon, strata, notes)
    else:
        values, psi, control_psi = _unstratified(panel, outcome, observable, horizon)

    estimate = values[1] - values[0]
    n = psi.size
    se = influence_se(psi, _codes_for(panel, psi), n)
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
        estimator="occupancy" + (" (stratified)" if strata else "") + (" (adjusted)" if covariates else ""),
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
        cluster=_codes_for(panel, psi),
        strata_used=list(strata) if strata else None,
        covariates_used=list(covariates) if covariates else None,
        randomization=randomization,
        notes=notes,
    )


# ----------------------------------------------------------------- internals


def _codes_for(panel, psi):
    """Cluster codes, restricted when stratification dropped some subscribers."""
    if panel.cluster is None:
        return None
    if psi.size == panel.n_subjects:
        return panel.cluster
    return None  # a dropped stratum leaves the mapping ambiguous; fall back to independent


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

    codes, levels = stratum_codes(panel.covariates, strata)

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


def _adjusted(panel, outcome, observable, horizon, covariates):
    """Per-period augmented estimator: an outcome model, corrected by its own residuals.

    For each period and arm, fit ``E[Y(t) | X]`` on the subscribers who could be
    observed in that period, predict it for everybody, then add back the observed
    residuals reweighted by how often that combination is seen::

        p_a(t) = mean_i [ m_a(t, X_i) + 1{A_i=a, observed}(Y_i(t) - m_a(t, X_i)) / (p_a pi(t)) ]

    The second term is what makes the model optional: under randomization its
    expectation is the estimation error of the first, so a poor model widens the
    interval and does not move the estimate. The influence function is exact --
    each period is a mean, not a product-limit -- so unlike the survival version
    this one needs no large-sample caveat.
    """
    from .logistic import design_matrix

    if panel.covariates is None:
        raise PanelError("Panel carries no covariates; rebuild it with covariates=[...].")
    missing = [c for c in covariates if c not in panel.covariates.columns]
    if missing:
        raise PanelError(f"Covariate(s) {missing} not in the panel's covariates.")

    features, _, _ = design_matrix(panel.covariates[covariates])
    n = panel.n_subjects
    design = np.column_stack([np.ones(n), features]) if features.size else np.ones((n, 1))

    values, psi, control_psi = {}, np.zeros(n), np.zeros(n)
    for a in (0, 1):
        in_arm = panel.arm == a
        share = float(in_arm.mean())
        influence = np.zeros(n)
        total = 0.0

        for t in range(horizon):
            seen = observable[:, t]
            fitted = _predict_period(design, outcome[:, t], in_arm & seen)
            pi = float(seen.mean())
            if pi <= 0:
                continue
            residual = np.zeros(n)
            rows = in_arm & seen
            residual[rows] = (outcome[rows, t] - fitted[rows]) / (share * pi)

            level = float(fitted.mean() + residual.mean())
            total += level
            influence += fitted - fitted.mean() + residual - residual.mean()

        values[a] = total
        psi += (1 if a == 1 else -1) * influence
        if a == 0:
            control_psi = influence
    return values, psi, control_psi


def _predict_period(design, y, rows):
    """Least-squares fit on ``rows``, predicted for everybody.

    Normal equations with a pseudo-inverse: the design is a handful of columns, and
    a collinear covariate (an all-one level inside a thin period) should widen the
    interval rather than raise.
    """
    if not rows.any():
        return np.zeros(design.shape[0])
    sub = design[rows]
    gram = sub.T @ sub
    coefficients = np.linalg.pinv(gram) @ (sub.T @ y[rows])
    return design @ coefficients


@dataclass(frozen=True)
class LapseCause:
    label: str
    periods_lost_control: float
    periods_lost_treatment: float
    estimate: float
    se: float
    ci: tuple[float, float]


@dataclass
class OccupancyDecomposition:
    """Periods paid for, split by what was keeping subscribers away."""

    horizon: int
    total: float
    total_se: float
    total_ci: tuple[float, float]
    causes: list[LapseCause]
    arm_labels: tuple[str, ...]
    alpha: float
    n_subjects: int

    def share(self, cause: LapseCause) -> float:
        return cause.estimate / self.total if self.total else float("nan")

    def summary(self) -> str:
        head = f"Periods paid for, by what was keeping subscribers away ({self.horizon} periods)"
        width = max(len(c.label) for c in self.causes)
        lines = [
            head,
            "=" * len(head),
            f"  total  {self.total:+.4f} periods per subscriber "
            f"[{self.total_ci[0]:+.4f}, {self.total_ci[1]:+.4f}]",
            "",
        ]
        for cause in self.causes:
            lines.append(
                f"  {cause.label:<{width}s}  {cause.estimate:+.4f}  "
                f"[{cause.ci[0]:+.4f}, {cause.ci[1]:+.4f}]   {self.share(cause):>6.0%} of the effect"
            )
        lines += [
            "",
            "  Causes sum to the total exactly. Unlike the first-spell split, a subscriber",
            "  can appear under more than one cause across the horizon -- cancelled in the",
            "  spring, card failed in the autumn -- and the periods are attributed to",
            "  whichever ending they were living under at the time.",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def occupancy_decomposition(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    alpha: float = 0.05,
    allow_extrapolation: bool = False,
) -> OccupancyDecomposition:
    """Split the incremental periods paid for by the cause of each lapse.

    The first-spell decomposition in :func:`sublift.churn_decomposition` attributes
    a subscriber to one cause, because it only ever sees them end once. Over a
    horizon in which people leave and return, one subscriber can lapse for
    different reasons at different times, and the periods belong to whichever
    ending they were living under.

    The split is exact for the same reason as the survival one: in every period a
    subscriber is either paying or not, and if not, exactly one ending is the most
    recent. So ``horizon = A + sum_j L_j`` and the contrast decomposes with no
    residual. Build the panel with ``from_spells(..., spell_cause=...)``.
    """
    if panel.lapsed_cause is None:
        raise PanelError(
            "This panel does not record why subscribers lapsed. Rebuild it with "
            "from_spells(..., spell_cause='<column>')."
        )
    if panel.potential_followup is None:
        raise NotIdentifiedError("occupancy_decomposition needs potential_followup.")

    horizon = _resolve_horizon(panel, horizon, allow_extrapolation)
    observable = np.arange(1, horizon + 1)[None, :] <= panel.potential_followup[:, None]
    grid = panel.lapsed_cause[:, :horizon]
    n = panel.n_subjects
    z = float(stats.norm.ppf(1 - alpha / 2))

    causes, total_psi, total = [], np.zeros(n), 0.0
    for index, label in enumerate(panel.lapsed_labels):
        outcome = (grid == index).astype(float)
        values, psi, _ = _unstratified(panel, outcome, observable, horizon)
        # Periods *saved* is minus the change in periods lost, so the sign flips.
        estimate = -(values[1] - values[0])
        psi = -psi
        se = influence_se(psi, panel.cluster, n)
        causes.append(
            LapseCause(
                label=label,
                periods_lost_control=values[0],
                periods_lost_treatment=values[1],
                estimate=estimate,
                se=se,
                ci=(estimate - z * se, estimate + z * se),
            )
        )
        total += estimate
        total_psi += psi

    total_se = influence_se(total_psi, panel.cluster, n)
    return OccupancyDecomposition(
        horizon=horizon,
        total=total,
        total_se=total_se,
        total_ci=(total - z * total_se, total + z * total_se),
        causes=causes,
        arm_labels=panel.arm_labels,
        alpha=alpha,
        n_subjects=n,
    )
