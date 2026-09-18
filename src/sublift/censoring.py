"""How much follow-up each subscriber could possibly have had.

Efficient estimation of a censored contrast needs the censoring distribution
``Gbar(s) = P(C >= s)`` -- the chance a subscriber is still *observable* in
period ``s``, as distinct from still subscribed. It appears as an inverse weight
in the augmentation term of the efficient influence function, correcting for the
fact that late-enrolling subscribers contribute to early periods only.

For a retention experiment this quantity is usually not something to estimate at
all. Censoring is administrative: a subscriber's potential follow-up is fixed
the day they enter the experiment, by the distance from their assignment date to
the data cut. That is known for *everyone*, including the subscribers who
churned long before the cut, so ``Gbar`` can be computed exactly rather than
inferred from the people who happened to survive.

:meth:`SubscriberPanel.from_spans` records it, because it has the dates. When it
is absent -- a panel built from period counts, where potential follow-up was
never captured -- this module falls back on the reverse Kaplan-Meier estimator,
treating censoring as the event. That is correct under independent censoring and
strictly noisier, and the fallback is reported rather than silent.
"""

from __future__ import annotations

import warnings

import numpy as np

from .exceptions import PanelError
from .panel import SubscriberPanel

__all__ = [
    "censoring_survival",
    "conditional_censoring_survival",
    "censoring_person_period",
    "CensoringWarning",
]

# Below this, inverse-censoring weights start to dominate the variance and the
# augmentation term becomes numerically unreliable.
FRAGILE_FOLLOWUP = 0.05


class CensoringWarning(UserWarning):
    """The horizon is long relative to how much follow-up the experiment has."""


def censoring_survival(
    panel: SubscriberPanel,
    horizon: int,
    *,
    warn: bool = True,
) -> tuple[np.ndarray, str]:
    """``Gbar[s-1] = P(C >= s)`` for ``s = 1..horizon``, and how it was obtained.

    Returns
    -------
    (gbar, source)
        ``gbar[0]`` is 1 by construction: everyone is observable in the period
        they are assigned. ``source`` is ``"exact"`` when potential follow-up was
        recorded, ``"reverse-km"`` when it had to be estimated.
    """
    if panel.potential_followup is not None:
        potential = np.asarray(panel.potential_followup, dtype=np.int64)
        gbar = np.array([(potential >= s).mean() for s in range(1, horizon + 1)], dtype=float)
        source = "exact"
    else:
        gbar = _reverse_km(panel.n_periods, panel.event, horizon)
        source = "reverse-km"

    if warn and gbar.size and gbar[-1] < FRAGILE_FOLLOWUP:
        warnings.warn(
            f"Only {gbar[-1]:.1%} of subscribers could have been observed through period "
            f"{horizon}. Inverse-censoring weights that small make the covariate-adjusted "
            "estimator unstable and its interval unreliable. Shorten the horizon, or use "
            "estimator='stratified', which does not need these weights.",
            CensoringWarning,
            stacklevel=3,
        )
    return np.clip(gbar, 1e-10, None), source


def _reverse_km(n_periods: np.ndarray, event: np.ndarray, horizon: int) -> np.ndarray:
    """Kaplan-Meier on the censoring process: censoring is the event, churn is the censoring.

    The subtlety is the tie. A panel records ``n_periods = s, event = True`` when a
    subscriber both churned at ``s`` and would have run out of follow-up at ``s``:
    churn wins, because it is what was observed. So those subscribers are *not*
    available to be seen censored at ``s``, and the risk set for the censoring
    event has to exclude them. Dividing by the full risk set instead understates
    the censoring hazard and biases ``Gbar`` upward -- by three points at period
    eight on this library's own simulator, which is more than enough to matter in
    an inverse-censoring weight.
    """
    n_periods = np.asarray(n_periods, dtype=np.int64)
    event = np.asarray(event, dtype=bool)

    capped = np.minimum(n_periods, horizon + 1)
    tail = np.cumsum(np.bincount(capped, minlength=horizon + 2)[::-1])[::-1]
    at_risk = tail[1 : horizon + 1].astype(float)

    censored_at = np.bincount(n_periods[~event], minlength=horizon + 2)[1 : horizon + 1]
    churned_at = np.bincount(n_periods[event], minlength=horizon + 2)[1 : horizon + 1]

    eligible = at_risk - churned_at
    hazard = np.divide(censored_at, eligible, out=np.zeros(horizon), where=eligible > 0)

    # Gbar[s-1] = P(C >= s) = product of (1 - censoring hazard) over the periods before s.
    gbar = np.ones(horizon, dtype=float)
    if horizon > 1:
        gbar[1:] = np.cumprod(1.0 - hazard[: horizon - 1])
    return gbar


def censoring_person_period(panel: SubscriberPanel, horizon: int):
    """Rows on which a subscriber was eligible to be *seen* censored.

    One row per subscriber-period at risk, with the outcome being "lost to
    follow-up here". A subscriber who churned in period ``s`` contributes no row
    for ``s``: churn preempted censoring, so they were never eligible to be
    observed censored there. Including that row would understate the censoring
    hazard, which is the same tie that biases the reverse Kaplan-Meier estimator
    if it is ignored.
    """
    capped = np.minimum(panel.n_periods, horizon)
    # Drop the final period for subscribers whose last observed period was a churn.
    eligible = capped - (panel.event & (panel.n_periods <= horizon)).astype(np.int64)
    eligible = np.maximum(eligible, 0)

    rows = np.repeat(np.arange(panel.n_subjects), eligible)
    starts = np.concatenate(([0], np.cumsum(eligible)[:-1]))
    period = np.arange(int(eligible.sum())) - np.repeat(starts, eligible) + 1

    censored = np.zeros(period.size, dtype=float)
    ends = np.cumsum(eligible) - 1
    lost = (~panel.event) & (panel.n_periods <= horizon) & (eligible > 0)
    censored[ends[lost]] = 1.0
    return rows, period.astype(np.int64), censored


def conditional_censoring_survival(
    panel: SubscriberPanel,
    horizon: int,
    covariates: list[str],
) -> np.ndarray:
    """``Gbar(s-1 | X_i)`` for every subscriber, shape ``(n_subjects, horizon)``.

    Fits a discrete-time censoring hazard on the covariates and accumulates it.
    Needed when subscribers are lost to follow-up for reasons related to their
    own state, where a single marginal ``Gbar`` is the wrong weight for everyone.
    """
    from .logistic import design_matrix, fit_logistic

    if panel.covariates is None:
        raise PanelError("Panel carries no covariates; rebuild it with covariates=[...].")
    missing = [c for c in covariates if c not in panel.covariates.columns]
    if missing:
        raise PanelError(f"Covariate(s) {missing} not in the panel's covariates.")

    X, _, _ = design_matrix(panel.covariates[covariates])
    rows, period, censored = censoring_person_period(panel, horizon)
    if rows.size == 0 or censored.sum() == 0:
        return np.ones((panel.n_subjects, horizon))

    fit = fit_logistic(_time_design(period, horizon, X[rows]), censored)
    hazard = _predict_hazard(fit, X, horizon)
    survival = np.cumprod(1.0 - hazard, axis=1)
    # Gbar(s-1 | X): probability of still being observable *entering* period s.
    return np.concatenate((np.ones((panel.n_subjects, 1)), survival[:, :-1]), axis=1)


def _time_design(period: np.ndarray, horizon: int, X_rows: np.ndarray) -> np.ndarray:
    dummies = np.zeros((period.size, horizon))
    dummies[np.arange(period.size), period - 1] = 1.0
    return np.hstack([dummies, X_rows]) if X_rows.size else dummies


def _predict_hazard(fit, X: np.ndarray, horizon: int) -> np.ndarray:
    alpha = fit.coef[:horizon]
    gamma = fit.coef[horizon:]
    offset = X @ gamma if gamma.size else np.zeros(X.shape[0])
    return 1.0 / (1.0 + np.exp(-(alpha[None, :] + offset[:, None])))
