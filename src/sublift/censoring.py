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

from .panel import SubscriberPanel

__all__ = ["censoring_survival", "CensoringWarning"]

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
