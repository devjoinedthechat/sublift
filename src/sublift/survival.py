"""Discrete-time survival: the primitive every estimator in sublift is built on.

Subscriptions do not die continuously. They die at renewal, so the hazard is a
spike train on billing boundaries and the natural object is the discrete hazard

    h(t) = P(T = t | T >= t)

read as: given the subscriber paid for period ``t``, the probability that ``t``
is the last period they pay for. Survival is the product-limit

    S(t) = P(T > t) = prod_{s <= t} (1 - h(s)),   S(0) = 1

and the quantity anyone actually wants is a *weighted* restricted mean,

    V(H) = sum_{t=1}^{H} w_t * S(t-1)

because "still paying in period t" is exactly the event that earns ``w_t``.
With ``w_t = 1`` this is restricted mean survival time -- expected periods
retained. With ``w_t`` a price it is lifetime value to horizon ``H``. One
primitive, both readouts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["DiscreteSurvival", "fit_survival", "weighted_value", "empirical_revenue_weights"]


@dataclass(frozen=True)
class DiscreteSurvival:
    """A fitted discrete-time survival curve over periods ``1..horizon``."""

    periods: np.ndarray  # 1..H
    at_risk: np.ndarray  # n(t): subjects who paid for period t
    events: np.ndarray  # d(t): subjects whose last paid period was t
    hazard: np.ndarray  # h(t)
    survival: np.ndarray  # S(t)
    n: int
    exhausted_at: int | None = None  # first period with an empty risk set, if any

    @property
    def horizon(self) -> int:
        return int(self.periods[-1])

    @property
    def survival_lagged(self) -> np.ndarray:
        """``[S(0), S(1), ..., S(H-1)]`` -- the probability of *earning* each period."""
        return np.concatenate(([1.0], self.survival[:-1]))

    @property
    def at_risk_fraction(self) -> np.ndarray:
        """``pi(t) = n(t)/n``, the observed at-risk share. Drives influence-function scaling."""
        return self.at_risk / self.n

    def rmst(self) -> float:
        """Restricted mean survival time: expected billing periods retained, to the horizon."""
        return float(self.survival_lagged.sum())


def fit_survival(
    n_periods: np.ndarray,
    event: np.ndarray,
    horizon: int,
    *,
    allow_extrapolation: bool = False,
) -> DiscreteSurvival:
    """Product-limit estimate of the discrete survival curve.

    Censoring is assumed *administrative*: subjects are censored because the
    data was cut, not because of anything correlated with their propensity to
    churn. That is the usual case for a retention experiment analysed at a
    fixed date, and it is the assumption the whole library rests on. Dropout
    that depends on the subscriber's state (a failed card that also predicts
    cancellation) violates it and is out of scope for v0.1.
    """
    n_periods = np.asarray(n_periods, dtype=np.int64)
    event = np.asarray(event, dtype=bool)
    n = n_periods.size
    if n == 0:
        raise ValueError("Cannot fit a survival curve to zero subjects.")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    periods = np.arange(1, horizon + 1, dtype=np.int64)
    # n(t) = #{i : n_periods_i >= t}. Capping at horizon+1 keeps subjects observed past the
    # horizon at risk throughout, then one reverse cumulative sum gives every period at once.
    capped = np.minimum(n_periods, horizon + 1)
    tail = np.cumsum(np.bincount(capped, minlength=horizon + 2)[::-1])[::-1]
    at_risk = tail[1 : horizon + 1].astype(np.int64)
    events = np.bincount(n_periods[event], minlength=horizon + 1)[1 : horizon + 1].astype(np.int64)

    empty = np.flatnonzero(at_risk == 0)
    exhausted_at = int(periods[empty[0]]) if empty.size else None
    if exhausted_at is not None:
        msg = (
            f"No subjects remain at risk from period {exhausted_at}, but horizon={horizon}. "
            "Beyond the data the curve is held flat, which assumes nobody ever churns again -- "
            "an assumption, not a measurement."
        )
        if not allow_extrapolation:
            raise ValueError(msg + " Lower the horizon, or pass allow_extrapolation=True.")
        warnings.warn(msg, stacklevel=2)

    with np.errstate(invalid="ignore", divide="ignore"):
        hazard = np.where(at_risk > 0, events / np.maximum(at_risk, 1), 0.0)
    survival = np.cumprod(1.0 - hazard)

    return DiscreteSurvival(
        periods=periods,
        at_risk=at_risk,
        events=events,
        hazard=hazard,
        survival=survival,
        n=int(n),
        exhausted_at=exhausted_at,
    )


def weighted_value(surv: DiscreteSurvival, weights: np.ndarray | float = 1.0) -> float:
    """``sum_t w_t * S(t-1)`` -- retained periods when ``weights=1``, LTV when they are prices."""
    w = _as_weights(weights, surv.horizon)
    return float((w * surv.survival_lagged).sum())


def empirical_revenue_weights(
    revenue: np.ndarray,
    n_periods: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Mean revenue per period among subjects actually at risk in that period.

    Estimated *per arm*, deliberately. A save offer that trades 50% off for
    three months against a lower churn rate changes both terms of
    ``sum_t w_t S(t-1)``, and an analysis that holds price fixed across arms
    books the retention win while hiding the discount that bought it.
    """
    revenue = np.asarray(revenue, dtype=float)
    n_periods = np.asarray(n_periods, dtype=np.int64)
    width = revenue.shape[1]
    out = np.zeros(horizon, dtype=float)
    for idx, t in enumerate(range(1, horizon + 1)):
        at_risk = n_periods >= t
        if t <= width and at_risk.any():
            vals = revenue[at_risk, t - 1]
            vals = vals[~np.isnan(vals)]
            out[idx] = float(vals.mean()) if vals.size else 0.0
    return out


def _as_weights(weights: np.ndarray | float, horizon: int) -> np.ndarray:
    if np.isscalar(weights):
        return np.full(horizon, float(weights))
    w = np.asarray(weights, dtype=float)
    if w.size < horizon:
        raise ValueError(f"Revenue weights cover {w.size} periods but the horizon is {horizon}.")
    return w[:horizon]
