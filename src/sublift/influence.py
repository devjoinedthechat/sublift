"""Influence functions for the weighted-survival value.

Every interval sublift reports comes from here. The influence function of an
estimator is the per-subject contribution to its sampling error,

    theta_hat - theta  ~=  (1/n) sum_i IF_i,      Var(theta_hat) = (1/n^2) sum_i IF_i^2

which buys three things at once: standard errors that do not assume the
Greenwood-plus-delta-method chain of approximations, a contrast variance that is
just a sum across independent arms, and -- the reason it is worth deriving by
hand -- a sequence of i.i.d. per-subject terms, which is precisely what an
anytime-valid confidence sequence needs. Bootstrap standard errors would give
the first two and not the third.

For the product-limit survival curve the influence function is standard:

    IF_i(S(t)) = -S(t) * sum_{s<=t} [dN_i(s) - Y_i(s) h(s)] / [pi(s) (1 - h(s))]

with ``Y_i(s) = 1{subject i paid for period s}`` and ``dN_i(s) = 1{s was their
last paid period}``. Summing against the weights and exchanging the order of
summation collapses the double sum to a single pass:

    IF_i(V) = -sum_s [dN_i(s) - Y_i(s) h(s)] / [pi(s)(1-h(s))] * G(s)
    G(s)    = sum_{t>s}^{H} w_t S(t-1)

When the revenue weights are themselves estimated from the data rather than
supplied as a known price schedule, they carry their own sampling error, and a
second term appears. Ignoring it understates the variance of any LTV readout
where the treatment moves realized revenue -- discounts, downgrades, win-back
pricing -- which is most of the interesting ones.
"""

from __future__ import annotations

import numpy as np

from .survival import DiscreteSurvival, _as_weights

__all__ = ["value_influence", "contrast_influence"]


def value_influence(
    surv: DiscreteSurvival,
    n_periods: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray | float = 1.0,
    *,
    revenue: np.ndarray | None = None,
) -> np.ndarray:
    """Per-subject influence values for ``weighted_value(surv, weights)``.

    Pass ``revenue`` (the subject-by-period matrix) when ``weights`` were
    estimated from that same data, so the weights' own sampling error is
    included. Leave it ``None`` for a known price schedule.

    Written without ever forming a subject-by-period array. Both terms of the
    sum collapse once you notice what the indicators are: a subscriber
    contributes their event term in exactly one period, and their at-risk term in
    a prefix of periods. So the whole thing is a lookup into two arrays of length
    ``horizon``::

        IF_i = -event_i * scale[n_i]  +  cumulative[min(n_i, H)]

    That is the difference between 440 MB and 8 MB at a million subscribers,
    which is the difference between this running on a real subscriber base and
    not. It is also several times faster, because nothing is allocated.
    """
    n_periods = np.asarray(n_periods, dtype=np.int64)
    event = np.asarray(event, dtype=bool)
    horizon = surv.horizon
    w = _as_weights(weights, horizon)

    s_lag = surv.survival_lagged
    # G(s) = sum_{t>s} w_t S(t-1): reverse cumulative sum of the per-period contributions,
    # excluding period s itself, so G(H) = 0.
    contrib = w * s_lag
    G = np.concatenate((np.cumsum(contrib[::-1])[::-1][1:], [0.0]))

    pi = surv.at_risk_fraction
    denom = pi * (1.0 - surv.hazard)
    # A period with an empty risk set, or a hazard of exactly 1, contributes nothing:
    # there is no further survival mass for it to perturb.
    scale = np.divide(G, denom, out=np.zeros_like(G), where=denom > 0)

    capped = np.minimum(n_periods, horizon)
    # sum_s Y_i(s) h(s) scale(s), a prefix sum evaluated at each subject's last period.
    at_risk_total = np.concatenate(([0.0], np.cumsum(surv.hazard * scale)))
    inf = at_risk_total[capped]

    # sum_s dN_i(s) scale(s): non-zero only for subscribers observed to churn by the horizon.
    churned = event & (n_periods <= horizon)
    inf[churned] -= scale[n_periods[churned] - 1]

    if revenue is not None:
        inf = inf + _revenue_influence(revenue, n_periods, capped, pi, s_lag, w, horizon)
    return inf


def _revenue_influence(
    revenue: np.ndarray,
    n_periods: np.ndarray,
    capped: np.ndarray,
    pi: np.ndarray,
    s_lag: np.ndarray,
    w: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """The term contributed by estimating w_t as the mean revenue among the at-risk.

    The observed-revenue half genuinely needs the subscriber-by-period matrix --
    revenue varies per subscriber, so there is nothing to collapse -- but it is a
    single matrix-vector product rather than a chain of temporaries. The expected
    half is a prefix sum like the rest.
    """
    revenue = np.asarray(revenue, dtype=float)
    width = min(horizon, revenue.shape[1])
    scale = np.divide(s_lag, pi, out=np.zeros_like(s_lag), where=pi > 0)

    observed = np.nan_to_num(revenue[:, :width], nan=0.0) @ scale[:width]
    expected = np.concatenate(([0.0], np.cumsum(w * scale)))[capped]
    return observed - expected


def contrast_influence(
    inf_treat: np.ndarray,
    inf_control: np.ndarray,
    n_total: int,
) -> np.ndarray:
    """Stack two within-arm influence vectors into one i.i.d. sequence over all subjects.

    Rescaling by the arm shares ``p_a = n_a/n`` puts both arms on the whole-sample
    scale, so ``mean(psi)`` estimates the contrast and ``Var(mean(psi))`` reproduces
    ``Var_treat + Var_control`` exactly. Returned in a single vector because the
    sequential layer consumes subjects in arrival order, not arm by arm.
    """
    p1 = inf_treat.size / n_total
    p0 = inf_control.size / n_total
    return np.concatenate((inf_treat / p1, -inf_control / p0))
