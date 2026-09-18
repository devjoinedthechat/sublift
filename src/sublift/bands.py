"""Uncertainty on the survival curves themselves.

The first thing anyone does with a retention experiment is plot the two curves.
Without bands that plot is an assertion: two lines that look different, with
nothing on the page saying whether they are. It is also the plot most likely to
be screenshotted into a deck, so it is worth getting the uncertainty onto it.

Two kinds of band, and the difference matters more here than almost anywhere
else. A **pointwise** interval is right for one period chosen in advance. A
**simultaneous** band is right for looking at the curve, which is what a plot
invites — and people do not look at a curve and then make a statement about
period seven; they look for where the lines separate. That is a search over
every period, and the pointwise interval does not cover it.

The two are computed together, because the only honest way to show a pointwise
band is next to the simultaneous one that says what it costs to have looked.

Cheaply
-------
The influence function of ``S(t)`` collapses, as the rest of this library's do::

    IF_i(S(t)) = -S(t) * C_i(t),  C_i(t) = event_i*1{k<=t}*a(k) - cumulative[min(k, t)]

where ``k`` is the subscriber's last observed period. So ``C_i(t)`` depends on the
subscriber only through ``(k, event)`` -- at most ``2H`` distinct values however
many million subscribers there are. The whole covariance across periods is then a
sum over those groups: ``O(H^2)`` memory and work, independent of the base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .clustering import cluster_sums
from .exceptions import NotIdentifiedError
from .family import calibrate, correlation
from .panel import SubscriberPanel
from .survival import fit_survival

__all__ = ["survival_curves"]


def survival_curves(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    alpha: float = 0.05,
    correction: str = "max-t",
    allow_extrapolation: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-arm survival curves with pointwise intervals and simultaneous bands.

    Returns one row per arm and period, plus rows for the difference between arms,
    which is usually the curve worth plotting: it is the one whose distance from
    zero is the finding.

    Columns are ``survival``, ``se``, ``ci_low``/``ci_high`` (pointwise) and
    ``band_low``/``band_high`` (simultaneous across every period shown). Use the
    band whenever the plot is being *read* rather than one pre-chosen period being
    quoted, which is nearly always.
    """
    if panel.n_arms > 2:
        raise NotIdentifiedError(
            f"This panel has {panel.n_arms} arms; use panel.contrast('<arm>') to pick a pair."
        )
    horizon = int(horizon) if horizon is not None else panel.followup
    if horizon > panel.followup and not allow_extrapolation:
        raise NotIdentifiedError(
            f"horizon={horizon} exceeds the {panel.followup} periods of follow-up both arms have."
        )

    n = panel.n_subjects
    periods = np.arange(1, horizon + 1)
    frames, curves, covariances = [], {}, {}

    for a, label in enumerate(panel.arm_labels):
        mask = panel.arm == a
        codes = None
        if panel.cluster is not None:
            codes = np.unique(panel.cluster[mask], return_inverse=True)[1].astype(np.int64)
        survival, cov = _arm_covariance(
            panel.n_periods[mask], panel.event[mask], horizon, codes, allow_extrapolation
        )
        curves[label] = survival
        covariances[label] = cov
        frames.append(_rows(label, periods, survival, cov, alpha, correction, seed))

    control, treatment = panel.arm_labels
    difference = curves[treatment] - curves[control]
    # Arms are independent, so the contrast's covariance is the sum of theirs.
    frames.append(
        _rows(
            "difference",
            periods,
            difference,
            covariances[treatment] + covariances[control],
            alpha,
            correction,
            seed,
        )
    )
    frame = pd.concat(frames, ignore_index=True)
    frame.attrs["n_subjects"] = n
    frame.attrs["alpha"] = alpha
    return frame


def _rows(label, periods, values, cov, alpha, correction, seed):
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    pointwise = float(stats.norm.ppf(1 - alpha / 2))
    critical = pointwise
    if correction != "none" and len(periods) > 1 and np.any(se > 0):
        critical, _ = calibrate(
            correction,
            correlation(cov),
            np.zeros(len(periods)),
            np.ones(len(periods)),
            alpha,
            len(periods),
            seed,
        )
    return pd.DataFrame(
        {
            "arm": label,
            "period": periods,
            "survival": values,
            "se": se,
            "ci_low": values - pointwise * se,
            "ci_high": values + pointwise * se,
            "band_low": values - critical * se,
            "band_high": values + critical * se,
        }
    )


def _arm_covariance(n_periods, event, horizon, codes, allow_extrapolation):
    """Survival curve and the covariance of its estimate across periods."""
    fitted = fit_survival(n_periods, event, horizon, allow_extrapolation=allow_extrapolation)
    survival = fitted.survival
    n = n_periods.size

    denominator = fitted.at_risk_fraction * (1.0 - fitted.hazard)
    scale = np.divide(1.0, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    cumulative = np.concatenate(([0.0], np.cumsum(fitted.hazard * scale)))

    last = np.minimum(n_periods, horizon)
    failed = event & (n_periods <= horizon)

    if codes is None:
        # C_i(t) depends on the subscriber only through (last period, failed), so the
        # whole covariance is a sum over at most 2H groups rather than over subscribers.
        weights = np.zeros((horizon, 2))
        for k, f in zip(last, failed, strict=True):
            weights[k - 1, int(f)] += 1.0
        design = np.empty((horizon * 2, horizon))
        counts = np.empty(horizon * 2)
        row = 0
        for k in range(1, horizon + 1):
            for f in (0, 1):
                design[row] = _c_values(k, bool(f), horizon, scale, cumulative)
                counts[row] = weights[k - 1, f]
                row += 1
        gram = design.T @ (design * counts[:, None])
    else:
        terms = np.empty((int(codes.max()) + 1, horizon)) if codes.size else np.zeros((0, horizon))
        per_subject = np.empty((n, horizon))
        for t in range(1, horizon + 1):
            per_subject[:, t - 1] = failed * (last <= t) * scale[last - 1] - cumulative[np.minimum(last, t)]
        terms = cluster_sums(per_subject.T, codes).T
        gram = terms.T @ terms

    return survival, np.outer(survival, survival) * gram / (n**2)


def _c_values(k, failed, horizon, scale, cumulative):
    """The accumulated martingale term for a subscriber last seen in period ``k``."""
    periods = np.arange(1, horizon + 1)
    return failed * (k <= periods) * scale[k - 1] - cumulative[np.minimum(k, periods)]
