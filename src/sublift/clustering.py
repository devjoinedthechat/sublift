"""Variance when the randomised unit is not the analysed unit.

Every interval in this library is built from an influence function, and the step
from influence values to a standard error assumes the subscribers are
independent. Often they are not. A household shares a card and a decision; an
account carries several subscriptions; a B2B seat count moves together. And the
randomisation is usually done at the level that *is* independent -- the account --
while the analysis is done per subscription, because that is what churns.

Nothing in the data announces this. The estimate stays correct; only the interval
is wrong, and it is wrong in the direction that matters, because dependence
within a cluster means the effective sample size is the number of clusters rather
than the number of rows. On a simulated base of three-subscription households the
reported interval comes out 13% too narrow, with no warning, which is exactly the
failure mode worth engineering against.

The correction is the standard one for an asymptotically linear estimator: sum
the influence values within each cluster and treat those sums as the independent
units. Passing ``cluster=`` to a panel constructor turns it on everywhere at
once -- contrasts, stratification, competing risks, occupancy, segments, arms,
and the confidence sequences, which then count clusters rather than subscribers.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cluster_sums", "influence_se", "sequence_terms"]


def cluster_sums(influence: np.ndarray, codes: np.ndarray | None) -> np.ndarray:
    """Influence values summed within cluster, or returned untouched if there are none.

    Works on a single influence vector or a stack of them, one row per comparison,
    so the segment and multi-arm covariance matrices get the same treatment as a
    scalar contrast.
    """
    if codes is None:
        return influence
    values = np.asarray(influence, dtype=float)
    groups = int(codes.max()) + 1 if codes.size else 0
    if values.ndim == 1:
        return np.bincount(codes, weights=values, minlength=groups)
    return np.vstack([np.bincount(codes, weights=row, minlength=groups) for row in values])


def influence_se(influence: np.ndarray, codes: np.ndarray | None, n: int) -> float:
    """Standard error from influence values, clustered if the panel says so.

    The scaling is unchanged by clustering: the estimator is still an average over
    ``n`` subscribers, so it is still ``sqrt(sum of squares) / n``. What changes is
    what gets squared -- a cluster's total rather than each member's share.
    """
    terms = cluster_sums(influence, codes)
    return float(np.sqrt((terms**2).sum()) / n)


def sequence_terms(influence: np.ndarray, codes: np.ndarray | None, n: int) -> np.ndarray:
    """Independent units for a confidence sequence, rescaled to keep its mean.

    A confidence sequence is a statement about a mean of i.i.d. terms, so with
    clustering the terms are cluster totals and there are as many of them as there
    are clusters -- not subscribers. Rescaling by ``n_clusters / n`` leaves the
    mean equal to the estimate while the count, which drives the boundary, becomes
    the honest one.
    """
    terms = cluster_sums(influence, codes)
    if codes is None:
        return terms
    return terms * (terms.size / n)
