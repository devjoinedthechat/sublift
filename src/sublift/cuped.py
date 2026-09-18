"""Variance reduction from data recorded before the experiment started.

The strongest variance reducer available to an experiment is almost never a
cleverer estimator. It is the subscriber's own behaviour in the weeks before
randomisation: how long they had already been subscribed, what they paid, how
often they opened the app. Someone who was on their way out stays on their way
out, and subtracting that predictable part leaves less noise to see through.

This is CUPED (Deng, Xu, Kohavi & Walker, 2013), and the usual presentation is
for a difference in means. The estimand here is not a difference in means -- it
is a contrast of weighted survival functionals -- so the usual derivation does
not apply directly.

The influence function makes it apply anyway. Any estimator that exposes one is,
to first order, the mean of that influence function; and the imbalance in a
pre-period covariate between arms is itself such a mean, with influence

    phi_i = (A_i/p1 - (1-A_i)/p0) (X_i - Xbar)

whose expectation is zero because assignment is random. Subtracting a multiple of
something with expectation zero changes nothing in expectation and changes the
variance by a lot, and the multiple that minimises the variance is the least
squares one. So the correction is::

    estimate  ->  estimate - theta . (Xbar_treatment - Xbar_control)
    IF_i      ->  IF_i - theta . phi_i

which reduces the variance by the fraction of the influence function that
pre-period behaviour explains, and works identically for survival, occupancy,
competing risks or a segment, because all it needs is the influence function.

The covariates must be **pre-assignment**. A post-assignment quantity has a
non-zero imbalance in expectation, and subtracting it moves the estimate.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .clustering import cluster_sums, influence_se
from .exceptions import NotIdentifiedError, PanelError
from .logistic import design_matrix
from .panel import SubscriberPanel

__all__ = ["cuped"]


def cuped(result, panel: SubscriberPanel, covariates: list[str], *, min_reduction: float = 0.0):
    """Reduce the variance of an existing result using pre-period covariates.

    Parameters
    ----------
    result
        Anything with ``.estimate`` and ``.influence`` -- any estimator here
        except ``adjusted`` with ``inference="bootstrap"``.
    covariates
        Pre-assignment columns, ideally the pre-period version of the outcome
        itself. Tenure before the experiment, revenue in the previous quarter,
        engagement in the fortnight before assignment: the more the column
        predicts what a subscriber was going to do anyway, the more it removes.
    min_reduction
        Return the original result untouched unless the variance falls by at
        least this fraction. Useful in an automated pipeline where a covariate
        that explains nothing should not silently add a step.

    Returns
    -------
    A result of the same type, with ``estimate``, ``se``, ``ci`` and
    ``influence`` updated, and a note recording how much was removed.

    Notes
    -----
    Not a substitute for ``estimator="stratified"`` or ``"adjusted"``, and not
    additive with them either: all three use the same information, so applying
    this on top of an adjusted estimate usually buys little. It is most useful
    exactly where outcome modelling is awkward -- occupancy, competing risks, a
    segment -- because it never touches the outcome model.
    """
    influence = getattr(result, "influence", None)
    if influence is None:
        raise NotIdentifiedError(
            "This result carries no influence function, so there is nothing to project onto. "
            "Estimators using inference='bootstrap' do not expose one."
        )
    if panel.covariates is None:
        raise PanelError("Panel carries no covariates; rebuild it with covariates=[...].")
    missing = [c for c in covariates if c not in panel.covariates.columns]
    if missing:
        raise PanelError(f"Covariate(s) {missing} not in the panel's covariates.")

    influence = np.asarray(influence, dtype=float)
    if influence.size != panel.n_subjects:
        raise NotIdentifiedError(
            f"The result covers {influence.size} subscribers and the panel has "
            f"{panel.n_subjects}. Estimators that drop thin strata cannot be corrected this way."
        )

    features, _, _ = design_matrix(panel.covariates[covariates])
    if features.size == 0:
        return result

    n = panel.n_subjects
    share = float((panel.arm == 1).mean())
    if not 0 < share < 1:
        raise NotIdentifiedError("Both arms must be non-empty.")

    # Influence of the covariate imbalance between arms -- mean zero under randomisation.
    weights = np.where(panel.arm == 1, 1.0 / share, -1.0 / (1.0 - share))
    centred = features - features.mean(axis=0)
    phi = centred * weights[:, None]

    codes = getattr(result, "cluster", None)
    theta = _projection(influence, phi, codes)
    if theta is None:
        return result

    imbalance = features[panel.arm == 1].mean(axis=0) - features[panel.arm == 0].mean(axis=0)
    corrected = influence - phi @ theta
    new_se = influence_se(corrected, codes, n)

    reduction = 1.0 - (new_se / result.se) ** 2 if result.se else 0.0
    if reduction < min_reduction:
        return result

    estimate = float(result.estimate - theta @ imbalance)
    from scipy import stats

    z = float(stats.norm.ppf(1 - result.alpha / 2))
    note = (
        f"CUPED on {', '.join(covariates)}: variance down {reduction:.1%}, se {result.se:.4f} -> {new_se:.4f}"
    )
    return replace(
        result,
        estimate=estimate,
        se=new_se,
        ci=(estimate - z * new_se, estimate + z * new_se),
        influence=corrected,
        notes=[*result.notes, note],
    )


def _projection(influence, phi, codes):
    """Least-squares coefficients of the influence function on the imbalance terms.

    Fitted on cluster totals when the panel is clustered, because those are the
    independent units whose variance is being reduced.
    """
    y = cluster_sums(influence, codes)
    x = cluster_sums(phi.T, codes).T if codes is not None else phi
    gram = x.T @ x
    if not np.all(np.isfinite(gram)) or np.linalg.matrix_rank(gram) < gram.shape[0]:
        # A constant or collinear pre-period column explains nothing; leave the result alone.
        return None
    return np.linalg.solve(gram, x.T @ y)
