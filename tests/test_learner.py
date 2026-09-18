"""Flexible nuisance models, cross-fitted."""

import numpy as np
import pytest

import sublift as sl

COVARIATES = ["engagement", "plan", "tenure_bucket"]


class Memoriser:
    """A deliberately over-flexible classifier, to prove cross-fitting is doing its job.

    Bins every column finely and stores the cell mean. Fitted and read on the same
    rows it overfits badly; cross-fitted it cannot, because the cells it memorised
    were built without the subscriber being predicted. Stands in for a gradient
    booster without adding a dependency.
    """

    def __init__(self, bins=12):
        self.bins = bins

    def _cells(self, X):
        binned = np.clip(((X - self.low) / self.span * self.bins).astype(int), 0, self.bins - 1)
        return (binned * (self.bins ** np.arange(X.shape[1]))).sum(axis=1)

    def fit(self, X, y):
        self.low = X.min(axis=0)
        self.span = np.maximum(X.max(axis=0) - self.low, 1e-9)
        self.prior = float(y.mean())
        cells = self._cells(X)
        self.table = {}
        for cell in np.unique(cells):
            mask = cells == cell
            self.table[cell] = (y[mask].sum() + 2 * self.prior) / (mask.sum() + 2)
        return self

    def predict_proba(self, X):
        p = np.array([self.table.get(c, self.prior) for c in self._cells(X)])
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def sim():
    return sl.simulate_experiment(n=20_000, seed=5, horizon=8, observation_window=12)


def test_a_custom_learner_is_accepted(sim):
    result = sl.retained_periods_lift(
        sim.panel, horizon=8, estimator="adjusted", covariates=COVARIATES, learner=Memoriser()
    )
    assert result.inference == "influence"
    assert result.se > 0


def test_it_lands_near_the_parametric_answer(sim):
    parametric = sl.retained_periods_lift(sim.panel, horizon=8, estimator="adjusted", covariates=COVARIATES)
    flexible = sl.retained_periods_lift(
        sim.panel, horizon=8, estimator="adjusted", covariates=COVARIATES, learner=Memoriser()
    )
    assert abs(flexible.estimate - parametric.estimate) < 3 * parametric.se


def test_fold_assignment_changes_the_estimate(sim):
    """If the folds made no difference, nothing was being held out."""
    first = sl.retained_periods_lift(
        sim.panel,
        horizon=8,
        estimator="adjusted",
        covariates=COVARIATES,
        learner=Memoriser(),
        seed=0,
    )
    second = sl.retained_periods_lift(
        sim.panel,
        horizon=8,
        estimator="adjusted",
        covariates=COVARIATES,
        learner=Memoriser(),
        seed=1,
    )
    assert first.estimate != second.estimate


def test_it_still_supports_a_confidence_sequence(sim):
    result = sl.retained_periods_lift(
        sim.panel, horizon=8, estimator="adjusted", covariates=COVARIATES, learner=Memoriser()
    )
    assert result.confidence_sequence().radius > 0


@pytest.mark.slow
def test_cross_fitting_keeps_an_over_flexible_learner_unbiased():
    """The property that separates double machine learning from using machine learning.

    A learner this flexible, fitted and read on the same rows, would carry a bias
    of the same order as the effect. Cross-fitted, it matches the parametric
    estimator.
    """
    reps = 40
    parametric, flexible = [], []
    truth = None
    for r in range(reps):
        sim = sl.simulate_experiment(n=8000, seed=3000 + r, horizon=8, observation_window=12)
        truth = sim.true_rmst_lift
        parametric.append(
            sl.retained_periods_lift(
                sim.panel, horizon=8, estimator="adjusted", covariates=COVARIATES
            ).estimate
        )
        flexible.append(
            sl.retained_periods_lift(
                sim.panel,
                horizon=8,
                estimator="adjusted",
                covariates=COVARIATES,
                learner=Memoriser(),
                n_folds=5,
            ).estimate
        )

    flexible = np.array(flexible)
    mc_se = flexible.std(ddof=1) / np.sqrt(reps)
    assert abs(flexible.mean() - truth) < 3.5 * mc_se
    assert abs(flexible.mean() - np.mean(parametric)) < 3.5 * mc_se
