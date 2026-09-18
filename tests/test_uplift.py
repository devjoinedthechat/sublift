"""Targeting: cross-fitted scores and the Qini curve."""

import numpy as np
import pytest

from sublift import qini, simulate_experiment, uplift_scores

COVS = ["engagement", "plan", "tenure_bucket"]


@pytest.fixture(scope="module")
def flat():
    """Constant odds ratio: no effect modification on the logit scale."""
    return simulate_experiment(n=10_000, seed=77, horizon=8, observation_window=12)


@pytest.fixture(scope="module")
def heterogeneous():
    """The intervention lands much harder on engaged subscribers, and backfires on others."""
    return simulate_experiment(
        n=10_000, seed=77, horizon=8, observation_window=12, effect_modification=1.5
    )


def test_every_subscriber_is_scored_out_of_fold(flat):
    scores = uplift_scores(flat.panel, horizon=8, covariates=COVS, n_folds=5)
    assert scores.shape == (flat.panel.n_subjects,)
    assert not np.isnan(scores).any()


def test_scores_track_the_true_individual_effect(heterogeneous):
    scores = uplift_scores(heterogeneous.panel, horizon=8, covariates=COVS)
    r = float(np.corrcoef(scores, heterogeneous.true_individual_rmst_lift)[0, 1])
    assert r > 0.85, f"correlation {r:.2f} with the true individual effect"


def test_shrinkage_beats_a_t_learner_when_there_is_no_effect_modification(flat):
    """The reason the pooled learner is the default: a T-learner fits noise here."""
    truth = flat.true_individual_rmst_lift
    pooled = uplift_scores(flat.panel, horizon=8, covariates=COVS, learner="interaction")
    naive = uplift_scores(flat.panel, horizon=8, covariates=COVS, learner="t")
    r_pooled = float(np.corrcoef(pooled, truth)[0, 1])
    r_naive = float(np.corrcoef(naive, truth)[0, 1])
    assert r_pooled > r_naive + 0.2, f"pooled {r_pooled:.2f} vs t-learner {r_naive:.2f}"
    # The T-learner does not merely rank worse; it invents heterogeneity that is not there.
    assert naive.std() > 2 * truth.std()
    assert pooled.std() < naive.std()


def test_shrinkage_does_not_destroy_real_heterogeneity(heterogeneous):
    """Cross-validated penalty has to relax when the data really is heterogeneous."""
    truth = heterogeneous.true_individual_rmst_lift
    pooled = uplift_scores(heterogeneous.panel, horizon=8, covariates=COVS, learner="interaction")
    assert pooled.std() > 0.6 * truth.std()
    assert float(np.corrcoef(pooled, truth)[0, 1]) > 0.9


def test_cross_fitting_changes_the_scores(flat):
    """If fold assignment made no difference, nothing was being held out."""
    a = uplift_scores(flat.panel, horizon=8, covariates=COVS, seed=0)
    b = uplift_scores(flat.panel, horizon=8, covariates=COVS, seed=1)
    assert not np.allclose(a, b)


def test_qini_curve_is_well_formed(heterogeneous):
    curve = qini(
        heterogeneous.panel, horizon=8, covariates=COVS, fractions=np.array([0.25, 0.5, 0.75, 1.0])
    )
    f = curve.frame
    assert list(f["fraction"]) == [0.25, 0.5, 0.75, 1.0]
    assert f["n_targeted"].is_monotonic_increasing
    # Targeting everyone is the overall effect, by construction.
    assert f["effect_in_targeted"].iloc[-1] == pytest.approx(curve.overall_effect, rel=1e-9)
    assert f["random_targeting"].iloc[-1] == pytest.approx(curve.overall_effect, rel=1e-9)


def test_targeting_the_best_half_beats_targeting_everyone(heterogeneous):
    curve = qini(heterogeneous.panel, horizon=8, covariates=COVS, fractions=np.array([0.5]))
    assert curve.frame["effect_in_targeted"].iloc[0] > curve.overall_effect
    assert curve.qini_auc is not None


def test_fewer_than_two_folds_is_rejected(flat):
    with pytest.raises(ValueError, match="at least 2"):
        uplift_scores(flat.panel, horizon=8, covariates=COVS, n_folds=1)


def test_unknown_covariate_is_rejected(flat):
    with pytest.raises(ValueError, match="not in the panel"):
        uplift_scores(flat.panel, horizon=8, covariates=["nonexistent"])


def test_unknown_learner_is_rejected(flat):
    with pytest.raises(ValueError, match="learner must be"):
        uplift_scores(flat.panel, horizon=8, covariates=COVS, learner="magic")


def test_bad_prior_sd_is_rejected(flat):
    with pytest.raises(ValueError, match="positive number"):
        uplift_scores(flat.panel, horizon=8, covariates=COVS, interaction_prior_sd=-1.0)
