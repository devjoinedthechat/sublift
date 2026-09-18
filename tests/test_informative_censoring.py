"""Censoring that depends on the subscriber: detecting it, and what can be done about it.

The library's central censoring assumption is that subscribers are censored
because the data was cut, not because of anything they did. This file covers the
case where that is false.
"""

import numpy as np
import pytest

from sublift import PanelError, check_censoring, simulate_experiment
from sublift.censoring import censoring_person_period, conditional_censoring_survival

COVS = ["engagement", "plan", "tenure_bucket"]


@pytest.fixture(scope="module")
def administrative():
    return simulate_experiment(n=8000, seed=1, horizon=8, observation_window=12)


@pytest.fixture(scope="module")
def informative():
    return simulate_experiment(
        n=20_000,
        seed=1,
        horizon=8,
        observation_window=14,
        dropout_hazard=0.08,
        dropout_depends_on_engagement=2.4,
    )


def test_administrative_censoring_needs_no_test(administrative):
    """When potential follow-up is recorded the question is settled by construction."""
    check = check_censoring(administrative.panel)
    assert check.known_exactly
    assert check.ok
    assert "administrative" in str(check)


def test_informative_dropout_is_detected(informative):
    check = check_censoring(informative.panel, COVS)
    assert not check.known_exactly
    assert check.depends_on_covariates
    assert not check.ok
    assert check.lr_p_value < 1e-6
    assert "NOT INDEPENDENT" in str(check)


def test_the_detected_driver_is_the_real_one(informative):
    """Only engagement drives dropout in this simulation; the check should say so."""
    check = check_censoring(informative.panel, COVS)
    strongest = check.coefficients.loc[check.coefficients["coefficient"].abs().idxmax()]
    assert strongest["covariate"] == "engagement"
    assert strongest["coefficient"] > 0.5


def test_independent_dropout_is_not_flagged():
    """Dropout that is unrelated to the subscriber must not trip the alarm."""
    sim = simulate_experiment(
        n=20_000,
        seed=2,
        horizon=8,
        observation_window=14,
        dropout_hazard=0.08,
        dropout_depends_on_engagement=0.0,
    )
    check = check_censoring(sim.panel, COVS)
    assert not check.known_exactly
    assert not check.depends_on_covariates
    assert check.ok


def test_a_panel_without_covariates_cannot_be_checked():
    from sublift import SubscriberPanel

    sim = simulate_experiment(n=3000, seed=3, horizon=6, observation_window=10, dropout_hazard=0.05)
    bare = SubscriberPanel.from_periods(
        sim.frame,
        subject="subscriber_id",
        period="billing_period",
        churned="churned",
        arm="variant",
        control="control",
    )
    with pytest.raises(PanelError, match="without baseline covariates"):
        check_censoring(bare)


def test_churners_are_not_eligible_to_be_seen_censored(informative):
    """The tie again: churn preempts censoring, so it contributes no censoring row."""
    panel = informative.panel
    rows, period, censored = censoring_person_period(panel, 8)
    capped = np.minimum(panel.n_periods, 8)
    churned_within = panel.event & (panel.n_periods <= 8)

    counts = np.bincount(rows, minlength=panel.n_subjects)
    np.testing.assert_array_equal(counts, np.maximum(capped - churned_within, 0))
    # Every censoring event belongs to someone who was not observed to churn.
    assert not churned_within[rows[censored == 1]].any()


def test_conditional_weights_separate_subscribers(informative):
    """One marginal censoring curve is the wrong weight for everybody here."""
    gbar = conditional_censoring_survival(informative.panel, 8, COVS)
    engagement = informative.panel.covariates["engagement"].to_numpy()
    low = gbar[engagement < -1].mean(axis=0)
    high = gbar[engagement > 1].mean(axis=0)
    assert low[-1] > 0.5
    assert high[-1] < 0.1
    assert gbar.shape == (informative.panel.n_subjects, 8)


def test_conditional_weights_are_a_valid_survival_curve(informative):
    gbar = conditional_censoring_survival(informative.panel, 8, COVS)
    assert np.all(gbar <= 1.0 + 1e-12)
    assert np.all(gbar > 0)
    assert np.all(np.diff(gbar, axis=1) <= 1e-12)
    np.testing.assert_allclose(gbar[:, 0], 1.0)


def test_censoring_covariates_are_reported_in_the_result(informative):
    from sublift import retained_periods_lift

    res = retained_periods_lift(
        informative.panel,
        horizon=8,
        estimator="adjusted",
        covariates=COVS,
        censoring_covariates=COVS,
    )
    assert any("modelled on" in n for n in res.notes)
