"""Randomization checks. These run before anyone is allowed to read an effect."""

import numpy as np
import pytest

from sublift import check_randomization, retained_periods_lift, simulate_experiment


@pytest.fixture(scope="module")
def sim():
    return simulate_experiment(n=20_000, seed=1, horizon=8, observation_window=12)


def test_a_clean_experiment_passes(sim):
    check = check_randomization(sim.panel)
    assert check.srm_p_value > 0.01
    assert not check.srm_flagged
    assert check.ok
    assert "No problems found" in str(check)


def test_a_broken_filter_is_caught(sim):
    """8% of the treatment arm silently dropped -- the classic logging bug."""
    rng = np.random.default_rng(0)
    drop = (sim.panel.arm == 1) & (rng.random(sim.panel.n_subjects) < 0.08)
    check = check_randomization(sim.panel.subset(~drop))
    assert check.srm_flagged
    assert check.srm_p_value < 1e-3
    assert "SAMPLE RATIO MISMATCH" in str(check)


def test_an_uneven_design_is_not_a_mismatch(sim):
    """A deliberate 90/10 holdout must not be reported as broken."""
    rng = np.random.default_rng(1)
    drop = (sim.panel.arm == 0) & (rng.random(sim.panel.n_subjects) < 0.8)
    uneven = sim.panel.subset(~drop)
    assert check_randomization(uneven, expected_ratio=0.5).srm_flagged
    observed = float((uneven.arm == 1).mean())
    assert not check_randomization(uneven, expected_ratio=observed).srm_flagged


def test_estimators_warn_on_sample_ratio_mismatch(sim):
    rng = np.random.default_rng(2)
    drop = (sim.panel.arm == 1) & (rng.random(sim.panel.n_subjects) < 0.1)
    broken = sim.panel.subset(~drop)
    with pytest.warns(UserWarning, match="Sample ratio mismatch"):
        res = retained_periods_lift(broken, horizon=8, estimator="unadjusted")
    assert "SAMPLE RATIO MISMATCH" in res.summary()


def test_balance_reports_standardized_differences(sim):
    check = check_randomization(sim.panel)
    assert check.balance is not None
    assert set(check.balance.columns) == {"covariate", "control_mean", "treatment_mean", "std_diff"}
    assert check.balance["std_diff"].abs().max() < 0.1
    assert check.imbalanced == []


def test_imbalance_is_surfaced(sim):
    """Drop most low-engagement subscribers from one arm and the balance check must notice."""
    eng = sim.panel.covariates["engagement"].to_numpy()
    rng = np.random.default_rng(3)
    drop = (sim.panel.arm == 1) & (eng < 0) & (rng.random(sim.panel.n_subjects) < 0.9)
    check = check_randomization(sim.panel.subset(~drop))
    assert "engagement" in check.imbalanced
    assert not check.ok


def test_bad_expected_ratio_is_rejected(sim):
    with pytest.raises(ValueError, match="expected_ratio"):
        check_randomization(sim.panel, expected_ratio=1.5)
