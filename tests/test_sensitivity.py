"""How wrong the censoring assumption would have to be to change the answer."""

import numpy as np
import pytest

from sublift import (
    NotIdentifiedError,
    censoring_sensitivity,
    retained_periods_lift,
    simulate_experiment,
)
from sublift.sensitivity import _imputed_rmst
from sublift.survival import fit_survival


@pytest.fixture(scope="module")
def sim():
    return simulate_experiment(n=30_000, seed=5, horizon=10, observation_window=14, treatment_odds_ratio=0.90)


def test_the_imputation_is_the_product_limit_exactly():
    """What makes the sweep readable against the headline instead of near it.

    Restricted mean survival time can be written as 'periods observed plus
    expected remaining', and with the expectation taken from the fitted curve
    that is not an approximation of the product-limit estimate -- it is the same
    number. The identity only holds with the right conditioning: the panel's
    convention puts a censored subscriber through that period's renewal decision,
    so the remaining tenure conditions on S(c), not S(c-1).
    """
    sim = simulate_experiment(n=50_000, seed=1, horizon=10, observation_window=15)
    panel = sim.panel
    for arm in (0, 1):
        mask = panel.arm == arm
        product_limit = fit_survival(panel.n_periods[mask], panel.event[mask], 10).rmst()
        imputed = _imputed_rmst(panel.n_periods[mask], panel.event[mask], 10, np.array([1.0]))
        assert imputed[0] == pytest.approx(product_limit, abs=1e-12)


def test_a_tiny_hand_checkable_case():
    """Six subscribers, horizon three, worked out by hand."""
    n_periods = np.array([1, 1, 1, 2, 3, 3])
    event = np.array([True, True, False, True, False, False])
    # The censored subscriber paid one period; conditional on surviving it their
    # expected total is 1 + S(1)/S(1) + S(2)/S(1).
    assert _imputed_rmst(n_periods, event, 3, np.array([1.0]))[0] == pytest.approx(
        fit_survival(n_periods, event, 3).rmst(), abs=1e-12
    )


def test_gamma_one_reproduces_the_headline(sim):
    result = censoring_sensitivity(sim.panel, horizon=10, n_boot=0)
    plain = retained_periods_lift(sim.panel, horizon=10, estimator="unadjusted")
    assert result.baseline.gamma == 1.0
    assert result.baseline.estimate == pytest.approx(plain.estimate, abs=1e-10)


def test_shrinking_the_favoured_arm_erodes_a_positive_effect(sim):
    result = censoring_sensitivity(sim.panel, horizon=10, n_boot=0)
    estimates = [p.estimate for p in result.points]
    assert estimates[0] > 0
    assert all(later <= earlier + 1e-12 for earlier, later in zip(estimates, estimates[1:], strict=False))


def test_it_defaults_to_the_unfavourable_direction(sim):
    """A sensitivity analysis that only makes the result stronger is not one."""
    result = censoring_sensitivity(sim.panel, horizon=10, n_boot=0)
    plain = retained_periods_lift(sim.panel, horizon=10, estimator="unadjusted")
    assert result.arm == ("treatment" if plain.estimate > 0 else "control")


def test_the_tipping_point_is_where_the_sign_flips(sim):
    result = censoring_sensitivity(sim.panel, horizon=10, n_boot=0)
    tipping = result.tipping_estimate
    assert tipping is not None
    by_gamma = {p.gamma: p.estimate for p in result.points}
    assert by_gamma[tipping] <= 0
    above = [g for g in by_gamma if g > tipping]
    assert all(by_gamma[g] > 0 for g in above)


def test_bootstrapping_gives_an_interval_tipping_point(sim):
    result = censoring_sensitivity(sim.panel, horizon=10, n_boot=80)
    assert np.isfinite(result.baseline.se)
    assert result.tipping_interval is not None
    # Losing significance happens before losing the sign.
    assert result.tipping_interval >= result.tipping_estimate


def test_a_specific_arm_can_be_chosen(sim):
    result = censoring_sensitivity(sim.panel, horizon=10, arm="control", n_boot=0)
    assert result.arm == "control"
    # Shrinking the control arm makes a positive effect look larger.
    assert result.points[-1].estimate > result.points[0].estimate


def test_an_unknown_arm_is_rejected(sim):
    with pytest.raises(ValueError, match="not one of"):
        censoring_sensitivity(sim.panel, horizon=10, arm="nonexistent", n_boot=0)


def test_multi_arm_panels_are_sent_back():
    from sublift import simulate_multi_arm

    multi = simulate_multi_arm(n=9_000, effects={"a": 0.9, "b": 1.0}, seed=1)
    with pytest.raises(NotIdentifiedError, match="contrast"):
        censoring_sensitivity(multi.panel, horizon=8, n_boot=0)


def test_summary_states_what_it_hands_back(sim):
    text = censoring_sensitivity(sim.panel, horizon=10, n_boot=60).summary()
    assert "gamma" in text
    assert "This is the part the numbers hand back" in text


def test_frame_covers_the_sweep(sim):
    frame = censoring_sensitivity(sim.panel, horizon=10, n_boot=0).to_frame()
    assert frame["gamma"].iloc[0] == 1.0
    assert frame["gamma"].is_monotonic_decreasing


def test_review_flags_a_fragile_result():
    """A weak effect with heavy censoring is exactly what should be caught."""
    from sublift import review

    weak = simulate_experiment(n=30_000, seed=9, horizon=12, observation_window=14, treatment_odds_ratio=0.97)
    result = review(weak.panel, horizon=12, strata=["plan"])
    assert any("Fragile to censoring" in f.title for f in result.findings)
    assert result.sensitivity is not None


def test_review_does_not_cry_wolf_on_a_solid_result():
    solid = simulate_experiment(n=60_000, seed=3, horizon=8, observation_window=20, treatment_odds_ratio=0.75)
    from sublift import review

    result = review(solid.panel, horizon=8, strata=["plan"])
    assert not any("Fragile to censoring" in f.title for f in result.findings)
