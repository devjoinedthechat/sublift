"""The planning path."""

import pytest

from sublift import duration_to_detect


@pytest.fixture(scope="module")
def plan():
    return duration_to_detect(
        arrivals_per_period=4000,
        horizon=6,
        baseline_hazard=0.08,
        treatment_odds_ratio=0.85,
        max_periods=12,
        pilot_n=8000,
    )


def test_longer_enrollment_means_more_subscribers_and_smaller_errors(plan):
    f = plan.frame
    assert f["subscribers"].is_monotonic_increasing
    assert f["se"].iloc[-1] < f["se"].iloc[0]
    assert f["fixed_power"].iloc[-1] >= f["fixed_power"].iloc[0]


def test_the_sequence_needs_at_least_as_long_as_a_single_look(plan):
    """Anytime-validity is bought with data; it should never look cheaper than a fixed test."""
    if plan.fixed_periods and plan.sequential_periods:
        assert plan.sequential_periods >= plan.fixed_periods


def test_sequential_detectable_effect_shrinks_with_enrollment(plan):
    assert plan.frame["sequential_detectable"].is_monotonic_decreasing


def test_planning_an_ltv_test_requires_a_price():
    with pytest.raises(ValueError, match="price"):
        duration_to_detect(arrivals_per_period=1000, horizon=6, metric="ltv", max_periods=7, pilot_n=2000)


def test_max_periods_below_horizon_is_rejected():
    with pytest.raises(ValueError, match="nothing to search"):
        duration_to_detect(arrivals_per_period=1000, horizon=12, max_periods=6, pilot_n=2000)


def test_summary_mentions_both_analysis_modes(plan):
    text = str(plan)
    assert "fixed-sample" in text and "anytime-valid" in text
