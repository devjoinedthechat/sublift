"""How the effect develops as the horizon extends."""

import numpy as np
import pytest

import sublift as sl


@pytest.fixture(scope="module")
def fading():
    """An offer whose effect decays: it buys time early and then stops."""
    return sl.simulate_experiment(
        n=80_000,
        seed=4,
        horizon=12,
        observation_window=18,
        treatment_odds_ratio=0.80,
        effect_decay=0.45,
    )


@pytest.fixture(scope="module")
def persisting():
    return sl.simulate_experiment(
        n=80_000,
        seed=4,
        horizon=12,
        observation_window=18,
        treatment_odds_ratio=0.90,
        effect_decay=0.0,
    )


def test_each_horizon_matches_the_ordinary_estimate(fading):
    curve = sl.lift_by_horizon(fading.panel, horizons=[3, 6, 9, 12])
    for point in curve.points:
        direct = sl.retained_periods_lift(fading.panel, horizon=point.horizon, estimator="unadjusted")
        assert point.estimate == pytest.approx(direct.estimate, rel=1e-10)
        assert point.se == pytest.approx(direct.se, rel=1e-10)


def test_nested_horizons_are_strongly_correlated(fading):
    """The twelve-period effect contains the six-period one, so they cannot be independent."""
    curve = sl.lift_by_horizon(fading.panel, horizons=[3, 6, 9, 12])
    assert curve.mean_correlation > 0.6
    assert curve.critical_value < curve.bonferroni_critical_value


def test_simultaneous_bands_are_wider_than_pointwise(fading):
    curve = sl.lift_by_horizon(fading.panel, horizons=[3, 6, 9, 12])
    for point in curve.points:
        assert point.band[0] <= point.ci[0]
        assert point.band[1] >= point.ci[1]


def test_a_decaying_effect_is_read_as_decaying(fading):
    curve = sl.lift_by_horizon(fading.panel, horizons=[3, 6, 9, 12])
    assert not curve.still_accumulating
    assert curve.points[curve.peak_index].horizon < 12
    assert curve.tail_share < 1.0
    assert "it buys early" in curve.summary()


def test_a_persisting_effect_is_read_as_still_growing(persisting):
    curve = sl.lift_by_horizon(persisting.panel, horizons=[3, 6, 9, 12])
    assert curve.still_accumulating
    assert curve.points[curve.peak_index].horizon == 12
    assert "lower bound" in curve.summary()


def test_the_shape_is_read_against_the_peak_not_the_start(fading):
    """Survival differences take time to open up, so the first stretch is nearly
    always the smallest even for an effect about to fade."""
    curve = sl.lift_by_horizon(fading.panel, horizons=[3, 6, 9, 12])
    rates = curve.per_period
    assert rates[0] < rates[curve.peak_index]
    assert curve.peak_index != 0


def test_per_period_rates_sum_back_to_the_total(fading):
    curve = sl.lift_by_horizon(fading.panel, horizons=[3, 6, 9, 12])
    spans = np.diff([0, *[p.horizon for p in curve.points]])
    assert float(curve.per_period @ spans) == pytest.approx(curve.points[-1].estimate, rel=1e-9)


def test_horizons_default_to_the_usable_follow_up(fading):
    curve = sl.lift_by_horizon(fading.panel)
    assert len(curve.points) >= 2
    assert curve.points[-1].horizon <= fading.panel.followup


def test_one_horizon_is_refused(fading):
    with pytest.raises(ValueError, match="at least two"):
        sl.lift_by_horizon(fading.panel, horizons=[6])


def test_a_horizon_past_the_data_is_refused(fading):
    with pytest.raises(ValueError, match="exceeds"):
        sl.lift_by_horizon(fading.panel, horizons=[6, 99])


def test_it_works_on_ltv_and_with_strata(fading):
    curve = sl.lift_by_horizon(
        fading.panel,
        horizons=[6, 12],
        metric="ltv",
        price=12.0,
        estimator="stratified",
        strata=["plan"],
    )
    assert len(curve.points) == 2
    assert curve.to_frame().shape[0] == 2
