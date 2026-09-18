"""Correcting across a family the library cannot infer for you."""

import pytest

from sublift import correct_family, incremental_ltv, retained_periods_lift, simulate_experiment


@pytest.fixture(scope="module")
def sim():
    return simulate_experiment(
        n=60_000,
        seed=4,
        horizon=10,
        observation_window=15,
        involuntary_hazard=0.015,
        price=12.0,
        treatment_odds_ratio=0.93,
    )


def metrics(panel):
    return {
        "retained periods": retained_periods_lift(panel, horizon=10, estimator="unadjusted"),
        "LTV": incremental_ltv(panel, horizon=10, estimator="unadjusted", price=12.0),
    }


def test_near_duplicate_metrics_are_priced_as_one_comparison(sim):
    """With a flat price, LTV is retained periods scaled -- literally the same statistic."""
    family = correct_family(metrics(sim.panel))
    assert family.mean_correlation > 0.98
    # Bonferroni would charge for two independent looks; max-t charges for barely one.
    assert family.critical_value < 0.9 * family.bonferroni_critical_value


def test_simultaneous_intervals_are_never_tighter_than_per_comparison(sim):
    family = correct_family(metrics(sim.panel))
    for member in family.members:
        assert member.ci[0] <= member.marginal_ci[0]
        assert member.ci[1] >= member.marginal_ci[1]
        assert member.adjusted_p_value >= member.p_value - 1e-9


@pytest.mark.parametrize("correction", ["max-t", "holm", "bonferroni", "none"])
def test_every_correction_runs(sim, correction):
    family = correct_family(metrics(sim.panel), correction=correction)
    assert len(family.members) == 2
    assert family.critical_value > 0


def test_arms_crossed_with_segments(sim):
    """Three offers over two segments is six comparisons, not three plus two."""
    from sublift import simulate_multi_arm

    multi = simulate_multi_arm(n=48_000, effects={"a": 0.85, "b": 0.95, "c": 1.0}, seed=2)
    family = {}
    for arm in multi.panel.treatment_labels:
        pair = multi.panel.contrast(arm)
        for plan in ("monthly", "annual"):
            mask = (pair.covariates["plan"].astype(str) == plan).to_numpy()
            family[f"{arm} / {plan}"] = retained_periods_lift(
                pair.subset(mask), horizon=8, estimator="unadjusted", allow_extrapolation=True
            )
    with pytest.raises(ValueError, match="different numbers of subscribers"):
        correct_family(family)


def test_a_family_of_one_is_refused(sim):
    with pytest.raises(ValueError, match="at least two"):
        correct_family({"only": retained_periods_lift(sim.panel, horizon=10, estimator="unadjusted")})


def test_bootstrap_results_cannot_join_a_family(sim):
    boot = retained_periods_lift(
        sim.panel,
        horizon=10,
        estimator="adjusted",
        covariates=["engagement", "plan"],
        inference="bootstrap",
        n_boot=15,
    )
    family = metrics(sim.panel)
    family["bootstrapped"] = boot
    with pytest.raises(ValueError, match="no influence function"):
        correct_family(family)


def test_frame_and_summary_cover_every_member(sim):
    family = correct_family(metrics(sim.panel))
    assert len(family.to_frame()) == 2
    text = family.summary()
    assert "retained periods" in text and "LTV" in text
