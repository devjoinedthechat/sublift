"""Estimator wiring, invariants and guardrails. Statistical behaviour lives in test_validation.py."""

import numpy as np
import pytest

from sublift import incremental_ltv, retained_periods_lift, simulate_experiment


@pytest.fixture(scope="module")
def sim():
    return simulate_experiment(n=6000, seed=42, horizon=8, observation_window=12)


def test_horizon_defaults_to_the_follow_up_both_arms_have(sim):
    res = retained_periods_lift(sim.panel, estimator="unadjusted")
    assert res.horizon == sim.panel.followup


def test_horizon_past_the_data_is_refused_with_a_usable_message(sim):
    with pytest.raises(ValueError, match="exceeds the .* follow-up"):
        retained_periods_lift(sim.panel, horizon=sim.panel.followup + 5, estimator="unadjusted")


def test_extrapolation_is_possible_but_must_be_asked_for(sim):
    with pytest.warns(UserWarning):
        res = retained_periods_lift(
            sim.panel, horizon=sim.panel.followup + 2, estimator="unadjusted", allow_extrapolation=True
        )
    assert res.horizon == sim.panel.followup + 2


def test_flat_price_makes_ltv_a_rescaled_retention_number(sim):
    periods = retained_periods_lift(sim.panel, horizon=8, estimator="unadjusted")
    ltv = incremental_ltv(sim.panel, horizon=8, estimator="unadjusted", price=9.0)
    assert ltv.estimate == pytest.approx(9.0 * periods.estimate)
    assert ltv.se == pytest.approx(9.0 * periods.se)


def test_arm_specific_price_captures_a_discount(sim):
    """A save offer priced into the treatment arm must reduce measured LTV."""
    flat = incremental_ltv(sim.panel, horizon=8, estimator="unadjusted", price=10.0)
    discounted = incremental_ltv(
        sim.panel,
        horizon=8,
        estimator="unadjusted",
        price={"control": 10.0, "treatment": np.array([5.0] * 3 + [10.0] * 5)},
    )
    assert discounted.estimate < flat.estimate


def test_stratifying_on_a_constant_reproduces_the_unadjusted_estimate():
    s = simulate_experiment(n=4000, seed=9, horizon=6, observation_window=10)
    frame = s.frame.assign(everyone="all")
    from sublift import SubscriberPanel

    panel = SubscriberPanel.from_periods(
        frame,
        subject="subscriber_id",
        period="billing_period",
        churned="churned",
        arm="variant",
        control="control",
        covariates=["everyone"],
        revenue="revenue",
    )
    plain = retained_periods_lift(panel, horizon=6, estimator="unadjusted")
    strat = retained_periods_lift(panel, horizon=6, estimator="stratified", strata=["everyone"])
    assert strat.estimate == pytest.approx(plain.estimate, rel=1e-10)
    assert strat.se == pytest.approx(plain.se, rel=1e-10)


def test_stratified_requires_strata(sim):
    with pytest.raises(ValueError, match="needs strata"):
        retained_periods_lift(sim.panel, horizon=8, estimator="stratified")


def test_adjusted_requires_covariates(sim):
    with pytest.raises(ValueError, match="needs covariates"):
        retained_periods_lift(sim.panel, horizon=8, estimator="adjusted")


def test_ltv_without_revenue_or_price_is_an_error():
    s = simulate_experiment(n=1500, seed=4, horizon=6, observation_window=9)
    from sublift import SubscriberPanel

    bare = SubscriberPanel.from_periods(
        s.frame,
        subject="subscriber_id",
        period="billing_period",
        churned="churned",
        arm="variant",
        control="control",
    )
    with pytest.raises(ValueError, match="price="):
        incremental_ltv(bare, horizon=6, estimator="unadjusted")


def test_adjusted_reports_influence_based_inference_by_default(sim):
    res = retained_periods_lift(sim.panel, horizon=6, estimator="adjusted", covariates=["engagement", "plan"])
    assert res.inference == "influence"
    assert res.confidence_sequence().radius > 0


def test_adjusted_can_still_be_bootstrapped(sim):
    """Kept as an escape hatch: no asymptotics, at the cost of no confidence sequence."""
    res = retained_periods_lift(
        sim.panel,
        horizon=6,
        estimator="adjusted",
        covariates=["engagement", "plan"],
        inference="bootstrap",
        n_boot=15,
    )
    assert res.inference == "bootstrap"
    with pytest.raises(ValueError, match="stratified"):
        res.confidence_sequence()


def test_small_samples_are_warned_about(sim):
    """The one-step estimator's variance is first-order; say so rather than run narrow."""
    small = sim.panel.subset(np.arange(sim.panel.n_subjects) < 1500)
    with pytest.warns(UserWarning, match="large-sample approximation"):
        retained_periods_lift(small, horizon=6, estimator="adjusted", covariates=["engagement", "plan"])


def test_curves_are_monotone_and_cover_the_horizon(sim):
    res = retained_periods_lift(sim.panel, horizon=8, estimator="unadjusted")
    curves = res.curves()
    assert set(curves["arm"]) == set(sim.panel.arm_labels)
    for _, g in curves.groupby("arm"):
        assert (np.diff(g["survival"].to_numpy()) <= 1e-12).all()
        assert (np.diff(g["cumulative_value"].to_numpy()) >= -1e-12).all()
    assert curves["period"].max() == 8


def test_unknown_estimator_is_rejected(sim):
    with pytest.raises(ValueError, match="estimator must be"):
        retained_periods_lift(sim.panel, estimator="magic")


def test_tiny_strata_are_dropped_and_reported():
    s = simulate_experiment(n=2500, seed=17, horizon=6, observation_window=9)
    rng = np.random.default_rng(0)
    rare = np.where(rng.random(len(s.frame)) < 0.0004, "rare", "common")
    frame = s.frame.assign(bucket=rare)
    frame["bucket"] = frame.groupby("subscriber_id")["bucket"].transform("first")
    from sublift import SubscriberPanel

    panel = SubscriberPanel.from_periods(
        frame,
        subject="subscriber_id",
        period="billing_period",
        churned="churned",
        arm="variant",
        control="control",
        covariates=["bucket"],
    )
    res = retained_periods_lift(panel, horizon=6, estimator="stratified", strata=["bucket"])
    assert any("dropped" in n for n in res.notes)
