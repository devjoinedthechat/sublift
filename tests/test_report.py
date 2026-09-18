"""Text and notebook rendering. A number without its context is how a wrong decision gets made."""

import numpy as np
import pytest

from sublift import (
    NotIdentifiedError,
    PanelError,
    SubliftError,
    churn_decomposition,
    incremental_ltv,
    retained_periods_lift,
    simulate_experiment,
)


@pytest.fixture(scope="module")
def sim():
    return simulate_experiment(
        n=12_000, seed=8, horizon=8, observation_window=12, involuntary_hazard=0.015, price=12.0
    )


def test_summary_states_the_horizon_and_the_estimator(sim):
    text = retained_periods_lift(sim.panel, horizon=8, strata=["plan"]).summary()
    assert "8 billing periods" in text
    assert "stratified" in text


def test_summary_surfaces_the_anytime_valid_interval(sim):
    """The honest number when you have been watching should be hard to miss."""
    text = retained_periods_lift(sim.panel, horizon=8, strata=["plan"]).summary()
    assert "anytime-valid" in text
    assert "monitoring this test" in text


def test_summary_warns_when_there_is_no_sequential_option(sim):
    text = retained_periods_lift(
        sim.panel,
        horizon=8,
        estimator="adjusted",
        covariates=["engagement", "plan"],
        inference="bootstrap",
        n_boot=15,
    ).summary()
    assert "only look you have taken" in text


def test_relative_lift_is_reported_with_an_interval(sim):
    text = incremental_ltv(sim.panel, horizon=8, strata=["plan"]).summary()
    assert "relative to control" in text
    assert text.count("%") >= 3  # point estimate plus both interval ends


def test_html_is_well_formed_and_escapes_labels(sim):
    html = retained_periods_lift(sim.panel, horizon=8, strata=["plan"])._repr_html_()
    assert html.startswith("<div") and html.endswith("</div>")
    assert html.count("<div") == html.count("</div>")
    assert "<script" not in html


def test_html_does_not_hardcode_a_background(sim):
    """Notebook themes vary; a fixed background is unreadable in half of them."""
    html = retained_periods_lift(sim.panel, horizon=8, strata=["plan"])._repr_html_()
    assert "background" not in html


def test_decomposition_renders_both_ways(sim):
    d = churn_decomposition(sim.panel, horizon=8)
    assert "voluntary" in str(d)
    html = d._repr_html_()
    assert html.startswith("<div") and "involuntary" in html


def test_html_escapes_arm_labels_from_data():
    """Arm labels come from user data and land in HTML; they must not be able to inject markup."""
    sim = simulate_experiment(n=2500, seed=2, horizon=6, observation_window=9)
    frame = sim.frame.copy()
    frame["variant"] = np.where(frame["variant"] == "control", "<script>x</script>", "treated")
    from sublift import SubscriberPanel

    panel = SubscriberPanel.from_periods(
        frame,
        subject="subscriber_id",
        period="billing_period",
        churned="churned",
        arm="variant",
        covariates=["plan"],
    )
    html = retained_periods_lift(panel, horizon=6, estimator="unadjusted")._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_error_taxonomy_separates_malformed_from_unanswerable(sim):
    with pytest.raises(NotIdentifiedError):
        retained_periods_lift(sim.panel, horizon=99, estimator="unadjusted")
    plain = simulate_experiment(n=1500, seed=3, horizon=6, observation_window=9)
    with pytest.raises(PanelError):
        churn_decomposition(plain.panel, horizon=6)


def test_errors_remain_catchable_as_value_errors(sim):
    """Existing `except ValueError` handlers must keep working."""
    assert issubclass(NotIdentifiedError, ValueError)
    assert issubclass(PanelError, ValueError)
    assert issubclass(PanelError, SubliftError)
    with pytest.raises(ValueError):
        retained_periods_lift(sim.panel, horizon=99, estimator="unadjusted")
