"""Variance reduction from pre-period behaviour."""

import numpy as np
import pandas as pd
import pytest

import sublift as sl


def sticky(n=40_000, seed=0, signal=3.0):
    """Subscribers with an underlying stickiness that shows in the pre-period too."""
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, 2, size=n)
    latent = rng.normal(size=n)
    pre = np.maximum(np.round(6 + signal * latent + rng.normal(0, 1.0, size=n)), 0)
    lifetime = np.maximum(np.ceil(rng.exponential(np.exp(1.3 + 0.45 * latent + 0.12 * arm))).astype(int), 1)
    censor = rng.integers(3, 16, size=n)
    return pd.DataFrame(
        {
            "uid": np.arange(n),
            "variant": np.where(arm == 1, "treat", "ctrl"),
            "n": np.minimum(lifetime, censor),
            "ev": lifetime <= censor,
            "pre_tenure": pre,
            "noise": rng.normal(size=n),
            "plan": np.where(rng.random(n) < 0.3, "annual", "monthly"),
        }
    )


def build(frame):
    return sl.SubscriberPanel.from_subjects(
        frame,
        subject="uid",
        arm="variant",
        periods="n",
        event="ev",
        control="ctrl",
        covariates=["pre_tenure", "noise", "plan"],
    )


@pytest.fixture(scope="module")
def panel():
    return build(sticky())


def test_a_predictive_pre_period_column_cuts_the_variance(panel):
    base = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    reduced = sl.cuped(base, panel, ["pre_tenure"])
    assert reduced.se < base.se
    assert 1 - (reduced.se / base.se) ** 2 > 0.05


def test_a_column_that_predicts_nothing_costs_nothing(panel):
    base = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    reduced = sl.cuped(base, panel, ["noise"])
    assert reduced.se <= base.se * 1.001  # never worse than the original, up to noise


def test_the_estimand_is_unchanged(panel):
    """CUPED subtracts something with expectation zero; it must not move the answer."""
    base = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    reduced = sl.cuped(base, panel, ["pre_tenure"])
    assert abs(reduced.estimate - base.estimate) < 2 * base.se


def test_it_records_what_it_removed(panel):
    reduced = sl.cuped(
        sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted"), panel, ["pre_tenure"]
    )
    assert any("CUPED" in note for note in reduced.notes)


def test_min_reduction_declines_a_pointless_correction(panel):
    base = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    same = sl.cuped(base, panel, ["noise"], min_reduction=0.10)
    assert same.se == base.se
    assert same is base or not any("CUPED" in n for n in same.notes)


def test_it_works_on_estimators_that_are_hard_to_model(panel):
    """The reason to have it: it never touches the outcome model."""
    spells = sl.simulate_experiment(n=20_000, seed=7, horizon=8, observation_window=13, winback_hazard=0.10)
    base = sl.occupancy_lift(spells.panel, horizon=8)
    reduced = sl.cuped(base, spells.panel, ["engagement"])
    assert reduced.se <= base.se


def test_the_influence_function_is_updated_too(panel):
    base = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    reduced = sl.cuped(base, panel, ["pre_tenure"])
    assert not np.allclose(reduced.influence, base.influence)
    assert reduced.confidence_sequence().radius < base.confidence_sequence().radius


def test_a_bootstrap_result_is_refused(panel):
    boot = sl.retained_periods_lift(
        panel,
        horizon=8,
        estimator="adjusted",
        covariates=["pre_tenure"],
        inference="bootstrap",
        n_boot=10,
    )
    with pytest.raises(sl.NotIdentifiedError, match="influence function"):
        sl.cuped(boot, panel, ["pre_tenure"])


def test_an_unknown_column_is_rejected(panel):
    base = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    with pytest.raises(sl.PanelError, match="not in the panel"):
        sl.cuped(base, panel, ["nonexistent"])
