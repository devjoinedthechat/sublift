"""Uncertainty on the survival curves."""

import numpy as np
import pytest

import sublift as sl


@pytest.fixture(scope="module")
def sim():
    return sl.simulate_experiment(
        n=20_000, seed=2, horizon=8, observation_window=12, treatment_odds_ratio=0.85
    )


def test_it_returns_both_arms_and_their_difference(sim):
    frame = sl.survival_curves(sim.panel, horizon=8)
    assert set(frame["arm"]) == {"control", "treatment", "difference"}
    assert len(frame) == 3 * 8


def test_the_difference_is_the_difference(sim):
    frame = sl.survival_curves(sim.panel, horizon=8)
    by_arm = {a: g.sort_values("period")["survival"].to_numpy() for a, g in frame.groupby("arm")}
    np.testing.assert_allclose(by_arm["difference"], by_arm["treatment"] - by_arm["control"], atol=1e-12)


def test_simultaneous_bands_are_wider_than_pointwise(sim):
    """A plot invites looking at every period, which a pointwise interval does not cover."""
    frame = sl.survival_curves(sim.panel, horizon=8)
    assert (frame["band_low"] <= frame["ci_low"] + 1e-12).all()
    assert (frame["band_high"] >= frame["ci_high"] - 1e-12).all()
    assert (frame["band_high"] - frame["band_low"] > frame["ci_high"] - frame["ci_low"]).any()


def test_uncertainty_grows_with_the_horizon(sim):
    """Later periods rest on fewer subscribers still at risk."""
    frame = sl.survival_curves(sim.panel, horizon=8)
    for _, group in frame.groupby("arm"):
        se = group.sort_values("period")["se"].to_numpy()
        assert se[-1] > se[0]


def test_the_standard_errors_match_the_bootstrap(sim):
    """The covariance is computed by grouping rather than over subscribers; check it."""
    panel = sim.panel
    reported = sl.survival_curves(panel, horizon=8)
    reported = reported[reported["arm"] == "difference"].sort_values("period")["se"].to_numpy()

    rng = np.random.default_rng(0)
    draws = []
    for _ in range(150):
        resampled = panel.take(rng.integers(0, panel.n_subjects, size=panel.n_subjects))
        curve = sl.survival_curves(resampled, horizon=8, allow_extrapolation=True)
        curve = curve[curve["arm"] == "difference"].sort_values("period")
        draws.append(curve["survival"].to_numpy())
    boot = np.std(draws, axis=0, ddof=1)
    np.testing.assert_allclose(reported, boot, rtol=0.15)


def test_the_grouping_shortcut_does_not_change_the_answer(sim):
    """Clustered panels take a different code path; on singleton clusters they must agree."""
    panel = sim.panel
    plain = sl.survival_curves(panel, horizon=8)
    solo = sl.survival_curves(panel.take(np.arange(panel.n_subjects)), horizon=8)
    np.testing.assert_allclose(plain["se"], solo["se"], rtol=1e-10)

    from dataclasses import replace

    clustered = replace(panel, cluster=np.arange(panel.n_subjects))
    with_codes = sl.survival_curves(clustered, horizon=8)
    np.testing.assert_allclose(plain["se"], with_codes["se"], rtol=1e-9)


def test_clustering_widens_the_bands_when_there_is_correlation_to_find():
    """And leaves them alone when there is not, which is the other half of being right.

    Grouping independent subscribers arbitrarily must not inflate anything --
    cluster-robust errors equal the independent ones in expectation when nothing
    is shared within a cluster. Only genuine within-household correlation should
    widen them.
    """
    import pandas as pd

    rng = np.random.default_rng(0)
    accounts, per_account = 6_000, 3
    account = np.repeat(np.arange(accounts), per_account)
    arm = np.repeat(rng.integers(0, 2, size=accounts), per_account)
    shared = np.repeat(rng.normal(size=accounts), per_account)  # household frailty
    lifetime = np.maximum(np.ceil(rng.exponential(np.exp(1.6 + 0.6 * shared + 0.15 * arm))).astype(int), 1)
    censor = rng.integers(3, 14, size=account.size)
    frame = pd.DataFrame(
        {
            "sub": np.arange(account.size),
            "account": account,
            "variant": np.where(arm == 1, "treat", "ctrl"),
            "n": np.minimum(lifetime, censor),
            "ev": lifetime <= censor,
        }
    )

    def build(**kwargs):
        return sl.SubscriberPanel.from_subjects(
            frame, subject="sub", arm="variant", periods="n", event="ev", control="ctrl", **kwargs
        )

    naive = sl.survival_curves(build(), horizon=8)["se"].mean()
    clustered = sl.survival_curves(build(cluster="account"), horizon=8)["se"].mean()
    assert clustered > naive * 1.05

    # Independent subscribers grouped arbitrarily: no correlation, so no inflation.
    from dataclasses import replace

    independent = sl.simulate_experiment(n=20_000, seed=2, horizon=8, observation_window=12).panel
    plain = sl.survival_curves(independent, horizon=8)["se"].mean()
    grouped = sl.survival_curves(
        replace(independent, cluster=np.arange(independent.n_subjects) // 3), horizon=8
    )["se"].mean()
    assert 0.95 < grouped / plain < 1.05


def test_a_multi_arm_panel_is_sent_back():
    multi = sl.simulate_multi_arm(n=9_000, effects={"a": 0.9, "b": 1.0}, seed=1)
    with pytest.raises(sl.NotIdentifiedError, match="contrast"):
        sl.survival_curves(multi.panel, horizon=6)


def test_a_horizon_past_the_data_is_refused(sim):
    with pytest.raises(sl.NotIdentifiedError, match="exceeds"):
        sl.survival_curves(sim.panel, horizon=99)


@pytest.mark.slow
def test_pointwise_and_simultaneous_coverage_are_both_nominal():
    """The hard one: the simultaneous band must cover the *whole* curve 95% of the time."""
    reference = sl.simulate_experiment(
        n=1000, seed=0, horizon=8, observation_window=12, treatment_odds_ratio=0.85
    )
    truth = reference.true_survival["treatment"] - reference.true_survival["control"]

    reps = 250
    pointwise = 0.0
    simultaneous = 0
    for r in range(reps):
        sim = sl.simulate_experiment(
            n=6000, seed=9000 + r, horizon=8, observation_window=12, treatment_odds_ratio=0.85
        )
        curve = sl.survival_curves(sim.panel, horizon=8)
        curve = curve[curve["arm"] == "difference"].sort_values("period")
        pointwise += float(((curve["ci_low"] <= truth) & (truth <= curve["ci_high"])).mean())
        simultaneous += bool(((curve["band_low"] <= truth) & (truth <= curve["band_high"])).all())

    assert 0.92 <= pointwise / reps <= 0.98, f"pointwise {pointwise / reps:.1%}"
    assert 0.90 <= simultaneous / reps <= 0.99, f"simultaneous {simultaneous / reps:.1%}"
