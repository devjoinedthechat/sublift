"""The censoring distribution: known exactly when it can be, estimated when it must be."""

import numpy as np
import pytest

from sublift import SubscriberPanel, simulate_experiment
from sublift.censoring import CensoringWarning, censoring_survival


def test_exact_when_potential_follow_up_is_recorded():
    """P(C >= s) computed from enrollment dates, not inferred from survivors."""
    sim = simulate_experiment(n=5000, seed=1, horizon=8, observation_window=12)
    panel = sim.panel
    assert panel.potential_followup is not None
    gbar, source = censoring_survival(panel, 8, warn=False)
    assert source == "exact"
    expected = np.array([(panel.potential_followup >= s).mean() for s in range(1, 9)])
    np.testing.assert_allclose(gbar, expected)
    assert gbar[0] == pytest.approx(1.0)
    assert np.all(np.diff(gbar) <= 1e-12)


def test_reverse_km_is_used_when_potential_follow_up_is_unknown():
    sim = simulate_experiment(n=8000, seed=2, horizon=8, observation_window=12)
    bare = SubscriberPanel.from_periods(
        sim.frame,
        subject="subscriber_id",
        period="billing_period",
        churned="churned",
        arm="variant",
        control="control",
    )
    assert bare.potential_followup is None
    gbar, source = censoring_survival(bare, 8, warn=False)
    assert source == "reverse-km"
    assert gbar[0] == pytest.approx(1.0)
    assert np.all(np.diff(gbar) <= 1e-12)


def test_the_two_routes_agree_on_the_same_experiment():
    """Reverse Kaplan-Meier is consistent for the same quantity, just noisier.

    It can only learn about censoring from subscribers who survived long enough to
    be censored; the exact route reads every subscriber's potential follow-up off
    their enrollment date, including those who churned in period one. They should
    still agree, and the gap should shrink with the sample -- a fixed gap would
    mean the estimator is biased, which it was until the tie between churn and
    censoring in the same period was handled.
    """
    gaps = []
    for n in (40_000, 200_000):
        sim = simulate_experiment(n=n, seed=3, horizon=8, observation_window=12)
        exact, _ = censoring_survival(sim.panel, 8, warn=False)
        bare = SubscriberPanel.from_periods(
            sim.frame,
            subject="subscriber_id",
            period="billing_period",
            churned="churned",
            arm="variant",
            control="control",
        )
        estimated, _ = censoring_survival(bare, 8, warn=False)
        gaps.append(float(np.abs(exact - estimated).max()))

    assert gaps[0] < 0.005, f"reverse-KM is off by {gaps[0]:.4f} at n=40,000"
    assert gaps[1] < gaps[0], f"gap did not shrink with the sample: {gaps}"


def test_reverse_km_would_be_biased_without_the_tie_correction():
    """Pins the bug the tie correction fixes, so it cannot come back.

    Dividing the censoring events by the whole risk set -- rather than by the
    subscribers who did not churn in that same period -- understates the censoring
    hazard, and the error compounds through the product-limit.
    """
    sim = simulate_experiment(n=200_000, seed=3, horizon=8, observation_window=12)
    exact, _ = censoring_survival(sim.panel, 8, warn=False)

    n_periods, event = sim.panel.n_periods, sim.panel.event
    capped = np.minimum(n_periods, 9)
    at_risk = np.cumsum(np.bincount(capped, minlength=10)[::-1])[::-1][1:9].astype(float)
    censored = np.bincount(n_periods[~event], minlength=10)[1:9]
    naive_hazard = censored / at_risk
    naive = np.ones(8)
    naive[1:] = np.cumprod(1.0 - naive_hazard[:7])

    assert np.abs(naive - exact).max() > 0.02
    corrected, _ = censoring_survival(
        SubscriberPanel.from_periods(
            sim.frame,
            subject="subscriber_id",
            period="billing_period",
            churned="churned",
            arm="variant",
            control="control",
        ),
        8,
        warn=False,
    )
    assert np.abs(corrected - exact).max() < 0.002


def test_a_horizon_beyond_the_follow_up_is_warned_about():
    """Inverse-censoring weights of 1/0.02 do not produce a usable interval."""
    sim = simulate_experiment(n=4000, seed=4, horizon=8, observation_window=40)
    with pytest.warns(CensoringWarning, match="could have been observed"):
        censoring_survival(sim.panel, 40)


def test_potential_follow_up_shorter_than_observation_is_rejected():
    """A subscriber cannot have been observed for longer than they could be observed."""
    from sublift.panel import PanelError

    with pytest.raises(PanelError, match="potential follow-up"):
        SubscriberPanel(
            subject=np.array([1, 2]),
            arm=np.array([0, 1], dtype=np.int8),
            n_periods=np.array([5, 3]),
            event=np.array([True, False]),
            potential_followup=np.array([2, 9]),
        )


def test_take_and_subset_carry_potential_follow_up():
    sim = simulate_experiment(n=1200, seed=5, horizon=6, observation_window=9)
    half = sim.panel.subset(np.arange(sim.panel.n_subjects) < 600)
    assert half.potential_followup is not None
    assert len(half.potential_followup) == 600
    rng = np.random.default_rng(0)
    boot = sim.panel.take(rng.integers(0, sim.panel.n_subjects, size=sim.panel.n_subjects))
    assert boot.potential_followup is not None
    assert len(boot.potential_followup) == sim.panel.n_subjects
