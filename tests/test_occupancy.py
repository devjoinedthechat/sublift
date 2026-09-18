"""Subscriptions that come back: spells, pauses, and the estimand that survives them."""

import numpy as np
import pandas as pd
import pytest

from sublift import (
    NotIdentifiedError,
    PanelError,
    SubscriberPanel,
    occupancy_lift,
    retained_periods_lift,
    simulate_experiment,
)

pytestmark = pytest.mark.filterwarnings("ignore:Arm .* has only")


def spells(**over):
    base = pd.DataFrame(
        {
            "uid": [1, 1, 2, 3, 3],
            "variant": ["a", "a", "b", "b", "b"],
            "entered": ["2025-01-01"] * 5,
            "start": ["2025-01-01", "2025-06-01", "2025-01-01", "2025-01-01", "2025-04-01"],
            "end": ["2025-03-01", None, "2025-02-01", "2025-02-01", None],
        }
    )
    return base.assign(**over) if over else base


def build(**kw):
    kw.setdefault("observed_through", "2025-09-01")
    return SubscriberPanel.from_spells(
        spells(),
        subject="uid",
        arm="variant",
        assigned_at="entered",
        spell_start="start",
        spell_end="end",
        **kw,
    )


def test_a_win_back_is_kept_rather_than_truncated():
    p = build()
    # uid 1 paid Jan-Mar, left, came back from June.
    assert list(p.active[0].astype(int)) == [1, 1, 1, 0, 0, 1, 1, 1, 1]
    assert p.has_spells


def test_the_first_spell_view_still_exists():
    """So the survival estimators keep working, and the two framings can be compared."""
    p = build()
    assert p.n_periods[0] == 3  # uid 1's first spell ended in March
    assert p.event[0]


def test_overlapping_spells_are_not_double_counted():
    """A plan change often opens a new row before the old one closes."""
    p = SubscriberPanel.from_spells(
        spells(start=["2025-01-01", "2025-02-01", "2025-01-01", "2025-01-01", "2025-04-01"]),
        subject="uid",
        arm="variant",
        assigned_at="entered",
        spell_start="start",
        spell_end="end",
        observed_through="2025-09-01",
    )
    assert set(np.unique(p.active)) <= {True, False}
    assert p.active[0][:3].all()


def test_a_gap_earns_nothing():
    p = build(price=10.0)
    grid = p.revenue_grid()
    # uid 1's lapsed periods 4 and 5 bring in no revenue, but are still observed.
    assert grid[0, 3] == 0.0
    assert grid[0, 5] == 10.0
    # And it is stored as one price, not as that price repeated down a row.
    assert p.flat_revenue is not None
    assert p.revenue is None


def test_assignment_must_not_vary_within_a_subscriber():
    bad = spells(entered=["2025-01-01", "2025-06-01", "2025-01-01", "2025-01-01", "2025-01-01"])
    with pytest.raises(PanelError, match="varies within a subscriber"):
        SubscriberPanel.from_spells(
            bad,
            subject="uid",
            arm="variant",
            assigned_at="entered",
            spell_start="start",
            spell_end="end",
            observed_through="2025-09-01",
        )


def test_a_spell_before_assignment_is_rejected():
    bad = spells(start=["2024-06-01", "2025-06-01", "2025-01-01", "2025-01-01", "2025-04-01"])
    with pytest.raises(PanelError, match="start before"):
        SubscriberPanel.from_spells(
            bad,
            subject="uid",
            arm="variant",
            assigned_at="entered",
            spell_start="start",
            spell_end="end",
            observed_through="2025-09-01",
        )


# -------------------------------------------------------------------- estimator


@pytest.fixture(scope="module")
def returning():
    return simulate_experiment(
        n=30_000,
        seed=7,
        horizon=8,
        observation_window=13,
        treatment_odds_ratio=0.85,
        winback_hazard=0.10,
    )


def test_occupancy_recovers_the_truth(returning):
    res = occupancy_lift(returning.panel, horizon=8)
    assert abs(res.estimate - returning.true_occupancy_lift) < 3 * res.se


def test_the_first_spell_view_overstates_the_effect(returning):
    """The finding that makes this estimand worth having.

    Time-to-first-cancellation counts a control subscriber who cancelled in March
    and resubscribed in May as lost. They were not. Ignoring the return inflates
    the measured win, and the inflation grows with the win-back rate -- 11% at a
    5% win-back hazard, 44% at 20%.
    """
    occupancy = occupancy_lift(returning.panel, horizon=8)
    survival = retained_periods_lift(returning.panel, horizon=8, estimator="unadjusted")
    truth = returning.true_occupancy_lift
    assert survival.estimate > occupancy.estimate
    assert abs(survival.estimate - truth) > abs(occupancy.estimate - truth)


def test_the_first_spell_view_cannot_see_win_backs_at_all(returning):
    """It is not merely biased; the quantity it measures is blind to what follows."""
    quiet = simulate_experiment(
        n=30_000,
        seed=7,
        horizon=8,
        observation_window=13,
        treatment_odds_ratio=0.85,
        winback_hazard=1e-9,
    )
    busy = returning  # same seed and parameters, 10% win-back hazard
    survival_quiet = retained_periods_lift(quiet.panel, horizon=8, estimator="unadjusted")
    survival_busy = retained_periods_lift(busy.panel, horizon=8, estimator="unadjusted")
    occupancy_quiet = occupancy_lift(quiet.panel, horizon=8)
    occupancy_busy = occupancy_lift(busy.panel, horizon=8)

    # The first-spell estimate barely moves; the occupancy estimate tracks the truth down.
    assert abs(survival_busy.estimate - survival_quiet.estimate) < 0.5 * survival_busy.se
    assert occupancy_busy.estimate < occupancy_quiet.estimate
    assert busy.true_occupancy_lift < quiet.true_occupancy_lift


def test_it_agrees_with_survival_when_nobody_returns():
    """Single-spell data: two consistent estimators of one estimand, so they should meet."""
    sim = simulate_experiment(n=40_000, seed=4, horizon=8, observation_window=12)
    base = pd.Timestamp("2025-01-01")
    p = sim.panel
    frame = pd.DataFrame(
        {
            "uid": p.subject,
            "variant": np.array(p.arm_labels)[p.arm],
            "entered": base,
            "start": base,
            "end": [
                base + pd.DateOffset(months=int(k) - 1) if ev else pd.NaT
                for k, ev in zip(p.n_periods, p.event, strict=True)
            ],
            "cut": [base + pd.DateOffset(months=int(k) - 1) for k in p.potential_followup],
        }
    )
    rebuilt = SubscriberPanel.from_spells(
        frame,
        subject="uid",
        arm="variant",
        assigned_at="entered",
        spell_start="start",
        spell_end="end",
        observed_through="cut",
    )
    np.testing.assert_array_equal(rebuilt.n_periods, p.n_periods)
    np.testing.assert_array_equal(rebuilt.event, p.event)

    survival = retained_periods_lift(p, horizon=8, estimator="unadjusted")
    occupancy = occupancy_lift(rebuilt, horizon=8)
    assert abs(survival.estimate - occupancy.estimate) < 0.5 * survival.se


def test_stratified_occupancy(returning):
    plain = occupancy_lift(returning.panel, horizon=8)
    strat = occupancy_lift(returning.panel, horizon=8, strata=["plan", "tenure_bucket"])
    assert "stratified" in strat.estimator
    assert abs(strat.estimate - plain.estimate) < 4 * plain.se


def test_it_supports_a_confidence_sequence(returning):
    res = occupancy_lift(returning.panel, horizon=8)
    assert res.inference == "influence"
    assert res.confidence_sequence().radius > 0


def test_ltv_uses_the_revenue_actually_earned(returning):
    periods = occupancy_lift(returning.panel, horizon=8)
    revenue = occupancy_lift(returning.panel, horizon=8, metric="ltv", price=10.0)
    assert revenue.estimate == pytest.approx(10.0 * periods.estimate, rel=1e-9)


def test_a_panel_without_spells_is_sent_back():
    sim = simulate_experiment(n=2000, seed=1, horizon=6, observation_window=9)
    with pytest.raises(PanelError, match="from_spells"):
        occupancy_lift(sim.panel, horizon=6)


def test_it_refuses_without_potential_follow_up(returning):
    stripped = SubscriberPanel(
        subject=returning.panel.subject,
        arm=returning.panel.arm,
        n_periods=returning.panel.n_periods,
        event=returning.panel.event,
        active=returning.panel.active,
    )
    with pytest.raises(NotIdentifiedError, match="potential_followup"):
        occupancy_lift(stripped, horizon=8)
