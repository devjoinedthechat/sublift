"""The guided entry point: right checks, right order, verdict written down."""

import numpy as np
import pandas as pd
import pytest

from sublift import SubscriberPanel, review, simulate_experiment

pytestmark = pytest.mark.filterwarnings("ignore:Sample ratio mismatch")

SCHEDULE = {
    "control": np.full(12, 12.0),
    "treatment": np.concatenate([np.full(3, 6.0), np.full(9, 12.0)]),
}


@pytest.fixture(scope="module")
def save_offer():
    """The motivating case: a discount that buys retention and destroys value."""
    return simulate_experiment(
        n=40_000,
        seed=2024,
        horizon=12,
        observation_window=18,
        baseline_hazard=0.065,
        treatment_odds_ratio=0.86,
        effect_decay=0.10,
        price=12.0,
        treatment_discount=0.50,
        discount_periods=3,
        involuntary_hazard=0.015,
    )


def test_a_clean_experiment_is_not_blocked(save_offer):
    result = review(save_offer.panel, horizon=12, price=SCHEDULE, strata=["plan"])
    assert not result.blocked
    assert "VERDICT" in result.summary()


def test_the_margin_finding_is_surfaced(save_offer):
    """The whole reason to report both numbers."""
    result = review(save_offer.panel, horizon=12, price=SCHEDULE, strata=["plan"])
    titles = [f.title for f in result.findings]
    assert any("lifetime value went down" in t for t in titles)
    assert result.retention.estimate > 0
    assert result.value.estimate < 0


def test_a_broken_randomization_blocks(save_offer):
    rng = np.random.default_rng(0)
    panel = save_offer.panel
    broken = panel.subset(~((panel.arm == 1) & (rng.random(panel.n_subjects) < 0.10)))
    result = review(broken, horizon=12, price=SCHEDULE, strata=["plan"])
    assert result.blocked
    assert "do not act on this" in result.summary()
    assert any(f.level == "blocker" for f in result.findings)


def test_the_estimate_is_still_computed_when_blocked(save_offer):
    """Hiding it would only invite someone to compute it a worse way."""
    rng = np.random.default_rng(1)
    panel = save_offer.panel
    broken = panel.subset(~((panel.arm == 1) & (rng.random(panel.n_subjects) < 0.10)))
    result = review(broken, horizon=12, price=SCHEDULE, strata=["plan"])
    assert result.headline is not None
    assert result.blocked


def test_monitoring_leads_with_the_anytime_valid_interval(save_offer):
    result = review(save_offer.panel, horizon=12, strata=["plan"], monitoring=True)
    titles = [f.title for f in result.findings]
    assert any("anytime-valid" in t for t in titles)
    assert not any("If anyone watched" in t for t in titles)


def test_without_monitoring_it_offers_the_sequence(save_offer):
    result = review(save_offer.panel, horizon=12, strata=["plan"], monitoring=False)
    assert any("If anyone watched" in f.title for f in result.findings)


def test_competing_risks_are_included_when_the_panel_has_causes(save_offer):
    result = review(save_offer.panel, horizon=12, price=SCHEDULE, strata=["plan"])
    assert result.decomposition is not None
    assert {c.label for c in result.decomposition.causes} == {"voluntary", "involuntary"}


def test_segments_are_scanned_only_when_asked(save_offer):
    without = review(save_offer.panel, horizon=12, strata=["plan"])
    with_scan = review(save_offer.panel, horizon=12, strata=["plan"], segments=["plan", "tenure_bucket"])
    assert without.segments is None
    assert with_scan.segments is not None
    assert any("segment" in f.title.lower() for f in with_scan.findings)


def test_a_small_arm_is_warned_about(save_offer):
    small = save_offer.panel.subset(np.arange(save_offer.panel.n_subjects) < 700)
    result = review(small, horizon=8, allow_extrapolation=True)
    assert any("Smallest arm" in f.title for f in result.findings)


def test_informative_censoring_is_warned_about():
    sim = simulate_experiment(
        n=20_000,
        seed=1,
        horizon=8,
        observation_window=14,
        dropout_hazard=0.08,
        dropout_depends_on_engagement=2.4,
    )
    result = review(sim.panel, horizon=8)
    assert any("Censoring is not independent" in f.title for f in result.findings)
    assert not result.blocked  # a caveat, not a stop


def test_returning_subscribers_switch_the_estimand():
    sim = simulate_experiment(
        n=20_000,
        seed=7,
        horizon=8,
        observation_window=13,
        treatment_odds_ratio=0.85,
        winback_hazard=0.10,
    )
    result = review(sim.panel, horizon=8)
    assert "occupancy" in result.retention.estimator
    assert any("Subscribers return" in f.title for f in result.findings)


def test_a_panel_without_revenue_reports_retention_only():
    sim = simulate_experiment(n=5_000, seed=3, horizon=6, observation_window=10)
    bare = SubscriberPanel.from_periods(
        sim.frame,
        subject="subscriber_id",
        period="billing_period",
        churned="churned",
        arm="variant",
        control="control",
    )
    result = review(bare, horizon=6)
    assert result.value is None
    assert result.retention is not None
    assert result.headline is result.retention


def test_summary_is_readable_end_to_end(save_offer):
    text = review(
        save_offer.panel,
        horizon=12,
        price=SCHEDULE,
        strata=["plan"],
        segments=["plan", "tenure_bucket"],
        monitoring=True,
    ).summary()
    assert text.startswith("Experiment review")
    assert "Incremental retained periods" in text
    assert "Incremental LTV" in text
    assert "by cause of churn" in text
    assert "Effect by segment" in text


def test_dates_in_reviews_out():
    """The realistic path: warehouse spans straight into a verdict."""
    base = pd.Timestamp("2025-01-01")
    rng = np.random.default_rng(0)
    n = 4_000
    arm = rng.integers(0, 2, size=n)
    lifetime = rng.geometric(np.where(arm == 1, 0.10, 0.13))
    frame = pd.DataFrame(
        {
            "uid": np.arange(n),
            "variant": np.where(arm == 1, "offer", "holdout"),
            "assigned": base,
            "ended": [base + pd.DateOffset(months=int(k) - 1) for k in lifetime],
            "plan": rng.choice(["monthly", "annual"], size=n),
        }
    )
    panel = SubscriberPanel.from_spans(
        frame,
        subject="uid",
        arm="variant",
        assigned_at="assigned",
        ended_at="ended",
        observed_through="2025-12-01",
        control="holdout",
        price=10.0,
        covariates=["plan"],
    )
    result = review(panel, horizon=8, strata=["plan"])
    assert result.retention is not None
    assert result.value is not None
    assert result.censoring is not None and result.censoring.known_exactly
