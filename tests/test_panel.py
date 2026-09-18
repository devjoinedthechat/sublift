"""The data contract rejects malformed experiments loudly rather than silently mis-answering."""

import numpy as np
import pandas as pd
import pytest

from sublift.panel import PanelError, SubscriberPanel

# The fixtures here are deliberately tiny; the small-arm warning is the guard working.
pytestmark = pytest.mark.filterwarnings("ignore:Arm .* has only")


def long_frame(**over):
    base = pd.DataFrame(
        {
            "uid": [1, 1, 1, 2, 2, 3],
            "cycle": [1, 2, 3, 1, 2, 1],
            "churned": [False, False, True, False, False, False],
            "variant": ["a", "a", "a", "b", "b", "b"],
            "revenue": [10.0, 10.0, 10.0, 10.0, 5.0, 10.0],
            "plan": ["m", "m", "m", "y", "y", "y"],
        }
    )
    return base.assign(**over) if over else base


def test_from_periods_collapses_to_one_record_per_subject():
    p = SubscriberPanel.from_periods(
        long_frame(), subject="uid", period="cycle", churned="churned", arm="variant"
    )
    assert p.n_subjects == 3
    assert list(p.n_periods) == [3, 2, 1]
    assert list(p.event) == [True, False, False]
    assert p.arm_labels == ("a", "b")


def test_from_subjects_matches_from_periods():
    wide = pd.DataFrame(
        {"uid": [1, 2, 3], "n": [3, 2, 1], "ev": [True, False, False], "variant": ["a", "b", "b"]}
    )
    a = SubscriberPanel.from_subjects(wide, subject="uid", arm="variant", periods="n", event="ev")
    b = SubscriberPanel.from_periods(
        long_frame(), subject="uid", period="cycle", churned="churned", arm="variant"
    )
    assert np.array_equal(a.n_periods, b.n_periods)
    assert np.array_equal(a.event, b.event)
    assert np.array_equal(a.arm, b.arm)


def test_gap_in_periods_is_rejected():
    bad = long_frame().drop(index=1)  # subject 1 jumps from period 1 to 3
    with pytest.raises(PanelError, match="contiguously"):
        SubscriberPanel.from_periods(bad, subject="uid", period="cycle", churned="churned", arm="variant")


def test_churn_before_final_period_is_rejected():
    bad = long_frame(churned=[True, False, True, False, False, False])
    with pytest.raises(PanelError, match="more than one period|final period"):
        SubscriberPanel.from_periods(bad, subject="uid", period="cycle", churned="churned", arm="variant")


def test_post_assignment_covariate_is_rejected():
    """Conditioning on something the treatment could have changed reintroduces bias."""
    bad = long_frame(plan=["m", "m", "y", "y", "y", "y"])
    with pytest.raises(PanelError, match="vary within a subject"):
        SubscriberPanel.from_periods(
            bad, subject="uid", period="cycle", churned="churned", arm="variant", covariates=["plan"]
        )


def test_three_arms_are_accepted_by_the_panel():
    """The panel holds any number of arms; the estimators decide what to do with them."""
    p = SubscriberPanel.from_periods(
        long_frame(variant=["a", "a", "a", "b", "b", "c"]),
        subject="uid",
        period="cycle",
        churned="churned",
        arm="variant",
    )
    assert p.n_arms == 3
    assert p.arm_labels == ("a", "b", "c")
    assert p.treatment_labels == ("b", "c")


def test_contrast_pulls_out_one_comparison():
    p = SubscriberPanel.from_periods(
        long_frame(variant=["a", "a", "a", "b", "b", "c"]),
        subject="uid",
        period="cycle",
        churned="churned",
        arm="variant",
    )
    pair = p.contrast("c")
    assert pair.n_arms == 2
    assert pair.arm_labels == ("a", "c")
    assert pair.n_subjects == 2  # the "b" subscriber is excluded
    with pytest.raises(PanelError, match="is the control arm"):
        p.contrast("a")


def test_a_single_arm_is_rejected():
    with pytest.raises(PanelError, match="at least"):
        SubscriberPanel.from_periods(
            long_frame(variant=["a"] * 6),
            subject="uid",
            period="cycle",
            churned="churned",
            arm="variant",
        )


def test_zero_period_rejected():
    bad = long_frame(cycle=[0, 1, 2, 1, 2, 1])
    with pytest.raises(PanelError, match=">= 1"):
        SubscriberPanel.from_periods(bad, subject="uid", period="cycle", churned="churned", arm="variant")


def test_duplicate_subject_in_wide_form_is_caught():
    wide = pd.DataFrame({"uid": [1, 1], "n": [2, 3], "ev": [True, False], "variant": ["a", "b"]})
    with pytest.raises(PanelError, match="must be unique"):
        SubscriberPanel.from_subjects(wide, subject="uid", arm="variant", periods="n", event="ev")


def test_control_label_can_be_chosen():
    p = SubscriberPanel.from_periods(
        long_frame(), subject="uid", period="cycle", churned="churned", arm="variant", control="b"
    )
    assert p.arm_labels == ("b", "a")


def test_followup_is_the_shorter_arm():
    p = SubscriberPanel.from_periods(
        long_frame(), subject="uid", period="cycle", churned="churned", arm="variant"
    )
    assert p.followup == 2  # arm "a" reaches 3, arm "b" only 2
    assert p.max_period == 3


def test_revenue_is_laid_out_by_period():
    p = SubscriberPanel.from_periods(
        long_frame(), subject="uid", period="cycle", churned="churned", arm="variant", revenue="revenue"
    )
    assert p.revenue.shape == (3, 3)
    assert p.revenue[1, 1] == 5.0
    assert np.isnan(p.revenue[2, 1])


def test_only_the_long_constructors_can_catch_a_post_assignment_covariate():
    """Pins what the README claims, because it claimed more than the code does.

    A covariate measured after assignment can be a consequence of the treatment,
    and adjusting for it reintroduces the bias randomisation removed. Constructors
    that see several rows per subscriber can notice one changing mid-subscription.
    Constructors that see a single row cannot -- there is nothing to compare it
    against -- and `from_spans` is the recommended one, so this is the common case
    rather than a corner.
    """
    long_form = long_frame(plan=["m", "m", "y", "y", "y", "y"])
    with pytest.raises(PanelError, match="vary within a subject"):
        SubscriberPanel.from_periods(
            long_form,
            subject="uid",
            period="cycle",
            churned="churned",
            arm="variant",
            covariates=["plan"],
        )

    wide = pd.DataFrame(
        {
            "uid": [1, 2],
            "variant": ["a", "b"],
            "n": [3, 3],
            "ev": [True, False],
            "measured_afterwards": [1.0, 0.0],
        }
    )
    accepted = SubscriberPanel.from_subjects(
        wide,
        subject="uid",
        arm="variant",
        periods="n",
        event="ev",
        covariates=["measured_afterwards"],
    )
    assert "measured_afterwards" in accepted.covariates.columns
