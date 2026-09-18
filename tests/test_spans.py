"""from_spans: dates in, billing periods out. Where real analyses actually go wrong."""

import numpy as np
import pandas as pd
import pytest

from sublift import SubscriberPanel
from sublift.panel import PanelError

pytestmark = pytest.mark.filterwarnings("ignore:Arm .* has only")


def frame(**over):
    base = pd.DataFrame(
        {
            "uid": [1, 2, 3, 4],
            "variant": ["a", "b", "a", "b"],
            "assigned": ["2025-01-15", "2025-01-15", "2025-01-31", "2025-03-01"],
            "ended": [None, "2025-04-15", None, "2025-03-10"],
        }
    )
    return base.assign(**over) if over else base


def build(**kw):
    kw.setdefault("observed_through", "2025-06-30")
    return SubscriberPanel.from_spans(
        frame(), subject="uid", arm="variant", assigned_at="assigned", ended_at="ended", **kw
    )


def test_assignment_period_is_period_one():
    """A subscriber randomized today has paid for one period, not zero."""
    p = build()
    assert p.n_periods[3] == 1  # assigned 2025-03-01, cancelled 2025-03-10
    assert p.event[3]


def test_monthly_anniversaries_not_thirty_day_blocks():
    p = build(billing_interval="month")
    # uid 2: 2025-01-15 -> 2025-04-15 is exactly three monthly anniversaries, so four paid periods.
    assert p.n_periods[1] == 4
    assert p.event[1]


def test_end_of_month_signup_bills_on_the_last_day_of_short_months():
    """2025-01-31 plus one month is 2025-02-28, which is what processors charge.

    Comparing day-of-month instead would skip February entirely and then credit
    two periods at once in March -- an off-by-one on everyone who signed up on
    the 29th, 30th or 31st, landing exactly where the hazard is steepest.
    """
    df = pd.DataFrame(
        {"uid": [1, 2], "variant": ["a", "b"], "assigned": ["2025-01-31"] * 2, "ended": [None] * 2}
    )

    def periods_at(cut):
        return list(
            SubscriberPanel.from_spans(
                df,
                subject="uid",
                arm="variant",
                assigned_at="assigned",
                ended_at="ended",
                observed_through=cut,
                billing_interval="month",
            ).n_periods
        )

    assert periods_at("2025-02-27") == [1, 1]  # second bill has not landed
    assert periods_at("2025-02-28") == [2, 2]  # it lands on the last day of February
    assert periods_at("2025-03-30") == [2, 2]  # March bill is on the 31st
    assert periods_at("2025-03-31") == [3, 3]


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2025-01-31", "2025-01-31", 0),
        ("2025-01-31", "2025-02-28", 1),
        ("2025-01-31", "2025-03-30", 1),
        ("2025-01-31", "2025-03-31", 2),
        ("2025-01-30", "2025-02-28", 1),
        ("2025-01-15", "2025-04-14", 2),
        ("2025-01-15", "2025-04-15", 3),
        ("2024-02-29", "2025-02-28", 12),  # leap-day signup, non-leap anniversary
    ],
)
def test_monthly_anniversary_edge_cases(start, end, expected):
    from sublift.panel import _periods_between

    got = _periods_between(pd.Series(pd.to_datetime([start])), pd.Series(pd.to_datetime([end])), "month")
    assert int(got.iloc[0]) == expected


def test_annual_anniversaries_handle_the_leap_day():
    from sublift.panel import _periods_between

    got = _periods_between(
        pd.Series(pd.to_datetime(["2024-02-29", "2024-02-29"])),
        pd.Series(pd.to_datetime(["2025-02-28", "2025-02-27"])),
        "year",
    )
    assert list(got) == [1, 0]


def test_still_active_subscribers_are_censored_at_the_cut():
    p = build()
    assert not p.event[0]
    assert p.n_periods[0] == 6  # 2025-01-15 through 2025-06-30 -> 5 anniversaries + 1


def test_end_after_the_cut_is_censored_with_a_warning():
    """Anything past the data cut is information the experiment did not have."""
    with pytest.warns(UserWarning, match="after the data cut"):
        p = SubscriberPanel.from_spans(
            frame(),
            subject="uid",
            arm="variant",
            assigned_at="assigned",
            ended_at="ended",
            observed_through="2025-03-31",
        )
    assert not p.event[1]  # uid 2 ended 2025-04-15, after the cut


def test_daily_and_weekly_intervals():
    weekly = build(billing_interval="week")
    daily = build(billing_interval="day")
    assert weekly.n_periods[1] == (pd.Timestamp("2025-04-15") - pd.Timestamp("2025-01-15")).days // 7 + 1
    assert daily.n_periods[1] == 91


def test_integer_interval_in_days():
    p = build(billing_interval=30)
    assert p.n_periods[1] == 90 // 30 + 1


def test_assignment_after_the_cut_is_an_error():
    with pytest.raises(PanelError, match="assigned after"):
        build(observed_through="2024-12-01")


def test_unparseable_dates_are_rejected():
    bad = frame(assigned=["2025-01-15", "not a date", "2025-01-31", "2025-03-01"])
    with pytest.raises(PanelError, match="not parseable dates"):
        SubscriberPanel.from_spans(
            bad,
            subject="uid",
            arm="variant",
            assigned_at="assigned",
            ended_at="ended",
            observed_through="2025-06-30",
        )


def test_per_subject_cut_column_is_supported():
    df = frame().assign(cut=["2025-06-30", "2025-06-30", "2025-02-28", "2025-06-30"])
    p = SubscriberPanel.from_spans(
        df,
        subject="uid",
        arm="variant",
        assigned_at="assigned",
        ended_at="ended",
        observed_through="cut",
        billing_interval="month",
    )
    # uid 3 assigned 2025-01-31 and cut on 2025-02-28: one anniversary, so two periods,
    # while everyone else runs to June.
    assert p.n_periods[2] == 2
    assert p.n_periods[0] == 6


def test_cause_requires_two_levels():
    df = frame(why=[None, "cancelled", None, "cancelled"])
    with pytest.raises(PanelError, match="at least two"):
        SubscriberPanel.from_spans(
            df,
            subject="uid",
            arm="variant",
            assigned_at="assigned",
            ended_at="ended",
            observed_through="2025-06-30",
            cause="why",
        )


def test_missing_cause_on_a_churned_subscriber_is_rejected():
    df = frame(why=[None, None, None, "cancelled"])
    with pytest.raises(PanelError, match="missing for"):
        SubscriberPanel.from_spans(
            df,
            subject="uid",
            arm="variant",
            assigned_at="assigned",
            ended_at="ended",
            observed_through="2025-06-30",
            cause="why",
        )


def test_a_flat_price_is_stored_once_and_expanded_on_demand():
    """Repeating one number across every period costs twelve times the memory."""
    p = build(price=9.0)
    assert p.revenue is None
    assert p.flat_revenue is not None and p.has_revenue

    grid = p.revenue_grid()
    assert grid.shape == (4, int(p.n_periods.max()))
    assert grid[0, 0] == 9.0
    assert np.isnan(grid[3, 1])  # beyond that subscriber's observed periods
