"""Survival curves and the influence functions that carry their uncertainty."""

import numpy as np
import pytest

from sublift.influence import value_influence
from sublift.survival import fit_survival, weighted_value


def test_hazard_and_survival_by_hand():
    # 5 subjects: two churn at period 1, one at 2, two censored at 3.
    n_periods = np.array([1, 1, 2, 3, 3])
    event = np.array([True, True, True, False, False])
    s = fit_survival(n_periods, event, horizon=3)

    assert list(s.at_risk) == [5, 3, 2]
    assert list(s.events) == [2, 1, 0]
    np.testing.assert_allclose(s.hazard, [2 / 5, 1 / 3, 0.0])
    np.testing.assert_allclose(s.survival, [0.6, 0.4, 0.4])
    np.testing.assert_allclose(s.rmst(), 1.0 + 0.6 + 0.4)


def test_weighted_value_is_rmst_times_price_for_a_flat_schedule():
    n_periods = np.array([1, 2, 3, 3, 2])
    event = np.array([True, True, False, False, True])
    s = fit_survival(n_periods, event, horizon=3)
    assert weighted_value(s, 9.0) == pytest.approx(9.0 * s.rmst())


def test_horizon_past_the_data_is_refused_by_default():
    s_args = (np.array([1, 2, 2]), np.array([True, True, True]))
    with pytest.raises(ValueError, match="allow_extrapolation"):
        fit_survival(*s_args, horizon=5)
    with pytest.warns(UserWarning, match="held flat"):
        s = fit_survival(*s_args, horizon=5, allow_extrapolation=True)
    assert s.survival[-1] == s.survival[2]  # carried forward, not invented


def test_influence_function_sums_to_zero():
    """A necessary property of a correctly derived influence function."""
    rng = np.random.default_rng(0)
    n = 5000
    T = rng.geometric(0.15, size=n)
    C = rng.integers(2, 12, size=n)
    obs, ev = np.minimum(T, C), T <= C
    s = fit_survival(obs, ev, horizon=8)
    inf = value_influence(s, obs, ev)
    assert abs(inf.mean()) < 1e-9 * max(1.0, abs(inf).max())


def test_influence_standard_error_matches_the_bootstrap():
    """The analytic variance has to agree with resampling, or it is decoration."""
    rng = np.random.default_rng(7)
    n = 4000
    T = rng.geometric(0.12, size=n)
    C = rng.integers(3, 14, size=n)
    obs, ev = np.minimum(T, C), T <= C

    s = fit_survival(obs, ev, horizon=10)
    analytic = np.sqrt((value_influence(s, obs, ev) ** 2).sum()) / n

    draws = []
    for _ in range(300):
        idx = rng.integers(0, n, size=n)
        draws.append(fit_survival(obs[idx], ev[idx], horizon=10).rmst())
    boot = float(np.std(draws, ddof=1))
    assert analytic == pytest.approx(boot, rel=0.12)


def test_estimated_revenue_weights_widen_the_interval():
    """Weights estimated from the data carry their own error; ignoring it understates variance."""
    rng = np.random.default_rng(3)
    n = 3000
    T = rng.geometric(0.15, size=n)
    C = rng.integers(3, 10, size=n)
    obs, ev = np.minimum(T, C), T <= C
    width = int(obs.max())
    revenue = np.where(
        np.arange(1, width + 1)[None, :] <= obs[:, None],
        rng.normal(20.0, 6.0, size=(n, width)),
        np.nan,
    )
    s = fit_survival(obs, ev, horizon=6)
    from sublift.survival import empirical_revenue_weights

    w = empirical_revenue_weights(revenue, obs, 6)
    known = np.sqrt((value_influence(s, obs, ev, w) ** 2).sum()) / n
    estimated = np.sqrt((value_influence(s, obs, ev, w, revenue=revenue) ** 2).sum()) / n
    assert estimated > known
