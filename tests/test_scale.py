"""Memory and time at the size the target users actually have.

A retention team at a media company has millions of subscribers, and an estimator
that allocates a subscriber-by-period matrix quietly needs gigabytes to answer a
question that should cost megabytes. These are regression tests: the easy mistake
is reintroducing an ``(n, horizon)`` temporary in a hot path, and it will not show
up in any correctness test.
"""

import time
import tracemalloc

import numpy as np
import pytest

from sublift import (
    churn_decomposition,
    incremental_ltv,
    retained_periods_lift,
    segment_scan,
    simulate_experiment,
)
from sublift.influence import value_influence
from sublift.survival import fit_survival


def peak_megabytes(call):
    tracemalloc.start()
    started = time.time()
    call()
    elapsed = time.time() - started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1e6, elapsed


def test_the_influence_function_does_not_scale_with_the_horizon():
    """Both of its terms collapse to lookups, so a longer horizon costs nothing."""
    rng = np.random.default_rng(0)
    n = 400_000
    lifetime = rng.geometric(0.08, size=n)
    censor = rng.integers(2, 40, size=n)
    observed = np.minimum(lifetime, censor).astype(np.int64)
    event = lifetime <= censor

    peaks = {}
    for horizon in (6, 24):
        fitted = fit_survival(observed, event, horizon)
        peaks[horizon], _ = peak_megabytes(lambda f=fitted: value_influence(f, observed, event))

    assert peaks[24] < 3 * peaks[6] + 5, (
        f"memory grew with the horizon: {peaks[6]:.0f}MB at 6, {peaks[24]:.0f}MB at 24"
    )
    assert peaks[24] < 60, f"{peaks[24]:.0f}MB for 400k subscribers is a matrix that should not exist"


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,budget_mb",
    [
        ("unadjusted", 200),
        ("stratified", 300),
        ("ltv", 320),
        ("decomposition", 250),
        ("segments", 350),
    ],
)
def test_a_million_subscribers_stays_within_budget(name, budget_mb):
    sim = simulate_experiment(
        n=1_000_000,
        seed=1,
        horizon=12,
        observation_window=18,
        involuntary_hazard=0.015,
        price=12.0,
    )
    panel = sim.panel
    calls = {
        "unadjusted": lambda: retained_periods_lift(panel, horizon=12, estimator="unadjusted"),
        "stratified": lambda: retained_periods_lift(
            panel, horizon=12, estimator="stratified", strata=["plan", "tenure_bucket"]
        ),
        "ltv": lambda: incremental_ltv(panel, horizon=12, estimator="unadjusted"),
        "decomposition": lambda: churn_decomposition(panel, horizon=12),
        "segments": lambda: segment_scan(
            panel, by=["plan", "tenure_bucket", "engagement_bucket"], horizon=12
        ),
    }
    peak, elapsed = peak_megabytes(calls[name])
    assert peak < budget_mb, f"{name}: {peak:.0f}MB exceeds the {budget_mb}MB budget"
    assert elapsed < 20, f"{name}: {elapsed:.1f}s at a million subscribers"


@pytest.mark.slow
def test_the_segment_scan_holds_one_influence_matrix_not_three():
    """It held three: a list of rows, the vstack of them, and the differenced copy.

    Two were removable outright -- fill in place, and expand the differenced
    variance into terms already computed -- and a fourth, the control arm's
    influence per segment, was dead weight left behind when the second
    heterogeneity scale changed.
    """
    sim = simulate_experiment(n=1_000_000, seed=1, horizon=12, observation_window=18, price=12.0)
    peak, _ = peak_megabytes(
        lambda: segment_scan(sim.panel, by=["plan", "tenure_bucket", "engagement_bucket"], horizon=12)
    )
    # Eight segments over a million subscribers is 64MB of influence values; anything
    # near a multiple of that means a copy came back.
    assert peak < 200, f"{peak:.0f}MB suggests the influence matrix exists more than once"


@pytest.mark.slow
def test_the_segment_odds_ratio_is_aggregated_not_expanded():
    """Its design is entirely categorical, so it has 2 x horizon distinct rows.

    Fitting the person-period rows instead would allocate hundreds of megabytes
    for a model with thirteen coefficients.
    """
    sim = simulate_experiment(n=1_000_000, seed=1, horizon=12, observation_window=18)
    peak, _ = peak_megabytes(lambda: segment_scan(sim.panel, by=["plan"], horizon=12, min_per_arm=10))
    assert peak < 250, f"{peak:.0f}MB suggests the person-period design is being materialised"
