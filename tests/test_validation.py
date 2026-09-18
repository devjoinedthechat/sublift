"""Statistical validation against known truth.

These are the tests that decide whether sublift is worth using. Everything else
checks that the code does what the code says; these check that what the code
says is true. Each one simulates experiments whose true effect is computable in
closed form and asserts a property of the estimator's sampling behaviour.

Seeds are fixed, so the assertions are deterministic rather than flaky.

    pytest -m slow
"""

import numpy as np
import pytest

from sublift import churn_decomposition, retained_periods_lift, simulate_experiment

pytestmark = pytest.mark.slow

ESTIMATORS = {
    "unadjusted": {"estimator": "unadjusted"},
    "stratified": {"estimator": "stratified", "strata": ["plan", "tenure_bucket"]},
}


def _replicate(reps, n, seed0, horizon=8, window=12, **sim_kwargs):
    for r in range(reps):
        yield simulate_experiment(
            n=n, horizon=horizon, observation_window=window, seed=seed0 + r, **sim_kwargs
        )


# --------------------------------------------------------------- unbiasedness


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_estimator_is_unbiased(name):
    """Averaged over replications, the estimate lands on the true effect."""
    reps = 400
    ests = []
    truth = None
    for sim in _replicate(reps, 4000, seed0=7000):
        res = retained_periods_lift(sim.panel, horizon=8, **ESTIMATORS[name])
        ests.append(res.estimate)
        truth = sim.true_rmst_lift

    ests = np.array(ests)
    mc_se = ests.std(ddof=1) / np.sqrt(reps)
    bias = ests.mean() - truth
    assert abs(bias) < 3.5 * mc_se, f"{name}: bias {bias:+.5f} vs MC se {mc_se:.5f}"


# -------------------------------------------------------------------- coverage


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_intervals_cover_at_their_nominal_rate(name):
    """A 95% interval has to contain the truth 95% of the time. This is the whole contract."""
    reps = 500
    covered = 0
    for sim in _replicate(reps, 4000, seed0=21_000):
        res = retained_periods_lift(sim.panel, horizon=8, alpha=0.05, **ESTIMATORS[name])
        covered += res.ci[0] <= sim.true_rmst_lift <= res.ci[1]

    coverage = covered / reps
    # Binomial se at 0.95 over 500 reps is 0.0097; this is roughly +/- 3 se.
    assert 0.92 <= coverage <= 0.98, f"{name}: {coverage:.1%} coverage over {reps} replications"


def test_adjusted_bootstrap_intervals_cover():
    """Separate, and smaller, because every replication refits 120 bootstrap models."""
    reps = 60
    covered = 0
    for sim in _replicate(reps, 3000, seed0=33_000):
        res = retained_periods_lift(
            sim.panel, horizon=8, estimator="adjusted",
            covariates=["engagement", "plan", "tenure_bucket"], n_boot=120,
        )
        covered += res.ci[0] <= sim.true_rmst_lift <= res.ci[1]
    assert 0.85 <= covered / reps <= 1.0, f"adjusted: {covered / reps:.1%} over {reps} replications"


# ----------------------------------------------------------- variance reduction


def test_adjustment_and_stratification_actually_reduce_variance():
    """Otherwise they are assumptions bought for nothing."""
    reps = 40
    ses = {"unadjusted": [], "stratified": [], "adjusted": []}
    for sim in _replicate(reps, 4000, seed0=51_000):
        ses["unadjusted"].append(retained_periods_lift(sim.panel, horizon=8, estimator="unadjusted").se)
        ses["stratified"].append(
            retained_periods_lift(
                sim.panel, horizon=8, estimator="stratified", strata=["plan", "tenure_bucket"]
            ).se
        )
        ses["adjusted"].append(
            retained_periods_lift(
                sim.panel, horizon=8, estimator="adjusted",
                covariates=["engagement", "plan", "tenure_bucket"], n_boot=40,
            ).se
        )
    med = {k: float(np.median(v)) for k, v in ses.items()}
    assert med["stratified"] < med["unadjusted"], med
    assert med["adjusted"] < med["unadjusted"], med


def test_influence_standard_errors_track_the_true_sampling_spread():
    """The reported se must match the spread of the estimates it is describing."""
    reps = 400
    ests, ses = [], []
    for sim in _replicate(reps, 4000, seed0=61_000):
        res = retained_periods_lift(sim.panel, horizon=8, estimator="unadjusted")
        ests.append(res.estimate)
        ses.append(res.se)
    assert float(np.mean(ses)) == pytest.approx(float(np.std(ests, ddof=1)), rel=0.08)


# ------------------------------------------------------------------- censoring


def test_censoring_is_handled_where_naive_averaging_fails():
    """The motivating bias: averaging observed tenure treats censored subscribers as churned."""
    reps = 60
    rows = []
    for window in (9, 12, 18):
        sub_bias, naive_bias, truth = [], [], None
        for sim in _replicate(reps, 4000, seed0=71_000, window=window):
            panel = sim.panel
            res = retained_periods_lift(panel, horizon=8, estimator="unadjusted")
            sub_bias.append(res.estimate)
            capped = np.minimum(panel.n_periods, 8)
            naive_bias.append(capped[panel.arm == 1].mean() - capped[panel.arm == 0].mean())
            truth = sim.true_rmst_lift
        rows.append(
            (window, float(np.mean(sub_bias) - truth), float(np.mean(naive_bias) - truth), truth)
        )

    for window, sub_b, _naive_b, truth in rows:
        assert abs(sub_b) < 0.1 * abs(truth), f"window={window}: sublift bias {sub_b:+.4f}"
    tightest = rows[0]
    assert abs(tightest[2]) > 3 * abs(tightest[1]), (
        f"naive bias {tightest[2]:+.4f} should dwarf sublift's {tightest[1]:+.4f} at window=9"
    )


# ------------------------------------------------------- the peeking guarantee


def test_confidence_sequence_survives_peeking_where_a_fixed_sample_test_does_not():
    """The library's motivating claim, as an assertion rather than a paragraph.

    A true null, monitored at sixteen interim looks. The fixed-sample interval is
    only honest if you look once; the confidence sequence is honest at every look.
    """
    reps = 250
    n_max = 8000
    looks = np.arange(500, n_max + 1, 500)

    fixed_false_alarms = 0
    cs_false_alarms = 0
    for r in range(reps):
        sim = simulate_experiment(
            n=n_max, horizon=6, observation_window=10,
            treatment_odds_ratio=1.0,  # exactly no effect
            with_covariates=False, seed=91_000 + r,
        )
        panel = sim.panel
        assert sim.true_rmst_lift == pytest.approx(0.0, abs=1e-12)

        fixed_hit = cs_hit = False
        for k in looks:
            mask = np.zeros(panel.n_subjects, dtype=bool)
            mask[:k] = True
            res = retained_periods_lift(
                panel.subset(mask), horizon=6, estimator="unadjusted", allow_extrapolation=True
            )
            if res.ci[0] > 0 or res.ci[1] < 0:
                fixed_hit = True
            if res.confidence_sequence(n_target=n_max).excludes_zero:
                cs_hit = True
        fixed_false_alarms += fixed_hit
        cs_false_alarms += cs_hit

    fixed_rate = fixed_false_alarms / reps
    cs_rate = cs_false_alarms / reps
    assert cs_rate <= 0.05, f"confidence sequence broke its guarantee: {cs_rate:.1%} over {reps} reps"
    assert fixed_rate > 2 * max(cs_rate, 0.02), (
        f"fixed-sample false alarms {fixed_rate:.1%} vs sequence {cs_rate:.1%} -- "
        "the peeking problem should be visible at 16 looks"
    )


def test_a_single_look_at_a_fixed_sample_test_is_still_calibrated():
    """The comparison above is about peeking, not about the fixed-sample test being broken."""
    reps = 500
    false_alarms = 0
    for r in range(reps):
        sim = simulate_experiment(
            n=6000, horizon=6, observation_window=10, treatment_odds_ratio=1.0,
            with_covariates=False, seed=110_000 + r,
        )
        res = retained_periods_lift(sim.panel, horizon=6, estimator="unadjusted")
        false_alarms += res.ci[0] > 0 or res.ci[1] < 0
    rate = false_alarms / reps
    assert 0.02 <= rate <= 0.09, f"single-look false alarm rate {rate:.1%}, expected ~5%"


# ------------------------------------------------------------- competing risks


def test_cause_specific_effects_are_unbiased_and_calibrated():
    """The authoritative check on the competing-risks influence functions.

    Compares the reported standard error against the actual spread of the
    estimates across independent experiments, which is what a standard error
    claims to be. A bootstrap on one dataset is a proxy for this; this is the
    thing itself.
    """
    reps = 400
    ests: dict[str, list[float]] = {}
    ses: dict[str, list[float]] = {}
    truth = None
    for r in range(reps):
        sim = simulate_experiment(
            n=8000, horizon=8, observation_window=13, seed=200_000 + r,
            involuntary_hazard=0.018, treatment_odds_ratio=0.80,
        )
        for c in churn_decomposition(sim.panel, horizon=8).causes:
            ests.setdefault(c.label, []).append(c.estimate)
            ses.setdefault(c.label, []).append(c.se)
        truth = sim.true_periods_saved

    for label, values in ests.items():
        e = np.array(values)
        mc_se = e.std(ddof=1) / np.sqrt(reps)
        assert abs(e.mean() - truth[label]) < 3.5 * mc_se, f"{label}: biased"
        ratio = float(np.mean(ses[label]) / e.std(ddof=1))
        assert 0.92 <= ratio <= 1.08, f"{label}: reported se is {ratio:.3f} of the actual spread"


def test_cause_specific_intervals_cover():
    reps = 400
    covered: dict[str, int] = {}
    for r in range(reps):
        sim = simulate_experiment(
            n=8000, horizon=8, observation_window=13, seed=300_000 + r,
            involuntary_hazard=0.018, treatment_odds_ratio=0.80,
        )
        for c in churn_decomposition(sim.panel, horizon=8).causes:
            hit = c.ci[0] <= sim.true_periods_saved[c.label] <= c.ci[1]
            covered[c.label] = covered.get(c.label, 0) + int(hit)
    for label, hits in covered.items():
        assert 0.92 <= hits / reps <= 0.98, f"{label}: {hits / reps:.1%} coverage"
