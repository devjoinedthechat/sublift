"""Every influence function, checked against leave-one-out.

The influence functions are the riskiest surface in this library: each one is a
hand derivation, they are what every interval is built from, and an error in one
would produce confident, plausible, wrong intervals rather than a crash. They are
also checked against the bootstrap elsewhere -- but the bootstrap and the
influence function are both *mine*, and agreement between two things derived by
the same person is weaker evidence than it looks.

So this checks them against something that shares no derivation at all. For an
asymptotically linear estimator the jackknife pseudo-value recovers the influence
function::

    IF_i  ~=  (n - 1) * (theta_full - theta_without_i)

which needs nothing but the ability to re-run the estimator on n-1 subscribers.
If a derivation is wrong, the two disagree, and no amount of internal consistency
hides it.

Agreement is judged on correlation and regression slope rather than exact
equality, because the jackknife is itself a finite-sample approximation.
Estimators whose influence function is exact should sit at r = 1.0000; the
one-step and AIPW ones are first-order and are allowed to be looser.
"""

import numpy as np
import pytest

import sublift as sl

pytestmark = [
    pytest.mark.slow,
    # These run at n=900 to keep the leave-one-out loop affordable; the adjusted
    # estimator warns below 5,000, which is the warning working, not a problem here.
    pytest.mark.filterwarnings("ignore:estimator=.adjusted."),
]

HORIZON = 6
SAMPLE = 120


def _jackknife(panel, estimate, subjects):
    n = panel.n_subjects
    full = estimate(panel)
    keep = np.ones(n, dtype=bool)
    out = np.empty(len(subjects))
    for j, i in enumerate(subjects):
        keep[i] = False
        out[j] = (n - 1) * (full - estimate(panel.subset(keep)))
        keep[i] = True
    return out


def assert_matches(panel, estimate, analytic, *, exact=True, seed=0):
    """Analytic influence values must track the leave-one-out ones."""
    subjects = np.random.default_rng(seed).choice(panel.n_subjects, size=SAMPLE, replace=False)
    jack = _jackknife(panel, estimate, subjects)
    ana = np.asarray(analytic)[subjects]

    correlation = float(np.corrcoef(jack, ana)[0, 1])
    slope = float(np.polyfit(ana, jack, 1)[0])
    floor = 0.999 if exact else 0.99
    assert correlation > floor, f"correlation with leave-one-out is {correlation:.4f}"
    assert abs(slope - 1) < 0.08, f"leave-one-out is {slope:.3f} times the analytic values"


@pytest.fixture(scope="module")
def panel():
    return sl.simulate_experiment(
        n=900,
        seed=3,
        horizon=HORIZON,
        observation_window=10,
        price=10.0,
        involuntary_hazard=0.02,
    ).panel


@pytest.fixture(scope="module")
def spells():
    return sl.simulate_experiment(
        n=900,
        seed=4,
        horizon=HORIZON,
        observation_window=10,
        winback_hazard=0.10,
        price=10.0,
    ).panel


def test_product_limit_contrast(panel):
    result = sl.retained_periods_lift(panel, horizon=HORIZON, estimator="unadjusted")
    assert_matches(
        panel,
        lambda p: sl.retained_periods_lift(p, horizon=HORIZON, estimator="unadjusted").estimate,
        result.influence,
    )


def test_the_revenue_term(panel):
    """Estimated revenue weights carry their own sampling error; this is that term."""
    result = sl.incremental_ltv(panel, horizon=HORIZON, estimator="unadjusted")
    assert_matches(
        panel,
        lambda p: sl.incremental_ltv(p, horizon=HORIZON, estimator="unadjusted").estimate,
        result.influence,
    )


def test_the_stratum_share_term(panel):
    """The term that is easy to omit, and understates variance when the effect varies."""
    result = sl.retained_periods_lift(panel, horizon=HORIZON, estimator="stratified", strata=["plan"])
    assert_matches(
        panel,
        lambda p: (
            sl.retained_periods_lift(p, horizon=HORIZON, estimator="stratified", strata=["plan"]).estimate
        ),
        result.influence,
    )


def test_the_one_step_efficient_influence_function(panel):
    covariates = ["engagement", "plan"]
    result = sl.retained_periods_lift(panel, horizon=HORIZON, estimator="adjusted", covariates=covariates)
    assert_matches(
        panel,
        lambda p: (
            sl.retained_periods_lift(p, horizon=HORIZON, estimator="adjusted", covariates=covariates).estimate
        ),
        result.influence,
        exact=False,  # first-order, and each refit moves the nuisance models
    )


@pytest.mark.parametrize("index", [0, 1])
def test_the_competing_risks_influence_functions(panel, index):
    from sublift.competing import _arm_decomposition

    n = panel.n_subjects
    psi = np.zeros(n)
    for a in (0, 1):
        mask = panel.arm == a
        piece = _arm_decomposition(
            panel.n_periods[mask],
            panel.event[mask],
            panel.cause[mask],
            HORIZON,
            len(panel.cause_labels),
            False,
        )
        # Periods *saved* is minus the change in periods lost, hence the flipped sign.
        psi[mask] = (-1 if a == 1 else 1) * piece["influence"][index] / (mask.sum() / n)

    assert_matches(
        panel,
        lambda p: sl.churn_decomposition(p, horizon=HORIZON).causes[index].estimate,
        psi,
    )


def test_occupancy(spells):
    result = sl.occupancy_lift(spells, horizon=HORIZON)
    assert_matches(spells, lambda p: sl.occupancy_lift(p, horizon=HORIZON).estimate, result.influence)


def test_occupancy_stratified(spells):
    result = sl.occupancy_lift(spells, horizon=HORIZON, strata=["plan"])
    assert_matches(
        spells,
        lambda p: sl.occupancy_lift(p, horizon=HORIZON, strata=["plan"]).estimate,
        result.influence,
    )


def test_occupancy_augmented(spells):
    covariates = ["engagement", "plan"]
    result = sl.occupancy_lift(spells, horizon=HORIZON, covariates=covariates)
    assert_matches(
        spells,
        lambda p: sl.occupancy_lift(p, horizon=HORIZON, covariates=covariates).estimate,
        result.influence,
        exact=False,
    )


def test_the_segment_contrast(panel):
    from sublift.estimators import _arm_weights
    from sublift.segments import _contrast

    weights = _arm_weights(panel, HORIZON, "retained_periods", None)
    mask = (panel.covariates["plan"].astype(str) == "monthly").to_numpy()
    _, (indices, values), _ = _contrast(panel, mask, HORIZON, weights, False)
    dense = np.zeros(panel.n_subjects)
    dense[indices] = values

    def estimate(p):
        inner = (p.covariates["plan"].astype(str) == "monthly").to_numpy()
        return _contrast(p, inner, HORIZON, _arm_weights(p, HORIZON, "retained_periods", None), True)[0]

    assert_matches(panel, estimate, dense)


def test_the_multi_arm_contrasts():
    multi = sl.simulate_multi_arm(n=1200, effects={"a": 0.85, "b": 1.0}, seed=2)
    shuffled = multi.panel.take(np.random.default_rng(0).permutation(multi.panel.n_subjects))
    result = sl.multi_arm_lift(shuffled, horizon=HORIZON, estimator="unadjusted", allow_extrapolation=True)
    for index in range(len(result.contrasts)):
        assert_matches(
            shuffled,
            lambda p, i=index: (
                sl.multi_arm_lift(p, horizon=HORIZON, estimator="unadjusted", allow_extrapolation=True)
                .contrasts[i]
                .estimate
            ),
            result.influence[index],
        )


def test_the_delta_method_ratio(panel):
    """`relative_ci` divides by an estimated denominator correlated with the numerator."""
    result = sl.retained_periods_lift(panel, horizon=HORIZON, estimator="unadjusted")
    psi = (result.influence - result.relative * result.control_influence) / result.control.value
    assert_matches(
        panel,
        lambda p: sl.retained_periods_lift(p, horizon=HORIZON, estimator="unadjusted").relative,
        psi,
    )
