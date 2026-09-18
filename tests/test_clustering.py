"""When the randomised unit is coarser than the analysed one."""

import numpy as np
import pandas as pd
import pytest

import sublift as sl


def households(n_accounts=6000, per_account=3, seed=0):
    """Accounts randomised as a unit, several subscriptions each, correlated within."""
    rng = np.random.default_rng(seed)
    account = np.repeat(np.arange(n_accounts), per_account)
    arm = np.repeat(rng.integers(0, 2, size=n_accounts), per_account)
    shared = np.repeat(rng.normal(size=n_accounts), per_account)  # household frailty
    lifetime = np.maximum(np.ceil(rng.exponential(np.exp(1.6 + 0.5 * shared + 0.15 * arm))).astype(int), 1)
    censor = rng.integers(3, 14, size=account.size)
    return pd.DataFrame(
        {
            "sub": np.arange(account.size),
            "account": account,
            "variant": np.where(arm == 1, "treat", "ctrl"),
            "n": np.minimum(lifetime, censor),
            "ev": lifetime <= censor,
            "plan": np.where(account % 2 == 0, "monthly", "annual"),
        }
    )


def build(frame, **kwargs):
    return sl.SubscriberPanel.from_subjects(
        frame,
        subject="sub",
        arm="variant",
        periods="n",
        event="ev",
        control="ctrl",
        covariates=["plan"],
        **kwargs,
    )


@pytest.fixture(scope="module")
def frame():
    return households()


def test_cluster_codes_are_dense_and_survive_subsetting(frame):
    panel = build(frame, cluster="account")
    assert panel.cluster is not None
    assert panel.cluster.min() == 0
    assert set(np.unique(panel.cluster)) == set(range(panel.cluster.max() + 1))

    half = panel.subset(np.arange(panel.n_subjects) < panel.n_subjects // 2)
    assert set(np.unique(half.cluster)) == set(range(half.cluster.max() + 1))


def test_ignoring_clusters_reports_an_interval_that_is_too_narrow(frame):
    """The failure this exists to prevent, and it is silent."""
    naive = build(frame)
    clustered = build(frame, cluster="account")
    assert (
        sl.retained_periods_lift(naive, horizon=8, estimator="unadjusted").se
        < sl.retained_periods_lift(clustered, horizon=8, estimator="unadjusted").se
    )


@pytest.mark.slow
def test_the_clustered_interval_matches_resampling_whole_households(frame):
    """The gold standard: resample the unit that was actually randomised."""
    naive = build(frame)
    clustered = build(frame, cluster="account")
    n_accounts, per_account = 6000, 3

    rng = np.random.default_rng(1)
    draws = []
    for _ in range(300):
        picks = rng.integers(0, n_accounts, size=n_accounts)
        idx = (picks[:, None] * per_account + np.arange(per_account)).ravel()
        draws.append(
            sl.retained_periods_lift(
                naive.take(idx), horizon=8, estimator="unadjusted", allow_extrapolation=True
            ).estimate
        )
    truth = float(np.std(draws, ddof=1))

    naive_se = sl.retained_periods_lift(naive, horizon=8, estimator="unadjusted").se
    clustered_se = sl.retained_periods_lift(clustered, horizon=8, estimator="unadjusted").se
    assert naive_se / truth < 0.95, f"naive se should be too small, was {naive_se / truth:.2f}x"
    assert 0.92 < clustered_se / truth < 1.08, f"clustered se is {clustered_se / truth:.2f}x"


def test_the_estimate_itself_does_not_move(frame):
    """Clustering is about the interval; the point estimate is unaffected."""
    naive = sl.retained_periods_lift(build(frame), horizon=8, estimator="unadjusted")
    clustered = sl.retained_periods_lift(build(frame, cluster="account"), horizon=8, estimator="unadjusted")
    assert clustered.estimate == pytest.approx(naive.estimate, rel=1e-12)


def test_confidence_sequences_count_clusters_not_subscriptions(frame):
    naive = sl.retained_periods_lift(build(frame), horizon=8, estimator="unadjusted")
    clustered = sl.retained_periods_lift(build(frame, cluster="account"), horizon=8, estimator="unadjusted")
    assert clustered.confidence_sequence().radius > naive.confidence_sequence().radius


def test_stratification_stays_clustered(frame):
    naive = sl.retained_periods_lift(build(frame), horizon=8, estimator="stratified", strata=["plan"])
    clustered = sl.retained_periods_lift(
        build(frame, cluster="account"), horizon=8, estimator="stratified", strata=["plan"]
    )
    assert clustered.se > naive.se
    assert clustered.estimate == pytest.approx(naive.estimate, rel=1e-12)


def test_relative_lift_interval_is_clustered_too(frame):
    naive = sl.retained_periods_lift(build(frame), horizon=8, estimator="unadjusted")
    clustered = sl.retained_periods_lift(build(frame, cluster="account"), horizon=8, estimator="unadjusted")
    width = lambda r: r.relative_ci[1] - r.relative_ci[0]  # noqa: E731
    assert width(clustered) > width(naive)


def test_one_subscription_per_cluster_changes_nothing(frame):
    """Clustering on a unique id must reduce to the independent case exactly."""
    unique = frame.assign(solo=frame["sub"])
    plain = sl.retained_periods_lift(build(frame), horizon=8, estimator="unadjusted")
    solo = sl.retained_periods_lift(build(unique, cluster="solo"), horizon=8, estimator="unadjusted")
    assert solo.se == pytest.approx(plain.se, rel=1e-12)


def test_a_missing_cluster_value_is_rejected(frame):
    broken = frame.copy()
    broken.loc[0, "account"] = np.nan
    with pytest.raises(sl.PanelError, match="missing values"):
        build(broken, cluster="account")
