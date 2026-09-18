"""Competing risks: voluntary versus involuntary churn."""

import numpy as np
import pytest

from sublift import (
    PanelError,
    churn_decomposition,
    retained_periods_lift,
    simulate_experiment,
)


@pytest.fixture(scope="module")
def sim():
    return simulate_experiment(
        n=30_000,
        seed=3,
        horizon=8,
        observation_window=13,
        involuntary_hazard=0.018,
        treatment_odds_ratio=0.80,
    )


def test_causes_sum_to_the_headline_effect_exactly(sim):
    """The split is an identity, not an attribution -- so it has no residual."""
    d = churn_decomposition(sim.panel, horizon=8)
    assert sum(c.estimate for c in d.causes) == pytest.approx(d.total, rel=1e-12)


def test_total_matches_the_ordinary_estimator(sim):
    """Decomposing must not change the number being decomposed."""
    d = churn_decomposition(sim.panel, horizon=8)
    plain = retained_periods_lift(sim.panel, horizon=8, estimator="unadjusted")
    assert d.total == pytest.approx(plain.estimate, rel=1e-10)
    assert d.total_se == pytest.approx(plain.se, rel=1e-10)


def test_each_cause_recovers_its_true_effect(sim):
    d = churn_decomposition(sim.panel, horizon=8)
    for c in d.causes:
        truth = sim.true_periods_saved[c.label]
        assert abs(c.estimate - truth) < 3.0 * c.se, f"{c.label}: {c.estimate:+.4f} vs {truth:+.4f}"


def test_the_effect_runs_through_voluntary_churn(sim):
    """The simulated intervention moves cancellations only, and the split should say so."""
    d = churn_decomposition(sim.panel, horizon=8)
    by_label = {c.label: c for c in d.causes}
    assert by_label["voluntary"].estimate > 0
    assert d.share(by_label["voluntary"]) > 0.9


def test_helping_one_cause_shifts_exposure_to_the_other(sim):
    """A competing-risks effect that a single all-cause hazard cannot express.

    The treatment never touches payment failures, yet periods lost to
    involuntary churn go *up*: subscribers it keeps alive get more billing
    cycles in which their card can fail. The decomposition surfaces that
    trade-off, and it is real -- the simulator's closed-form truth has the same
    sign -- rather than an artefact of the estimator.
    """
    d = churn_decomposition(sim.panel, horizon=8)
    involuntary = next(c for c in d.causes if c.label == "involuntary")
    assert sim.true_periods_saved["involuntary"] < 0
    assert involuntary.estimate < 0


def test_influence_functions_agree_with_the_bootstrap(sim):
    """A cheap sanity check on the new derivation.

    The authoritative calibration test is in test_validation.py, which compares
    the analytic standard error against the actual sampling spread over many
    independent experiments. A bootstrap over one dataset is noisier -- and
    noisiest exactly for the rarer cause -- so the tolerance here is loose on
    purpose.
    """
    panel = sim.panel
    d = churn_decomposition(panel, horizon=8)
    rng = np.random.default_rng(0)
    draws = {c.label: [] for c in d.causes}
    for _ in range(150):
        b = churn_decomposition(
            panel.take(rng.integers(0, panel.n_subjects, size=panel.n_subjects)),
            horizon=8,
            allow_extrapolation=True,
        )
        for c in b.causes:
            draws[c.label].append(c.estimate)
    for c in d.causes:
        boot = float(np.std(draws[c.label], ddof=1))
        assert c.se == pytest.approx(boot, rel=0.25), f"{c.label}: {c.se:.5f} vs boot {boot:.5f}"


def test_a_panel_without_causes_says_what_to_do():
    plain = simulate_experiment(n=1200, seed=1, horizon=6, observation_window=9)
    with pytest.raises(ValueError, match="cause="):
        churn_decomposition(plain.panel, horizon=6)


def test_summary_mentions_every_cause(sim):
    text = str(churn_decomposition(sim.panel, horizon=8))
    assert "voluntary" in text and "involuntary" in text and "total" in text


def test_frame_has_a_row_per_cause(sim):
    f = churn_decomposition(sim.panel, horizon=8).to_frame()
    assert len(f) == 2
    assert f["share_of_effect"].sum() == pytest.approx(1.0, rel=1e-9)


# ------------------------------------------------------------------- stratified

STRATA = ["plan", "tenure_bucket"]


def test_stratified_causes_still_sum_exactly(sim):
    split = churn_decomposition(sim.panel, horizon=8, strata=STRATA)
    assert sum(c.estimate for c in split.causes) == pytest.approx(split.total, rel=1e-12)


def test_the_decomposition_reconciles_with_the_headline_it_decomposes(sim):
    """A report that stratifies its headline and not its split shows two numbers that
    should agree and do not. They must match on both the estimate and the interval."""
    for strata, estimator in ((None, "unadjusted"), (STRATA, "stratified")):
        headline = retained_periods_lift(sim.panel, horizon=8, estimator=estimator, strata=strata)
        split = churn_decomposition(sim.panel, horizon=8, strata=strata)
        assert split.total == pytest.approx(headline.estimate, rel=1e-10)
        assert split.total_se == pytest.approx(headline.se, rel=1e-10)


def test_stratifying_moves_the_answer(sim):
    """Otherwise the argument is not being passed through."""
    pooled = churn_decomposition(sim.panel, horizon=8)
    split = churn_decomposition(sim.panel, horizon=8, strata=STRATA)
    assert split.total != pooled.total
    assert abs(split.total - pooled.total) < 4 * pooled.total_se


def test_each_cause_still_recovers_its_truth_when_stratified(sim):
    split = churn_decomposition(sim.panel, horizon=8, strata=STRATA)
    for cause in split.causes:
        truth = sim.true_periods_saved[cause.label]
        assert abs(cause.estimate - truth) < 3.0 * cause.se, cause.label


def test_unknown_strata_are_rejected(sim):
    with pytest.raises(PanelError, match="not in the panel"):
        churn_decomposition(sim.panel, horizon=8, strata=["nonexistent"])


def test_review_decomposes_with_the_same_strata_as_its_headline(sim):
    from sublift import review

    result = review(sim.panel, horizon=8, strata=STRATA)
    assert result.decomposition is not None
    assert result.decomposition.total == pytest.approx(result.retention.estimate, rel=1e-10)
