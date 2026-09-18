"""Several arms against one control."""

import numpy as np
import pytest

from sublift import (
    NotIdentifiedError,
    PanelError,
    multi_arm_lift,
    retained_periods_lift,
    simulate_multi_arm,
)

STRATA = ["plan", "tenure_bucket"]


@pytest.fixture(scope="module")
def sim():
    return simulate_multi_arm(n=40_000, effects={"offer_a": 0.85, "offer_b": 0.95, "offer_c": 1.0}, seed=1)


def test_two_arm_estimators_refuse_and_point_somewhere(sim):
    """Looping them over arms and keeping the best is the mistake being prevented."""
    with pytest.raises(NotIdentifiedError, match="multi_arm_lift"):
        retained_periods_lift(sim.panel, horizon=8, estimator="unadjusted")


def test_contrast_lets_a_single_comparison_through(sim):
    pair = sim.panel.contrast("offer_a")
    res = retained_periods_lift(pair, horizon=8, estimator="unadjusted")
    assert res.arms["offer_a"].n > 0
    assert abs(res.estimate - sim.true_arm_lift["offer_a"]) < 4 * res.se


def test_each_arm_recovers_its_true_effect(sim):
    res = multi_arm_lift(sim.panel, horizon=8, estimator="stratified", strata=STRATA)
    for contrast in res.contrasts:
        truth = sim.true_arm_lift[contrast.label]
        assert abs(contrast.estimate - truth) < 4 * contrast.se, contrast.label


def test_the_null_arm_does_not_win(sim):
    res = multi_arm_lift(sim.panel, horizon=8, estimator="stratified", strata=STRATA)
    by_label = {c.label: c for c in res.contrasts}
    assert by_label["offer_a"].significant
    assert not by_label["offer_c"].significant
    assert res.best().label == "offer_a"


def test_contrasts_share_a_control_and_are_therefore_correlated(sim):
    """The fact that makes max-t less conservative than Bonferroni."""
    res = multi_arm_lift(sim.panel, horizon=8, estimator="unadjusted")
    off_diagonal = res.correlation[np.triu_indices(len(res.contrasts), 1)]
    assert np.all(off_diagonal > 0.3)
    assert res.critical_value < res.bonferroni_critical_value


@pytest.mark.parametrize("correction", ["max-t", "holm", "bonferroni", "none"])
def test_every_correction_runs_and_orders_sensibly(sim, correction):
    res = multi_arm_lift(sim.panel, horizon=8, estimator="unadjusted", correction=correction)
    for contrast in res.contrasts:
        assert contrast.adjusted_p_value >= contrast.p_value - 1e-9
        if correction != "none":
            lo, hi = contrast.ci
            m_lo, m_hi = contrast.marginal_ci
            assert lo <= m_lo and hi >= m_hi  # simultaneous is never tighter


def test_uncorrected_intervals_are_offered_but_not_the_default(sim):
    corrected = multi_arm_lift(sim.panel, horizon=8, estimator="unadjusted")
    naive = multi_arm_lift(sim.panel, horizon=8, estimator="unadjusted", correction="none")
    assert corrected.critical_value > naive.critical_value


def test_best_returns_none_when_nothing_survives():
    """In a null experiment some arm always has the largest estimate. It is not a winner."""
    null = simulate_multi_arm(n=12_000, effects={"a": 1.0, "b": 1.0, "c": 1.0}, seed=7)
    res = multi_arm_lift(null.panel, horizon=8, estimator="unadjusted")
    assert res.best() is None
    assert "not a winner" in res.summary()


def test_confidence_sequences_split_the_family(sim):
    res = multi_arm_lift(sim.panel, horizon=8, estimator="stratified", strata=STRATA)
    sequences = res.confidence_sequences(n_target=80_000)
    assert set(sequences) == {c.label for c in res.contrasts}
    for label, cs in sequences.items():
        assert cs.alpha == pytest.approx(0.05 / len(res.contrasts))
        assert cs.lower < cs.upper
        assert label in sim.true_arm_lift


def test_a_two_arm_panel_is_sent_back(sim):
    pair = sim.panel.contrast("offer_a")
    with pytest.raises(PanelError, match="no family to correct for"):
        multi_arm_lift(pair, horizon=8, estimator="unadjusted")


def test_the_adjusted_estimator_is_declined_with_a_reason(sim):
    with pytest.raises(ValueError, match="panel.contrast"):
        multi_arm_lift(sim.panel, horizon=8, estimator="adjusted")


def test_frame_and_summary_cover_every_arm(sim):
    res = multi_arm_lift(sim.panel, horizon=8, estimator="stratified", strata=STRATA)
    frame = res.to_frame()
    assert len(frame) == 3
    assert set(frame["arm"]) == {"offer_a", "offer_b", "offer_c"}
    text = res.summary()
    for label in frame["arm"]:
        assert label in text
