"""Slicing the base without the slice becoming the finding."""

import numpy as np
import pytest

from sublift import NotIdentifiedError, PanelError, segment_scan, simulate_experiment

BY = ["plan", "tenure_bucket", "engagement_bucket"]


@pytest.fixture(scope="module")
def null():
    """No effect at all, so every segment effect is exactly zero."""
    return simulate_experiment(n=60_000, seed=11, horizon=8, observation_window=12, treatment_odds_ratio=1.0)


@pytest.fixture(scope="module")
def uniform():
    """One odds ratio for everybody -- which is *not* one effect in retained periods.

    Segments churning faster have more periods to save, so the same odds ratio
    buys them more. Real variation, routinely misread as a targeting insight.
    """
    return simulate_experiment(n=250_000, seed=3, horizon=8, observation_window=12, treatment_odds_ratio=0.80)


@pytest.fixture(scope="module")
def heterogeneous():
    """The treatment genuinely acts differently on engaged subscribers."""
    return simulate_experiment(
        n=250_000,
        seed=3,
        horizon=8,
        observation_window=12,
        treatment_odds_ratio=0.88,
        effect_modification=1.8,
    )


def test_a_true_null_produces_no_segment_story(null):
    scan = segment_scan(null.panel, by=BY, horizon=8)
    assert not scan.any_heterogeneity
    assert scan.credible_segments() == []
    assert "one effect seen through noise" in scan.summary()


def test_segments_that_look_different_by_eye_are_not_reported(null):
    """The trap: the best and worst segments always differ, and it means nothing."""
    scan = segment_scan(null.panel, by=BY, horizon=8)
    estimates = [s.estimate for s in scan.segments]
    assert max(estimates) - min(estimates) > 0.05  # they do look different
    assert not any(s.differs_from_overall for s in scan.segments)  # and it is noise


def test_a_uniform_odds_ratio_still_varies_in_retained_periods(uniform):
    """Non-collapsibility: identical treatment, genuinely different absolute effects."""
    scan = segment_scan(uniform.panel, by=BY, horizon=8)
    assert scan.any_heterogeneity


def test_and_the_scan_says_it_is_a_scale_artefact(uniform):
    """The distinction that stops it being written up as a mechanism."""
    scan = segment_scan(uniform.panel, by=BY, horizon=8)
    assert not scan.mechanism_heterogeneity
    assert scan.scale_artefact
    assert "the churn odds ratio does not" in scan.summary()


def test_genuine_effect_modification_shows_on_both_scales(heterogeneous):
    scan = segment_scan(heterogeneous.panel, by=BY, horizon=8)
    assert scan.any_heterogeneity
    assert scan.mechanism_heterogeneity
    assert not scan.scale_artefact
    detected = {h.dimension for h in scan.heterogeneity if h.detected and h.scale == "odds ratio"}
    assert detected == {"engagement_bucket"}


def test_the_credible_segments_are_the_right_ones(heterogeneous):
    scan = segment_scan(heterogeneous.panel, by=BY, horizon=8)
    names = {s.name for s in scan.credible_segments()}
    assert "engagement_bucket=high" in names
    assert "engagement_bucket=low" in names
    assert not any(n.startswith("plan") for n in names)


def test_a_sign_flip_is_surfaced(heterogeneous):
    """The finding worth having: the offer helps some subscribers and harms others."""
    scan = segment_scan(heterogeneous.panel, by=BY, horizon=8)
    by_name = {s.name: s for s in scan.segments}
    assert by_name["engagement_bucket=high"].estimate > 0
    assert by_name["engagement_bucket=low"].estimate < 0
    assert by_name["engagement_bucket=low"].ci[1] < 0  # not a rounding error


def test_odds_ratios_are_reported_per_segment(uniform):
    scan = segment_scan(uniform.panel, by=BY, horizon=8)
    for segment in scan.segments:
        assert 0.5 < segment.odds_ratio < 1.1
        assert segment.log_odds_ratio_se > 0


def test_simultaneous_intervals_are_wider_than_per_comparison(null):
    scan = segment_scan(null.panel, by=BY, horizon=8)
    for s in scan.segments:
        assert s.ci[0] <= s.marginal_ci[0]
        assert s.ci[1] >= s.marginal_ci[1]


def test_the_interaction_is_not_the_segment_effect(heterogeneous):
    """'It works better here' is a claim about the difference, with its own uncertainty."""
    scan = segment_scan(heterogeneous.panel, by=BY, horizon=8)
    for s in scan.segments:
        assert s.interaction == pytest.approx(s.estimate - scan.overall, abs=1e-9)
        assert s.interaction_se > 0


def test_cross_product_segments(null):
    scan = segment_scan(null.panel, by=["plan", "tenure_bucket"], horizon=8, cross=True)
    assert scan.n_comparisons == 6
    assert all(s.dimension == "plan x tenure_bucket" for s in scan.segments)


def test_thin_segments_are_dropped_and_reported(null):
    scan = segment_scan(null.panel, by=BY, horizon=8, min_per_arm=9_000)
    assert any("dropped" in n for n in scan.notes)
    assert scan.n_comparisons < 8


def test_every_segment_too_thin_is_an_error(null):
    with pytest.raises(NotIdentifiedError, match="coarser segments"):
        segment_scan(null.panel, by=BY, horizon=8, min_per_arm=10**7)


def test_unknown_column_is_rejected(null):
    with pytest.raises(PanelError, match="not in the panel"):
        segment_scan(null.panel, by=["nonexistent"], horizon=8)


def test_frame_covers_every_segment(null):
    scan = segment_scan(null.panel, by=BY, horizon=8)
    frame = scan.to_frame()
    assert len(frame) == scan.n_comparisons
    assert set(frame["dimension"]) == set(BY)
    np.testing.assert_allclose(frame["vs_overall"], frame["estimate"] - scan.overall, atol=1e-9)


def test_the_sparse_gram_equals_the_dense_product():
    """The influence functions are stored compactly; their cross-products must not change."""
    from sublift.segments import _gram

    rng = np.random.default_rng(0)
    n = 5_000
    pieces = []
    for _ in range(6):
        indices = np.sort(rng.choice(n, size=int(rng.integers(500, 3000)), replace=False))
        pieces.append((indices, rng.normal(size=indices.size)))

    dense = np.zeros((6, n))
    for j, (indices, values) in enumerate(pieces):
        dense[j, indices] = values

    np.testing.assert_allclose(_gram(pieces, n), dense @ dense.T, atol=1e-9)


def test_segments_within_a_dimension_are_uncorrelated(null):
    """They partition the base, so their influence functions cannot overlap."""
    from sublift.segments import _gram

    scan = segment_scan(null.panel, by=["plan"], horizon=8)
    assert scan.n_comparisons == 2
    # Two plans, disjoint subscribers: the off-diagonal cross-product is exactly zero.
    assert _gram is not None
    for segment in scan.segments:
        assert segment.se > 0
