"""The anytime-valid layer."""

import math

import numpy as np
import pytest
from scipy import stats

from sublift.sequential import confidence_sequence, cs_radius


def test_sequence_is_wider_than_the_fixed_sample_interval():
    r = cs_radius(1.0, 10_000, alpha=0.05, n_target=10_000)
    fixed = stats.norm.ppf(0.975) / math.sqrt(10_000)
    assert 1.3 < r / fixed < 2.0  # the documented price of being allowed to peek


def test_radius_is_narrowest_at_the_tuning_point():
    n = 20_000
    at_target = cs_radius(1.0, n, n_target=n)
    mistuned_low = cs_radius(1.0, n, n_target=n // 20)
    mistuned_high = cs_radius(1.0, n, n_target=n * 20)
    assert at_target < mistuned_low
    assert at_target < mistuned_high


def test_radius_shrinks_with_sample_size():
    radii = [cs_radius(1.0, n, n_target=100_000) for n in (1_000, 10_000, 100_000)]
    assert radii[0] > radii[1] > radii[2]


def test_confidence_sequence_brackets_the_estimate():
    psi = np.random.default_rng(0).normal(size=5000)
    cs = confidence_sequence(psi, estimate=0.3, alpha=0.05)
    assert cs.lower < 0.3 < cs.upper
    assert cs.radius == pytest.approx((cs.upper - cs.lower) / 2)
    assert cs.peeking_cost > 1.0


def test_too_few_subjects_is_an_error():
    with pytest.raises(ValueError):
        confidence_sequence(np.array([1.0]), estimate=0.0)
