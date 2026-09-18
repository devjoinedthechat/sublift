"""Anytime-valid inference, so that watching the dashboard is not cheating.

A fixed-sample 95% interval is only a 95% interval if you look once. Growth
teams look every morning, and under continuous monitoring the false-positive
rate of a repeatedly-checked z-test climbs well past 5% -- toward 100% as the
number of looks grows, because a random walk eventually crosses any fixed
boundary. Every "we called it early and it didn't replicate" post-mortem is
this effect.

A confidence sequence is an interval valid *simultaneously at every sample
size*: the probability that it ever excludes the truth, across an unlimited
number of looks, is at most alpha. You may stop whenever you like, including
because of what you saw.

sublift implements the asymptotic confidence sequence of Waudby-Smith, Arbour,
Sinha, Kennedy and Ramdas (*Time-uniform central limit theory and asymptotic
confidence sequences*), which needs only an i.i.d. sequence with a finite
variance -- which is exactly what the influence function gives us.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import lambertw

__all__ = ["ConfidenceSequence", "confidence_sequence", "cs_radius"]


@dataclass(frozen=True)
class ConfidenceSequence:
    estimate: float
    lower: float
    upper: float
    radius: float
    n: int
    n_target: int
    alpha: float
    fixed_radius: float

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0

    @property
    def peeking_cost(self) -> float:
        """How much wider than the (invalid, if you peeked) fixed-sample interval."""
        return self.radius / self.fixed_radius if self.fixed_radius else float("nan")

    def __str__(self) -> str:
        verdict = "excludes 0" if self.excludes_zero else "includes 0"
        return (
            f"anytime-valid {1 - self.alpha:.0%} CS at n={self.n:,}: "
            f"[{self.lower:+.4f}, {self.upper:+.4f}] ({verdict}); "
            f"{self.peeking_cost:.2f}x the fixed-sample width"
        )


def cs_radius(sd: float, n: int, *, alpha: float = 0.05, n_target: int | None = None) -> float:
    """Half-width of the asymptotic confidence sequence.

    ``n_target`` is the sample size the boundary is tuned to be tightest at --
    normally the size you expect the experiment to reach. The guarantee holds at
    every ``n`` regardless; tuning only decides where the sequence is narrowest,
    so set it to the planned end of the test and leave it alone. Re-tuning it
    after seeing the data is exactly the optional stopping the sequence was
    meant to protect against.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    target = int(n_target or n)
    if target < 1:
        raise ValueError("n_target must be at least 1.")
    rho2 = _rho_squared(alpha, target)
    inner = n * rho2 + 1.0
    return float(sd * math.sqrt((2.0 * inner) / (n**2 * rho2) * math.log(math.sqrt(inner) / alpha)))


def confidence_sequence(
    influence: np.ndarray,
    *,
    estimate: float,
    alpha: float = 0.05,
    n_target: int | None = None,
) -> ConfidenceSequence:
    """Anytime-valid interval around ``estimate`` from per-subject influence values."""
    psi = np.asarray(influence, dtype=float)
    n = psi.size
    if n < 2:
        raise ValueError("Need at least two subjects for a confidence sequence.")
    sd = float(psi.std(ddof=1))
    radius = cs_radius(sd, n, alpha=alpha, n_target=n_target)
    fixed = float(stats.norm.ppf(1 - alpha / 2) * sd / math.sqrt(n))
    return ConfidenceSequence(
        estimate=float(estimate),
        lower=float(estimate - radius),
        upper=float(estimate + radius),
        radius=radius,
        n=n,
        n_target=int(n_target or n),
        alpha=alpha,
        fixed_radius=fixed,
    )


def _rho_squared(alpha: float, n_target: int) -> float:
    """Tuning constant that minimizes the boundary width at ``n_target``.

    ``rho^2 = (-W_{-1}(-alpha^2) - 1) / n_target``, with the lower branch of the
    Lambert W function.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    w = lambertw(-(alpha**2), k=-1).real
    return float((-w - 1.0) / n_target)
