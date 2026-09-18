"""How the effect develops as the horizon extends.

Every readout in this library is at one stated horizon, which is the right
discipline -- but it answers "how much" and never "when". Those are different
questions, and for retention interventions the second is often the one that
decides whether to ship.

A save offer that buys three months and then fades has the same twelve-period
number as one that buys a little every month forever, and they are not the same
product. A discount whose lift is still climbing at the edge of your data is
telling you the horizon is too short. An effect that peaks and then *declines* --
which happens when a treatment pulls churn forward rather than preventing it --
looks like a win at six periods and a wash at twelve.

Reading the effect at several horizons is the obvious way to see this and a
multiple-comparisons problem the moment you report the best one. The horizons are
also nested, so the estimates are strongly correlated -- the twelve-period effect
contains the six-period one -- which is exactly the case where treating them as
independent tests is most wasteful. The bands here are simultaneous across the
whole trajectory, calibrated on the correlation the influence functions actually
have, so a claim about the shape is a claim the interval supports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .family import correct_family
from .panel import SubscriberPanel

__all__ = ["lift_by_horizon", "HorizonCurve", "HorizonPoint"]


@dataclass(frozen=True)
class HorizonPoint:
    horizon: int
    estimate: float
    se: float
    ci: tuple[float, float]
    band: tuple[float, float]
    adjusted_p_value: float

    @property
    def significant(self) -> bool:
        """Non-zero after correcting for every horizon examined."""
        return self.band[0] > 0 or self.band[1] < 0


@dataclass
class HorizonCurve:
    metric: str
    alpha: float
    points: list[HorizonPoint]
    critical_value: float
    bonferroni_critical_value: float
    mean_correlation: float
    n_subjects: int

    @property
    def per_period(self) -> np.ndarray:
        """Effect added by each successive stretch of horizon, not the running total."""
        values = np.array([p.estimate for p in self.points])
        spans = np.diff([0, *[p.horizon for p in self.points]])
        return np.diff([0.0, *values]) / spans

    @property
    def peak_index(self) -> int:
        """Which stretch of horizon added the most.

        Not the first one, usually. Survival differences take time to open up, so
        the earliest stretch is nearly always the smallest even for an effect that
        is about to fade -- which is why the shape is read against the peak rather
        than against the start.
        """
        return int(np.argmax(self.per_period))

    @property
    def still_accumulating(self) -> bool:
        """The last stretch of horizon was the most productive one.

        The data ran out before the effect did, so the headline number is a lower
        bound on the full effect.
        """
        return self.peak_index == len(self.points) - 1

    @property
    def tail_share(self) -> float:
        """What the final stretch added, as a fraction of the most productive one."""
        rates = self.per_period
        peak = rates[self.peak_index]
        return float(rates[-1] / peak) if peak else float("nan")

    def to_frame(self) -> pd.DataFrame:
        """One row per horizon, pointwise and simultaneous."""
        return pd.DataFrame(
            [
                {
                    "horizon": p.horizon,
                    "estimate": p.estimate,
                    "se": p.se,
                    "ci_low": p.ci[0],
                    "ci_high": p.ci[1],
                    "band_low": p.band[0],
                    "band_high": p.band[1],
                    "adjusted_p": p.adjusted_p_value,
                }
                for p in self.points
            ]
        )

    def summary(self) -> str:
        """The effect at each horizon, and what the trajectory's shape means."""

        head = "Effect as the horizon extends"
        lines = [
            head,
            "=" * len(head),
            "",
            f"  {'horizon':>8s}  {'effect':>10s}  {'simultaneous band':>24s}  {'per period':>11s}",
        ]
        rates = self.per_period
        for point, rate in zip(self.points, rates, strict=True):
            mark = "*" if point.significant else " "
            lines.append(
                f" {mark}{point.horizon:>8d}  {point.estimate:>+10.4f}  "
                f"[{point.band[0]:>+10.4f}, {point.band[1]:>+10.4f}]  {rate:>+11.4f}"
            )

        lines += [
            "",
            f"  {1 - self.alpha:.0%} bands, simultaneous over {len(self.points)} horizons "
            f"(critical value {self.critical_value:.3f} against Bonferroni "
            f"{self.bonferroni_critical_value:.3f};",
            f"  nested horizons correlate at {self.mean_correlation:+.2f}, which is why the two differ).",
            "",
        ]
        if self.still_accumulating:
            lines.append("  The last stretch of horizon was the most productive one, so the data ran out")
            lines.append("  before the effect did. The headline number is a lower bound, and a longer")
            lines.append("  follow-up would report more.")
        else:
            peak = self.points[self.peak_index].horizon
            lines.append(f"  The effect accumulated fastest up to horizon {peak}, and the final stretch")
            lines.append(f"  added {self.tail_share:.0%} of that rate. Most of what this intervention")
            lines.append("  buys, it buys early; extending the horizon adds less than the headline implies.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def lift_by_horizon(
    panel: SubscriberPanel,
    *,
    horizons: Sequence[int] | None = None,
    metric: str = "retained_periods",
    estimator: str = "unadjusted",
    strata: list[str] | None = None,
    price=None,
    alpha: float = 0.05,
    correction: str = "max-t",
    seed: int = 0,
    **kwargs,
) -> HorizonCurve:
    """The effect at several horizons, with bands simultaneous across all of them.

    Parameters
    ----------
    horizons
        Which horizons to read. Defaults to roughly four evenly spaced points up
        to the follow-up both arms have.
    correction
        As elsewhere. ``"max-t"`` is worth keeping here: nested horizons are
        strongly correlated, so Bonferroni pays for independence that is
        emphatically absent.

    Notes
    -----
    This is a description of a trajectory, not licence to pick the horizon that
    looks best. Choose the horizon your decision needs beforehand; read this to
    understand the shape around it, and to notice when the data ran out before the
    effect did.
    """
    from .estimators import incremental_ltv, retained_periods_lift

    usable = panel.followup
    if horizons is None:
        step = max(usable // 4, 1)
        horizons = sorted({min(step * k, usable) for k in range(1, 5)})
    horizons = sorted({int(h) for h in horizons})
    if len(horizons) < 2:
        raise ValueError("Give at least two horizons; one is just an ordinary estimate.")
    if max(horizons) > usable:
        raise ValueError(f"horizon {max(horizons)} exceeds the {usable} periods of follow-up both arms have.")

    estimate = incremental_ltv if metric == "ltv" else retained_periods_lift
    shared = {"estimator": estimator, "strata": strata, "alpha": alpha, **kwargs}
    if metric == "ltv":
        shared["price"] = price

    results = {str(h): estimate(panel, horizon=h, **shared) for h in horizons}
    family = correct_family(results, alpha=alpha, correction=correction, seed=seed)

    points = [
        HorizonPoint(
            horizon=h,
            estimate=member.estimate,
            se=member.se,
            ci=member.marginal_ci,
            band=member.ci,
            adjusted_p_value=member.adjusted_p_value,
        )
        for h, member in zip(horizons, family.members, strict=True)
    ]
    return HorizonCurve(
        metric=metric,
        alpha=alpha,
        points=points,
        critical_value=family.critical_value,
        bonferroni_critical_value=family.bonferroni_critical_value,
        mean_correlation=family.mean_correlation,
        n_subjects=panel.n_subjects,
    )
