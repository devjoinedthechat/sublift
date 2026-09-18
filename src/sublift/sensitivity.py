"""How wrong would the censoring assumption have to be to change the answer?

Independent censoring is the one assumption in this library that cannot be
verified from the data. :func:`sublift.check_censoring` tests whether censoring
depends on *observed* covariates, and covariate adjustment removes most of that
bias, but nothing rules out the case that actually worries people: subscribers
leaving the data for reasons related to how much longer they would have stayed,
in a way no recorded column captures.

The honest response to an untestable assumption is not to keep looking for a
better estimator. It is to say how badly it would have to fail before the
conclusion changes, and let the reader decide whether that is plausible for their
business. That is what this does.

The parameterisation
--------------------
Restricted mean survival time can be written exactly as an imputation::

    RMST = mean_i [ periods observed_i  +  1{censored} * expected remaining_i ]

with the expected remaining tenure taken from the fitted curve. At the
independent-censoring assumption this reproduces the product-limit estimate to
machine precision, which is what makes it a usable anchor rather than a second
estimator with its own quirks.

The sensitivity parameter ``gamma`` scales that remaining tenure for censored
subscribers in one arm. ``gamma = 0.8`` says they would have stayed 20% less long
than otherwise-identical subscribers who were *not* censored -- a selection the
data cannot see. Sweeping it and finding where the conclusion flips gives a
statement in the units the business argues in: *the subscribers we lost track of
in the treatment arm would have had to be a quarter worse than their peers for
this result to go away.*

Whether a quarter is plausible is not a statistical question, which is exactly
why it is handed back rather than answered here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import NotIdentifiedError
from .panel import SubscriberPanel
from .survival import fit_survival

__all__ = ["censoring_sensitivity", "CensoringSensitivity", "SensitivityPoint"]

_DEFAULT_GAMMAS = np.round(np.arange(1.0, 0.39, -0.05), 2)


@dataclass(frozen=True)
class SensitivityPoint:
    gamma: float
    estimate: float
    se: float
    ci: tuple[float, float]

    @property
    def excludes_zero(self) -> bool:
        if self.ci[0] != self.ci[0]:  # NaN: no bootstrap was run
            return abs(self.estimate) > 0
        return self.ci[0] > 0 or self.ci[1] < 0


@dataclass
class CensoringSensitivity:
    horizon: int
    arm: str
    alpha: float
    censoring_rate: dict[str, float]
    points: list[SensitivityPoint]
    tipping_estimate: float | None
    tipping_interval: float | None
    n_subjects: int

    @property
    def baseline(self) -> SensitivityPoint:
        """The point at ``gamma = 1``: the ordinary estimate under independent censoring."""
        return self.points[0]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "gamma": p.gamma,
                    "estimate": p.estimate,
                    "se": p.se,
                    "ci_low": p.ci[0],
                    "ci_high": p.ci[1],
                    "excludes_zero": p.excludes_zero,
                }
                for p in self.points
            ]
        )

    def summary(self) -> str:
        head = f"Censoring sensitivity over {self.horizon} billing periods"
        base = self.baseline
        lines = [
            head,
            "=" * len(head),
            f"  Under independent censoring: {base.estimate:+.4f} [{base.ci[0]:+.4f}, {base.ci[1]:+.4f}]",
            "  Censored: " + ", ".join(f"{k} {v:.0%}" for k, v in self.censoring_rate.items()),
            "",
            f"  gamma scales the remaining tenure of censored {self.arm} subscribers.",
            "",
            f"  {'gamma':>7s}  {'estimate':>10s}  {'interval':>24s}",
        ]
        for point in self.points:
            mark = "" if point.excludes_zero else "   <- includes zero"
            lines.append(
                f"  {point.gamma:>7.2f}  {point.estimate:>+10.4f}  "
                f"[{point.ci[0]:>+10.4f}, {point.ci[1]:>+10.4f}]{mark}"
            )

        lines.append("")
        if not np.isfinite(base.se):
            if self.tipping_estimate is None:
                lines.append("  The estimate keeps its sign at every value tested.")
            else:
                lines.append(
                    f"  The estimate changes sign at gamma = {self.tipping_estimate:.2f}. Run with"
                    " n_boot to bracket it."
                )
            return "\n".join(lines)
        if self.tipping_interval is None:
            lines.append(
                f"  The conclusion survives every value tested, down to gamma="
                f"{self.points[-1].gamma:.2f}. Censored {self.arm} subscribers would have had to"
            )
            lines.append(
                f"  stay less than {self.points[-1].gamma:.0%} as long as their observed peers"
                " to overturn it."
            )
        else:
            shortfall = 1 - self.tipping_interval
            lines.append(f"  Tipping point: gamma = {self.tipping_interval:.2f}. Censored {self.arm}")
            lines.append(
                f"  subscribers would have had to stay {shortfall:.0%} less long than otherwise-identical"
            )
            lines.append("  subscribers who were not censored, for a reason no recorded column captures,")
            lines.append("  before this result loses significance.")
        lines += [
            "",
            "  Whether that is plausible is a question about your data pipeline, not a",
            "  statistical one. This is the part the numbers hand back.",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def censoring_sensitivity(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    arm: str | None = None,
    gammas: np.ndarray | None = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
    allow_extrapolation: bool = False,
) -> CensoringSensitivity:
    """How far independent censoring must fail before the conclusion changes.

    Parameters
    ----------
    arm
        Which arm's censored subscribers to make pessimistic about. Defaults to
        whichever direction *erodes* the observed effect, since that is the
        question worth asking -- a sensitivity analysis that only makes the
        result stronger is not one.
    gammas
        Multipliers on the expected remaining tenure of censored subscribers in
        that arm. Defaults to 1.00 down to 0.40 in steps of 0.05.
    n_boot
        Bootstrap resamples. The whole sweep is evaluated inside one set of
        resamples, so the cost is the same as bootstrapping a single estimate.
        Pass ``0`` to skip it and get point estimates only, which is enough to
        find where the estimate changes sign and is what :func:`sublift.review`
        uses so that it can afford to run this on every experiment.
    """
    if panel.n_arms > 2:
        raise NotIdentifiedError("Censoring sensitivity compares two arms; use panel.contrast().")
    horizon = int(horizon) if horizon is not None else panel.followup
    if horizon > panel.followup and not allow_extrapolation:
        raise NotIdentifiedError(
            f"horizon={horizon} exceeds the {panel.followup} periods of follow-up both arms have."
        )
    grid = np.asarray(_DEFAULT_GAMMAS if gammas is None else gammas, dtype=float)
    if grid.size == 0 or grid[0] != 1.0:
        grid = np.concatenate(([1.0], grid[grid != 1.0]))

    baseline = _contrast(panel.n_periods, panel.event, panel.arm, horizon, grid, target=1)
    # Make the *unfavourable* direction the default: shrink whichever arm's censored
    # subscribers would pull the estimate towards zero.
    target = 1 if baseline[0] > 0 else 0
    if arm is not None:
        if arm not in panel.arm_labels:
            raise ValueError(f"{arm!r} is not one of {list(panel.arm_labels)}.")
        target = panel.arm_labels.index(arm)

    estimates = _contrast(panel.n_periods, panel.event, panel.arm, horizon, grid, target)

    n = panel.n_subjects
    if n_boot > 1:
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot, grid.size))
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            draws[b] = _contrast(
                panel.n_periods[idx], panel.event[idx], panel.arm[idx], horizon, grid, target
            )
        se = draws.std(axis=0, ddof=1)
    else:
        se = np.full(grid.size, np.nan)
    z = float(stats.norm.ppf(1 - alpha / 2))

    points = [
        SensitivityPoint(
            gamma=float(g),
            estimate=float(estimates[i]),
            se=float(se[i]),
            ci=(float(estimates[i] - z * se[i]), float(estimates[i] + z * se[i])),
        )
        for i, g in enumerate(grid)
    ]
    return CensoringSensitivity(
        horizon=horizon,
        arm=panel.arm_labels[target],
        alpha=alpha,
        censoring_rate={
            label: float((~panel.event[panel.arm == a]).mean()) for a, label in enumerate(panel.arm_labels)
        },
        points=points,
        tipping_estimate=_tipping(points, lambda p: np.sign(p.estimate) == np.sign(estimates[0])),
        tipping_interval=_tipping(points, lambda p: p.excludes_zero),
        n_subjects=n,
    )


# ----------------------------------------------------------------- internals


def _contrast(n_periods, event, arm, horizon, gammas, target) -> np.ndarray:
    """The arm contrast at every gamma, from one pass over each arm's curve."""
    values = {}
    for a in (0, 1):
        mask = arm == a
        shifts = gammas if a == target else np.ones_like(gammas)
        values[a] = _imputed_rmst(n_periods[mask], event[mask], horizon, shifts)
    return values[1] - values[0]


def _imputed_rmst(n_periods, event, horizon, gammas) -> np.ndarray:
    """RMST written as an imputation, evaluated at each multiplier.

    At ``gamma = 1`` this is the product-limit estimate exactly -- verified to
    machine precision in the tests -- which is what lets the sweep be read against
    the headline number rather than against a slightly different one.
    """
    with warnings.catch_warnings():
        # Extrapolation past an emptied risk set is deliberate here: a bootstrap
        # resample can run out of subscribers in a late period, and holding the
        # curve flat there is the same thing the sweep is already reasoning about.
        warnings.simplefilter("ignore")
        fitted = fit_survival(n_periods, event, horizon, allow_extrapolation=True)
    survival = np.concatenate(([1.0], fitted.survival))

    # Expected additional periods for a subscriber still paying after period c. The
    # panel's convention puts a censored subscriber through period c's renewal
    # decision, so the conditioning is on S(c), not S(c-1).
    remaining = np.zeros(horizon + 1)
    for c in range(horizon + 1):
        if survival[c] > 0:
            remaining[c] = survival[c:horizon].sum() / survival[c]

    capped = np.minimum(n_periods, horizon)
    censored = (~event) & (n_periods < horizon)
    observed = capped.mean()
    imputed = np.where(censored, remaining[capped], 0.0).mean()
    return observed + np.asarray(gammas, dtype=float) * imputed


def _tipping(points, holds) -> float | None:
    """The first gamma at which the conclusion stops holding."""
    for point in points:
        if not holds(point):
            return point.gamma
    return None
