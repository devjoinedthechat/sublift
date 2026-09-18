"""How long until this test can answer the question?

Asked before a test, not after. Most retention experiments are called on a
30-day proxy because nobody worked out that the real metric needed nine months,
and a 30-day proxy is a different question with a convenient answer.

The calculation is done by simulation rather than by a closed-form formula,
because the variance of a censored survival contrast depends on the enrollment
pattern in a way no clean formula captures: subscribers enrolled late in the
window contribute a few periods of follow-up each, and their contribution to
the variance of a 12-period estimand is not proportional to their headcount.
Simulating the actual enrollment schedule prices that in. The cost is a few
seconds, which is nothing against the weeks of calendar time being planned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .datasets import simulate_experiment
from .estimators import incremental_ltv, retained_periods_lift
from .sequential import cs_radius

__all__ = ["duration_to_detect", "PowerCurve"]


@dataclass
class PowerCurve:
    frame: pd.DataFrame
    true_effect: float
    metric: str
    alpha: float
    power: float
    fixed_periods: int | None
    sequential_periods: int | None

    def __str__(self) -> str:
        unit = "revenue" if self.metric == "ltv" else "periods"
        lines = [
            f"Effect being planned for: {self.true_effect:+.4f} {unit} per subscriber",
            f"Target: {self.power:.0%} power at alpha={self.alpha}",
            "",
        ]
        if self.fixed_periods:
            lines.append(
                f"  fixed-sample analysis (one look only): {self.fixed_periods} periods of enrollment"
            )
        else:
            lines.append("  fixed-sample analysis: not powered within the horizon searched")
        if self.sequential_periods:
            lines.append(f"  anytime-valid monitoring:                {self.sequential_periods} periods")
        else:
            lines.append("  anytime-valid monitoring: not conclusive within the horizon searched")
        lines += ["", self.frame.to_string(index=False, float_format=lambda v: f"{v:,.4f}")]
        return "\n".join(lines)


def duration_to_detect(
    arrivals_per_period: int,
    *,
    horizon: int = 12,
    baseline_hazard: float | np.ndarray = 0.07,
    treatment_odds_ratio: float = 0.85,
    metric: str = "retained_periods",
    price: float | None = None,
    treatment_discount: float = 0.0,
    discount_periods: int = 3,
    treat_fraction: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    max_periods: int = 36,
    pilot_n: int = 40_000,
    sequential_target: int | None = None,
    seed: int = 0,
    **sim_kwargs,
) -> PowerCurve:
    """Periods of enrollment needed before the effect is detectable.

    Parameters
    ----------
    arrivals_per_period
        Subscribers entering the experiment each billing period, across both arms.
    treatment_odds_ratio
        The effect worth detecting, as an odds ratio on per-period churn. Choose
        the smallest effect that would change a decision, not the effect you
        hope for.
    sequential_target
        Sample size the confidence sequence is tuned to. Defaults to the
        enrollment reached at ``max_periods``. Fix it before starting the test.

    Returns
    -------
    PowerCurve
        A row per candidate enrollment duration, with the standard error,
        fixed-sample power, and the always-valid detectable effect at that point.
    """
    if metric not in ("ltv", "retained_periods"):
        raise ValueError("metric must be 'ltv' or 'retained_periods'.")
    if metric == "ltv" and price is None:
        raise ValueError("Planning an LTV test needs price=... to convert periods into revenue.")
    if max_periods < horizon:
        raise ValueError(f"max_periods={max_periods} is below horizon={horizon}; nothing to search.")

    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    target_n = int(sequential_target or arrivals_per_period * max_periods)

    rows = []
    true_effect = float("nan")
    for duration in range(horizon, max_periods + 1):
        sim = simulate_experiment(
            n=pilot_n,
            horizon=horizon,
            observation_window=duration,
            baseline_hazard=baseline_hazard,
            treatment_odds_ratio=treatment_odds_ratio,
            treat_fraction=treat_fraction,
            price=price if price is not None else 1.0,
            treatment_discount=treatment_discount,
            discount_periods=discount_periods,
            with_covariates=False,
            seed=seed + duration,
            **sim_kwargs,
        )
        true_effect = sim.true_ltv_lift if metric == "ltv" else sim.true_rmst_lift

        fn = incremental_ltv if metric == "ltv" else retained_periods_lift
        kwargs = {"price": price} if metric == "ltv" else {}
        pilot = fn(sim.panel, horizon=horizon, estimator="unadjusted", **kwargs)

        # Per-subject variance, so it can be rescaled to any enrollment size.
        per_subject_var = (pilot.se**2) * pilot.n_subjects
        n = arrivals_per_period * duration
        se = float(np.sqrt(per_subject_var / n))
        achieved = float(stats.norm.sf(z_alpha - abs(true_effect) / se))
        radius = cs_radius(float(np.sqrt(per_subject_var)), n, alpha=alpha, n_target=target_n)

        rows.append(
            {
                "periods_enrolling": duration,
                "subscribers": n,
                "se": se,
                "fixed_power": achieved,
                "fixed_detectable": z_alpha * se + stats.norm.ppf(power) * se,
                "sequential_detectable": radius,
                "sequential_conclusive": bool(abs(true_effect) > radius),
            }
        )

    frame = pd.DataFrame(rows)
    powered = frame.loc[frame["fixed_power"] >= power, "periods_enrolling"]
    seq = frame.loc[frame["sequential_conclusive"], "periods_enrolling"]
    return PowerCurve(
        frame=frame,
        true_effect=true_effect,
        metric=metric,
        alpha=alpha,
        power=power,
        fixed_periods=int(powered.iloc[0]) if len(powered) else None,
        sequential_periods=int(seq.iloc[0]) if len(seq) else None,
    )
