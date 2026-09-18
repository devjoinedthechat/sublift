"""A subscription retention experiment whose answer is known in advance.

A library that reports confidence intervals is only worth using if the
intervals cover, and you cannot check coverage against real data, because real
data does not come with a true effect attached. So the validation suite is
built on a generator whose estimand is computable in closed form, and the
simulator is part of the public API rather than hidden in ``tests/`` -- if you
want to know whether sublift will call your test correctly at your sample size,
base rate and censoring, simulate it first.

The generating process is a discrete-time logistic hazard

    logit h(t | x, a) = alpha_t + gamma'x + beta_t * a

with a declining baseline (churn is front-loaded after signup), optional
renewal spikes at annual boundaries, a treatment effect that may decay with
time since the intervention, staggered enrollment producing realistic
administrative censoring, and an optional treatment-side discount, so that the
revenue arm of the estimand has something to bite on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .panel import SubscriberPanel

__all__ = ["simulate_experiment", "SimulatedExperiment"]

_TRUTH_DRAWS = 250_000
_OFFSET_CACHE: dict[tuple, np.ndarray] = {}
_CURVE_CACHE: dict[tuple, dict] = {}
_LOST_CACHE: dict[tuple, dict] = {}


@dataclass
class SimulatedExperiment:
    """A simulated experiment together with the answer it should produce."""

    panel: SubscriberPanel
    frame: pd.DataFrame
    true_rmst_lift: float
    true_ltv_lift: float
    true_survival: dict[str, np.ndarray]
    true_individual_rmst_lift: np.ndarray | None = None
    true_periods_saved: dict[str, float] | None = None
    horizon: int = 12
    params: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (
            f"SimulatedExperiment(n={self.panel.n_subjects}, horizon={self.horizon}, "
            f"true_rmst_lift={self.true_rmst_lift:+.4f} periods, "
            f"true_ltv_lift={self.true_ltv_lift:+.3f})"
        )


def simulate_experiment(
    n: int = 20_000,
    *,
    horizon: int = 12,
    observation_window: int = 18,
    baseline_hazard: float | np.ndarray = 0.07,
    early_churn_multiplier: float = 2.0,
    early_churn_decay: float = 0.45,
    renewal_spike: dict[int, float] | None = None,
    treatment_odds_ratio: float = 0.85,
    effect_decay: float = 0.0,
    effect_modification: float = 0.0,
    involuntary_hazard: float = 0.0,
    treat_fraction: float = 0.5,
    with_covariates: bool = True,
    covariate_strength: float = 0.6,
    price: float = 12.0,
    treatment_discount: float = 0.0,
    discount_periods: int = 3,
    staggered_enrollment: bool = True,
    seed: int | None = 0,
) -> SimulatedExperiment:
    """Simulate one randomized retention experiment.

    Parameters
    ----------
    baseline_hazard
        Asymptotic per-period churn probability, or a full per-period array (in
        which case the shape arguments below are ignored).
    early_churn_multiplier, early_churn_decay
        Early periods churn harder. The baseline is scaled by
        ``1 + (multiplier-1) * exp(-decay * (t-1))``, which is the shape real
        subscription curves take: a steep first few renewals settling onto a
        flatter tail.
    renewal_spike
        ``{period: multiplier}`` -- e.g. ``{12: 2.5}`` for the annual renewal
        cliff, where a year's worth of ambivalence gets resolved at once.
    treatment_odds_ratio
        Effect on the odds of churning per period. ``0.85`` is a solid, and
        realistic, retention win. ``1.0`` generates a true null.
    effect_decay
        Per-period exponential fade of the treatment effect. Retention
        interventions rarely keep working forever; ``0.15`` roughly halves the
        effect over five periods.
    effect_modification
        Genuine heterogeneity: the treatment's log-odds effect is scaled by
        ``1 + effect_modification * engagement``, so at ``1.0`` the intervention
        lands twice as hard on subscribers one standard deviation above mean
        engagement and not at all on those one below. ``0.0`` (the default)
        means a constant odds ratio -- which is *not* a constant effect on the
        retention scale, and the difference is what uplift models must survive.
        Requires ``with_covariates=True``.
    treatment_discount
        Fraction off ``price`` given to the treatment arm for its first
        ``discount_periods`` periods. Set it to make the estimand honest: a save
        offer that buys retention with margin should not look free.
    involuntary_hazard
        Per-period probability of *involuntary* churn -- a failed payment that
        dunning does not recover. Set above zero and the simulated experiment
        has two competing causes, the treatment moves only the voluntary one
        (a save offer does not fix a dead card), and the panel carries a
        ``churn_reason`` column for :func:`sublift.churn_decomposition`. Values
        around ``0.01``-``0.02`` against a 6-7% total hazard reproduce the
        20-40% involuntary share typical of consumer subscription media.
    """
    rng = np.random.default_rng(seed)
    if not 0 < treat_fraction < 1:
        raise ValueError("treat_fraction must be strictly between 0 and 1.")
    if observation_window < horizon:
        raise ValueError(
            f"observation_window={observation_window} is shorter than horizon={horizon}; "
            "the experiment would be extrapolating from its first day."
        )

    alpha = _baseline_logit(
        baseline_hazard, observation_window, early_churn_multiplier, early_churn_decay, renewal_spike
    )
    beta = np.log(treatment_odds_ratio) * np.exp(-effect_decay * np.arange(observation_window))

    arm = (rng.random(n) < treat_fraction).astype(np.int8)
    X, cov_frame, gamma = _covariates(rng, n, with_covariates, covariate_strength)

    if effect_modification and not with_covariates:
        raise ValueError("effect_modification needs with_covariates=True to have something to modify.")
    modifier = 1.0 + effect_modification * (X[:, 0] if with_covariates else np.zeros(n))
    lin = alpha[None, :] + (X @ gamma)[:, None] + arm[:, None] * modifier[:, None] * beta[None, :]
    hazard = _expit(lin)

    if not 0.0 <= involuntary_hazard < 1.0:
        raise ValueError("involuntary_hazard must be in [0, 1).")
    lifetime, cause = _draw_lifetimes(rng, hazard, involuntary_hazard)
    censor = _censoring(rng, n, observation_window, staggered_enrollment)
    n_periods = np.minimum(lifetime, censor)
    event = lifetime <= censor

    weights = _price_schedule(price, treatment_discount, discount_periods, observation_window)
    observed_cause = np.where(event, cause, -1)
    frame = _to_frame(arm, n_periods, event, cov_frame, weights, observed_cause, involuntary_hazard > 0)

    panel = SubscriberPanel.from_periods(
        frame,
        subject="subscriber_id",
        period="billing_period",
        churned="churned",
        arm="variant",
        control="control",
        covariates=list(cov_frame.columns) if cov_frame is not None else None,
        revenue="revenue",
        cause="churn_reason" if involuntary_hazard > 0 else None,
    )

    truth = _truth(
        rng, alpha, beta, gamma, horizon, weights, with_covariates, covariate_strength, X,
        effect_modification, involuntary_hazard,
    )
    return SimulatedExperiment(
        panel=panel,
        frame=frame,
        horizon=horizon,
        params={
            "n": n,
            "treatment_odds_ratio": treatment_odds_ratio,
            "effect_decay": effect_decay,
            "price": price,
            "treatment_discount": treatment_discount,
        },
        **truth,
    )


# ---------------------------------------------------------------- generating


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _baseline_logit(
    baseline: float | np.ndarray,
    periods: int,
    multiplier: float,
    decay: float,
    spike: dict[int, float] | None,
) -> np.ndarray:
    t = np.arange(1, periods + 1)
    if np.isscalar(baseline):
        shape = 1.0 + (multiplier - 1.0) * np.exp(-decay * (t - 1))
        h = np.clip(float(baseline) * shape, 1e-6, 0.95)
    else:
        h = np.asarray(baseline, dtype=float)
        if h.size < periods:
            raise ValueError(f"baseline_hazard covers {h.size} periods, need {periods}.")
        h = h[:periods].copy()
    for p, mult in (spike or {}).items():
        if 1 <= p <= periods:
            h[p - 1] = min(h[p - 1] * mult, 0.95)
    return _logit(h)


def _covariates(rng, n: int, enabled: bool, strength: float):
    if not enabled:
        return np.zeros((n, 1)), None, np.zeros(1)
    engagement = rng.normal(size=n)  # standardized sessions in the pre-period
    annual = (rng.random(n) < 0.3).astype(float)
    tenure = rng.integers(0, 3, size=n).astype(float)  # 0 new, 1 established, 2 long-tenured
    X = np.column_stack([engagement, annual, tenure - 1.0])
    # Engaged, annual and long-tenured subscribers churn less. Signs matter: if the
    # covariates carried no real signal, adjustment would have nothing to remove.
    gamma = strength * np.array([-0.55, -0.70, -0.35])
    frame = pd.DataFrame(
        {
            "engagement": engagement,
            "plan": np.where(annual > 0, "annual", "monthly"),
            "tenure_bucket": pd.Categorical.from_codes(tenure.astype(int), ["new", "established", "long"]),
        }
    )
    return X, frame, gamma


def _draw_lifetimes(rng, hazard: np.ndarray, involuntary: float):
    """Sequential renewal decisions; returns the last period paid for and the cause.

    Each period the payment fails with probability ``involuntary``; if it does
    not, the subscriber may still choose to cancel. Cause codes follow the
    alphabetical label order used by the panel: 0 = involuntary, 1 = voluntary.
    """
    n, periods = hazard.shape
    alive = np.ones(n, dtype=bool)
    lifetime = np.full(n, periods, dtype=np.int64)
    cause = np.full(n, -1, dtype=np.int64)
    vol_draw = rng.random((n, periods))
    inv_draw = rng.random((n, periods)) if involuntary > 0 else None

    for t in range(periods):
        if involuntary > 0:
            failed = alive & (inv_draw[:, t] < involuntary)
            lifetime[failed] = t + 1
            cause[failed] = 0
            alive &= ~failed
        cancelled = alive & (vol_draw[:, t] < hazard[:, t])
        lifetime[cancelled] = t + 1
        cause[cancelled] = 1
        alive &= ~cancelled

    lifetime[alive] = periods + 1  # survived the whole window; censored in practice
    cause[alive] = -1
    return lifetime, cause


def _censoring(rng, n: int, window: int, staggered: bool) -> np.ndarray:
    if not staggered:
        return np.full(n, window, dtype=np.int64)
    # Uniform enrollment across the window: subjects who joined late have short follow-up.
    enrolled_ago = rng.integers(1, window + 1, size=n)
    return enrolled_ago.astype(np.int64)


def _price_schedule(price: float, discount: float, discount_periods: int, periods: int) -> dict[str, np.ndarray]:
    control = np.full(periods, float(price))
    treatment = control.copy()
    if discount > 0:
        treatment[: max(discount_periods, 0)] *= 1.0 - discount
    return {"control": control, "treatment": treatment}


def _to_frame(arm, n_periods, event, cov_frame, weights, cause=None, with_cause=False) -> pd.DataFrame:
    n = arm.size
    labels = np.where(arm == 1, "treatment", "control")
    rows = np.repeat(np.arange(n), n_periods)
    period = np.concatenate([np.arange(1, k + 1) for k in n_periods])
    churned = np.zeros(period.size, dtype=bool)
    churned[np.cumsum(n_periods) - 1] = event

    revenue = np.where(
        labels[rows] == "treatment",
        weights["treatment"][period - 1],
        weights["control"][period - 1],
    )
    data = {
        "subscriber_id": rows,
        "billing_period": period,
        "churned": churned,
        "variant": labels[rows],
        "revenue": revenue,
    }
    frame = pd.DataFrame(data)
    if with_cause:
        labels = np.array(["involuntary", "voluntary"])
        reason = np.where(cause >= 0, labels[np.clip(cause, 0, 1)], None)
        per_row = np.where(churned, reason[rows], None)
        frame["churn_reason"] = per_row
    if cov_frame is not None:
        for col in cov_frame.columns:
            frame[col] = cov_frame[col].to_numpy()[rows]
    return frame


# -------------------------------------------------------------------- truth


def _truth(rng, alpha, beta, gamma, horizon, weights, with_covariates, strength, X_realized, em=0.0, involuntary=0.0):
    """The estimand, computed from the generating model rather than from a sample.

    Averaged over a large independent draw of covariates, so the target is the
    super-population effect an experiment is trying to estimate -- not the
    particular realized sample, which would flatter the coverage numbers.
    """
    key = (alpha[:horizon].tobytes(), beta[:horizon].tobytes(), horizon, bool(with_covariates), float(strength), float(em), float(involuntary))
    if key in _CURVE_CACHE:
        curves, lost = _CURVE_CACHE[key], _LOST_CACHE[key]
    else:
        if with_covariates:
            okey = (float(strength), _TRUTH_DRAWS)
            if okey not in _OFFSET_CACHE:
                Xt, _, gamma_t = _covariates(np.random.default_rng(12345), _TRUTH_DRAWS, True, strength)
                _OFFSET_CACHE[okey] = (Xt @ gamma_t, Xt[:, 0])
            offset, eng = _OFFSET_CACHE[okey]
        else:
            offset, eng = np.zeros(1), np.zeros(1)
        mod = 1.0 + em * eng

        curves, lost = {}, {}
        for label, on in (("control", 0.0), ("treatment", 1.0)):
            lin = alpha[None, :horizon] + offset[:, None] + on * mod[:, None] * beta[None, :horizon]
            h_vol = _expit(lin) * (1.0 - involuntary)
            h_inv = np.full_like(h_vol, involuntary)
            surv_i = np.cumprod(1.0 - h_vol - h_inv, axis=1)
            curves[label] = surv_i.mean(axis=0)
            if involuntary > 0:
                lag_i = np.concatenate((np.ones((surv_i.shape[0], 1)), surv_i[:, :-1]), axis=1)
                g = np.maximum(horizon - np.arange(1, horizon + 1), 0).astype(float)
                lost[label] = {
                    "involuntary": float((g * (lag_i * h_inv).mean(axis=0)).sum()),
                    "voluntary": float((g * (lag_i * h_vol).mean(axis=0)).sum()),
                }
        _CURVE_CACHE[key] = curves
        _LOST_CACHE[key] = lost

    lagged = {k: np.concatenate(([1.0], v[:-1])) for k, v in curves.items()}
    rmst_lift = float(lagged["treatment"].sum() - lagged["control"].sum())
    ltv_lift = float(
        (weights["treatment"][:horizon] * lagged["treatment"]).sum()
        - (weights["control"][:horizon] * lagged["control"]).sum()
    )

    individual = None
    if with_covariates:
        off_i = X_realized @ (strength * np.array([-0.55, -0.70, -0.35]))
        mod_i = 1.0 + em * X_realized[:, 0]
        per_arm = []
        for on in (0.0, 1.0):
            lin = alpha[None, :horizon] + off_i[:, None] + on * mod_i[:, None] * beta[None, :horizon]
            s = np.cumprod(1.0 - _expit(lin), axis=1)
            per_arm.append(np.concatenate((np.ones((s.shape[0], 1)), s[:, :-1]), axis=1).sum(axis=1))
        individual = per_arm[1] - per_arm[0]

    saved = None
    if lost:
        saved = {
            cause: -(lost["treatment"][cause] - lost["control"][cause]) for cause in lost["control"]
        }

    return {
        "true_rmst_lift": rmst_lift,
        "true_ltv_lift": ltv_lift,
        "true_survival": curves,
        "true_individual_rmst_lift": individual,
        "true_periods_saved": saved,
    }
