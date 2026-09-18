"""The three estimators, and the result object they all return.

All three target the same estimand -- the difference between arms in
``sum_t w_t S(t-1)`` to a stated horizon -- and differ only in what they assume
in exchange for variance:

``unadjusted``
    Nonparametric product-limit per arm. Assumes randomization and independent
    (administrative) censoring, and nothing else.
``stratified``
    The same, within pre-assignment strata, recombined on stratum shares. Buys
    real variance reduction while keeping an exact per-subject influence
    function -- which is what lets it be monitored sequentially.
``adjusted``
    Covariate-adjusted discrete hazard model, standardized over the covariate
    distribution (g-computation). Fit separately by arm with a saturated time
    baseline, the configuration under which the estimator stays consistent
    under randomization even when the covariate model is wrong (Moore & van der
    Laan, 2009). Inference by nonparametric bootstrap over subjects.

Default to ``stratified``. It is the one that both reduces variance and
supports anytime-valid monitoring, and on real subscriber data a handful of
prognostic strata -- plan, tenure bucket, pre-period engagement quantile --
recovers most of what full covariate adjustment would.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .censoring import censoring_survival, conditional_censoring_survival
from .diagnostics import warn_on_srm
from .exceptions import NotIdentifiedError
from .influence import contrast_influence, value_influence
from .logistic import design_matrix, fit_logistic
from .panel import SubscriberPanel
from .survival import empirical_revenue_weights, fit_survival, weighted_value

__all__ = ["incremental_ltv", "retained_periods_lift", "LiftResult", "ArmSummary"]

_ESTIMATORS = ("unadjusted", "stratified", "adjusted")

# Below this the one-step estimator's first-order variance is measurably optimistic.
_ADJUSTED_MIN_N = 5_000


def _require_two_arms(panel: SubscriberPanel) -> None:
    """These estimators compare one treatment against one control.

    Running them once per arm and reporting whichever looked best is a
    multiple-comparisons problem, so the refusal points at the two ways to do it
    deliberately rather than by accident.
    """
    if panel.n_arms > 2:
        raise NotIdentifiedError(
            f"This panel has {panel.n_arms} arms {list(panel.arm_labels)}, and this estimator "
            "compares two. Use sublift.multi_arm_lift(), which tests every arm against the "
            "control and controls the family-wise error rate, or panel.contrast('<arm>') to "
            "pull out one comparison deliberately. Looping over arms and reporting the best "
            "one inflates the false-positive rate by roughly the number of arms."
        )


@dataclass(frozen=True)
class ArmSummary:
    label: str
    n: int
    value: float
    survival: np.ndarray
    at_risk: np.ndarray
    weights: np.ndarray


@dataclass
class LiftResult:
    """The readout: a point estimate, an interval, and the curves behind them."""

    estimator: str
    metric: str
    horizon: int
    estimate: float
    se: float
    ci: tuple[float, float]
    alpha: float
    arms: dict[str, ArmSummary]
    n_subjects: int
    inference: str
    influence: np.ndarray | None = None
    bootstrap_draws: np.ndarray | None = field(default=None, repr=False)
    control_influence: np.ndarray | None = field(default=None, repr=False)
    bootstrap_relative: np.ndarray | None = field(default=None, repr=False)
    randomization: object | None = field(default=None, repr=False)
    strata_used: list[str] | None = None
    covariates_used: list[str] | None = None
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- readouts

    @property
    def control(self) -> ArmSummary:
        return self.arms[next(iter(self.arms))]

    @property
    def treatment(self) -> ArmSummary:
        return self.arms[list(self.arms)[1]]

    @property
    def relative(self) -> float:
        """Lift as a fraction of the control arm's value -- the "+4.8%" everyone reports."""
        return self.estimate / self.control.value if self.control.value else float("nan")

    @property
    def relative_ci(self) -> tuple[float, float]:
        """Interval for the relative lift.

        Not the absolute interval divided by the control value: the denominator
        is estimated too, and it is correlated with the numerator. This is the
        delta-method interval for the ratio, built from the influence function
        of both parts::

            IF(Delta / V0) = [IF(Delta) - (Delta/V0) IF(V0)] / V0

        For the bootstrap estimator the ratio is resampled directly. Reporting a
        percentage lift without this is the most common way a correct analysis
        still ends up with a wrong interval on the slide.
        """
        v0 = self.control.value
        if not v0:
            return (float("nan"), float("nan"))
        z = float(stats.norm.ppf(1 - self.alpha / 2))
        if self.bootstrap_relative is not None and self.bootstrap_relative.size:
            lo, hi = np.percentile(
                self.bootstrap_relative, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)]
            )
            return float(lo), float(hi)
        if self.influence is None or self.control_influence is None:
            return (float("nan"), float("nan"))
        ratio = self.relative
        psi = (self.influence - ratio * self.control_influence) / v0
        se = float(np.sqrt((psi**2).sum()) / psi.size)
        return float(ratio - z * se), float(ratio + z * se)

    @property
    def p_value(self) -> float:
        """Two-sided fixed-sample p-value.

        Valid only if this is the *one* analysis you run. If you have been
        watching the test, read :meth:`confidence_sequence` instead -- that is
        the whole point of it.
        """
        if not self.se:
            return float("nan")
        return float(2 * stats.norm.sf(abs(self.estimate / self.se)))

    def confidence_sequence(self, *, n_target: int | None = None, alpha: float | None = None):
        """Anytime-valid interval: correct even if you have looked every day.

        Requires per-subject influence values, so it is available for the
        ``unadjusted`` and ``stratified`` estimators and not for ``adjusted``.
        """
        from .sequential import confidence_sequence as _cs

        if self.influence is None:
            raise ValueError(
                f"The {self.estimator!r} estimator reports bootstrap inference, which does not "
                "expose per-subject influence values. Use estimator='stratified' to monitor a "
                "running experiment; it reduces variance and stays sequentially valid."
            )
        return _cs(
            self.influence,
            estimate=self.estimate,
            alpha=self.alpha if alpha is None else alpha,
            n_target=n_target,
        )

    def curves(self) -> pd.DataFrame:
        """Per-arm survival and cumulative value, period by period."""
        rows = []
        for label, arm in self.arms.items():
            lagged = np.concatenate(([1.0], arm.survival[:-1]))
            rows.append(
                pd.DataFrame(
                    {
                        "arm": label,
                        "period": np.arange(1, self.horizon + 1),
                        "at_risk": arm.at_risk,
                        "survival": arm.survival,
                        "period_weight": arm.weights,
                        "cumulative_value": np.cumsum(arm.weights * lagged),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def summary(self) -> str:
        from .report import format_result

        return format_result(self)

    def __str__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        from .report import html_result

        return html_result(self)


# --------------------------------------------------------------- public API


def incremental_ltv(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    estimator: str = "stratified",
    strata: Sequence[str] | None = None,
    covariates: Sequence[str] | None = None,
    censoring_covariates: Sequence[str] | None = None,
    price: float | np.ndarray | dict | None = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    allow_extrapolation: bool = False,
    expected_ratio: float = 0.5,
    inference: str = "influence",
    seed: int | None = 0,
) -> LiftResult:
    """Incremental lifetime value to ``horizon`` billing periods.

    Parameters
    ----------
    horizon
        Billing periods to measure over, counting the assignment period as 1.
        Defaults to the longest follow-up both arms actually have. There is no
        default "lifetime": a lifetime value without a horizon is an
        extrapolation wearing a measurement's clothes, so sublift makes you name
        the horizon and refuses to run past the data unless you insist.
    price
        A known revenue schedule -- a scalar, a per-period array, or
        ``{"control": ..., "treatment": ...}``. Prefer this when you have it.
        Falling back on revenue observed in the panel is supported but weaker:
        the observed mean conditions on *being at risk*, and once the treatment
        has changed who is still subscribed, the two arms' at-risk populations
        are no longer comparable. With a stated schedule that problem does not
        arise.
    """
    return _estimate(
        panel,
        horizon=horizon,
        estimator=estimator,
        metric="ltv",
        strata=strata,
        covariates=covariates,
        censoring_covariates=censoring_covariates,
        price=price,
        alpha=alpha,
        n_boot=n_boot,
        allow_extrapolation=allow_extrapolation,
        expected_ratio=expected_ratio,
        inference=inference,
        seed=seed,
    )


def retained_periods_lift(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    estimator: str = "stratified",
    strata: Sequence[str] | None = None,
    covariates: Sequence[str] | None = None,
    censoring_covariates: Sequence[str] | None = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    allow_extrapolation: bool = False,
    expected_ratio: float = 0.5,
    inference: str = "influence",
    seed: int | None = 0,
) -> LiftResult:
    """Incremental billing periods retained -- restricted mean survival time, contrasted.

    The revenue-free readout. Report it alongside LTV: when the two disagree in
    sign, the intervention bought retention with margin, and that is the finding.
    """
    return _estimate(
        panel,
        horizon=horizon,
        estimator=estimator,
        metric="retained_periods",
        strata=strata,
        covariates=covariates,
        censoring_covariates=censoring_covariates,
        price=None,
        alpha=alpha,
        n_boot=n_boot,
        allow_extrapolation=allow_extrapolation,
        expected_ratio=expected_ratio,
        inference=inference,
        seed=seed,
    )


# ------------------------------------------------------------------ routing


def _estimate(
    panel,
    *,
    horizon,
    estimator,
    metric,
    strata,
    covariates,
    censoring_covariates=None,
    price=None,
    alpha,
    n_boot,
    allow_extrapolation,
    seed,
    expected_ratio=0.5,
    inference="influence",
):
    if estimator not in _ESTIMATORS:
        raise ValueError(f"estimator must be one of {_ESTIMATORS}, got {estimator!r}.")
    _require_two_arms(panel)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    horizon = _resolve_horizon(panel, horizon, allow_extrapolation)
    weights = _arm_weights(panel, horizon, metric, price)
    notes: list[str] = []
    randomization = warn_on_srm(panel, expected_ratio)

    if estimator == "unadjusted":
        if strata:
            warnings.warn("strata are ignored by the 'unadjusted' estimator.", stacklevel=3)
        out = _unadjusted(panel, horizon, weights, allow_extrapolation)
    elif estimator == "stratified":
        if not strata:
            raise ValueError(
                "estimator='stratified' needs strata=[...]: the pre-assignment columns to "
                "stratify on (plan, tenure bucket, pre-period engagement quantile). With no "
                "strata it is just estimator='unadjusted'."
            )
        out = _stratified(panel, horizon, weights, list(strata), allow_extrapolation, notes)
    else:
        if not covariates:
            raise ValueError("estimator='adjusted' needs covariates=[...] to adjust for.")
        out = _adjusted(
            panel,
            horizon,
            weights,
            list(covariates),
            n_boot,
            alpha,
            allow_extrapolation,
            seed,
            inference=inference,
            censoring_covariates=censoring_covariates,
        )

    ci = _interval(out["estimate"], out["se"], alpha, out.get("boot"))
    estimate, se = out["estimate"], out["se"]
    notes.extend(out.get("notes", []))

    return LiftResult(
        estimator=estimator,
        metric=metric,
        horizon=horizon,
        estimate=estimate,
        se=se,
        ci=ci,
        alpha=alpha,
        arms=out["arms"],
        n_subjects=panel.n_subjects,
        inference=out["inference"],
        influence=out.get("influence"),
        bootstrap_draws=out.get("boot"),
        control_influence=out.get("control_influence"),
        bootstrap_relative=out.get("boot_relative"),
        randomization=randomization,
        strata_used=list(strata) if strata and estimator == "stratified" else None,
        covariates_used=list(covariates) if covariates and estimator == "adjusted" else None,
        notes=notes,
    )


def _resolve_horizon(panel: SubscriberPanel, horizon: int | None, allow_extrapolation: bool) -> int:
    usable = panel.followup
    if horizon is None:
        return usable
    horizon = int(horizon)
    if horizon > usable and not allow_extrapolation:
        raise NotIdentifiedError(
            f"horizon={horizon} exceeds the {usable} periods of follow-up both arms have. "
            "Past that point at least one arm's curve is carried forward on no data, so the "
            f"contrast is not a measurement. Use horizon={usable}, wait for more follow-up, or "
            "pass allow_extrapolation=True and label the number as a projection."
        )
    return horizon


def _arm_weights(panel, horizon, metric, price) -> dict[int, tuple[np.ndarray, bool]]:
    if metric == "retained_periods":
        return {a: (np.ones(horizon), False) for a in (0, 1)}

    if price is not None:
        if isinstance(price, dict):
            missing = set(panel.arm_labels) - set(price)
            if missing:
                raise ValueError(f"price dict is missing arm(s) {sorted(missing)}.")
            return {a: (_fit_schedule(price[panel.arm_labels[a]], horizon), False) for a in (0, 1)}
        sched = _fit_schedule(price, horizon)
        return dict.fromkeys((0, 1), (sched, False))

    if panel.revenue is None:
        raise ValueError(
            "No revenue in the panel and no price= given, so LTV cannot be formed. Pass "
            "price=..., build the panel with revenue=..., or ask for retained_periods_lift()."
        )
    out = {}
    for a in (0, 1):
        m = panel.arm == a
        out[a] = (empirical_revenue_weights(panel.revenue[m], panel.n_periods[m], horizon), True)
    return out


def _fit_schedule(price, horizon: int) -> np.ndarray:
    if np.isscalar(price):
        return np.full(horizon, float(price))
    arr = np.asarray(price, dtype=float)
    if arr.size < horizon:
        raise ValueError(f"price schedule covers {arr.size} periods but the horizon is {horizon}.")
    return arr[:horizon]


def _interval(estimate, se, alpha, boot) -> tuple[float, float]:
    if boot is not None and boot.size:
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)
    z = float(stats.norm.ppf(1 - alpha / 2))
    return float(estimate - z * se), float(estimate + z * se)


# ------------------------------------------------------------- unadjusted


def _fit_arm(panel, mask, horizon, weights, allow_extrapolation):
    w, estimated = weights
    surv = fit_survival(
        panel.n_periods[mask], panel.event[mask], horizon, allow_extrapolation=allow_extrapolation
    )
    value = weighted_value(surv, w)
    inf = value_influence(
        surv,
        panel.n_periods[mask],
        panel.event[mask],
        w,
        revenue=panel.revenue[mask] if (estimated and panel.revenue is not None) else None,
    )
    return surv, value, inf, w


def _unadjusted(panel, horizon, weights, allow_extrapolation):
    arms: dict[str, ArmSummary] = {}
    values, infs = {}, {}
    for a in (0, 1):
        mask = panel.arm == a
        surv, value, inf, w = _fit_arm(panel, mask, horizon, weights[a], allow_extrapolation)
        values[a], infs[a] = value, inf
        arms[panel.arm_labels[a]] = ArmSummary(
            label=panel.arm_labels[a],
            n=int(mask.sum()),
            value=value,
            survival=surv.survival,
            at_risk=surv.at_risk,
            weights=w,
        )

    estimate = values[1] - values[0]
    n = panel.n_subjects
    psi_stacked = contrast_influence(infs[1], infs[0], n)
    psi = np.empty(n)
    psi[panel.arm == 1] = psi_stacked[: infs[1].size]
    psi[panel.arm == 0] = psi_stacked[infs[1].size :]

    # The control arm's own value, on the whole-sample scale, for the ratio interval.
    psi_control = np.zeros(n)
    psi_control[panel.arm == 0] = infs[0] / (infs[0].size / n)

    return {
        "estimate": estimate,
        "se": float(np.sqrt((psi**2).sum()) / n),
        "arms": arms,
        "influence": psi,
        "control_influence": psi_control,
        "inference": "influence",
    }


# -------------------------------------------------------------- stratified


def _stratified(panel, horizon, weights, strata, allow_extrapolation, notes):
    """Post-stratified contrast with an exact influence function.

    Writing ``Delta = sum_k pi_k Delta_k``, the influence function picks up two
    terms: the within-stratum estimation error, rescaled by the stratum's arm
    shares, and the error in the stratum shares themselves::

        IF_i = (+/-) iota_i / p_{a|k_i}  +  (Delta_{k_i} - Delta)

    Dropping the second term is a common and quiet mistake; it understates the
    variance whenever the effect genuinely differs across strata, which is
    exactly when someone reaches for stratification.
    """
    if panel.covariates is None:
        raise ValueError("Panel carries no covariates; rebuild it with covariates=[...] to stratify.")
    missing = [c for c in strata if c not in panel.covariates.columns]
    if missing:
        raise ValueError(f"Strata column(s) {missing} not in the panel's covariates.")

    as_str = panel.covariates[strata].astype(str)
    first = as_str[strata[0]]
    key = (
        first.str.cat([as_str[c] for c in strata[1:]], sep="|").to_numpy()
        if len(strata) > 1
        else first.to_numpy()
    )
    levels, codes = np.unique(key, return_inverse=True)

    n = panel.n_subjects
    psi = np.zeros(n)
    psi_control = np.zeros(n)
    control_values = np.zeros(len(levels))
    deltas = np.zeros(len(levels))
    shares = np.zeros(len(levels))
    per_arm_value = {0: 0.0, 1: 0.0}
    per_arm_surv = {0: np.zeros(horizon), 1: np.zeros(horizon)}
    per_arm_risk = {0: np.zeros(horizon, dtype=np.int64), 1: np.zeros(horizon, dtype=np.int64)}
    kept = np.zeros(n, dtype=bool)

    for k in range(len(levels)):
        in_k = codes == k
        sizes = {a: int((in_k & (panel.arm == a)).sum()) for a in (0, 1)}
        if min(sizes.values()) < 2:
            notes.append(f"stratum {levels[k]!r} dropped: {sizes[0]} control / {sizes[1]} treatment subjects")
            continue
        kept |= in_k
        shares[k] = in_k.sum()
        vals, infos = {}, {}
        for a in (0, 1):
            mask = in_k & (panel.arm == a)
            surv, value, inf, w = _fit_arm(panel, mask, horizon, weights[a], allow_extrapolation)
            vals[a], infos[a] = value, inf
            p_ak = sizes[a] / in_k.sum()
            psi[mask] = (1 if a == 1 else -1) * inf / p_ak
            if a == 0:
                psi_control[mask] = inf / p_ak
                control_values[k] = value
            per_arm_surv[a] += surv.survival * in_k.sum()
            per_arm_risk[a] += surv.at_risk
        deltas[k] = vals[1] - vals[0]

    total = shares.sum()
    if total == 0:
        raise NotIdentifiedError("Every stratum was too small to estimate; use coarser strata.")
    if kept.sum() < n:
        notes.append(f"{n - int(kept.sum())} of {n} subjects fell in dropped strata and were excluded")
    pi = shares / total
    estimate = float(pi @ deltas)

    control_total = float(pi @ control_values)
    psi = psi[kept] + (deltas[codes[kept]] - estimate)
    psi_control = psi_control[kept] + (control_values[codes[kept]] - control_total)
    m = int(kept.sum())
    se = float(np.sqrt((psi**2).sum()) / m)

    arms = {}
    for a in (0, 1):
        label = panel.arm_labels[a]
        surv = per_arm_surv[a] / total
        value = float(np.sum(weights[a][0] * np.concatenate(([1.0], surv[:-1]))))
        per_arm_value[a] = value
        arms[label] = ArmSummary(
            label=label,
            n=int(((panel.arm == a) & kept).sum()),
            value=value,
            survival=surv,
            at_risk=per_arm_risk[a],
            weights=weights[a][0],
        )

    return {
        "estimate": estimate,
        "se": se,
        "arms": arms,
        "influence": psi,
        "control_influence": psi_control,
        "inference": "influence",
    }


# ---------------------------------------------------------------- adjusted


def _person_period(panel: SubscriberPanel, horizon: int):
    """Expand to one row per subject-period at risk, the discrete-survival likelihood's unit."""
    capped = np.minimum(panel.n_periods, horizon)
    starts = np.concatenate(([0], np.cumsum(capped)[:-1]))
    rows = np.repeat(np.arange(panel.n_subjects), capped)
    period = np.arange(capped.sum()) - np.repeat(starts, capped) + 1
    churn = np.zeros(period.size, dtype=float)
    ends = np.cumsum(capped) - 1
    churn[ends] = (panel.event & (panel.n_periods <= horizon)).astype(float)
    return rows, period.astype(np.int64), churn, capped, starts


def _adjusted(
    panel,
    horizon,
    weights,
    covariates,
    n_boot,
    alpha,
    allow_extrapolation,
    seed,
    inference="influence",
    censoring_covariates=None,
):
    """Covariate-adjusted g-computation, corrected by its efficient influence function.

    The plain g-computation estimator standardizes a fitted hazard model over the
    covariate distribution. It is consistent under randomization with a saturated
    time baseline, but it has no tractable influence function, which is why it
    used to be stuck with the bootstrap -- and therefore with no way to monitor a
    running experiment.

    Adding the augmentation term of the efficient influence function fixes both
    problems at once. The one-step estimator

        V_a = (1/n) sum_i S_a(t|X_i) * (1 - Q_i(t)),  summed against the weights

    where ``Q_i`` accumulates inverse-censoring-weighted residuals from the
    subject's own observed renewal decisions, is asymptotically linear with a
    known influence function. That buys three things: a standard error that does
    not cost 200 model refits, an anytime-valid confidence sequence for the
    estimator that reduces variance the most, and double robustness -- the
    augmentation keeps the estimate consistent even where the hazard model is
    wrong, because assignment is randomized and the censoring distribution is
    known rather than modelled.

    The price is the inverse-censoring weight ``1/Gbar(s-1)``. When very few
    subscribers could have been observed to the horizon those weights blow up,
    which :mod:`sublift.censoring` warns about rather than absorbing silently.

    The other price is asymptotic. Being asymptotically linear is a large-sample
    property, and in simulation this estimator's intervals cover at about 93% at
    4,000 subscribers, reaching the nominal 95% by roughly 16,000. Below a few
    thousand, prefer ``estimator="stratified"``, whose influence function is
    exact in finite samples rather than first-order.
    """
    if panel.covariates is None:
        raise ValueError("Panel carries no covariates; rebuild it with covariates=[...] to adjust.")
    missing = [c for c in covariates if c not in panel.covariates.columns]
    if missing:
        raise ValueError(f"Covariate(s) {missing} not in the panel's covariates.")
    if inference not in ("influence", "bootstrap"):
        raise ValueError("inference must be 'influence' or 'bootstrap'.")

    X_sub, _, _ = design_matrix(panel.covariates[covariates])
    rows, period, churn, capped, starts = _person_period(panel, horizon)

    if inference == "bootstrap":
        estimate, arms = _gcomp(panel, horizon, weights, X_sub, rows, period, churn)
        rng = np.random.default_rng(seed)
        n = panel.n_subjects
        draws, rel_draws = np.empty(n_boot), np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            diff, ctrl = _gcomp_boot(panel, horizon, weights, X_sub, idx, capped, starts, period, churn)
            draws[b] = diff
            rel_draws[b] = diff / ctrl if ctrl else np.nan
        draws = draws[np.isfinite(draws)]
        rel_draws = rel_draws[np.isfinite(rel_draws)]
        return {
            "estimate": estimate,
            "se": float(draws.std(ddof=1)) if draws.size > 1 else float("nan"),
            "arms": arms,
            "boot": draws,
            "boot_relative": rel_draws,
            "inference": "bootstrap",
        }

    if censoring_covariates:
        gbar = conditional_censoring_survival(panel, horizon, list(censoring_covariates))
        gbar_source = "modelled on " + ", ".join(censoring_covariates)
    else:
        marginal, gbar_source = censoring_survival(panel, horizon)
        gbar = marginal[None, :]
    n = panel.n_subjects
    if n < _ADJUSTED_MIN_N:
        warnings.warn(
            f"estimator='adjusted' with {n:,} subscribers: its influence function is a "
            "large-sample approximation, and below a few thousand subjects the intervals run "
            "slightly narrow (about 93% coverage at 4,000 in simulation). Prefer "
            "estimator='stratified', whose influence function is exact at any size, or pass "
            "inference='bootstrap'.",
            stacklevel=4,
        )
    t_grid = np.arange(1, horizon + 1, dtype=np.int64)[None, :]
    at_risk = panel.n_periods[:, None] >= t_grid
    churned_at = (panel.n_periods[:, None] == t_grid) & panel.event[:, None]

    values, eifs, arms, notes = {}, {}, {}, []
    for a in (0, 1):
        mask = panel.arm[rows] == a
        fit = _fit_hazards(mask, X_sub, rows, period, churn, horizon)
        hazard_i = _individual_hazard(fit, X_sub, horizon)
        surv_i = np.cumprod(1.0 - hazard_i, axis=1)

        share = float((panel.arm == a).mean())
        in_arm = (panel.arm == a)[:, None]
        residual = churned_at.astype(float) - at_risk * hazard_i
        # Inverse-censoring-weighted residuals, divided through by the subject's own
        # survival so the running sum telescopes into S(t|X)/S(s|X) -- a ratio that is
        # always <= 1, which is what keeps the augmentation bounded.
        contribution = in_arm * residual / (share * gbar)
        accumulated = np.cumsum(contribution / np.maximum(surv_i, 1e-12), axis=1)

        corrected = surv_i * (1.0 - accumulated)
        lagged = np.concatenate((np.ones((n, 1)), corrected[:, :-1]), axis=1)
        curve_lagged = lagged.mean(axis=0)
        curve = corrected.mean(axis=0)

        w = weights[a][0]
        values[a] = float((w * curve_lagged).sum())
        # Mean zero by construction, since the curve is the sample mean of `lagged`.
        eifs[a] = (w * (lagged - curve_lagged[None, :])).sum(axis=1)

        if np.any(np.diff(curve) > 1e-9):
            notes.append(
                f"the {panel.arm_labels[a]!r} corrected survival curve is not monotone, which "
                "means the influence-function correction is large -- the hazard model is fitting "
                "poorly, so prefer estimator='stratified' here"
            )
        arms[panel.arm_labels[a]] = ArmSummary(
            label=panel.arm_labels[a],
            n=int((panel.arm == a).sum()),
            value=values[a],
            survival=curve,
            at_risk=np.asarray(
                [int((panel.n_periods[panel.arm == a] >= t).sum()) for t in range(1, horizon + 1)]
            ),
            weights=w,
        )

    psi = eifs[1] - eifs[0]
    notes.append(f"censoring distribution: {gbar_source}")
    return {
        "estimate": values[1] - values[0],
        "se": float(np.sqrt((psi**2).sum()) / n),
        "arms": arms,
        "influence": psi,
        "control_influence": -eifs[0],
        "inference": "influence",
        "notes": notes,
    }


def _individual_hazard(fit, X_all, horizon):
    """Per-subject hazard h(t | X_i) under one arm, shape (n_subjects, horizon)."""
    alpha_t = fit.coef[:horizon]
    gamma = fit.coef[horizon:]
    offset = X_all @ gamma if gamma.size else np.zeros(X_all.shape[0])
    lin = alpha_t[None, :] + offset[:, None]
    return 1.0 / (1.0 + np.exp(-lin))


def _fit_hazards(arm_rows_mask, X_sub, rows, period, churn, horizon):
    """Arm-specific hazard model: saturated time baseline plus covariates."""
    r, t, y = rows[arm_rows_mask], period[arm_rows_mask], churn[arm_rows_mask]
    time_dummies = np.zeros((t.size, horizon))
    time_dummies[np.arange(t.size), t - 1] = 1.0
    X = np.hstack([time_dummies, X_sub[r]]) if X_sub.size else time_dummies
    return fit_logistic(X, y)


def _predict_curve(fit, X_all, horizon):
    """Standardized survival: predict every subject under one arm, then average."""
    n = X_all.shape[0]
    alpha_t = fit.coef[:horizon]
    gamma = fit.coef[horizon:]
    offset = X_all @ gamma if gamma.size else np.zeros(n)
    lin = alpha_t[None, :] + offset[:, None]
    haz = 1.0 / (1.0 + np.exp(-lin))
    surv_i = np.cumprod(1.0 - haz, axis=1)
    return surv_i.mean(axis=0)


def _assemble(panel, horizon, weights, curves, counts):
    arms, values = {}, {}
    for a in (0, 1):
        label = panel.arm_labels[a]
        surv = curves[a]
        w = weights[a][0]
        value = float(np.sum(w * np.concatenate(([1.0], surv[:-1]))))
        values[a] = value
        arms[label] = ArmSummary(
            label=label,
            n=counts[a],
            value=value,
            survival=surv,
            at_risk=np.array(
                [int((panel.n_periods[panel.arm == a] >= t).sum()) for t in range(1, horizon + 1)]
            ),
            weights=w,
        )
    return values[1] - values[0], arms


def _gcomp(panel, horizon, weights, X_sub, rows, period, churn):
    curves = {}
    for a in (0, 1):
        mask = panel.arm[rows] == a
        fit = _fit_hazards(mask, X_sub, rows, period, churn, horizon)
        curves[a] = _predict_curve(fit, X_sub, horizon)
    counts = {a: int((panel.arm == a).sum()) for a in (0, 1)}
    return _assemble(panel, horizon, weights, curves, counts)


def _gcomp_boot(panel, horizon, weights, X_sub, idx, capped, starts, period, churn):
    lens = capped[idx]
    total = int(lens.sum())
    if total == 0:
        return np.nan, np.nan
    offsets = np.concatenate(([0], np.cumsum(lens)[:-1]))
    gather = np.repeat(starts[idx] - offsets, lens) + np.arange(total)
    b_rows = np.repeat(np.arange(idx.size), lens)
    b_period, b_churn = period[gather], churn[gather]
    X_b = X_sub[idx] if X_sub.size else X_sub
    arm_b = panel.arm[idx]

    curves = {}
    for a in (0, 1):
        mask = arm_b[b_rows] == a
        if not mask.any():
            return np.nan, np.nan
        fit = _fit_hazards(mask, X_b, b_rows, b_period, b_churn, horizon)
        curves[a] = _predict_curve(fit, X_b, horizon)
    values = [float(np.sum(weights[a][0] * np.concatenate(([1.0], curves[a][:-1])))) for a in (0, 1)]
    return values[1] - values[0], values[0]
