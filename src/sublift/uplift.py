"""Who to target: cross-fitted uplift scores and a censoring-correct Qini curve.

An average effect answers "did this work". A targeting question -- "who should
we send it to" -- needs per-subscriber effects, and the usual recipe (train an
uplift model on a binary 30-day churn flag, plot a Qini curve) reintroduces
every problem the rest of this library exists to avoid. A binary flag throws
away the censored subscribers. Evaluating the model on the data that fit it
makes the curve a picture of the model's confidence rather than of the effect.

So: per-subscriber effects are on the same restricted-mean scale as the headline
estimate; scores are **cross-fitted**, so nobody is ranked by a model that saw
them; and each point of the Qini curve is a real censoring-aware estimate re-run
inside the targeted subset, not a sum of predicted scores.

Why the default learner is not a T-learner
------------------------------------------
The obvious construction -- fit a hazard model per arm, difference the predicted
curves -- is a T-learner, and on subscription data it is usually worse than
useless. Predicted heterogeneity is driven by ``(gamma_treat - gamma_control)'x``,
the difference between two independently estimated covariate vectors. When the
treatment does not genuinely modify the effect of a covariate on the logit
scale, that difference is *pure estimation noise*, and it does not average out
per subscriber: it fans the scores out around the truth. Measured on this
library's own simulator with a constant odds ratio, a T-learner produces scores
nearly three times more dispersed than the true individual effects, correlating
about 0.26 with them, and targeting on it does worse than targeting at random.

The default learner instead fits **one pooled model** with shared time and
covariate effects, a treatment main effect, and treatment-by-covariate
interactions that are **ridge-penalized toward zero**. Real effect modification
survives the penalty; noise does not. What remains when the interactions shrink
away is the heterogeneity that is actually there in a survival contrast even
with a constant odds ratio -- subscribers on a steeper part of the hazard curve
have more retention to gain -- and that part is carried by the shared main
effects, which are estimated from both arms at once and are correspondingly
precise.

The penalty is chosen from the data by held-out likelihood, so this is not a
bet on heterogeneity being small. Against the same simulator, correlation with
the true individual effect:

===========================  ==========  ================
regime                       T-learner   pooled + shrunk
===========================  ==========  ================
constant odds ratio               0.26              0.62
strong effect modification        0.94              0.97
===========================  ==========  ================

The pooled model wins in both, which is the argument for it being the default.
Pass ``learner="t"`` for the naive version if you want to see the difference on
your own data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimators import _arm_weights, _fit_hazards, _person_period, incremental_ltv, retained_periods_lift
from .logistic import design_matrix, fit_logistic
from .panel import SubscriberPanel

__all__ = ["uplift_scores", "qini", "QiniCurve"]

_LEARNERS = ("interaction", "t")


@dataclass
class QiniCurve:
    frame: pd.DataFrame
    overall_effect: float
    qini_auc: float
    metric: str
    horizon: int

    def best_fraction(self) -> pd.Series:
        """The targeting fraction with the largest population-level incremental value."""
        valid = self.frame.dropna(subset=["incremental_per_subscriber"])
        if valid.empty:
            raise ValueError("No targeting fraction produced a usable estimate.")
        return valid.loc[valid["incremental_per_subscriber"].idxmax()]

    def __str__(self) -> str:
        unit = "revenue" if self.metric == "ltv" else "periods"
        try:
            best = self.best_fraction()
            head = (
                f"Best targeting fraction: top {best['fraction']:.0%} "
                f"({int(best['n_targeted']):,} subscribers), "
                f"{best['incremental_per_subscriber']:+.4f} {unit} per subscriber in the population"
            )
        except ValueError:
            head = "No usable targeting fraction."
        return "\n".join(
            [
                head,
                f"Treating everyone: {self.overall_effect:+.4f} {unit} per subscriber",
                f"Qini AUC over random targeting: {self.qini_auc:+.4f}",
                "",
                self.frame.to_string(index=False, float_format=lambda v: f"{v:,.4f}"),
            ]
        )


def uplift_scores(
    panel: SubscriberPanel,
    *,
    horizon: int,
    covariates: list[str],
    metric: str = "retained_periods",
    price=None,
    learner: str = "interaction",
    interaction_prior_sd: float | str = "cv",
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Cross-fitted per-subscriber incremental value, on the restricted-mean scale.

    Parameters
    ----------
    learner
        ``"interaction"`` (default) fits one pooled hazard model with penalized
        treatment-by-covariate interactions. ``"t"`` fits an independent model
        per arm; see the module docstring for why that is rarely what you want.
    interaction_prior_sd
        Scale of the shrinkage on the interaction terms, as a prior standard
        deviation on the logit scale with covariates standardized. The default
        ``"cv"`` picks it from the data by held-out person-period likelihood
        inside each training fold, which is the point: when the intervention
        really does land differently across segments the penalty relaxes, and
        when it does not the penalty collapses the interactions and the score
        falls back on the shared main effects. Pass a float to fix it.
    """
    if panel.covariates is None:
        raise ValueError("Panel carries no covariates; rebuild it with covariates=[...].")
    missing = [c for c in covariates if c not in panel.covariates.columns]
    if missing:
        raise ValueError(f"Covariate(s) {missing} not in the panel's covariates.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2 for cross-fitting.")
    if learner not in _LEARNERS:
        raise ValueError(f"learner must be one of {_LEARNERS}, got {learner!r}.")
    if not (interaction_prior_sd == "cv" or (np.isscalar(interaction_prior_sd) and interaction_prior_sd > 0)):
        raise ValueError('interaction_prior_sd must be a positive number or "cv".')

    weights = _arm_weights(panel, horizon, metric, price)
    X_all, _, _ = design_matrix(panel.covariates[covariates])

    n = panel.n_subjects
    fold = np.random.default_rng(seed).permutation(n) % n_folds
    scores = np.full(n, np.nan)

    for k in range(n_folds):
        test_mask = fold == k
        train = panel.subset(~test_mask)
        if len(np.unique(train.arm)) < 2:
            continue
        curves = _fit_and_predict(
            train, X_all[~test_mask], X_all[test_mask], horizon, learner, interaction_prior_sd
        )
        if curves is None:
            continue
        values = {}
        for a in (0, 1):
            lagged = np.concatenate((np.ones((curves[a].shape[0], 1)), curves[a][:, :-1]), axis=1)
            values[a] = (lagged * weights[a][0][None, :]).sum(axis=1)
        scores[test_mask] = values[1] - values[0]

    return scores


def qini(
    panel: SubscriberPanel,
    *,
    horizon: int,
    covariates: list[str],
    metric: str = "retained_periods",
    price=None,
    learner: str = "interaction",
    interaction_prior_sd: float | str = "cv",
    fractions: np.ndarray | None = None,
    n_folds: int = 5,
    min_per_arm: int = 200,
    seed: int = 0,
) -> QiniCurve:
    """Incremental value as a function of how much of the base you target.

    For each fraction ``q``, the effect is re-estimated *inside* the top-``q``
    subscribers by predicted uplift, using the same censoring-aware estimator as
    the headline number, then scaled by ``q`` to express it per subscriber in the
    whole population. The comparison point is random targeting,
    ``q * overall_effect``: a model only earns its place by beating that line.
    """
    scores = uplift_scores(
        panel,
        horizon=horizon,
        covariates=covariates,
        metric=metric,
        price=price,
        learner=learner,
        interaction_prior_sd=interaction_prior_sd,
        n_folds=n_folds,
        seed=seed,
    )
    if np.isnan(scores).all():
        raise ValueError("No fold produced usable uplift scores; check arm sizes.")

    fn = incremental_ltv if metric == "ltv" else retained_periods_lift
    kwargs = {"price": price} if metric == "ltv" else {}
    overall = fn(panel, horizon=horizon, estimator="unadjusted", allow_extrapolation=True, **kwargs)
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]

    if fractions is None:
        fractions = np.arange(0.1, 1.01, 0.1)
    rows = []
    n = panel.n_subjects

    for q in fractions:
        take = int(round(q * n))
        mask = np.zeros(n, dtype=bool)
        mask[order[:take]] = True
        sub = panel.subset(mask)
        effect, se = np.nan, np.nan
        if min(int((sub.arm == a).sum()) for a in (0, 1)) >= min_per_arm:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = fn(sub, horizon=horizon, estimator="unadjusted", allow_extrapolation=True, **kwargs)
                effect, se = res.estimate, res.se
            except (ValueError, np.linalg.LinAlgError):
                pass
        rows.append(
            {
                "fraction": float(q),
                "n_targeted": take,
                "effect_in_targeted": effect,
                "se": se,
                "incremental_per_subscriber": q * effect,
                "random_targeting": q * overall.estimate,
            }
        )

    frame = pd.DataFrame(rows)
    valid = frame.dropna(subset=["incremental_per_subscriber"])
    auc = (
        float(
            np.trapezoid(valid["incremental_per_subscriber"] - valid["random_targeting"], valid["fraction"])
        )
        if len(valid) > 1
        else float("nan")
    )
    return QiniCurve(
        frame=frame, overall_effect=overall.estimate, qini_auc=auc, metric=metric, horizon=horizon
    )


# ------------------------------------------------------------------- learners


def _fit_and_predict(train, X_train, X_test, horizon, learner, prior_sd):
    rows, period, churn, _, _ = _person_period(train, horizon)
    if learner == "t":
        curves = {}
        for a in (0, 1):
            arm_rows = train.arm[rows] == a
            if not arm_rows.any():
                return None
            fit = _fit_hazards(arm_rows, X_train, rows, period, churn, horizon)
            curves[a] = _survival_from(fit.coef[:horizon], X_test @ fit.coef[horizon:], 0.0)
        return curves
    return _pooled(train, X_train, X_test, rows, period, churn, horizon, prior_sd)


_CV_GRID = (0.005, 0.02, 0.05, 0.12, 0.3, 0.8)


def _pooled_design(train, X_train, rows, period, horizon):
    arm = train.arm[rows].astype(float)
    Xr = X_train[rows]
    time_d = np.zeros((period.size, horizon))
    time_d[np.arange(period.size), period - 1] = 1.0
    return np.hstack([time_d, Xr, arm[:, None], arm[:, None] * Xr])


def _ridge_vector(horizon, p, prior_sd):
    """No penalty on the baseline, the main effects, or the average treatment effect."""
    return np.concatenate([np.zeros(horizon + p + 1), np.full(p, 1.0 / prior_sd**2)]) + 1e-8


def _select_prior_sd(design, churn, rows, horizon, p, n_subjects, seed, inner_folds=3):
    """Pick the shrinkage by held-out person-period log-likelihood.

    Split on *subjects*, not person-period rows: rows from one subscriber are
    not independent, and splitting on rows would leak a subscriber's own future
    into their training set and make every penalty look equally good.
    """
    fold_of_subject = np.random.default_rng(seed).permutation(n_subjects) % inner_folds
    fold_of_row = fold_of_subject[rows]
    scores = np.zeros(len(_CV_GRID))

    for k in range(inner_folds):
        tr, te = fold_of_row != k, fold_of_row == k
        if not te.any() or not tr.any():
            continue
        for j, sd in enumerate(_CV_GRID):
            fit = fit_logistic(design[tr], churn[tr], ridge=_ridge_vector(horizon, p, sd))
            eta = design[te] @ fit.coef
            scores[j] += float(np.sum(churn[te] * eta - np.logaddexp(0.0, eta)))
    return _CV_GRID[int(np.argmax(scores))]


def _pooled(train, X_train, X_test, rows, period, churn, horizon, prior_sd):
    """One model: shared baseline and covariate effects, penalized interactions."""
    p = X_train.shape[1]
    design = _pooled_design(train, X_train, rows, period, horizon)

    if prior_sd == "cv":
        prior_sd = _select_prior_sd(design, churn, rows, horizon, p, train.n_subjects, seed=0)

    fit = fit_logistic(design, churn, ridge=_ridge_vector(horizon, p, prior_sd))

    alpha = fit.coef[:horizon]
    gamma = fit.coef[horizon : horizon + p]
    delta = float(fit.coef[horizon + p])
    theta = fit.coef[horizon + p + 1 :]

    base = X_test @ gamma if p else np.zeros(X_test.shape[0])
    tau = delta + (X_test @ theta if p else 0.0)
    return {0: _survival_from(alpha, base, 0.0), 1: _survival_from(alpha, base, tau)}


def _survival_from(alpha, offset, shift):
    """Per-subject survival curves, shape (n_subjects, horizon)."""
    offset = np.asarray(offset, dtype=float)
    shift = np.asarray(shift, dtype=float)
    lin = alpha[None, :] + offset[:, None] + (shift.reshape(-1, 1) if shift.ndim else shift)
    return np.cumprod(1.0 - 1.0 / (1.0 + np.exp(-lin)), axis=1)
