"""A small, self-contained logistic regression.

sublift depends on numpy, scipy and pandas and nothing else, so the discrete
hazard model is fitted here rather than pulled in from statsmodels or sklearn.
That is not only about dependency weight: Newton-IRLS hands back the Hessian as
a by-product, and the Hessian is the bread of the sandwich estimator used for
standard errors on hazard coefficients. Borrowing a solver would mean borrowing
its variance assumptions too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["LogisticFit", "fit_logistic", "design_matrix"]


@dataclass(frozen=True)
class LogisticFit:
    coef: np.ndarray
    hessian: np.ndarray
    n_iter: int
    converged: bool
    names: list[str] | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _expit(np.asarray(X, dtype=float) @ self.coef)

    def sandwich_se(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Robust standard errors. Person-period rows within a subject are not
        independent, so the model-based errors are optimistic; callers that need
        honest coefficient errors should cluster, and callers that need honest
        *effect* errors should use the influence-function or bootstrap paths
        instead of reading anything off the coefficients."""
        X = np.asarray(X, dtype=float)
        resid = (np.asarray(y, dtype=float) - self.predict(X))[:, None]
        meat = (X * resid).T @ (X * resid)
        bread = np.linalg.pinv(self.hessian)
        return np.sqrt(np.diag(bread @ meat @ bread))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float | np.ndarray = 1e-8,
    max_iter: int = 100,
    tol: float = 1e-10,
    names: list[str] | None = None,
) -> LogisticFit:
    """Newton-IRLS with a ridge floor and step halving.

    The ridge term is tiny by default -- enough to keep the Hessian invertible
    under separation (a covariate level where nobody ever churns, which happens
    constantly in real subscriber data once you stratify), not enough to shrink
    estimates anyone would notice.

    ``ridge`` may also be a per-coefficient vector, which is how the uplift model
    penalizes its treatment-by-covariate interactions while leaving the main
    effects alone.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X has {n} rows but y has {y.shape[0]}.")

    coef = np.zeros(p)
    ridge_vec = np.broadcast_to(np.asarray(ridge, dtype=float), (p,)).copy()
    if (ridge_vec < 0).any():
        raise ValueError("ridge penalties must be non-negative.")
    penalty = np.diag(ridge_vec)
    prev_ll = -np.inf
    converged = False
    it = 0

    for it in range(1, max_iter + 1):  # noqa: B007 -- read after the loop, as n_iter
        eta = X @ coef
        mu = _expit(eta)
        w = np.clip(mu * (1 - mu), 1e-10, None)
        grad = X.T @ (y - mu) - ridge_vec * coef
        hess = (X * w[:, None]).T @ X + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]

        # Step halving: guard against the overshoot IRLS produces near separation.
        scale = 1.0
        for _ in range(30):
            cand = coef + scale * step
            ll = _loglik(X, y, cand, ridge_vec)
            if ll >= prev_ll - 1e-12:
                break
            scale *= 0.5
        else:
            cand = coef
            ll = prev_ll

        shift = np.max(np.abs(cand - coef))
        coef = cand
        if shift < tol or abs(ll - prev_ll) < tol:
            prev_ll = ll
            converged = True
            break
        prev_ll = ll

    eta = X @ coef
    mu = _expit(eta)
    w = np.clip(mu * (1 - mu), 1e-10, None)
    hess = (X * w[:, None]).T @ X + penalty
    return LogisticFit(coef=coef, hessian=hess, n_iter=it, converged=converged, names=names)


def design_matrix(
    frame: pd.DataFrame | None,
    *,
    reference: dict[str, object] | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Numeric design matrix from mixed covariates, with a reusable encoding.

    Categoricals become one-hot with the first observed level dropped; numerics
    are standardized. The returned ``encoding`` must be reused when scoring new
    rows (bootstrap resamples, cross-fitting folds, counterfactual predictions),
    or columns silently reorder and the coefficients apply to the wrong variable.
    """
    if frame is None or frame.shape[1] == 0:
        return np.zeros((0, 0)), [], {}

    enc = dict(reference or {})
    cols: list[np.ndarray] = []
    names: list[str] = []

    for col in frame.columns:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            if np.isnan(vals).any():
                raise ValueError(f"Covariate {col!r} has missing values; impute before adjusting.")
            key = f"num::{col}"
            if key not in enc:
                mu, sd = float(vals.mean()), float(vals.std())
                enc[key] = (mu, sd if sd > 1e-12 else 1.0)
            mu, sd = enc[key]
            cols.append((vals - mu) / sd)
            names.append(col)
        else:
            key = f"cat::{col}"
            if key not in enc:
                levels = [str(v) for v in pd.Series(s.astype(str)).dropna().unique()]
                enc[key] = sorted(levels)
            levels = enc[key]
            as_str = s.astype(str).to_numpy()
            for level in levels[1:]:  # drop first level as reference
                cols.append((as_str == level).astype(float))
                names.append(f"{col}={level}")

    X = np.column_stack(cols) if cols else np.zeros((len(frame), 0))
    return X, names, enc


def _expit(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _loglik(X, y, coef, ridge) -> float:
    eta = X @ coef
    ll = np.sum(y * eta - np.logaddexp(0.0, eta))
    return float(ll - 0.5 * float(np.sum(np.asarray(ridge) * coef * coef)))
