"""Voluntary versus involuntary churn, as separate competing causes.

In consumer subscriptions a large share of churn -- routinely 20-40% in media
-- is *involuntary*: a card expires, a payment fails, dunning runs out of
retries and the subscription lapses. The subscriber never decided anything.
Voluntary churn is somebody clicking cancel.

These are different events with different remedies, and treating them as one
quietly corrupts a retention programme in both directions. A save offer acts on
voluntary churn; measured against all-cause churn its effect is diluted by an
involuntary baseline it cannot move. Meanwhile a card-updater or a smarter
dunning schedule shows up as "retention improved", and the retention team gets
the credit for a payments fix.

The decomposition
-----------------
With competing causes the estimand does not change -- it is still the all-cause
survival curve that earns revenue -- but it splits exactly. Writing
``L_j`` for the billing periods lost to cause ``j`` within the horizon,

    RMST(H) = H - sum_j L_j,       L_j = sum_{s<H} (H-s) * S(s-1) * h_j(s)

where ``h_j(s)`` is the cause-specific hazard. So the incremental retained
periods decompose additively::

    Delta_total = sum_j Delta_j,   Delta_j = -(L_j^treatment - L_j^control)

and each ``Delta_j`` reads as "billing periods per subscriber saved from *this*
cause". They sum to the headline number with no residual, which is what makes
the split worth trusting: it is an identity, not an attribution heuristic.

Note what is deliberately *not* reported: a cause-specific survival curve with
the other cause censored out. That quantity ("what if nobody ever had a failed
payment?") is not identified from this data without assuming the two causes are
independent, which for subscriptions they plainly are not -- a subscriber
halfway out the door is also the one who does not bother updating their card.
The cumulative incidence above needs no such assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import NotIdentifiedError, PanelError
from .panel import SubscriberPanel
from .survival import fit_survival

__all__ = ["churn_decomposition", "ChurnDecomposition", "CauseEffect"]


@dataclass(frozen=True)
class CauseEffect:
    label: str
    periods_lost_control: float
    periods_lost_treatment: float
    estimate: float  # periods per subscriber saved from this cause
    se: float
    ci: tuple[float, float]

    @property
    def p_value(self) -> float:
        return float(2 * stats.norm.sf(abs(self.estimate / self.se))) if self.se else float("nan")


@dataclass
class ChurnDecomposition:
    horizon: int
    total: float
    total_se: float
    total_ci: tuple[float, float]
    causes: list[CauseEffect]
    arm_labels: tuple[str, str]
    alpha: float
    n_subjects: int

    def share(self, cause: CauseEffect) -> float:
        """Fraction of the overall retention effect running through this cause."""
        return cause.estimate / self.total if self.total else float("nan")

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "cause": c.label,
                f"periods_lost_{self.arm_labels[0]}": c.periods_lost_control,
                f"periods_lost_{self.arm_labels[1]}": c.periods_lost_treatment,
                "periods_saved": c.estimate,
                "se": c.se,
                "ci_low": c.ci[0],
                "ci_high": c.ci[1],
                "share_of_effect": self.share(c),
            }
            for c in self.causes
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        head = f"Retention effect by cause of churn, over {self.horizon} billing periods"
        lines = [
            head,
            "=" * len(head),
            f"  total  {self.total:+.4f} periods per subscriber "
            f"[{self.total_ci[0]:+.4f}, {self.total_ci[1]:+.4f}]",
            "",
        ]
        width = max(len(c.label) for c in self.causes)
        for c in self.causes:
            lines.append(
                f"  {c.label:<{width}s}  {c.estimate:+.4f}  "
                f"[{c.ci[0]:+.4f}, {c.ci[1]:+.4f}]   {self.share(c):>6.0%} of the effect"
            )
        lines += [
            "",
            "  Causes sum to the total exactly; the split is an identity, not an attribution.",
            f"  {1 - self.alpha:.0%} intervals, {self.n_subjects:,} subscribers.",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        from .report import html_decomposition

        return html_decomposition(self)


def churn_decomposition(
    panel: SubscriberPanel,
    *,
    horizon: int | None = None,
    alpha: float = 0.05,
    allow_extrapolation: bool = False,
) -> ChurnDecomposition:
    """Split the incremental retained periods by cause of churn.

    Requires a panel built with ``cause=`` (see :meth:`SubscriberPanel.from_spans`).
    Nonparametric, like ``estimator="unadjusted"``; the total it reports is
    identical to :func:`retained_periods_lift` at the same horizon.
    """
    if panel.cause is None:
        raise PanelError(
            "This panel has no cause of churn. Rebuild it with cause='<column>' -- typically a "
            "column distinguishing voluntary cancellation from involuntary/payment churn."
        )
    if panel.n_arms > 2:
        raise NotIdentifiedError(
            f"churn_decomposition compares two arms; this panel has {panel.n_arms}. "
            "Use panel.contrast('<arm>') to choose one comparison."
        )
    horizon = int(horizon) if horizon is not None else panel.followup
    if horizon > panel.followup and not allow_extrapolation:
        raise NotIdentifiedError(
            f"horizon={horizon} exceeds the {panel.followup} periods of follow-up both arms have."
        )

    n_causes = len(panel.cause_labels)
    per_arm = {}
    for a in (0, 1):
        mask = panel.arm == a
        per_arm[a] = _arm_decomposition(
            panel.n_periods[mask],
            panel.event[mask],
            panel.cause[mask],
            horizon,
            n_causes,
            allow_extrapolation,
        )

    n = panel.n_subjects
    p = {a: float((panel.arm == a).sum()) / n for a in (0, 1)}

    causes: list[CauseEffect] = []
    total_psi = np.zeros(n)
    z = float(stats.norm.ppf(1 - alpha / 2))

    for j, label in enumerate(panel.cause_labels):
        lost = {a: per_arm[a]["lost"][j] for a in (0, 1)}
        estimate = -(lost[1] - lost[0])
        psi = np.zeros(n)
        # Saved periods are minus the change in periods lost, so the influence
        # terms carry the opposite sign to the usual contrast.
        psi[panel.arm == 1] = -per_arm[1]["influence"][j] / p[1]
        psi[panel.arm == 0] = per_arm[0]["influence"][j] / p[0]
        se = float(np.sqrt((psi**2).sum()) / n)
        causes.append(
            CauseEffect(
                label=label,
                periods_lost_control=lost[0],
                periods_lost_treatment=lost[1],
                estimate=estimate,
                se=se,
                ci=(estimate - z * se, estimate + z * se),
            )
        )
        total_psi += psi

    total = float(sum(c.estimate for c in causes))
    total_se = float(np.sqrt((total_psi**2).sum()) / n)
    return ChurnDecomposition(
        horizon=horizon,
        total=total,
        total_se=total_se,
        total_ci=(total - z * total_se, total + z * total_se),
        causes=causes,
        arm_labels=panel.arm_labels,
        alpha=alpha,
        n_subjects=n,
    )


def _arm_decomposition(n_periods, event, cause, horizon, n_causes, allow_extrapolation):
    """Periods lost to each cause, and the influence function of each.

    ``L_j = sum_{s<H} (H-s) S(s-1) h_j(s)`` perturbs through both the survival
    curve and the cause-specific hazard, so the influence function has two
    terms::

        IF(L_j) = sum_s (H-s) [ h_j(s) IF(S(s-1)) + S(s-1) IF(h_j(s)) ]

    with the product-limit influence ``IF(S(t)) = -S(t) * cumsum_u A(u)`` and
    ``IF(h_j(s)) = [dN_j(s) - Y(s) h_j(s)] / pi(s)``.
    """
    surv = fit_survival(n_periods, event, horizon, allow_extrapolation=allow_extrapolation)
    n = n_periods.size
    H = horizon

    t_grid = np.arange(1, H + 1, dtype=np.int64)[None, :]
    at_risk_i = n_periods[:, None] >= t_grid  # Y_i(t)
    churn_i = (n_periods[:, None] == t_grid) & event[:, None]  # dN_i(t), any cause

    pi = surv.at_risk_fraction
    haz = surv.hazard
    s_lag = surv.survival_lagged  # S(t-1) for t = 1..H

    # A_i(u) and its running sum C_i(t) = sum_{u<=t} A_i(u), with C_i(0) = 0.
    denom = pi * (1.0 - haz)
    a_scale = np.divide(1.0, denom, out=np.zeros_like(denom), where=denom > 0)
    martingale = churn_i.astype(float) - at_risk_i * haz[None, :]
    A = martingale * a_scale[None, :]
    C = np.cumsum(A, axis=1)
    C_lag = np.concatenate((np.zeros((n, 1)), C[:, :-1]), axis=1)  # C_i(t-1)

    # Only periods s = 1..H-1 can cost anything inside the horizon: a subscription
    # ending in period H forfeits no period that the horizon would have counted.
    g = np.maximum(H - np.arange(1, H + 1), 0).astype(float)
    inv_pi = np.divide(1.0, pi, out=np.zeros_like(pi), where=pi > 0)

    lost = np.zeros(n_causes)
    influence = np.zeros((n_causes, n))
    for j in range(n_causes):
        churn_ij = (n_periods[:, None] == t_grid) & event[:, None] & (cause[:, None] == j)
        d_j = churn_ij.sum(axis=0)
        haz_j = np.divide(d_j, surv.at_risk, out=np.zeros(H), where=surv.at_risk > 0)

        lost[j] = float(np.sum(g * s_lag * haz_j))

        term_survival = -(g * haz_j * s_lag)[None, :] * C_lag
        term_hazard = (g * s_lag * inv_pi)[None, :] * (churn_ij.astype(float) - at_risk_i * haz_j[None, :])
        influence[j] = (term_survival + term_hazard).sum(axis=1)

    return {"lost": lost, "influence": influence, "survival": surv}
