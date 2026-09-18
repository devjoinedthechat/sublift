"""Correcting across a family of comparisons, whatever the family is made of.

Three different things in this library produce a family: several treatment arms
against one control, several segments of one comparison, and several metrics read
off the same experiment. They are the same statistical problem -- a set of
contrasts, each with an influence function, of which you will report the
interesting ones -- and they get the same correction here rather than three
copies of it.

The correction is single-step **max-t**. Every contrast in a family is computed
from the same subscribers, so they are correlated, and usually strongly:
contrasts sharing a control arm correlate at about 0.5, and lifetime value
against retained periods on the same panel correlates above 0.9. Bonferroni
assumes the worst about that dependence and pays for independence the family does
not have. max-t reads the correlation off the influence functions and calibrates
the critical value against it, which controls the same family-wise error rate
while being less conservative -- dramatically so for near-duplicate metrics,
mildly so for arms.

What this module cannot do is know what family you looked at. If you ran the scan
and then also eyeballed a segment, the family is larger than the code can see.
That part is yours.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["correct_family", "CorrectedFamily", "FamilyMember", "CORRECTIONS"]

CORRECTIONS = ("max-t", "holm", "bonferroni", "none")
_MC_DRAWS = 200_000


@dataclass(frozen=True)
class FamilyMember:
    label: str
    estimate: float
    se: float
    ci: tuple[float, float]
    marginal_ci: tuple[float, float]
    p_value: float
    adjusted_p_value: float

    @property
    def significant(self) -> bool:
        """Differs from zero after correcting for everything else in the family."""
        return self.ci[0] > 0 or self.ci[1] < 0


@dataclass
class CorrectedFamily:
    members: list[FamilyMember]
    correction: str
    alpha: float
    critical_value: float
    bonferroni_critical_value: float
    correlation: np.ndarray
    n_subjects: int

    @property
    def survivors(self) -> list[FamilyMember]:
        """Members that survive the correction, largest effect first."""
        return sorted(
            (m for m in self.members if m.significant),
            key=lambda m: abs(m.estimate),
            reverse=True,
        )

    @property
    def mean_correlation(self) -> float:
        """Average correlation between the family's comparisons."""
        k = len(self.members)
        if k < 2:
            return float("nan")
        return float(self.correlation[np.triu_indices(k, 1)].mean())

    def to_frame(self) -> pd.DataFrame:
        """One row per comparison, raw and adjusted."""
        return pd.DataFrame(
            [
                {
                    "comparison": m.label,
                    "estimate": m.estimate,
                    "se": m.se,
                    "ci_low": m.ci[0],
                    "ci_high": m.ci[1],
                    "p_value": m.p_value,
                    "adjusted_p": m.adjusted_p_value,
                    "significant": m.significant,
                }
                for m in self.members
            ]
        )

    def summary(self) -> str:
        """Each comparison with its simultaneous interval, and what the correction cost."""
        head = f"{len(self.members)} comparisons, corrected together"
        width = max(len(m.label) for m in self.members)
        lines = [head, "=" * len(head), ""]
        for m in self.members:
            mark = "*" if m.significant else " "
            lines.append(
                f" {mark}{m.label:<{width}s}  {m.estimate:>+12.4f}  "
                f"[{m.ci[0]:>+12.4f}, {m.ci[1]:>+12.4f}]   p={m.adjusted_p_value:.4f}"
            )
        saving = 1 - self.critical_value / self.bonferroni_critical_value
        lines += [
            "",
            f"  {1 - self.alpha:.0%} simultaneous intervals, {self.correction} correction.",
            f"  critical value {self.critical_value:.3f}"
            + (
                f" vs Bonferroni {self.bonferroni_critical_value:.3f} "
                f"({saving:.0%} narrower; mean correlation {self.mean_correlation:+.2f})"
                if self.correction == "max-t" and len(self.members) > 1
                else ""
            ),
        ]
        if not self.survivors:
            lines.append("  Nothing survives the correction.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def correct_family(
    results: Mapping[str, object] | Sequence[tuple[str, object]],
    *,
    alpha: float = 0.05,
    correction: str = "max-t",
    seed: int = 0,
) -> CorrectedFamily:
    """Correct a set of results that were all read off the same experiment.

    Takes anything with ``.estimate`` and ``.influence`` -- a
    :class:`~sublift.LiftResult` from any estimator that exposes an influence
    function, which is all of them except ``adjusted`` with
    ``inference="bootstrap"``.

    This is the escape hatch for families the library cannot infer. Three arms
    scanned across four segments is twelve comparisons, not three plus four;
    lifetime value and retained periods and involuntary churn is three. Assemble
    what you actually looked at and pass it here.

    Examples
    --------
    Several metrics from one experiment::

        results = {
            "ltv": sl.incremental_ltv(panel, horizon=12, price=12.0),
            "periods": sl.retained_periods_lift(panel, horizon=12),
        }
        print(sl.correct_family(results))

    Arms crossed with segments::

        family = {}
        for arm in panel.treatment_labels:
            pair = panel.contrast(arm)
            for plan in ("monthly", "annual"):
                mask = (pair.covariates["plan"] == plan).to_numpy()
                family[f"{arm} / {plan}"] = sl.retained_periods_lift(
                    pair.subset(mask), horizon=12, estimator="unadjusted"
                )
        print(sl.correct_family(family))
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}.")
    items = list(results.items()) if isinstance(results, Mapping) else list(results)
    if len(items) < 2:
        raise ValueError(
            f"A family needs at least two comparisons; got {len(items)}. With one there is "
            "nothing to correct for."
        )

    influences, estimates, labels = [], [], []
    for label, result in items:
        influence = getattr(result, "influence", None)
        if influence is None:
            raise ValueError(
                f"{label!r} carries no influence function, so it cannot be combined with the "
                "others. Estimators using inference='bootstrap' do not expose one; use the "
                "influence-based default."
            )
        labels.append(str(label))
        estimates.append(float(result.estimate))
        influences.append(np.asarray(influence, dtype=float))

    sizes = {inf.size for inf in influences}
    if len(sizes) > 1:
        raise ValueError(
            f"These results cover different numbers of subscribers ({sorted(sizes)}). A family "
            "correction assumes one experiment; comparisons on disjoint or differently-filtered "
            "panels are not jointly calibrated here."
        )

    psi = np.vstack(influences)
    n = psi.shape[1]
    estimates = np.asarray(estimates)
    cov = (psi @ psi.T) / (n**2)
    se = np.sqrt(np.diag(cov))
    z_scores = np.divide(estimates, se, out=np.zeros_like(estimates), where=se > 0)
    raw_p = 2 * stats.norm.sf(np.abs(z_scores))

    k = len(items)
    critical, adjusted_p = calibrate(correction, correlation(cov), z_scores, raw_p, alpha, k, seed)
    marginal = float(stats.norm.ppf(1 - alpha / 2))

    members = [
        FamilyMember(
            label=labels[i],
            estimate=float(estimates[i]),
            se=float(se[i]),
            ci=(
                float(estimates[i] - critical * se[i]),
                float(estimates[i] + critical * se[i]),
            ),
            marginal_ci=(
                float(estimates[i] - marginal * se[i]),
                float(estimates[i] + marginal * se[i]),
            ),
            p_value=float(raw_p[i]),
            adjusted_p_value=float(adjusted_p[i]),
        )
        for i in range(k)
    ]
    return CorrectedFamily(
        members=members,
        correction=correction,
        alpha=alpha,
        critical_value=float(critical),
        bonferroni_critical_value=float(stats.norm.ppf(1 - alpha / (2 * k))),
        correlation=correlation(cov),
        n_subjects=n,
    )


# ------------------------------------------------------- shared calibration


def correlation(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    outer = np.outer(sd, sd)
    corr = np.divide(cov, outer, out=np.eye(cov.shape[0]), where=outer > 0)
    return np.clip(corr, -1.0, 1.0)


def calibrate(correction, corr, z_scores, raw_p, alpha, k, seed):
    """Critical value and adjusted p-values for the chosen correction."""
    if correction == "none":
        return float(stats.norm.ppf(1 - alpha / 2)), raw_p
    if correction == "bonferroni":
        return float(stats.norm.ppf(1 - alpha / (2 * k))), np.minimum(1.0, raw_p * k)
    if correction == "holm":
        order = np.argsort(raw_p)
        adjusted = np.empty_like(raw_p)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (k - rank) * raw_p[idx])
            adjusted[idx] = min(1.0, running)
        # Holm is a testing procedure, not an interval one; intervals stay Bonferroni.
        return float(stats.norm.ppf(1 - alpha / (2 * k))), adjusted

    rng = np.random.default_rng(seed)
    draws = multivariate_normal(rng, corr, _MC_DRAWS)
    maxima = np.abs(draws).max(axis=1)
    critical = float(np.quantile(maxima, 1 - alpha))
    adjusted = np.array([(maxima > abs(z)).mean() for z in z_scores])
    return critical, np.clip(adjusted, 1.0 / _MC_DRAWS, 1.0)


def multivariate_normal(rng, corr: np.ndarray, draws: int) -> np.ndarray:
    """Correlated normals via Cholesky, falling back to an eigen decomposition.

    An estimated correlation matrix can be numerically indefinite when two
    comparisons are near-duplicates -- two metrics that are almost the same
    quantity, say -- which Cholesky will not tolerate and an analysis should not
    fail over.
    """
    k = corr.shape[0]
    try:
        factor = np.linalg.cholesky(corr + 1e-10 * np.eye(k))
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(corr)
        factor = vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None)))
    return rng.standard_normal((draws, k)) @ factor.T


def effective_multiplicity(corr: np.ndarray, *, alpha: float = 0.05, seed: int = 0) -> float:
    """How many *independent* comparisons this family behaves like.

    Bonferroni charges a family of ``k`` comparisons as if all ``k`` were separate
    chances to be wrong. When they are correlated, fewer of them are. This reads
    the max-t critical value off the estimated correlation and asks what
    Bonferroni divisor would have produced the same threshold::

        k_eff = alpha / (2 * P(Z > c))

    Independent comparisons give back ``k``; perfectly correlated ones give back
    1, which is right -- looking twice at the same number is one look.

    It exists because a confidence sequence has no max-t form: its boundary is
    derived for a scalar process, and the correlation between comparisons does not
    enter it. Splitting ``alpha`` by the effective multiplicity instead of by
    ``k`` is an approximation, and it is validated by simulation rather than
    argued for -- see ``tests/test_validation.py``.
    """
    k = corr.shape[0]
    if k < 2:
        return 1.0
    critical, _ = calibrate("max-t", corr, np.zeros(k), np.ones(k), alpha, k, seed)
    tail = float(stats.norm.sf(critical))
    if tail <= 0:
        return float(k)
    return float(np.clip(alpha / (2 * tail), 1.0, k))
