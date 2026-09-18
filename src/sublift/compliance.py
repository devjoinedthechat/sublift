"""When assignment is not exposure.

Retention interventions are usually *triggered*. A save offer fires when someone
opens the cancel flow; a win-back email goes only to subscribers who have already
lapsed. Everyone is randomised, but most of them never meet the thing being
tested, and the ones who do are not a random subset -- they are the subscribers
who were leaving.

Two different numbers follow, and confusing them is the common error.

**Intention to treat** is what every estimator here reports by default: the effect
of *being assigned*, averaged over everybody including the subscribers the offer
never reached. It is the number a business decision usually wants, because
shipping the programme means shipping the trigger too, and it is unbiased by
construction -- randomisation is the only assumption.

**The complier effect** is the effect among subscribers who actually saw the
offer. It is larger, often much larger, because intention to treat divides the
same effect across a base most of which was untouched. It is the right number for
"does this offer work", and the wrong one for "should we launch it".

With one-sided non-compliance -- the control arm cannot receive the treatment,
which is the usual shape for a triggered intervention -- the complier effect is
the Wald ratio: intention to treat divided by the share of the treatment arm that
was exposed. That rests on an **exclusion restriction**: being assigned changes
nothing for a subscriber who never triggers. If assignment itself changes
behaviour -- a holdout flagged in a CRM that agents then treat differently -- the
ratio is not identified and only the intention-to-treat number survives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .clustering import influence_se, sequence_terms
from .exceptions import NotIdentifiedError, PanelError
from .panel import SubscriberPanel

__all__ = ["complier_effect", "ComplierEffect"]


@dataclass
class ComplierEffect:
    """The effect among subscribers who actually met the intervention."""

    horizon: int
    metric: str
    alpha: float
    intention_to_treat: float
    itt_ci: tuple[float, float]
    exposure_rate: float
    control_exposure_rate: float
    estimate: float
    se: float
    ci: tuple[float, float]
    n_exposed: int
    n_subjects: int
    influence: np.ndarray
    cluster: np.ndarray | None = None

    @property
    def p_value(self) -> float:
        return float(2 * stats.norm.sf(abs(self.estimate / self.se))) if self.se else float("nan")

    def confidence_sequence(self, *, n_target: int | None = None, alpha: float | None = None):
        """Anytime-valid interval for the complier effect."""
        from .sequential import confidence_sequence

        return confidence_sequence(
            sequence_terms(self.influence, self.cluster, self.influence.size),
            estimate=self.estimate,
            alpha=self.alpha if alpha is None else alpha,
            n_target=n_target,
        )

    def summary(self) -> str:
        unit = "revenue" if self.metric == "ltv" else "periods"
        head = f"Effect among the exposed, over {self.horizon} billing periods"
        lines = [
            head,
            "=" * len(head),
            f"  intention to treat  {self.intention_to_treat:+.4f} {unit} "
            f"[{self.itt_ci[0]:+.4f}, {self.itt_ci[1]:+.4f}]",
            f"  exposed             {self.exposure_rate:.1%} of the treatment arm "
            f"({self.n_exposed:,} subscribers)",
            "",
            f"  complier effect     {self.estimate:+.4f} {unit} [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]",
            "",
            "  The first number is what shipping the programme does to the whole base, and is",
            "  what a launch decision wants. The second is what the offer does to someone who",
            "  sees it, and is what a design decision wants. Quoting the second as though it",
            "  were the first overstates the programme by the reciprocal of the exposure rate.",
        ]
        if self.control_exposure_rate > 0:
            lines += [
                "",
                f"  note: {self.control_exposure_rate:.1%} of the control arm was also exposed, so",
                "  this is a two-sided compliance ratio and leans on monotonicity as well as the",
                "  exclusion restriction.",
            ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def complier_effect(
    result,
    panel: SubscriberPanel,
    *,
    exposed: str,
    alpha: float | None = None,
) -> ComplierEffect:
    """Scale an intention-to-treat result up to the subscribers who were exposed.

    Parameters
    ----------
    result
        Any result carrying an influence function.
    exposed
        A boolean column on the panel's covariates: did this subscriber actually
        receive the intervention? It must be a *consequence* of assignment, not a
        baseline trait, which is the one case where the panel's usual rule about
        post-assignment columns is deliberately relaxed -- so pass it explicitly
        rather than through ``covariates=``.

    Notes
    -----
    Rests on the exclusion restriction: assignment affects the outcome only
    through exposure. That is plausible for a triggered offer and implausible
    whenever the holdout is visible to anyone who acts on it.
    """
    influence = getattr(result, "influence", None)
    if influence is None:
        raise NotIdentifiedError(
            "The complier effect is a ratio of two estimates, so it needs an influence "
            "function. Estimators using inference='bootstrap' do not expose one."
        )
    if panel.covariates is None or exposed not in panel.covariates.columns:
        raise PanelError(
            f"{exposed!r} is not on the panel. Build the panel with it in covariates=[...], "
            "even though it is measured after assignment -- here that is the point."
        )

    influence = np.asarray(influence, dtype=float)
    n = panel.n_subjects
    if influence.size != n:
        raise NotIdentifiedError(f"The result covers {influence.size} subscribers and the panel has {n}.")

    flags = panel.covariates[exposed].to_numpy()
    if flags.dtype != bool:
        flags = flags.astype(bool)

    treated = panel.arm == 1
    share = float(treated.mean())
    rate = float(flags[treated].mean())
    control_rate = float(flags[~treated].mean())
    compliance = rate - control_rate
    if compliance <= 0:
        raise NotIdentifiedError(
            f"Exposure is no higher in the treatment arm ({rate:.1%} against "
            f"{control_rate:.1%}), so there is nothing to scale by. Check that {exposed!r} "
            "records the intervention rather than an outcome of it."
        )
    if compliance < 0.02:
        raise NotIdentifiedError(
            f"Only {compliance:.2%} of subscribers were moved from unexposed to exposed by "
            "assignment. Dividing by a number that small turns a modest interval into a "
            "meaningless one; report intention to treat."
        )

    # Influence of the compliance rate, then the delta method for the ratio.
    phi = np.where(treated, (flags - rate) / share, -(flags - control_rate) / (1 - share))
    estimate = float(result.estimate / compliance)
    psi = (influence - estimate * phi) / compliance

    codes = getattr(result, "cluster", None)
    level = result.alpha if alpha is None else alpha
    se = influence_se(psi, codes, n)
    z = float(stats.norm.ppf(1 - level / 2))

    return ComplierEffect(
        horizon=result.horizon,
        metric=result.metric,
        alpha=level,
        intention_to_treat=float(result.estimate),
        itt_ci=tuple(result.ci),
        exposure_rate=rate,
        control_exposure_rate=control_rate,
        estimate=estimate,
        se=se,
        ci=(estimate - z * se, estimate + z * se),
        n_exposed=int(flags[treated].sum()),
        n_subjects=n,
        influence=psi,
        cluster=codes,
    )
