"""Triggered interventions: intention to treat against the effect on the exposed."""

import numpy as np
import pandas as pd
import pytest

import sublift as sl


def triggered(n=60_000, trigger_rate=0.25, seed=1):
    """Only a quarter ever hit the cancel flow, and only the treatment arm sees an offer."""
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, 2, size=n)
    triggers = rng.random(n) < trigger_rate
    exposed = triggers & (arm == 1)
    lifetime = np.maximum(
        np.ceil(rng.exponential(np.exp(1.2) * np.where(exposed, np.exp(0.55), 1.0))).astype(int), 1
    )
    censor = rng.integers(3, 16, size=n)
    return pd.DataFrame(
        {
            "uid": np.arange(n),
            "variant": np.where(arm == 1, "treat", "ctrl"),
            "n": np.minimum(lifetime, censor),
            "ev": lifetime <= censor,
            "saw_offer": exposed,
            "would_trigger": triggers,
        }
    )


def build(frame):
    return sl.SubscriberPanel.from_subjects(
        frame,
        subject="uid",
        arm="variant",
        periods="n",
        event="ev",
        control="ctrl",
        covariates=["saw_offer", "would_trigger"],
    )


@pytest.fixture(scope="module")
def panel():
    return build(triggered())


def test_the_complier_effect_is_the_wald_ratio(panel):
    itt = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    cace = sl.complier_effect(itt, panel, exposed="saw_offer")
    assert cace.estimate == pytest.approx(itt.estimate / cace.exposure_rate, rel=1e-9)
    assert cace.estimate > itt.estimate


def test_it_recovers_the_effect_among_those_who_saw_it(panel):
    """Restricting to the triggered subscribers is the direct estimate of the same thing."""
    itt = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    cace = sl.complier_effect(itt, panel, exposed="saw_offer")

    triggers = panel.covariates["would_trigger"].to_numpy().astype(bool)
    direct = sl.retained_periods_lift(
        panel.subset(triggers), horizon=8, estimator="unadjusted", allow_extrapolation=True
    )
    assert abs(cace.estimate - direct.estimate) < 3 * max(cace.se, direct.se)


def test_the_interval_widens_with_the_scaling(panel):
    itt = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    cace = sl.complier_effect(itt, panel, exposed="saw_offer")
    assert cace.se > itt.se
    assert cace.ci[1] - cace.ci[0] > itt.ci[1] - itt.ci[0]


def test_it_reports_both_numbers_and_says_which_is_which(panel):
    itt = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    text = sl.complier_effect(itt, panel, exposed="saw_offer").summary()
    assert "intention to treat" in text
    assert "complier effect" in text
    assert "launch decision" in text


def test_a_confidence_sequence_is_available(panel):
    itt = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    cs = sl.complier_effect(itt, panel, exposed="saw_offer").confidence_sequence()
    assert cs.lower < cs.upper


def test_negligible_compliance_is_refused():
    """Dividing by a tiny number turns a modest interval into a meaningless one."""
    panel = build(triggered(n=20_000, trigger_rate=0.005, seed=2))
    itt = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    with pytest.raises(sl.NotIdentifiedError, match="report intention to treat"):
        sl.complier_effect(itt, panel, exposed="saw_offer")


def test_exposure_that_does_not_follow_assignment_is_refused(panel):
    with pytest.raises(sl.NotIdentifiedError, match="no higher in the treatment arm"):
        sl.complier_effect(
            sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted"),
            panel,
            exposed="would_trigger",  # the same in both arms by construction
        )


def test_a_missing_exposure_column_says_what_to_do(panel):
    itt = sl.retained_periods_lift(panel, horizon=8, estimator="unadjusted")
    with pytest.raises(sl.PanelError, match="covariates"):
        sl.complier_effect(itt, panel, exposed="nonexistent")
