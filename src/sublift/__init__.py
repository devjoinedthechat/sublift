"""sublift -- incremental LTV measurement for subscription retention experiments.

    import sublift as sl

    panel = sl.SubscriberPanel.from_periods(
        df, subject="user_id", period="cycle", churned="churned",
        arm="variant", revenue="amount",
        covariates=["plan", "tenure_bucket", "engagement"],
    )
    result = sl.incremental_ltv(panel, horizon=12, strata=["plan", "tenure_bucket"])
    print(result)
    print(result.confidence_sequence())

Four things make this different from running a t-test on 30-day retention:
the outcome is censored and handled as such, churn is modelled in discrete
billing periods, the horizon is explicit, and the intervals stay valid when you
look at them every day.
"""

from .datasets import SimulatedExperiment, simulate_experiment
from .estimators import ArmSummary, LiftResult, incremental_ltv, retained_periods_lift
from .panel import SubscriberPanel
from .power import PowerCurve, duration_to_detect
from .sequential import ConfidenceSequence, confidence_sequence
from .survival import DiscreteSurvival, fit_survival, weighted_value
from .uplift import QiniCurve, qini, uplift_scores

__version__ = "0.1.0"

__all__ = [
    "SubscriberPanel",
    "incremental_ltv",
    "retained_periods_lift",
    "LiftResult",
    "ArmSummary",
    "confidence_sequence",
    "ConfidenceSequence",
    "duration_to_detect",
    "PowerCurve",
    "qini",
    "QiniCurve",
    "uplift_scores",
    "simulate_experiment",
    "SimulatedExperiment",
    "fit_survival",
    "weighted_value",
    "DiscreteSurvival",
    "__version__",
]
