"""sublift -- incremental LTV measurement for subscription retention experiments.

    import sublift as sl

    panel = sl.SubscriberPanel.from_spans(
        df, subject="user_id", arm="variant",
        assigned_at="assigned_at", ended_at="cancelled_at",
        observed_through="2026-09-01", billing_interval="month",
        price="mrr", cause="churn_reason",
        covariates=["plan", "tenure_bucket", "engagement"],
    )
    result = sl.incremental_ltv(panel, horizon=12, strata=["plan", "tenure_bucket"])
    print(result)                              # point estimate, interval, curves
    print(result.confidence_sequence())        # valid even if you peeked
    print(sl.churn_decomposition(panel))       # voluntary vs involuntary
    print(sl.check_randomization(panel))       # SRM and baseline balance

Five things make this different from a t-test on 30-day retention: the outcome
is censored and handled as such, churn is modelled in discrete billing periods,
the horizon is explicit, voluntary and involuntary churn are separable, and the
intervals stay valid when you look at them every day.
"""

from .competing import CauseEffect, ChurnDecomposition, churn_decomposition
from .compliance import ComplierEffect, complier_effect
from .cuped import cuped
from .datasets import SimulatedExperiment, simulate_experiment, simulate_multi_arm
from .diagnostics import (
    CensoringCheck,
    RandomizationCheck,
    check_censoring,
    check_randomization,
)
from .estimators import ArmSummary, LiftResult, incremental_ltv, retained_periods_lift
from .exceptions import EstimationError, NotIdentifiedError, PanelError, SubliftError
from .family import CorrectedFamily, FamilyMember, correct_family
from .multiarm import ArmContrast, MultiArmResult, multi_arm_lift
from .occupancy import (
    LapseCause,
    OccupancyDecomposition,
    occupancy_decomposition,
    occupancy_lift,
)
from .panel import SubscriberPanel
from .power import PowerCurve, duration_to_detect
from .review import ExperimentReview, Finding, review
from .segments import Heterogeneity, SegmentEffect, SegmentScan, segment_scan
from .sensitivity import (
    CensoringSensitivity,
    SensitivityPoint,
    censoring_sensitivity,
)
from .sequential import ConfidenceSequence, confidence_sequence
from .survival import DiscreteSurvival, fit_survival, weighted_value
from .uplift import QiniCurve, qini, uplift_scores

__version__ = "0.1.0"

__all__ = [
    "SubscriberPanel",
    "incremental_ltv",
    "retained_periods_lift",
    "cuped",
    "complier_effect",
    "ComplierEffect",
    "occupancy_lift",
    "occupancy_decomposition",
    "OccupancyDecomposition",
    "LapseCause",
    "review",
    "ExperimentReview",
    "Finding",
    "segment_scan",
    "SegmentScan",
    "SegmentEffect",
    "Heterogeneity",
    "correct_family",
    "CorrectedFamily",
    "FamilyMember",
    "multi_arm_lift",
    "MultiArmResult",
    "ArmContrast",
    "churn_decomposition",
    "ChurnDecomposition",
    "CauseEffect",
    "check_randomization",
    "RandomizationCheck",
    "check_censoring",
    "censoring_sensitivity",
    "CensoringSensitivity",
    "SensitivityPoint",
    "CensoringCheck",
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
    "simulate_multi_arm",
    "SimulatedExperiment",
    "SubliftError",
    "PanelError",
    "NotIdentifiedError",
    "EstimationError",
    "fit_survival",
    "weighted_value",
    "DiscreteSurvival",
    "__version__",
]
