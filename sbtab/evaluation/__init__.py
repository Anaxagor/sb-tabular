"""Model-independent metrics operating on raw real and synthetic tables."""

from __future__ import annotations

from sbtab.evaluation.artifacts import write_evaluation_artifacts
from sbtab.evaluation.final import (
    CategoricalQualitySummary,
    ContinuousQualitySummary,
    CrossValidationEvaluation,
    DiscreteQualitySummary,
    FinalEvaluationSummary,
    FoldEvaluation,
    ScalarSummary,
    UtilitySummary,
    evaluate_cross_validation,
)
from sbtab.evaluation.quality import (
    CategoricalQuality,
    ContinuousColumnQuality,
    ContinuousQuality,
    DiscreteQuality,
    FiniteColumnQuality,
    QualityScore,
    evaluate_quality,
)
from sbtab.evaluation.tuning import (
    ColumnTuningScore,
    TuningMetric,
    TuningScore,
    evaluate_tuning_score,
)
from sbtab.evaluation.utility import (
    UtilityMetric,
    UtilityScore,
    evaluate_utility,
)

__all__ = [
    "CategoricalQualitySummary",
    "CategoricalQuality",
    "ColumnTuningScore",
    "ContinuousQualitySummary",
    "ContinuousColumnQuality",
    "ContinuousQuality",
    "CrossValidationEvaluation",
    "DiscreteQualitySummary",
    "DiscreteQuality",
    "FinalEvaluationSummary",
    "FiniteColumnQuality",
    "FoldEvaluation",
    "QualityScore",
    "ScalarSummary",
    "TuningMetric",
    "TuningScore",
    "UtilityMetric",
    "UtilityScore",
    "UtilitySummary",
    "evaluate_cross_validation",
    "evaluate_quality",
    "evaluate_tuning_score",
    "evaluate_utility",
    "write_evaluation_artifacts",
]
