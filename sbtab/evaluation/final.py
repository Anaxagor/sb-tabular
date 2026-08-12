"""Final fold evaluation and model-independent cross-fold aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from sbtab.benchmark.runner import CrossValidationResult
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation.quality import (
    CategoricalQuality,
    ContinuousQuality,
    DiscreteQuality,
    QualityScore,
    evaluate_quality,
)
from sbtab.evaluation.utility import (
    UtilityMetric,
    UtilityScore,
    evaluate_utility,
)


@dataclass(frozen=True)
class ScalarSummary:
    """Population summary of one scalar across the complete fold set.

    Parameters
    ----------
    mean:
        Arithmetic mean over all folds.
    std:
        Population standard deviation with ``ddof=0``.
    """

    mean: float
    std: float


@dataclass(frozen=True)
class ContinuousQualitySummary:
    """Cross-fold summaries of continuous final-quality metrics.

    Parameters
    ----------
    mean_wasserstein:
        Summary of per-fold means across continuous columns.
    mean_kl:
        Summary of per-fold mean 50-bin KL values.
    mmd_rbf:
        Summary of per-fold train-standardized RBF MMD-squared values.
    pearson_frobenius:
        Summary of per-fold association distances, or ``None`` when fewer than
        two continuous columns exist.
    """

    mean_wasserstein: ScalarSummary
    mean_kl: ScalarSummary
    mmd_rbf: ScalarSummary
    pearson_frobenius: ScalarSummary | None


@dataclass(frozen=True)
class DiscreteQualitySummary:
    """Cross-fold summaries of discrete final-quality metrics.

    Parameters
    ----------
    mean_kl:
        Summary of per-fold mean exact-support KL values.
    spearman_frobenius:
        Summary of per-fold association distances, or ``None`` when fewer than
        two discrete columns exist.
    """

    mean_kl: ScalarSummary
    spearman_frobenius: ScalarSummary | None


@dataclass(frozen=True)
class CategoricalQualitySummary:
    """Cross-fold summaries of categorical final-quality metrics.

    Parameters
    ----------
    mean_kl:
        Summary of per-fold mean exact-support KL values.
    nmi_frobenius:
        Summary of per-fold pairwise-NMI distances, or ``None`` when fewer than
        two categorical columns exist.
    """

    mean_kl: ScalarSummary
    nmi_frobenius: ScalarSummary | None


@dataclass(frozen=True)
class UtilitySummary:
    """Cross-fold TSTR summaries for one declared downstream task.

    Parameters
    ----------
    metric:
        One task-selected metric shared by every fold.
    real_score, synthetic_score, absolute_change:
        Population summaries of the corresponding per-fold utility fields.
    relative_change_percent:
        Population summary only when every fold has a non-zero real baseline.
        It is ``None`` if at least one fold's percentage is undefined.
    """

    metric: UtilityMetric
    real_score: ScalarSummary
    synthetic_score: ScalarSummary
    absolute_change: ScalarSummary
    relative_change_percent: ScalarSummary | None


@dataclass(frozen=True)
class FinalEvaluationSummary:
    """All applicable cross-fold quality and utility summaries.

    Parameters
    ----------
    continuous, discrete, categorical:
        Semantic-group summaries, or ``None`` when that group is absent from
        the dataset declaration.
    utility:
        TSTR summary, or ``None`` when the dataset has no declared target.
    """

    continuous: ContinuousQualitySummary | None
    discrete: DiscreteQualitySummary | None
    categorical: CategoricalQualitySummary | None
    utility: UtilitySummary | None


@dataclass(frozen=True)
class FoldEvaluation:
    """Statistical quality and optional TSTR utility for one final fold.

    Parameters
    ----------
    fold_id:
        Positional fold identifier copied from the generation result.
    quality:
        Marginal and association metrics against this fold's real test table.
    utility:
        TSTR result, or ``None`` when the dataset has no target.
    """

    fold_id: int
    quality: QualityScore
    utility: UtilityScore | None


@dataclass(frozen=True)
class CrossValidationEvaluation:
    """Final evaluation tied to the exact in-memory generation result.

    Parameters
    ----------
    generation:
        Fixed-config decoded generation result that supplied all fold tables.
    folds:
        Per-fold metrics in the same order as ``generation.folds``.
    summary:
        Population mean/std across the complete fold set.
    """

    generation: CrossValidationResult
    folds: tuple[FoldEvaluation, ...]
    summary: FinalEvaluationSummary


def _summarize(values: tuple[float, ...]) -> ScalarSummary:
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise ContractViolation(
            f"Cannot aggregate non-finite fold values: {values!r}."
        )
    return ScalarSummary(
        mean=float(np.mean(array)),
        std=float(np.std(array, ddof=0)),
    )


def _continuous_summary(
    folds: tuple[FoldEvaluation, ...],
    *,
    pairwise_applicable: bool,
) -> ContinuousQualitySummary:
    scores = tuple(
        cast(ContinuousQuality, fold.quality.continuous)
        for fold in folds
    )
    return ContinuousQualitySummary(
        mean_wasserstein=_summarize(
            tuple(score.mean_wasserstein for score in scores)
        ),
        mean_kl=_summarize(tuple(score.mean_kl for score in scores)),
        mmd_rbf=_summarize(tuple(score.mmd_rbf for score in scores)),
        pearson_frobenius=(
            _summarize(
                tuple(
                    cast(float, score.pearson_frobenius)
                    for score in scores
                )
            )
            if pairwise_applicable
            else None
        ),
    )


def _discrete_summary(
    folds: tuple[FoldEvaluation, ...],
    *,
    pairwise_applicable: bool,
) -> DiscreteQualitySummary:
    scores = tuple(
        cast(DiscreteQuality, fold.quality.discrete)
        for fold in folds
    )
    return DiscreteQualitySummary(
        mean_kl=_summarize(tuple(score.mean_kl for score in scores)),
        spearman_frobenius=(
            _summarize(
                tuple(
                    cast(float, score.spearman_frobenius)
                    for score in scores
                )
            )
            if pairwise_applicable
            else None
        ),
    )


def _categorical_summary(
    folds: tuple[FoldEvaluation, ...],
    *,
    pairwise_applicable: bool,
) -> CategoricalQualitySummary:
    scores = tuple(
        cast(CategoricalQuality, fold.quality.categorical)
        for fold in folds
    )
    return CategoricalQualitySummary(
        mean_kl=_summarize(tuple(score.mean_kl for score in scores)),
        nmi_frobenius=(
            _summarize(
                tuple(
                    cast(float, score.nmi_frobenius)
                    for score in scores
                )
            )
            if pairwise_applicable
            else None
        ),
    )


def _utility_summary(
    folds: tuple[FoldEvaluation, ...],
) -> UtilitySummary:
    scores = tuple(cast(UtilityScore, fold.utility) for fold in folds)
    metric = scores[0].metric
    if any(score.metric is not metric for score in scores[1:]):
        raise ContractViolation("Utility metric changed between folds.")
    relative_values = tuple(
        score.relative_change_percent for score in scores
    )
    return UtilitySummary(
        metric=metric,
        real_score=_summarize(tuple(score.real_score for score in scores)),
        synthetic_score=_summarize(
            tuple(score.synthetic_score for score in scores)
        ),
        absolute_change=_summarize(
            tuple(score.absolute_change for score in scores)
        ),
        relative_change_percent=(
            None
            if any(value is None for value in relative_values)
            else _summarize(
                tuple(cast(float, value) for value in relative_values)
            )
        ),
    )


def evaluate_cross_validation(
    generation: CrossValidationResult,
) -> CrossValidationEvaluation:
    """Evaluate every decoded generation fold and aggregate final metrics."""

    if not isinstance(generation, CrossValidationResult):
        raise ContractViolation("generation must be CrossValidationResult.")

    fold_evaluations = tuple(
        FoldEvaluation(
            fold_id=fold.split.fold_id,
            quality=evaluate_quality(
                generation.dataset,
                fold.train_raw,
                fold.test_raw,
                fold.synthetic_raw,
                seed=(
                    generation.config.training_seed + fold.split.fold_id
                ),
            ),
            utility=(
                evaluate_utility(
                    generation.dataset,
                    fold.train_raw,
                    fold.test_raw,
                    fold.synthetic_raw,
                    seed=generation.config.training_seed
                    + fold.split.fold_id,
                )
                if generation.dataset.target is not None
                else None
            ),
        )
        for fold in generation.folds
    )
    if not fold_evaluations:
        raise ContractViolation("Final evaluation received no folds.")

    continuous_count = len(generation.dataset.continuous_columns)
    discrete_count = len(generation.dataset.discrete_columns)
    categorical_count = len(generation.dataset.categorical_columns)
    summary = FinalEvaluationSummary(
        continuous=(
            _continuous_summary(
                fold_evaluations,
                pairwise_applicable=continuous_count >= 2,
            )
            if continuous_count
            else None
        ),
        discrete=(
            _discrete_summary(
                fold_evaluations,
                pairwise_applicable=discrete_count >= 2,
            )
            if discrete_count
            else None
        ),
        categorical=(
            _categorical_summary(
                fold_evaluations,
                pairwise_applicable=categorical_count >= 2,
            )
            if categorical_count
            else None
        ),
        utility=(
            _utility_summary(fold_evaluations)
            if generation.dataset.target is not None
            else None
        ),
    )
    return CrossValidationEvaluation(
        generation=generation,
        folds=fold_evaluations,
        summary=summary,
    )
