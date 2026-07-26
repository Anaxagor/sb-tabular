"""Tests for complete fold evaluation and population aggregation."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from sbtab.benchmark import (
    BenchmarkConfig,
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    DiscreteView,
    InputSpec,
    KFoldConfig,
    MissingPolicy,
    PreparedTable,
    RunContext,
    StratifiedKFoldConfig,
    TabularDataset,
    TaskType,
    run_cross_validation,
)
from sbtab.evaluation import (
    UtilityMetric,
    evaluate_cross_validation,
)


class _EchoAdapter:
    name = "evaluation-echo"
    input_spec = InputSpec(
        continuous_view=ContinuousView.RAW,
        discrete_view=DiscreteView.FINITE_STATE_CODES,
        categorical_view=CategoricalView.FINITE_STATE_CODES,
    )

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        self.train = train

    def sample(self, n: int, seed: int) -> PreparedTable:
        repeats = (n + len(self.train.frame) - 1) // len(self.train.frame)
        frame = pd.concat(
            [self.train.frame] * repeats,
            ignore_index=True,
        ).iloc[:n]
        return PreparedTable(frame=frame, schema=self.train.schema)


def _generation():
    rows = 20
    frame = pd.DataFrame(
        {
            "value": [float(index % 5) for index in range(rows)],
            "amount": [float((rows - index) % 7) for index in range(rows)],
            "count": [index % 3 for index in range(rows)],
            "rank": [(index // 2) % 3 for index in range(rows)],
            "group": ["a" if index % 2 == 0 else "b" for index in range(rows)],
            "target": [bool(index % 2) for index in range(rows)],
        }
    )
    dataset = TabularDataset(
        name="final-evaluation",
        frame=frame,
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("amount", ColumnKind.CONTINUOUS),
            ColumnSpec("count", ColumnKind.DISCRETE),
            ColumnSpec("rank", ColumnKind.DISCRETE),
            ColumnSpec("group", ColumnKind.CATEGORICAL),
            ColumnSpec("target", ColumnKind.CATEGORICAL),
        ),
        target="target",
        task=TaskType.CLASSIFICATION,
    )
    return run_cross_validation(
        dataset,
        _EchoAdapter,
        BenchmarkConfig(
            split=StratifiedKFoldConfig(n_splits=2, seed=42),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            training_seed=42,
            artifact_dir=Path("evaluation-test"),
        ),
    )


class FinalEvaluationTests(unittest.TestCase):
    """Verify fold ownership, semantic applicability, utility, and ddof=0."""

    def test_evaluates_all_folds_and_aggregates_population_statistics(
        self,
    ) -> None:
        generation = _generation()

        result = evaluate_cross_validation(generation)

        self.assertIs(result.generation, generation)
        self.assertEqual(
            tuple(fold.fold_id for fold in result.folds),
            (0, 1),
        )
        self.assertIs(
            result.summary.utility.metric,
            UtilityMetric.MACRO_F1,
        )
        values = [
            fold.quality.continuous.mean_wasserstein
            for fold in result.folds
        ]
        expected_mean = sum(values) / len(values)
        expected_std = (
            sum((value - expected_mean) ** 2 for value in values)
            / len(values)
        ) ** 0.5
        self.assertAlmostEqual(
            result.summary.continuous.mean_wasserstein.mean,
            expected_mean,
        )
        self.assertAlmostEqual(
            result.summary.continuous.mean_wasserstein.std,
            expected_std,
        )
        self.assertIsNotNone(result.summary.continuous.pearson_frobenius)
        self.assertIsNotNone(result.summary.discrete.spearman_frobenius)
        self.assertIsNotNone(result.summary.categorical.nmi_frobenius)

    def test_absent_groups_and_target_remain_inapplicable(self) -> None:
        frame = pd.DataFrame({"value": [float(index) for index in range(8)]})
        dataset = TabularDataset(
            name="continuous-only",
            frame=frame,
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
        )
        generation = run_cross_validation(
            dataset,
            _EchoAdapter,
            BenchmarkConfig(
                split=KFoldConfig(n_splits=2, seed=42),
                missing_policy=MissingPolicy.COMPLETE_CASE,
            ),
        )

        result = evaluate_cross_validation(generation)

        self.assertIsNone(result.summary.continuous.pearson_frobenius)
        self.assertIsNone(result.summary.discrete)
        self.assertIsNone(result.summary.categorical)
        self.assertIsNone(result.summary.utility)


if __name__ == "__main__":
    unittest.main()
