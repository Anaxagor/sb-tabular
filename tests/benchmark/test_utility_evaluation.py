"""Tests for the fixed CatBoost train-on-synthetic utility protocol."""

from __future__ import annotations

import unittest

import pandas as pd

from sbtab.benchmark import (
    ColumnKind,
    ColumnSpec,
    ContractViolation,
    TabularDataset,
    TaskType,
)
from sbtab.evaluation import (
    UtilityMetric,
    evaluate_utility,
)


def _classification_dataset(frame: pd.DataFrame) -> TabularDataset:
    return TabularDataset(
        name="classification-utility",
        frame=frame,
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("group", ColumnKind.CATEGORICAL),
            ColumnSpec("target", ColumnKind.CATEGORICAL),
        ),
        target="target",
        task=TaskType.CLASSIFICATION,
    )


def _classification_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [float(index % 6) for index in range(rows)],
            "group": ["a" if index % 2 == 0 else "b" for index in range(rows)],
            "target": [bool(index % 2) for index in range(rows)],
        }
    )


class UtilityEvaluationTests(unittest.TestCase):
    """Verify task dispatch, signed changes, degenerate labels, and semantics."""

    def test_classification_accepts_single_class_synthetic_target(self) -> None:
        real_train = _classification_frame(20)
        real_test = _classification_frame(8)
        synthetic = real_train.copy()
        synthetic["target"] = False

        score = evaluate_utility(
            _classification_dataset(real_train),
            real_train,
            real_test,
            synthetic,
            seed=42,
        )

        self.assertIs(score.metric, UtilityMetric.MACRO_F1)
        self.assertGreaterEqual(score.real_score, 0.0)
        self.assertLessEqual(score.real_score, 1.0)
        self.assertGreaterEqual(score.synthetic_score, 0.0)
        self.assertLessEqual(score.synthetic_score, 1.0)
        self.assertAlmostEqual(
            score.absolute_change,
            score.synthetic_score - score.real_score,
        )
        if score.real_score != 0.0:
            self.assertAlmostEqual(
                score.relative_change_percent,
                100.0 * score.absolute_change / abs(score.real_score),
            )

    def test_regression_uses_r2_and_preserves_signed_change(self) -> None:
        real_train = pd.DataFrame(
            {
                "feature": [float(index) for index in range(20)],
                "target": [float(2 * index + 1) for index in range(20)],
            }
        )
        real_test = pd.DataFrame(
            {
                "feature": [float(index) for index in range(20, 28)],
                "target": [float(2 * index + 1) for index in range(20, 28)],
            }
        )
        synthetic = real_train.copy()
        synthetic["target"] = list(reversed(synthetic["target"].tolist()))
        dataset = TabularDataset(
            name="regression-utility",
            frame=real_train,
            columns=(
                ColumnSpec("feature", ColumnKind.CONTINUOUS),
                ColumnSpec("target", ColumnKind.CONTINUOUS),
            ),
            target="target",
            task=TaskType.REGRESSION,
        )

        score = evaluate_utility(
            dataset,
            real_train,
            real_test,
            synthetic,
            seed=7,
        )

        self.assertIs(score.metric, UtilityMetric.R2)
        self.assertAlmostEqual(
            score.absolute_change,
            score.synthetic_score - score.real_score,
        )
        self.assertAlmostEqual(
            score.relative_change_percent,
            100.0 * score.absolute_change / abs(score.real_score),
        )

    def test_utility_rejects_missing_or_incompatible_target_semantics(
        self,
    ) -> None:
        frame = pd.DataFrame({"value": [0.0, 1.0]})
        no_target = TabularDataset(
            name="no-target",
            frame=frame,
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
        )
        with self.assertRaisesRegex(ContractViolation, "target and task"):
            evaluate_utility(
                no_target,
                frame,
                frame,
                frame,
                seed=1,
            )

        continuous_target = TabularDataset(
            name="continuous-class",
            frame=pd.DataFrame(
                {
                    "feature": [0.0, 1.0],
                    "target": [0.0, 1.0],
                }
            ),
            columns=(
                ColumnSpec("feature", ColumnKind.CONTINUOUS),
                ColumnSpec("target", ColumnKind.CONTINUOUS),
            ),
            target="target",
            task=TaskType.CLASSIFICATION,
        )
        with self.assertRaisesRegex(ContractViolation, "Classification target"):
            evaluate_utility(
                continuous_target,
                continuous_target.frame,
                continuous_target.frame,
                continuous_target.frame,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
