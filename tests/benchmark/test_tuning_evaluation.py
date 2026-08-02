"""Tests for the common scale-balanced tuning objective."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from sbtab.benchmark import (
    ColumnKind,
    ColumnSpec,
    ContractViolation,
    TabularDataset,
    TaskType,
)
from sbtab.evaluation import TuningMetric, evaluate_tuning_score


def _mixed_dataset() -> TabularDataset:
    return TabularDataset(
        name="mixed-tuning-score",
        frame=pd.DataFrame(
            {
                "value": [0.0, 2.0],
                "count": [0.0, 1.0],
                "label": ["no", "yes"],
            }
        ),
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("count", ColumnKind.DISCRETE),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
    )


class TuningEvaluationTests(unittest.TestCase):
    """Verify semantic metric selection, aggregation, and strict raw boundary."""

    def test_mixed_score_adds_mean_wasserstein_and_mean_js(self) -> None:
        dataset = _mixed_dataset()
        real = dataset.frame.copy()
        synthetic = pd.DataFrame(
            {
                "value": [1.0, 3.0],
                "count": [1.0, 1.0],
                "label": ["no", "no"],
            }
        )
        expected_js = 0.75 * math.log(4.0 / 3.0)

        score = evaluate_tuning_score(dataset, real, real, synthetic)

        self.assertAlmostEqual(score.mean_wasserstein, 1.0)
        self.assertAlmostEqual(score.mean_jensen_shannon, expected_js)
        self.assertAlmostEqual(score.total, 1.0 + expected_js)
        self.assertEqual(
            tuple(item.column for item in score.columns),
            dataset.column_order,
        )
        self.assertEqual(
            tuple(item.metric for item in score.columns),
            (
                TuningMetric.STANDARDIZED_WASSERSTEIN,
                TuningMetric.JENSEN_SHANNON,
                TuningMetric.JENSEN_SHANNON,
            ),
        )
        self.assertEqual(
            tuple(item.reference_scale for item in score.columns),
            (1.0, None, None),
        )

    def test_pure_continuous_and_finite_scores_use_only_existing_group(self) -> None:
        continuous = TabularDataset(
            name="continuous",
            frame=pd.DataFrame({"value": [0.0, 2.0]}),
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
        )
        continuous_score = evaluate_tuning_score(
            continuous,
            continuous.frame,
            continuous.frame,
            pd.DataFrame({"value": [1.0, 3.0]}),
        )
        self.assertEqual(continuous_score.total, 1.0)
        self.assertIsNone(continuous_score.mean_jensen_shannon)

        finite = TabularDataset(
            name="finite",
            frame=pd.DataFrame({"value": ["a", "b"]}),
            columns=(ColumnSpec("value", ColumnKind.CATEGORICAL),),
        )
        finite_score = evaluate_tuning_score(
            finite,
            finite.frame,
            finite.frame,
            pd.DataFrame({"value": ["a", "a"]}),
        )
        self.assertEqual(
            finite_score.total,
            finite_score.mean_jensen_shannon,
        )
        self.assertIsNone(finite_score.mean_wasserstein)

    def test_discrete_values_are_compared_exactly_without_rounding(self) -> None:
        dataset = TabularDataset(
            name="float-states",
            frame=pd.DataFrame({"state": [0.0, 1.0]}),
            columns=(ColumnSpec("state", ColumnKind.DISCRETE),),
        )

        score = evaluate_tuning_score(
            dataset,
            dataset.frame,
            dataset.frame,
            pd.DataFrame({"state": [0.0, 1.1]}),
        )

        self.assertGreater(score.total, 0.0)

    def test_tuning_score_requires_canonical_non_missing_raw_tables(self) -> None:
        dataset = _mixed_dataset()
        reordered = dataset.frame.loc[:, ["count", "value", "label"]]
        with self.assertRaisesRegex(ContractViolation, "canonical"):
            evaluate_tuning_score(
                dataset,
                dataset.frame,
                reordered,
                dataset.frame,
            )

        missing = dataset.frame.copy()
        missing.loc[0, "label"] = None
        with self.assertRaisesRegex(ContractViolation, "missing values"):
            evaluate_tuning_score(
                dataset,
                dataset.frame,
                dataset.frame,
                missing,
            )

    def test_continuous_score_is_invariant_to_column_units(self) -> None:
        dataset = TabularDataset(
            name="scale-invariant",
            frame=pd.DataFrame(
                {
                    "small": [0.0, 2.0],
                    "large": [0.0, 2_000.0],
                }
            ),
            columns=(
                ColumnSpec("small", ColumnKind.CONTINUOUS),
                ColumnSpec("large", ColumnKind.CONTINUOUS),
            ),
        )
        synthetic = pd.DataFrame(
            {
                "small": [1.0, 3.0],
                "large": [1_000.0, 3_000.0],
            }
        )

        score = evaluate_tuning_score(
            dataset,
            dataset.frame,
            dataset.frame,
            synthetic,
        )

        self.assertEqual(
            tuple(item.value for item in score.columns),
            (1.0, 1.0),
        )
        self.assertEqual(
            tuple(item.reference_scale for item in score.columns),
            (1.0, 1_000.0),
        )
        self.assertEqual(score.mean_wasserstein, 1.0)


if __name__ == "__main__":
    unittest.main()
