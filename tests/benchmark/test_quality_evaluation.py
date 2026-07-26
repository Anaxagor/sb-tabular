"""Tests for decoded raw-space final statistical quality metrics."""

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
from sbtab.evaluation import evaluate_quality


def _mixed_dataset(frame: pd.DataFrame) -> TabularDataset:
    return TabularDataset(
        name="mixed-quality",
        frame=frame,
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("amount", ColumnKind.CONTINUOUS),
            ColumnSpec("count", ColumnKind.DISCRETE),
            ColumnSpec("rank", ColumnKind.DISCRETE),
            ColumnSpec("group", ColumnKind.CATEGORICAL),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
    )


def _real_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [0.0, 1.0, 2.0, 3.0],
            "amount": [3.0, 2.0, 1.0, 0.0],
            "count": [0, 0, 1, 1],
            "rank": [0, 1, 0, 1],
            "group": ["a", "a", "b", "b"],
            "label": ["no", "yes", "no", "yes"],
        }
    )


class QualityEvaluationTests(unittest.TestCase):
    """Verify metric grouping, exact supports, associations, and raw boundary."""

    def test_identical_tables_have_zero_applicable_distances(self) -> None:
        real = _real_frame()
        score = evaluate_quality(_mixed_dataset(real), real, real.copy())

        self.assertEqual(score.continuous.mean_wasserstein, 0.0)
        self.assertEqual(score.continuous.mean_kl, 0.0)
        self.assertEqual(score.continuous.pearson_frobenius, 0.0)
        self.assertEqual(score.discrete.mean_kl, 0.0)
        self.assertEqual(score.discrete.spearman_frobenius, 0.0)
        self.assertEqual(score.categorical.mean_kl, 0.0)
        self.assertEqual(score.categorical.nmi_frobenius, 0.0)

    def test_quality_accepts_unrelated_row_counts_and_retains_columns(
        self,
    ) -> None:
        real = _real_frame()
        synthetic = pd.concat(
            (real, real.iloc[[0, 1]]),
            ignore_index=True,
        )

        score = evaluate_quality(_mixed_dataset(real), real, synthetic)

        self.assertEqual(
            tuple(item.column for item in score.continuous.columns),
            ("value", "amount"),
        )
        self.assertEqual(
            tuple(item.column for item in score.discrete.columns),
            ("count", "rank"),
        )
        self.assertEqual(
            tuple(item.column for item in score.categorical.columns),
            ("group", "label"),
        )

    def test_finite_kl_uses_exact_union_support_and_real_to_synthetic_direction(
        self,
    ) -> None:
        frame = pd.DataFrame({"state": ["a", "b"]})
        dataset = TabularDataset(
            name="finite-kl",
            frame=frame,
            columns=(ColumnSpec("state", ColumnKind.CATEGORICAL),),
        )

        score = evaluate_quality(
            dataset,
            frame,
            pd.DataFrame({"state": ["a", "a"]}),
        )

        pseudocount = 1e-12
        real_a = (1.0 + pseudocount) / (2.0 + 2.0 * pseudocount)
        synthetic_a = (2.0 + pseudocount) / (2.0 + 2.0 * pseudocount)
        synthetic_b = pseudocount / (2.0 + 2.0 * pseudocount)
        expected = real_a * math.log(real_a / synthetic_a)
        expected += real_a * math.log(real_a / synthetic_b)
        self.assertAlmostEqual(score.categorical.mean_kl, expected)
        self.assertIsNone(score.categorical.nmi_frobenius)

    def test_constant_correlations_zero_only_the_undefined_entries(self) -> None:
        real = pd.DataFrame(
            {
                "left": [1.0, 1.0, 1.0, 1.0],
                "right": [0.0, 1.0, 2.0, 3.0],
            }
        )
        synthetic = pd.DataFrame(
            {
                "left": [0.0, 1.0, 2.0, 3.0],
                "right": [0.0, 1.0, 2.0, 3.0],
            }
        )
        dataset = TabularDataset(
            name="constant-correlation",
            frame=real,
            columns=(
                ColumnSpec("left", ColumnKind.CONTINUOUS),
                ColumnSpec("right", ColumnKind.CONTINUOUS),
            ),
        )

        score = evaluate_quality(dataset, real, synthetic)

        self.assertAlmostEqual(
            score.continuous.pearson_frobenius,
            math.sqrt(2.0),
        )

    def test_quality_requires_canonical_non_missing_raw_tables(self) -> None:
        real = _real_frame()
        dataset = _mixed_dataset(real)
        reordered = real.loc[
            :,
            ["amount", "value", "count", "rank", "group", "label"],
        ]
        with self.assertRaisesRegex(ContractViolation, "canonical"):
            evaluate_quality(dataset, reordered, real)

        missing = real.copy()
        missing.loc[0, "group"] = None
        with self.assertRaisesRegex(ContractViolation, "missing values"):
            evaluate_quality(dataset, real, missing)


if __name__ == "__main__":
    unittest.main()
