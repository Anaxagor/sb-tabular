"""Tests for one global pre-split missing-value policy."""

from __future__ import annotations

import unittest

import pandas as pd

from sbtab.benchmark import (
    ColumnKind,
    ColumnSpec,
    ContractViolation,
    MissingPolicy,
    MissingReport,
    MissingValuesError,
    TabularDataset,
    TaskType,
    apply_missing_policy,
)


def _classification_dataset() -> TabularDataset:
    frame = pd.DataFrame(
        {
            "row_id": [None, "b", "c", "d"],
            "amount": [1.0, None, 3.0, 4.0],
            "segment": ["new", "returning", None, "new"],
            "label": ["no", "yes", "yes", None],
        }
    )
    return TabularDataset(
        name="missing-fixture",
        frame=frame,
        columns=(
            ColumnSpec("amount", ColumnKind.CONTINUOUS),
            ColumnSpec("segment", ColumnKind.CATEGORICAL),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
        identifier="row_id",
    )


class MissingPolicyTests(unittest.TestCase):
    """Verify common rows, evidence, and identifier exclusion."""

    def test_error_policy_raises_with_complete_report_without_filtering(self) -> None:
        dataset = _classification_dataset()
        original = dataset.frame.copy(deep=True)

        with self.assertRaises(MissingValuesError) as raised:
            apply_missing_policy(dataset, MissingPolicy.ERROR)

        report = raised.exception.report
        self.assertEqual(
            dict(report.missing_by_column),
            {"amount": 1, "segment": 1, "label": 1},
        )
        self.assertEqual(report.rows_before, 4)
        self.assertEqual(report.rows_after, 4)
        self.assertEqual(report.dropped_count, 0)
        self.assertNotIn("row_id", report.missing_by_column)
        pd.testing.assert_frame_equal(dataset.frame, original)

    def test_complete_case_filters_modeled_columns_and_ignores_id(self) -> None:
        dataset = _classification_dataset()

        result = apply_missing_policy(dataset, MissingPolicy.COMPLETE_CASE)

        self.assertEqual(result.dataset.frame.index.tolist(), [0])
        self.assertTrue(pd.isna(result.dataset.frame.loc[0, "row_id"]))
        self.assertEqual(result.report.rows_before, 4)
        self.assertEqual(result.report.rows_after, 1)
        self.assertEqual(result.report.dropped_count, 3)
        self.assertEqual(result.report.dropped_fraction, 0.75)
        self.assertEqual(len(dataset.frame), 4)

    def test_class_distribution_is_recorded_before_and_after_filtering(self) -> None:
        result = apply_missing_policy(
            _classification_dataset(),
            MissingPolicy.COMPLETE_CASE,
        )

        self.assertIsNotNone(result.report.class_counts_before)
        self.assertIsNotNone(result.report.class_counts_after)
        before = [
            (count.label, count.count)
            for count in result.report.class_counts_before or ()
        ]
        after = [
            (count.label, count.count)
            for count in result.report.class_counts_after or ()
        ]
        self.assertEqual(before[:2], [("no", 1), ("yes", 2)])
        self.assertEqual(len(before), 3)
        self.assertTrue(pd.isna(before[2][0]))
        self.assertEqual(before[2][1], 1)
        self.assertEqual(after, [("no", 1)])

    def test_error_policy_returns_original_when_modeled_values_are_complete(
        self,
    ) -> None:
        dataset = _classification_dataset()
        complete_dataset = TabularDataset(
            name=dataset.name,
            frame=dataset.frame.iloc[[0]].copy(),
            columns=dataset.columns,
            target=dataset.target,
            task=dataset.task,
            identifier=dataset.identifier,
        )

        result = apply_missing_policy(complete_dataset, MissingPolicy.ERROR)

        self.assertIs(result.dataset, complete_dataset)
        self.assertEqual(result.report.dropped_count, 0)

    def test_policy_must_be_explicit_enum(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "MissingPolicy"):
            apply_missing_policy(  # type: ignore[arg-type]
                _classification_dataset(),
                "complete_case",
            )

    def test_missing_report_rejects_inconsistent_row_evidence(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "dropped_count"):
            MissingReport(
                policy=MissingPolicy.COMPLETE_CASE,
                rows_before=2,
                rows_after=1,
                dropped_count=0,
                dropped_fraction=0.5,
                missing_by_column={"value": 1},
            )

    def test_missing_report_snapshots_column_counts(self) -> None:
        result = apply_missing_policy(
            _classification_dataset(),
            MissingPolicy.COMPLETE_CASE,
        )

        with self.assertRaises(TypeError):
            result.report.missing_by_column["amount"] = 99  # type: ignore[index]

    def test_error_report_cannot_claim_removed_rows(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "must not contain"):
            MissingReport(
                policy=MissingPolicy.ERROR,
                rows_before=2,
                rows_after=1,
                dropped_count=1,
                dropped_fraction=0.5,
                missing_by_column={"value": 1},
            )

    def test_column_missing_count_cannot_exceed_source_rows(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "exceed rows_before"):
            MissingReport(
                policy=MissingPolicy.ERROR,
                rows_before=2,
                rows_after=2,
                dropped_count=0,
                dropped_fraction=0.0,
                missing_by_column={"value": 3},
            )


if __name__ == "__main__":
    unittest.main()
