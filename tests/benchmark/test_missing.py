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
    """Return modeled misses on three rows plus an identifier-only miss."""

    return TabularDataset(
        name="missing-fixture",
        frame=pd.DataFrame(
            {
                "row_id": [None, "b", "c", "d"],
                "amount": [1.0, None, 3.0, 4.0],
                "segment": ["new", None, None, "new"],
                "label": ["no", "yes", "yes", None],
            }
        ),
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
    """Verify common rows, audit evidence, and identifier exclusion."""

    def test_only_error_and_complete_case_are_approved(self) -> None:
        self.assertEqual(
            tuple(MissingPolicy),
            (MissingPolicy.ERROR, MissingPolicy.COMPLETE_CASE),
        )

    def test_error_raises_with_report_without_mutating_source(self) -> None:
        dataset = _classification_dataset()
        original_frame = dataset.frame.copy(deep=True)

        with self.assertRaises(MissingValuesError) as raised:
            apply_missing_policy(dataset, MissingPolicy.ERROR)

        report = raised.exception.report
        self.assertEqual(
            dict(report.missing_by_column),
            {"amount": 1, "segment": 2, "label": 1},
        )
        self.assertEqual(tuple(report.missing_by_column), dataset.column_order)
        self.assertEqual(report.rows_before, 4)
        self.assertEqual(report.rows_after, 4)
        self.assertEqual(report.dropped_count, 0)
        self.assertEqual(report.dropped_fraction, 0.0)
        self.assertNotIn("row_id", report.missing_by_column)
        pd.testing.assert_frame_equal(dataset.frame, original_frame)

    def test_complete_case_drops_row_union_and_ignores_identifier(self) -> None:
        dataset = _classification_dataset()
        original_frame = dataset.frame.copy(deep=True)

        result = apply_missing_policy(dataset, MissingPolicy.COMPLETE_CASE)

        self.assertIsNot(result.dataset, dataset)
        self.assertIsNot(result.dataset.frame, dataset.frame)
        self.assertEqual(result.report.policy, MissingPolicy.COMPLETE_CASE)
        self.assertEqual(result.dataset.name, dataset.name)
        self.assertIs(result.dataset.columns, dataset.columns)
        self.assertEqual(result.dataset.target, dataset.target)
        self.assertEqual(result.dataset.task, dataset.task)
        self.assertEqual(result.dataset.identifier, dataset.identifier)
        self.assertEqual(result.dataset.frame.index.tolist(), [0])
        self.assertTrue(pd.isna(result.dataset.frame.loc[0, "row_id"]))
        self.assertEqual(result.report.rows_before, 4)
        self.assertEqual(result.report.rows_after, 1)
        self.assertEqual(result.report.dropped_count, 3)
        self.assertEqual(result.report.dropped_fraction, 0.75)
        pd.testing.assert_frame_equal(dataset.frame, original_frame)

        result.dataset.frame.loc[0, "amount"] = 999.0
        pd.testing.assert_frame_equal(dataset.frame, original_frame)

    def test_class_distribution_is_recorded_before_and_after_filtering(self) -> None:
        result = apply_missing_policy(
            _classification_dataset(),
            MissingPolicy.COMPLETE_CASE,
        )

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

    def test_identifier_missing_alone_does_not_trigger_error(self) -> None:
        dataset = _classification_dataset()
        complete_modeled = TabularDataset(
            name=dataset.name,
            frame=dataset.frame.iloc[[0]].copy(),
            columns=dataset.columns,
            target=dataset.target,
            task=dataset.task,
            identifier=dataset.identifier,
        )

        result = apply_missing_policy(complete_modeled, MissingPolicy.ERROR)

        self.assertIs(result.dataset, complete_modeled)
        self.assertEqual(
            dict(result.report.missing_by_column),
            {"amount": 0, "segment": 0, "label": 0},
        )
        self.assertEqual(result.report.dropped_count, 0)

    def test_regression_report_has_no_class_distribution(self) -> None:
        dataset = TabularDataset(
            name="regression",
            frame=pd.DataFrame({"feature": [1.0, None], "target": [2.0, 3.0]}),
            columns=(
                ColumnSpec("feature", ColumnKind.CONTINUOUS),
                ColumnSpec("target", ColumnKind.CONTINUOUS),
            ),
            target="target",
            task=TaskType.REGRESSION,
        )

        result = apply_missing_policy(dataset, MissingPolicy.COMPLETE_CASE)

        self.assertIsNone(result.report.class_counts_before)
        self.assertIsNone(result.report.class_counts_after)

    def test_dataset_without_target_has_no_class_distribution(self) -> None:
        dataset = TabularDataset(
            name="unsupervised",
            frame=pd.DataFrame({"feature": [1.0, None]}),
            columns=(ColumnSpec("feature", ColumnKind.CONTINUOUS),),
        )

        result = apply_missing_policy(dataset, MissingPolicy.COMPLETE_CASE)

        self.assertIsNone(result.report.class_counts_before)
        self.assertIsNone(result.report.class_counts_after)

    def test_empty_typed_dataset_has_zero_drop_fraction(self) -> None:
        dataset = TabularDataset(
            name="empty",
            frame=pd.DataFrame(
                {
                    "feature": pd.Series(dtype="float64"),
                    "label": pd.Series(dtype="object"),
                }
            ),
            columns=(
                ColumnSpec("feature", ColumnKind.CONTINUOUS),
                ColumnSpec("label", ColumnKind.CATEGORICAL),
            ),
            target="label",
            task=TaskType.CLASSIFICATION,
        )

        result = apply_missing_policy(dataset, MissingPolicy.COMPLETE_CASE)

        self.assertEqual(result.report.rows_before, 0)
        self.assertEqual(result.report.rows_after, 0)
        self.assertEqual(result.report.dropped_count, 0)
        self.assertEqual(result.report.dropped_fraction, 0.0)
        self.assertEqual(result.report.class_counts_before, ())
        self.assertEqual(result.report.class_counts_after, ())

    def test_complete_case_can_remove_every_row(self) -> None:
        dataset = TabularDataset(
            name="all-dropped",
            frame=pd.DataFrame(
                {
                    "feature": pd.Series([None, None], dtype="float64"),
                    "label": pd.Series([None, None], dtype="object"),
                }
            ),
            columns=(
                ColumnSpec("feature", ColumnKind.CONTINUOUS),
                ColumnSpec("label", ColumnKind.CATEGORICAL),
            ),
            target="label",
            task=TaskType.CLASSIFICATION,
        )

        result = apply_missing_policy(dataset, MissingPolicy.COMPLETE_CASE)

        self.assertEqual(result.report.rows_before, 2)
        self.assertEqual(result.report.rows_after, 0)
        self.assertEqual(result.report.dropped_count, 2)
        self.assertEqual(result.report.dropped_fraction, 1.0)
        self.assertEqual(
            sum(count.count for count in result.report.class_counts_before or ()),
            2,
        )
        self.assertEqual(result.report.class_counts_after, ())

    def test_policy_must_be_an_explicit_enum(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "MissingPolicy"):
            apply_missing_policy(  # type: ignore[arg-type]
                _classification_dataset(),
                "complete_case",
            )

    def test_report_missing_counts_are_read_only(self) -> None:
        result = apply_missing_policy(
            _classification_dataset(),
            MissingPolicy.COMPLETE_CASE,
        )

        with self.assertRaises(TypeError):
            result.report.missing_by_column["amount"] = 99  # type: ignore[index]

    def test_report_snapshots_source_missing_counts(self) -> None:
        source_counts = {"feature": 1}
        report = MissingReport(
            policy=MissingPolicy.ERROR,
            rows_before=1,
            rows_after=1,
            dropped_count=0,
            dropped_fraction=0.0,
            missing_by_column=source_counts,
        )

        source_counts["feature"] = 99

        self.assertEqual(dict(report.missing_by_column), {"feature": 1})


if __name__ == "__main__":
    unittest.main()
