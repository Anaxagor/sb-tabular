"""Unit tests for benchmark declarations and boundary validation."""

from __future__ import annotations

import unittest
from dataclasses import fields

import pandas as pd

from sbtab.benchmark import (
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    ContractViolation,
    DiscreteView,
    InputSpec,
    PreparedSchema,
    PreparedTable,
    StateColumn,
    TabularDataset,
    TaskType,
    validate_input_spec,
    validate_prepared_table,
    validate_tabular_dataset,
)
from sbtab.benchmark.validation import validate_prepared_schema


class TabularDatasetValidationTests(unittest.TestCase):
    """Exercise raw dataset invariants without legacy schema inference."""

    def test_valid_target_remains_in_canonical_modeled_order(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [21.0, 35.0],
                "visits": [1, 4],
                "segment": ["new", "returning"],
                "label": ["no", "yes"],
            }
        )
        dataset = TabularDataset(
            name="valid",
            frame=frame,
            columns=(
                ColumnSpec("age", ColumnKind.CONTINUOUS),
                ColumnSpec("visits", ColumnKind.DISCRETE),
                ColumnSpec("segment", ColumnKind.CATEGORICAL),
                ColumnSpec("label", ColumnKind.CATEGORICAL),
            ),
            target="label",
            task=TaskType.CLASSIFICATION,
        )

        validate_tabular_dataset(dataset)

        self.assertEqual(dataset.column_order, tuple(frame.columns))
        self.assertIn(dataset.target, dataset.categorical_columns)

    def test_duplicate_modeled_column_is_rejected(self) -> None:
        dataset = TabularDataset(
            name="duplicate",
            frame=pd.DataFrame({"value": [1.0]}),
            columns=(
                ColumnSpec("value", ColumnKind.CONTINUOUS),
                ColumnSpec("value", ColumnKind.CONTINUOUS),
            ),
        )

        with self.assertRaisesRegex(ContractViolation, "duplicate names"):
            validate_tabular_dataset(dataset)

    def test_duplicate_raw_frame_column_is_rejected(self) -> None:
        dataset = TabularDataset(
            name="duplicate-frame-label",
            frame=pd.DataFrame([[1.0, 2.0]], columns=["value", "value"]),
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
        )

        with self.assertRaisesRegex(ContractViolation, "duplicate column labels"):
            validate_tabular_dataset(dataset)

    def test_invalid_column_name_raises_contextual_contract_error(self) -> None:
        dataset = TabularDataset(
            name="invalid-name",
            frame=pd.DataFrame({"value": [1.0]}),
            columns=(
                ColumnSpec(  # type: ignore[arg-type]
                    name=[],
                    kind=ColumnKind.CONTINUOUS,
                ),
            ),
        )

        with self.assertRaisesRegex(ContractViolation, "non-empty string"):
            validate_tabular_dataset(dataset)

    def test_undeclared_raw_column_is_rejected(self) -> None:
        dataset = TabularDataset(
            name="extra",
            frame=pd.DataFrame({"value": [1.0], "hidden": [2.0]}),
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
        )

        with self.assertRaisesRegex(ContractViolation, "undeclared raw columns"):
            validate_tabular_dataset(dataset)

    def test_identifier_must_be_present_and_unmodeled(self) -> None:
        dataset = TabularDataset(
            name="identifier",
            frame=pd.DataFrame({"row_id": [1], "value": [2.0]}),
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
            identifier="row_id",
        )

        validate_tabular_dataset(dataset)

    def test_missing_declared_identifier_is_rejected(self) -> None:
        dataset = TabularDataset(
            name="missing-identifier",
            frame=pd.DataFrame({"value": [2.0]}),
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
            identifier="row_id",
        )

        with self.assertRaisesRegex(ContractViolation, "absent from the raw frame"):
            validate_tabular_dataset(dataset)

    def test_identifier_cannot_also_be_modeled(self) -> None:
        dataset = TabularDataset(
            name="modeled-identifier",
            frame=pd.DataFrame({"row_id": [1]}),
            columns=(ColumnSpec("row_id", ColumnKind.DISCRETE),),
            identifier="row_id",
        )

        with self.assertRaisesRegex(ContractViolation, "cannot be a modeled column"):
            validate_tabular_dataset(dataset)

    def test_target_and_task_must_be_declared_together(self) -> None:
        dataset = TabularDataset(
            name="missing-task",
            frame=pd.DataFrame({"label": ["yes"]}),
            columns=(ColumnSpec("label", ColumnKind.CATEGORICAL),),
            target="label",
        )

        with self.assertRaisesRegex(ContractViolation, "both present"):
            validate_tabular_dataset(dataset)

    def test_raw_target_kind_must_match_task(self) -> None:
        cases = (
            (TaskType.CLASSIFICATION, ColumnKind.CONTINUOUS, "discrete or categorical"),
            (TaskType.REGRESSION, ColumnKind.CATEGORICAL, "continuous or discrete"),
        )
        for task, kind, expected_message in cases:
            with self.subTest(task=task, kind=kind):
                dataset = TabularDataset(
                    name="incompatible-target",
                    frame=pd.DataFrame({"target": [1.0]}),
                    columns=(ColumnSpec("target", kind),),
                    target="target",
                    task=task,
                )

                with self.assertRaisesRegex(ContractViolation, expected_message):
                    validate_tabular_dataset(dataset)

    def test_numeric_discrete_target_is_valid_for_either_task(self) -> None:
        for task in TaskType:
            with self.subTest(task=task):
                dataset = TabularDataset(
                    name="numeric-discrete-target",
                    frame=pd.DataFrame({"target": [0, 1]}),
                    columns=(ColumnSpec("target", ColumnKind.DISCRETE),),
                    target="target",
                    task=task,
                )

                validate_tabular_dataset(dataset)

    def test_numeric_semantics_reject_strings_without_reclassification(self) -> None:
        dataset = TabularDataset(
            name="wrong-kind",
            frame=pd.DataFrame({"count": ["one", "two"]}),
            columns=(ColumnSpec("count", ColumnKind.DISCRETE),),
        )

        with self.assertRaisesRegex(ContractViolation, "must be real numeric"):
            validate_tabular_dataset(dataset)

    def test_integer_codes_are_valid_raw_categorical_values(self) -> None:
        dataset = TabularDataset(
            name="numeric-categories",
            frame=pd.DataFrame({"region": [1, 3]}),
            columns=(ColumnSpec("region", ColumnKind.CATEGORICAL),),
        )

        validate_tabular_dataset(dataset)

    def test_only_categorical_column_can_declare_ordered_values(self) -> None:
        for kind in (ColumnKind.CONTINUOUS, ColumnKind.DISCRETE):
            with self.subTest(kind=kind):
                dataset = TabularDataset(
                    name="bad-order",
                    frame=pd.DataFrame({"value": [1.0, 2.0]}),
                    columns=(
                        ColumnSpec(
                            "value",
                            kind,
                            ordered_values=(1.0, 2.0),
                        ),
                    ),
                )

                with self.assertRaisesRegex(ContractViolation, "Only categorical"):
                    validate_tabular_dataset(dataset)

    def test_ordered_domain_must_cover_observed_values(self) -> None:
        dataset = TabularDataset(
            name="incomplete-order",
            frame=pd.DataFrame({"grade": ["low", "high"]}),
            columns=(
                ColumnSpec(
                    "grade",
                    ColumnKind.CATEGORICAL,
                    ordered_values=("low", "medium"),
                ),
            ),
        )

        with self.assertRaisesRegex(ContractViolation, "absent from ordered_values"):
            validate_tabular_dataset(dataset)

    def test_ordered_timestamp_categories_preserve_raw_scalar_identity(self) -> None:
        first = pd.Timestamp("2020-01-01")
        second = pd.Timestamp("2020-02-01")
        dataset = TabularDataset(
            name="ordered-timestamps",
            frame=pd.DataFrame({"period": [second, first]}),
            columns=(
                ColumnSpec(
                    "period",
                    ColumnKind.CATEGORICAL,
                    ordered_values=(first, second),
                ),
            ),
        )

        validate_tabular_dataset(dataset)

    def test_nested_unhashable_ordered_value_is_a_contract_violation(self) -> None:
        dataset = TabularDataset(
            name="nested-unhashable",
            frame=pd.DataFrame({"category": ["known"]}),
            columns=(
                ColumnSpec(
                    "category",
                    ColumnKind.CATEGORICAL,
                    ordered_values=(("nested", ["list"]),),  # type: ignore[list-item]
                ),
            ),
        )

        with self.assertRaisesRegex(ContractViolation, "unhashable"):
            validate_tabular_dataset(dataset)


class InputAndPreparedValidationTests(unittest.TestCase):
    """Exercise semantic model views and canonical prepared-table checks."""

    def _schema(self) -> PreparedSchema:
        return PreparedSchema(
            column_order=("amount", "count", "label"),
            continuous_columns=("amount",),
            discrete_columns=("count",),
            categorical_columns=("label",),
            target_col="label",
            task_type=TaskType.CLASSIFICATION,
            state_columns={
                "count": StateColumn(cardinality=3, ordered=True),
                "label": StateColumn(cardinality=2, ordered=False),
            },
        )

    def test_input_spec_accepts_only_semantic_enums(self) -> None:
        spec = InputSpec(
            continuous_view=ContinuousView.STANDARD,
            discrete_view=DiscreteView.FINITE_STATE_CODES,
            categorical_view=CategoricalView.FINITE_STATE_CODES,
        )
        validate_input_spec(spec)

        invalid = InputSpec(  # type: ignore[arg-type]
            continuous_view="standard",
            discrete_view=DiscreteView.FINITE_STATE_CODES,
            categorical_view=CategoricalView.FINITE_STATE_CODES,
        )
        with self.assertRaisesRegex(ContractViolation, "ContinuousView"):
            validate_input_spec(invalid)

    def test_input_spec_has_only_three_semantic_fields(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(InputSpec)),
            ("continuous_view", "discrete_view", "categorical_view"),
        )

    def test_view_enums_contain_only_approved_mvp_values(self) -> None:
        self.assertEqual(
            tuple(ContinuousView),
            (
                ContinuousView.RAW,
                ContinuousView.STANDARD,
                ContinuousView.UNSUPPORTED,
            ),
        )
        self.assertEqual(
            tuple(DiscreteView),
            (
                DiscreteView.RAW_VALUES,
                DiscreteView.FINITE_STATE_CODES,
                DiscreteView.UNSUPPORTED,
            ),
        )
        self.assertEqual(
            tuple(CategoricalView),
            (
                CategoricalView.RAW_VALUES,
                CategoricalView.FINITE_STATE_CODES,
                CategoricalView.UNSUPPORTED,
            ),
        )

    def test_prepared_schema_snapshots_state_mapping(self) -> None:
        source = {"count": StateColumn(cardinality=3, ordered=True)}
        schema = PreparedSchema(
            column_order=("count",),
            continuous_columns=(),
            discrete_columns=("count",),
            categorical_columns=(),
            target_col=None,
            task_type=None,
            state_columns=source,
        )

        source["count"] = StateColumn(cardinality=99, ordered=False)

        self.assertEqual(schema.state_columns["count"].cardinality, 3)
        with self.assertRaises(TypeError):
            schema.state_columns["new"] = StateColumn(  # type: ignore[index]
                cardinality=2,
                ordered=False,
            )

    def test_prepared_groups_must_partition_canonical_order(self) -> None:
        schema = PreparedSchema(
            column_order=("amount", "label"),
            continuous_columns=("amount",),
            discrete_columns=(),
            categorical_columns=(),
            target_col="label",
            task_type=TaskType.CLASSIFICATION,
        )

        with self.assertRaisesRegex(ContractViolation, "partition"):
            validate_prepared_schema(schema)

    def test_prepared_groups_must_follow_canonical_order(self) -> None:
        schema = PreparedSchema(
            column_order=("first", "second"),
            continuous_columns=("second", "first"),
            discrete_columns=(),
            categorical_columns=(),
            target_col=None,
            task_type=None,
        )

        with self.assertRaisesRegex(ContractViolation, "must follow column_order"):
            validate_prepared_schema(schema)

    def test_prepared_target_kind_must_match_task(self) -> None:
        cases = (
            (TaskType.CLASSIFICATION, ("target",), (), "discrete or categorical"),
            (TaskType.REGRESSION, (), ("target",), "continuous or discrete"),
        )
        for task, continuous, categorical, expected_message in cases:
            with self.subTest(task=task):
                schema = PreparedSchema(
                    column_order=("target",),
                    continuous_columns=continuous,
                    discrete_columns=(),
                    categorical_columns=categorical,
                    target_col="target",
                    task_type=task,
                )

                with self.assertRaisesRegex(ContractViolation, expected_message):
                    validate_prepared_schema(schema)

    def test_numeric_discrete_state_must_be_ordered(self) -> None:
        schema = PreparedSchema(
            column_order=("count",),
            continuous_columns=(),
            discrete_columns=("count",),
            categorical_columns=(),
            target_col=None,
            task_type=None,
            state_columns={
                "count": StateColumn(cardinality=3, ordered=False),
            },
        )

        with self.assertRaisesRegex(ContractViolation, "must be ordered"):
            validate_prepared_schema(schema)

    def test_state_metadata_fields_are_strictly_typed_and_positive(self) -> None:
        invalid_states = (
            (StateColumn(cardinality=True, ordered=True), "must be an integer"),
            (StateColumn(cardinality=0, ordered=True), "must be positive"),
            (StateColumn(cardinality=2, ordered=1), "must be bool"),
        )
        for state, expected_message in invalid_states:
            with self.subTest(state=state):
                schema = PreparedSchema(
                    column_order=("count",),
                    continuous_columns=(),
                    discrete_columns=("count",),
                    categorical_columns=(),
                    target_col=None,
                    task_type=None,
                    state_columns={"count": state},
                )

                with self.assertRaisesRegex(ContractViolation, expected_message):
                    validate_prepared_schema(schema)

    def test_state_metadata_cannot_cover_only_part_of_semantic_group(self) -> None:
        schema = PreparedSchema(
            column_order=("first", "second"),
            continuous_columns=(),
            discrete_columns=("first", "second"),
            categorical_columns=(),
            target_col=None,
            task_type=None,
            state_columns={
                "first": StateColumn(cardinality=3, ordered=True),
            },
        )

        with self.assertRaisesRegex(ContractViolation, "whole semantic group"):
            validate_prepared_schema(schema)

    def test_prepared_table_accepts_valid_named_state_codes(self) -> None:
        table = PreparedTable(
            frame=pd.DataFrame(
                {
                    "amount": [0.1, -0.5],
                    "count": pd.Series([0, 2], dtype="int64"),
                    "label": pd.Series([1, 0], dtype="int64"),
                }
            ),
            schema=self._schema(),
        )

        validate_prepared_table(table, expected_rows=2)

    def test_expected_row_count_is_strictly_validated(self) -> None:
        table = PreparedTable(
            frame=pd.DataFrame(
                {
                    "amount": [0.1],
                    "count": pd.Series([1], dtype="int64"),
                    "label": pd.Series([1], dtype="int64"),
                }
            ),
            schema=self._schema(),
        )
        cases = (
            (True, "must be an integer"),
            (1.0, "must be an integer"),
            (-1, "must be non-negative"),
            (2, "row count"),
        )
        for expected_rows, expected_message in cases:
            with self.subTest(expected_rows=expected_rows):
                with self.assertRaisesRegex(ContractViolation, expected_message):
                    validate_prepared_table(
                        table,
                        expected_rows=expected_rows,  # type: ignore[arg-type]
                    )

    def test_prepared_table_rejects_missing_values(self) -> None:
        table = PreparedTable(
            frame=pd.DataFrame(
                {
                    "amount": [None],
                    "count": pd.Series([1], dtype="int64"),
                    "label": pd.Series([1], dtype="int64"),
                }
            ),
            schema=self._schema(),
        )

        with self.assertRaisesRegex(ContractViolation, "contains missing values"):
            validate_prepared_table(table)

    def test_prepared_table_rejects_invalid_state_without_clipping(self) -> None:
        table = PreparedTable(
            frame=pd.DataFrame(
                {
                    "amount": [0.1],
                    "count": pd.Series([3], dtype="int64"),
                    "label": pd.Series([1], dtype="int64"),
                }
            ),
            schema=self._schema(),
        )

        with self.assertRaisesRegex(ContractViolation, "invalid codes"):
            validate_prepared_table(table)

    def test_prepared_state_requires_integer_dtype(self) -> None:
        table = PreparedTable(
            frame=pd.DataFrame(
                {
                    "amount": [0.1],
                    "count": [1.0],
                    "label": pd.Series([1], dtype="int64"),
                }
            ),
            schema=self._schema(),
        )

        with self.assertRaisesRegex(ContractViolation, "integer dtype"):
            validate_prepared_table(table)

    def test_prepared_continuous_output_rejects_infinity(self) -> None:
        table = PreparedTable(
            frame=pd.DataFrame(
                {
                    "amount": [float("inf")],
                    "count": pd.Series([1], dtype="int64"),
                    "label": pd.Series([1], dtype="int64"),
                }
            ),
            schema=self._schema(),
        )

        with self.assertRaisesRegex(ContractViolation, "non-finite"):
            validate_prepared_table(table)

    def test_prepared_raw_discrete_column_must_remain_numeric(self) -> None:
        schema = PreparedSchema(
            column_order=("count",),
            continuous_columns=(),
            discrete_columns=("count",),
            categorical_columns=(),
            target_col=None,
            task_type=None,
        )
        table = PreparedTable(
            frame=pd.DataFrame({"count": ["one"]}),
            schema=schema,
        )

        with self.assertRaisesRegex(ContractViolation, "must be real numeric"):
            validate_prepared_table(table)

    def test_prepared_raw_category_must_remain_hashable(self) -> None:
        schema = PreparedSchema(
            column_order=("category",),
            continuous_columns=(),
            discrete_columns=(),
            categorical_columns=("category",),
            target_col=None,
            task_type=None,
        )
        table = PreparedTable(
            frame=pd.DataFrame({"category": [["nested"]]}),
            schema=schema,
        )

        with self.assertRaisesRegex(ContractViolation, "unhashable"):
            validate_prepared_table(table)


if __name__ == "__main__":
    unittest.main()
