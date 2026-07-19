"""Strict validation at unified benchmark contract boundaries.

Validation rejects ambiguous declarations and invalid prepared model output.
It never repairs data by clipping, rounding, dropping, reordering, or guessing
semantics. The caller receives evidence identifying the broken field or
column.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from enum import Enum
from typing import TypeVar

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from sbtab.benchmark.contracts import (
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    DiscreteView,
    InputSpec,
    PreparedSchema,
    PreparedTable,
    StateColumn,
    TabularDataset,
    TaskType,
)


class ContractViolation(ValueError):
    """Raised when benchmark data contradicts its declared contract."""


EnumT = TypeVar("EnumT", bound=Enum)


def _require_enum(value: object, enum_type: type[EnumT], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise ContractViolation(
            f"{field_name} must be {enum_type.__name__}, got {value!r}."
        )


def _require_name(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(f"{field_name} must be a non-empty string.")


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _require_name_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ContractViolation(f"{field_name} must be a tuple of column names.")
    for index, value in enumerate(values):
        _require_name(value, f"{field_name}[{index}]")
    duplicates = _duplicates(values)
    if duplicates:
        raise ContractViolation(
            f"{field_name} contains duplicate columns: {list(duplicates)!r}."
        )
    return values


def _require_hashable_domain(
    values: tuple[object, ...],
    field_name: str,
) -> set[object]:
    domain: set[object] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Hashable):
            raise ContractViolation(
                f"{field_name}[{index}]={value!r} is not hashable."
            )
        try:
            is_missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_missing = False
        if is_missing:
            raise ContractViolation(
                f"{field_name}[{index}] must not be a missing value."
            )
        if value in domain:
            raise ContractViolation(
                f"{field_name} contains duplicate value {value!r}."
            )
        domain.add(value)
    return domain


def _require_real_numeric_series(series: pd.Series, semantic_label: str) -> None:
    if (
        not is_numeric_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
    ):
        raise ContractViolation(
            f"{semantic_label} must be real numeric, got dtype {series.dtype!r}."
        )
    observed_numeric = series.dropna().to_numpy(dtype=np.float64)
    if observed_numeric.size and not bool(np.isfinite(observed_numeric).all()):
        raise ContractViolation(f"{semantic_label} contains non-finite values.")


def _require_hashable_series(series: pd.Series, semantic_label: str) -> None:
    try:
        observed_values = tuple(pd.unique(series.dropna()).tolist())
    except TypeError as error:
        raise ContractViolation(
            f"{semantic_label} contains unhashable values."
        ) from error
    _require_hashable_domain(observed_values, f"observed support for {series.name!r}")


def validate_column_spec(column: ColumnSpec, series: pd.Series | None = None) -> None:
    """Validate one column declaration and its optional observed raw support."""

    if not isinstance(column, ColumnSpec):
        raise ContractViolation(
            f"columns entries must be ColumnSpec, got {type(column).__name__}."
        )
    _require_name(column.name, "ColumnSpec.name")
    _require_enum(column.kind, ColumnKind, f"ColumnSpec[{column.name!r}].kind")

    if series is not None and column.kind in {
        ColumnKind.CONTINUOUS,
        ColumnKind.DISCRETE,
    }:
        _require_real_numeric_series(
            series,
            f"Raw {column.kind.value} column {column.name!r}",
        )

    if series is not None and column.kind is ColumnKind.CATEGORICAL:
        _require_hashable_series(
            series,
            f"Raw categorical column {column.name!r}",
        )

    ordered_values = column.ordered_values
    if ordered_values is None:
        return
    if not isinstance(ordered_values, tuple):
        raise ContractViolation(
            f"ColumnSpec[{column.name!r}].ordered_values must be a tuple or None."
        )
    if column.kind is ColumnKind.CONTINUOUS:
        raise ContractViolation(
            f"Continuous column {column.name!r} cannot declare ordered_values."
        )
    if not ordered_values:
        raise ContractViolation(
            f"ColumnSpec[{column.name!r}].ordered_values cannot be empty."
        )

    declared_domain = _require_hashable_domain(
        ordered_values,
        f"ColumnSpec[{column.name!r}].ordered_values",
    )
    if series is None:
        return

    observed_values = tuple(pd.unique(series.dropna()).tolist())
    observed_domain = _require_hashable_domain(
        observed_values,
        f"observed support for {column.name!r}",
    )
    missing_values = observed_domain - declared_domain
    if missing_values:
        ordered_missing = [
            value for value in observed_values if value in missing_values
        ]
        raise ContractViolation(
            f"Column {column.name!r} has observed values absent from "
            f"ordered_values: {ordered_missing!r}."
        )


def validate_tabular_dataset(dataset: TabularDataset) -> None:
    """Validate raw schema, modeled order, target, task, and identifier rules."""

    if not isinstance(dataset, TabularDataset):
        raise ContractViolation(
            f"dataset must be TabularDataset, got {type(dataset).__name__}."
        )
    _require_name(dataset.name, "TabularDataset.name")
    if not isinstance(dataset.frame, pd.DataFrame):
        raise ContractViolation("TabularDataset.frame must be a pandas DataFrame.")
    if not isinstance(dataset.columns, tuple):
        raise ContractViolation("TabularDataset.columns must be a tuple.")
    if not dataset.columns:
        raise ContractViolation("TabularDataset.columns must not be empty.")

    duplicate_frame_columns = tuple(
        str(name) for name in dataset.frame.columns[dataset.frame.columns.duplicated()]
    )
    if duplicate_frame_columns:
        raise ContractViolation(
            "TabularDataset.frame contains duplicate column labels: "
            f"{list(duplicate_frame_columns)!r}."
        )

    modeled_names: list[str] = []
    for column in dataset.columns:
        if not isinstance(column, ColumnSpec):
            raise ContractViolation(
                "TabularDataset.columns entries must be ColumnSpec, got "
                f"{type(column).__name__}."
            )
        validate_column_spec(column)
        modeled_names.append(column.name)
    duplicates = _duplicates(modeled_names)
    if duplicates:
        raise ContractViolation(
            f"TabularDataset.columns contains duplicate names: {list(duplicates)!r}."
        )

    missing_columns = [
        name for name in modeled_names if name not in dataset.frame.columns
    ]
    if missing_columns:
        raise ContractViolation(
            f"Dataset {dataset.name!r} is missing modeled columns: {missing_columns!r}."
        )

    if (dataset.target is None) != (dataset.task is None):
        raise ContractViolation(
            "TabularDataset.target and TabularDataset.task must be both present "
            "or both absent."
        )
    if dataset.target is not None:
        _require_name(dataset.target, "TabularDataset.target")
        _require_enum(dataset.task, TaskType, "TabularDataset.task")
        if dataset.target not in modeled_names:
            raise ContractViolation(
                f"Target {dataset.target!r} must be one of the modeled columns."
            )

    expected_frame_columns = set(modeled_names)
    if dataset.identifier is not None:
        _require_name(dataset.identifier, "TabularDataset.identifier")
        if dataset.identifier not in dataset.frame.columns:
            raise ContractViolation(
                f"Identifier {dataset.identifier!r} is absent from the raw frame."
            )
        if dataset.identifier in modeled_names:
            raise ContractViolation(
                f"Identifier {dataset.identifier!r} cannot be a modeled column."
            )
        expected_frame_columns.add(dataset.identifier)

    undeclared_columns = [
        str(name)
        for name in dataset.frame.columns
        if name not in expected_frame_columns
    ]
    if undeclared_columns:
        raise ContractViolation(
            f"Dataset {dataset.name!r} contains undeclared raw columns: "
            f"{undeclared_columns!r}. Declare one as identifier or remove it."
        )

    for column in dataset.columns:
        validate_column_spec(column, dataset.frame[column.name])


def validate_input_spec(spec: InputSpec) -> None:
    """Validate that a model requests exactly the approved semantic enums."""

    if not isinstance(spec, InputSpec):
        raise ContractViolation(
            f"spec must be InputSpec, got {type(spec).__name__}."
        )
    _require_enum(spec.continuous_view, ContinuousView, "continuous_view")
    _require_enum(spec.discrete_view, DiscreteView, "discrete_view")
    _require_enum(spec.categorical_view, CategoricalView, "categorical_view")


def validate_state_column(name: str, state: StateColumn) -> None:
    """Validate cardinality and order meaning for one prepared state column."""

    _require_name(name, "state column name")
    if not isinstance(state, StateColumn):
        raise ContractViolation(
            f"State metadata for {name!r} must be StateColumn, "
            f"got {type(state).__name__}."
        )
    if isinstance(state.cardinality, bool) or not isinstance(state.cardinality, int):
        raise ContractViolation(
            f"State column {name!r} cardinality must be an integer."
        )
    if state.cardinality < 1:
        raise ContractViolation(
            f"State column {name!r} cardinality must be positive, "
            f"got {state.cardinality}."
        )
    if not isinstance(state.ordered, bool):
        raise ContractViolation(
            f"State column {name!r} ordered must be bool, got {state.ordered!r}."
        )


def validate_prepared_schema(schema: PreparedSchema) -> None:
    """Validate canonical order, semantic partition, target, and state metadata."""

    if not isinstance(schema, PreparedSchema):
        raise ContractViolation(
            f"schema must be PreparedSchema, got {type(schema).__name__}."
        )
    column_order = _require_name_tuple(schema.column_order, "column_order")
    if not column_order:
        raise ContractViolation("PreparedSchema.column_order must not be empty.")

    groups = {
        "continuous_columns": _require_name_tuple(
            schema.continuous_columns,
            "continuous_columns",
        ),
        "discrete_columns": _require_name_tuple(
            schema.discrete_columns,
            "discrete_columns",
        ),
        "categorical_columns": _require_name_tuple(
            schema.categorical_columns,
            "categorical_columns",
        ),
    }
    assigned = [name for values in groups.values() for name in values]
    duplicates = _duplicates(assigned)
    if duplicates:
        raise ContractViolation(
            "Prepared columns belong to multiple semantic groups: "
            f"{list(duplicates)!r}."
        )
    missing_from_groups = [name for name in column_order if name not in assigned]
    unknown_group_columns = [name for name in assigned if name not in column_order]
    if missing_from_groups or unknown_group_columns:
        raise ContractViolation(
            "Prepared semantic groups must partition column_order exactly; "
            f"missing={missing_from_groups!r}, unknown={unknown_group_columns!r}."
        )
    for group_name, group_columns in groups.items():
        group_set = set(group_columns)
        canonical_group_order = tuple(
            name for name in column_order if name in group_set
        )
        if group_columns != canonical_group_order:
            raise ContractViolation(
                f"PreparedSchema.{group_name} must follow column_order; "
                f"actual={group_columns!r}, expected={canonical_group_order!r}."
            )

    if (schema.target_col is None) != (schema.task_type is None):
        raise ContractViolation(
            "PreparedSchema.target_col and task_type must be both present or "
            "both absent."
        )
    if schema.target_col is not None:
        _require_name(schema.target_col, "PreparedSchema.target_col")
        _require_enum(schema.task_type, TaskType, "PreparedSchema.task_type")
        if schema.target_col not in column_order:
            raise ContractViolation(
                f"Prepared target {schema.target_col!r} is absent from column_order."
            )

    allowed_state_columns = set(schema.discrete_columns) | set(
        schema.categorical_columns
    )
    for name, state in schema.state_columns.items():
        if name not in allowed_state_columns:
            raise ContractViolation(
                f"State metadata for {name!r} is valid only for a declared "
                "discrete or categorical column."
            )
        validate_state_column(name, state)
        if name in schema.discrete_columns and not state.ordered:
            raise ContractViolation(
                f"Numeric discrete state column {name!r} must be ordered."
            )

    state_names = set(schema.state_columns)
    for group_name, group_columns in (
        ("discrete", schema.discrete_columns),
        ("categorical", schema.categorical_columns),
    ):
        group_set = set(group_columns)
        covered = state_names & group_set
        if covered and covered != group_set:
            missing_state_metadata = tuple(
                name for name in group_columns if name not in covered
            )
            raise ContractViolation(
                f"Finite-state metadata for the {group_name} group must cover "
                f"the whole semantic group; missing={missing_state_metadata!r}."
            )


def validate_prepared_table(
    table: PreparedTable,
    *,
    expected_rows: int | None = None,
) -> None:
    """Validate one prepared adapter input or output without repairing it."""

    if not isinstance(table, PreparedTable):
        raise ContractViolation(
            f"table must be PreparedTable, got {type(table).__name__}."
        )
    if not isinstance(table.frame, pd.DataFrame):
        raise ContractViolation("PreparedTable.frame must be a pandas DataFrame.")
    validate_prepared_schema(table.schema)

    actual_columns = tuple(table.frame.columns.tolist())
    if actual_columns != table.schema.column_order:
        raise ContractViolation(
            "PreparedTable columns must exactly match schema.column_order; "
            f"actual={actual_columns!r}, expected={table.schema.column_order!r}."
        )
    if expected_rows is not None:
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
            raise ContractViolation("expected_rows must be an integer or None.")
        if expected_rows < 0:
            raise ContractViolation("expected_rows must be non-negative.")
        if len(table.frame) != expected_rows:
            raise ContractViolation(
                f"PreparedTable row count is {len(table.frame)}, expected "
                f"{expected_rows}."
            )

    missing_counts = table.frame.isna().sum()
    columns_with_missing = {
        str(name): int(count)
        for name, count in missing_counts.items()
        if int(count) > 0
    }
    if columns_with_missing:
        raise ContractViolation(
            "PreparedTable contains missing values after the benchmark missing "
            f"policy: {columns_with_missing!r}."
        )

    for name in table.schema.continuous_columns:
        _require_real_numeric_series(
            table.frame[name],
            f"Prepared continuous column {name!r}",
        )

    state_names = set(table.schema.state_columns)
    for name in table.schema.discrete_columns:
        if name not in state_names:
            _require_real_numeric_series(
                table.frame[name],
                f"Prepared raw discrete column {name!r}",
            )
    for name in table.schema.categorical_columns:
        if name not in state_names:
            _require_hashable_series(
                table.frame[name],
                f"Prepared raw categorical column {name!r}",
            )

    for name, state in table.schema.state_columns.items():
        series = table.frame[name]
        if is_bool_dtype(series.dtype) or not is_integer_dtype(series.dtype):
            raise ContractViolation(
                f"Prepared state column {name!r} must have an integer dtype, "
                f"got {series.dtype!r}."
            )
        invalid_mask = (series < 0) | (series >= state.cardinality)
        if bool(invalid_mask.any()):
            invalid_values = tuple(pd.unique(series[invalid_mask]).tolist())
            raise ContractViolation(
                f"Prepared state column {name!r} contains invalid codes "
                f"{invalid_values!r}; valid range is "
                f"[0, {state.cardinality})."
            )
