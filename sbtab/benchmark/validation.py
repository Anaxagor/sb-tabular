"""Strict validation at the unified benchmark contract boundaries.

This module validates four different objects, each at the boundary that owns
its invariants:

``TabularDataset``
    The raw table and its semantic declaration, before missing-value handling
    and splitting.
``InputSpec``
    The three semantic representations requested by a model family.
``PreparedSchema``
    The column order and representation metadata produced by a fold-local
    codec.
``PreparedTable``
    The actual codec or adapter data checked against ``PreparedSchema``.

Validation is deliberately fail-fast and non-corrective. It never clips state
codes, rounds generated values, drops rows, reorders columns, or guesses a
column kind. Repairing data here would make different models run under
different effective benchmark conditions and would hide the component that
broke its contract.
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


# These small validators keep public boundary functions focused on semantic
# relationships while preserving precise field names in error messages.
def _require_enum(value: object, enum_type: type[EnumT], field_name: str) -> None:
    """Require an enum member, not an equivalent string or integer value."""

    if not isinstance(value, enum_type):
        raise ContractViolation(
            f"{field_name} must be {enum_type.__name__}, got {value!r}."
        )


def _require_name(value: object, field_name: str) -> None:
    """Require a usable column or object name without normalizing it."""

    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(f"{field_name} must be a non-empty string.")


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    """Return each duplicate once, in the order it first repeats."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _require_name_tuple(values: object, field_name: str) -> tuple[str, ...]:
    """Validate immutable, ordered and duplicate-free column names."""

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
    """Validate values suitable for an exact finite-state codebook.

    Hashability is required because the codec maps raw values to integer state
    codes by dictionary lookup. Missing and duplicate entries would make that
    mapping ambiguous even if pandas happened to accept them as labels.
    """

    domain: set[object] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Hashable):
            raise ContractViolation(
                f"{field_name}[{index}]={value!r} is unhashable."
            )
        # ``pd.isna`` is the cross-dtype missing predicate used by the raw
        # pipeline. Some exotic scalar objects do not support it cleanly; that
        # alone does not make a hashable category invalid.
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
    """Require finite real numbers while rejecting bool and complex dtypes."""

    # Pandas considers booleans numeric, but treating False/True as magnitudes
    # would silently change categorical semantics. Complex values likewise
    # have no agreed ordering or real-valued transport interpretation here.
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
    """Require trainable categorical scalars without imposing a storage dtype."""

    try:
        observed_values = tuple(series.dropna().drop_duplicates().tolist())
    except TypeError as error:
        raise ContractViolation(
            f"{semantic_label} contains unhashable values."
        ) from error
    _require_hashable_domain(observed_values, f"observed support for {series.name!r}")


def validate_column_spec(column: ColumnSpec, series: pd.Series | None = None) -> None:
    """Validate one column declaration and its optional observed raw support.

    Declaration-only validation is used before the frame is indexed, so a bad
    ``ColumnSpec`` produces a contract error rather than an incidental pandas
    exception. When ``series`` is supplied, the same declaration is checked
    against values actually present in the raw table.
    """

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

    # ``ordered_values`` is semantic metadata, not merely a list of known
    # categories. Its tuple position determines adjacency for models that use
    # ordered finite-state transitions.
    declared_domain = _require_hashable_domain(
        ordered_values,
        f"ColumnSpec[{column.name!r}].ordered_values",
    )
    if series is None:
        return

    observed_values = tuple(series.dropna().drop_duplicates().tolist())
    observed_domain = _require_hashable_domain(
        observed_values,
        f"observed support for {column.name!r}",
    )
    # A declared order may contain states absent from this particular dataset,
    # but every observed state must have an explicitly declared position.
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
    """Validate the complete raw dataset declaration before orchestration.

    The raw frame must consist exactly of modeled columns plus, optionally, one
    identifier. The target remains an ordinary modeled column; ``task`` only
    tells downstream evaluation how to interpret it. This function validates
    raw values but does not filter missing rows or learn any transform state.
    """

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

    # Validate declarations before using their names to index the frame. This
    # keeps malformed metadata errors separate from missing-frame-column errors.
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

    # A task without a target is unusable, while a target without a task would
    # force evaluators to infer classification versus regression from dtype.
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

    # Identifiers are retained only for audit/output reconstruction. They never
    # enter PreparedTable and therefore cannot also be modeled features.
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

    # Reject extra raw columns instead of silently dropping them. Otherwise a
    # misspelled ColumnSpec or forgotten target could yield a successful but
    # semantically different experiment.
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

    # Only after structural checks succeed do we validate each observed raw
    # support against its declaration.
    for column in dataset.columns:
        validate_column_spec(column, dataset.frame[column.name])


def validate_input_spec(spec: InputSpec) -> None:
    """Validate the model's three semantic representation requests.

    Exact enum membership is intentional: accepting strings here would turn
    typos and future enum additions into implicit compatibility behavior.
    Native tensor layout, dtype and target extraction are adapter concerns and
    therefore have no corresponding checks in this function.
    """

    if not isinstance(spec, InputSpec):
        raise ContractViolation(
            f"spec must be InputSpec, got {type(spec).__name__}."
        )
    _require_enum(spec.continuous_view, ContinuousView, "continuous_view")
    _require_enum(spec.discrete_view, DiscreteView, "discrete_view")
    _require_enum(spec.categorical_view, CategoricalView, "categorical_view")


def validate_state_column(name: str, state: StateColumn) -> None:
    """Validate finite-state cardinality and ordering metadata for one column.

    ``cardinality`` defines the complete valid prepared-code interval
    ``[0, cardinality)``. ``ordered`` describes whether code adjacency has
    semantic meaning; it is not inferred later from numeric-looking values.
    """

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
    """Validate canonical order, semantic partition, target, and state metadata.

    Every prepared column belongs to exactly one semantic group, and each group
    preserves its relative position from ``column_order``. State metadata is
    keyed by column name so adapters never have to infer cardinality or ordering
    from an array position.
    """

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
    # The three groups are a partition, not independent convenience lists. A
    # duplicate would give two conflicting meanings to the same prepared data.
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
    # Adapters may form separate native blocks, but the order inside each block
    # must remain a stable projection of the canonical table order.
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

    # Prepared data keeps the target in the canonical table. These fields mark
    # its evaluation semantics; they do not authorize a shared X/y split.
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

    # Continuous columns never use finite-state metadata. Their representation
    # is governed solely by ContinuousView and codec transform state.
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

    # InputSpec chooses one view for an entire semantic group. Partial metadata
    # would imply that different columns silently received different views, so
    # a finite-state group must be covered completely or not at all.
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
    """Validate one codec or adapter table without repairing model output.

    This function is used at both sides of the adapter boundary. It checks that
    physical DataFrame values implement the attached semantic schema, including
    exact column order, optional expected row count, missingness, numeric
    finiteness, raw-view value types and integer state-code ranges.
    """

    if not isinstance(table, PreparedTable):
        raise ContractViolation(
            f"table must be PreparedTable, got {type(table).__name__}."
        )
    if not isinstance(table.frame, pd.DataFrame):
        raise ContractViolation("PreparedTable.frame must be a pandas DataFrame.")
    validate_prepared_schema(table.schema)

    # Equality rather than set comparison protects the canonical order used to
    # assemble native tensors and to decode the returned sample.
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

    # MissingPolicy is applied once before splitting. Missing values here mean
    # that a codec or adapter violated the trusted boundary; this layer must not
    # apply a second, model-specific fallback.
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

    # A finite column without StateColumn metadata is intentionally in its raw
    # view. Numeric discrete values remain real numeric; categorical values may
    # use any exact hashable scalar representation.
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

    # Encoded states are strict integer codes. Numerically integral floats are
    # rejected instead of rounded because rounding would conceal invalid model
    # output and could change the generated distribution.
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
