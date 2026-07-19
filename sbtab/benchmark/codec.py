"""Fold-local preprocessing between raw tables and model adapters.

The codec learns only from one training partition. It has deliberately no
method for transforming held-out rows: generators train on prepared train data,
sample new prepared rows, and evaluation receives the inversely decoded sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

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
)
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_input_spec,
    validate_prepared_table,
    validate_tabular_dataset,
)


@dataclass(frozen=True)
class _StandardTransform:
    """Train-fitted population location and non-zero scale for one column."""

    mean: float
    scale: float


@dataclass(frozen=True)
class _StateTransform:
    """Reversible train-observed finite support for one column."""

    values: tuple[object, ...]
    value_to_code: Mapping[object, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value_to_code",
            MappingProxyType(dict(self.value_to_code)),
        )


class ModelCodec:
    """One single-use codec fitted for one dataset, model, and fold.

    Construct through :func:`compile_codec`, then call :meth:`fit_transform`
    exactly once. The object snapshots declarations but does not retain the
    complete dataset frame, which prevents later transform state from being
    derived from held-out values.
    """

    def __init__(self, dataset: TabularDataset, input_spec: InputSpec) -> None:
        validate_tabular_dataset(dataset)
        validate_input_spec(input_spec)
        self._reject_unsupported_groups(dataset, input_spec)

        self._dataset_name = dataset.name
        self._columns = dataset.columns
        self._column_order = dataset.column_order
        self._raw_column_order = tuple(dataset.frame.columns.tolist())
        self._target = dataset.target
        self._task = dataset.task
        self._identifier = dataset.identifier
        self._input_spec = input_spec
        self._schema: PreparedSchema | None = None
        self._standard: Mapping[str, _StandardTransform] = MappingProxyType({})
        self._states: Mapping[str, _StateTransform] = MappingProxyType({})
        self._raw_supports: Mapping[str, frozenset[object]] = MappingProxyType({})

    @staticmethod
    def _reject_unsupported_groups(
        dataset: TabularDataset,
        input_spec: InputSpec,
    ) -> None:
        groups = (
            (
                "continuous",
                dataset.continuous_columns,
                input_spec.continuous_view is ContinuousView.UNSUPPORTED,
            ),
            (
                "discrete",
                dataset.discrete_columns,
                input_spec.discrete_view is DiscreteView.UNSUPPORTED,
            ),
            (
                "categorical",
                dataset.categorical_columns,
                input_spec.categorical_view is CategoricalView.UNSUPPORTED,
            ),
        )
        for group_name, names, unsupported in groups:
            if names and unsupported:
                raise ContractViolation(
                    f"InputSpec marks the non-empty {group_name} group as "
                    f"UNSUPPORTED: {names!r}."
                )

    @property
    def input_spec(self) -> InputSpec:
        """Return the immutable semantic views compiled for this codec."""

        return self._input_spec

    @property
    def schema(self) -> PreparedSchema:
        """Return the fitted prepared schema or reject premature access."""

        if self._schema is None:
            raise ContractViolation("Codec is not fitted yet.")
        return self._schema

    def _training_dataset(self, train_raw: pd.DataFrame) -> TabularDataset:
        if not isinstance(train_raw, pd.DataFrame):
            raise ContractViolation("train_raw must be a pandas DataFrame.")
        actual_columns = tuple(train_raw.columns.tolist())
        if actual_columns != self._raw_column_order:
            raise ContractViolation(
                "train_raw columns must match the compiled raw dataset schema; "
                f"actual={actual_columns!r}, expected={self._raw_column_order!r}."
            )
        if train_raw.empty:
            raise ContractViolation("Cannot fit a codec on an empty train partition.")

        train_dataset = TabularDataset(
            name=self._dataset_name,
            frame=train_raw,
            columns=self._columns,
            target=self._target,
            task=self._task,
            identifier=self._identifier,
        )
        validate_tabular_dataset(train_dataset)
        missing = {
            name: int(train_raw[name].isna().sum())
            for name in self._column_order
            if train_raw[name].isna().any()
        }
        if missing:
            raise ContractViolation(
                "train_raw contains modeled missing values. Apply one benchmark "
                f"MissingPolicy before splitting: {missing!r}."
            )
        return train_dataset

    @staticmethod
    def _observed_values(series: pd.Series) -> tuple[object, ...]:
        # ``pd.unique(...).tolist()`` converts datetime64/timedelta64 values to
        # integer nanoseconds in some pandas versions. Series preserves the raw
        # scalar objects required by a reversible categorical codebook.
        return tuple(series.drop_duplicates().tolist())

    @classmethod
    def _state_values(
        cls,
        column: ColumnSpec,
        series: pd.Series,
    ) -> tuple[object, ...]:
        observed = cls._observed_values(series)
        if column.kind is ColumnKind.DISCRETE:
            return tuple(sorted(observed))
        if column.ordered_values is None:
            return observed
        observed_set = set(observed)
        return tuple(
            value for value in column.ordered_values if value in observed_set
        )

    def fit_transform(self, train_raw: pd.DataFrame) -> PreparedTable:
        """Fit on raw train only and return one canonical prepared table."""

        if self._schema is not None:
            raise ContractViolation("A fold-local codec can be fitted only once.")
        train_dataset = self._training_dataset(train_raw)

        standard: dict[str, _StandardTransform] = {}
        states: dict[str, _StateTransform] = {}
        raw_supports: dict[str, frozenset[object]] = {}
        prepared_columns: list[pd.Series] = []

        for column in self._columns:
            source = train_dataset.frame[column.name].reset_index(drop=True)
            if column.kind is ColumnKind.CONTINUOUS:
                prepared = self._prepare_continuous(column.name, source, standard)
            else:
                prepared = self._prepare_finite(
                    column,
                    source,
                    states,
                    raw_supports,
                )
            prepared_columns.append(prepared.rename(column.name))

        state_metadata = {
            name: StateColumn(
                cardinality=len(transform.values),
                ordered=(
                    self._column(name).kind is ColumnKind.DISCRETE
                    or self._column(name).ordered_values is not None
                ),
            )
            for name, transform in states.items()
        }
        schema = PreparedSchema(
            column_order=self._column_order,
            continuous_columns=train_dataset.continuous_columns,
            discrete_columns=train_dataset.discrete_columns,
            categorical_columns=train_dataset.categorical_columns,
            target_col=self._target,
            task_type=self._task,
            state_columns=state_metadata,
        )
        prepared_frame = pd.concat(prepared_columns, axis=1)
        prepared_table = PreparedTable(frame=prepared_frame, schema=schema)
        validate_prepared_table(prepared_table, expected_rows=len(train_raw))

        self._standard = MappingProxyType(standard)
        self._states = MappingProxyType(states)
        self._raw_supports = MappingProxyType(raw_supports)
        self._schema = schema
        return prepared_table

    def _column(self, name: str) -> ColumnSpec:
        for column in self._columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def _prepare_continuous(
        self,
        name: str,
        source: pd.Series,
        standard: dict[str, _StandardTransform],
    ) -> pd.Series:
        if self._input_spec.continuous_view is ContinuousView.RAW:
            return source.copy()

        values = source.to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        observed_scale = float(np.std(values, ddof=0))
        scale = observed_scale if observed_scale > 0.0 else 1.0
        standard[name] = _StandardTransform(mean=mean, scale=scale)
        return pd.Series((values - mean) / scale, dtype="float64")

    def _prepare_finite(
        self,
        column: ColumnSpec,
        source: pd.Series,
        states: dict[str, _StateTransform],
        raw_supports: dict[str, frozenset[object]],
    ) -> pd.Series:
        if column.kind is ColumnKind.DISCRETE:
            finite_state = (
                self._input_spec.discrete_view
                is DiscreteView.FINITE_STATE_CODES
            )
        else:
            finite_state = (
                self._input_spec.categorical_view
                is CategoricalView.FINITE_STATE_CODES
            )

        values = self._state_values(column, source)
        if not finite_state:
            raw_supports[column.name] = frozenset(values)
            return source.copy()

        value_to_code = {value: code for code, value in enumerate(values)}
        transform = _StateTransform(
            values=values,
            value_to_code=value_to_code,
        )
        encoded = source.map(transform.value_to_code)
        if encoded.isna().any():
            raise ContractViolation(
                f"Failed to encode train-observed values for {column.name!r}."
            )
        states[column.name] = transform
        return pd.Series(encoded.to_numpy(dtype=np.int64), dtype="int64")

    def inverse_transform(self, sample: PreparedTable) -> pd.DataFrame:
        """Decode a validated model sample to canonical raw semantic values."""

        schema = self.schema
        if not isinstance(sample, PreparedTable):
            raise ContractViolation("sample must be PreparedTable.")
        if sample.schema is not schema:
            raise ContractViolation(
                "Sample must carry the exact PreparedSchema fitted by this codec."
            )
        validate_prepared_table(sample)

        decoded_columns: list[pd.Series] = []
        for column in self._columns:
            source = sample.frame[column.name].reset_index(drop=True)
            decoded = self._decode_column(column, source)
            decoded_columns.append(decoded.rename(column.name))
        decoded_frame = pd.concat(decoded_columns, axis=1)

        decoded_dataset = TabularDataset(
            name=f"{self._dataset_name}:decoded-sample",
            frame=decoded_frame,
            columns=self._columns,
            target=self._target,
            task=self._task,
        )
        validate_tabular_dataset(decoded_dataset)
        return decoded_frame

    def _decode_column(
        self,
        column: ColumnSpec,
        source: pd.Series,
    ) -> pd.Series:
        standard = self._standard.get(column.name)
        if standard is not None:
            values = source.to_numpy(dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                decoded = values * standard.scale + standard.mean
            return pd.Series(decoded, dtype="float64")

        state = self._states.get(column.name)
        if state is not None:
            codes = source.to_numpy(dtype=np.int64)
            support = pd.Series(state.values)
            return support.iloc[codes].reset_index(drop=True)

        support = self._raw_supports.get(column.name)
        if support is not None:
            unknown = tuple(
                value
                for value in self._observed_values(source)
                if value not in support
            )
            if unknown:
                raise ContractViolation(
                    f"Raw finite-state sample column {column.name!r} contains "
                    f"values absent from train support: {unknown!r}."
                )
        return source.copy()


def compile_codec(dataset: TabularDataset, input_spec: InputSpec) -> ModelCodec:
    """Validate declarations and create an unfitted single-fold codec."""

    return ModelCodec(dataset, input_spec)
