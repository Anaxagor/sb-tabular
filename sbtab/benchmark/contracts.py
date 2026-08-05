"""Public data contracts for the unified tabular benchmark.

This module describes semantic data exchanged by the benchmark core. It does
not describe any model's native tensor layout, dtype, data loader, or temporary
``X``/``y`` call shape. Adapters own those integration details.

Contract objects are frozen so their fields cannot be replaced after
construction. Pandas frames remain mutable Python objects and must be treated
as read-only by benchmark components that receive them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import pandas as pd


class TaskType(str, Enum):
    """Semantics of a declared target used by utility evaluation."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ColumnKind(str, Enum):
    """Semantic kind of a modeled raw table column.

    Continuous columns have real-valued support. Discrete columns have finite
    numeric support whose rank is meaningful. Categorical columns have finite
    nominal support unless :attr:`ColumnSpec.ordered_values` declares an
    explicit ordinal domain.
    """

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


class ContinuousView(str, Enum):
    """Train-fitted representation requested for continuous columns.

    ``RAW`` preserves raw numeric values. ``STANDARD`` requests location/scale
    normalization fitted on train only. ``UNSUPPORTED`` rejects a dataset when
    its continuous modeled group is non-empty.
    """

    RAW = "raw"
    STANDARD = "standard"
    UNSUPPORTED = "unsupported"


class DiscreteView(str, Enum):
    """Train-fitted representation requested for numeric discrete columns.

    ``RAW_VALUES`` preserves numeric support. ``FINITE_STATE_CODES`` requests a
    reversible train-fitted mapping to ``0..K-1``. ``UNSUPPORTED`` rejects a
    non-empty discrete modeled group.
    """

    RAW_VALUES = "raw_values"
    FINITE_STATE_CODES = "finite_state_codes"
    UNSUPPORTED = "unsupported"


class CategoricalView(str, Enum):
    """Train-fitted representation requested for categorical columns.

    ``RAW_VALUES`` preserves category values. ``FINITE_STATE_CODES`` requests a
    reversible train-fitted mapping to ``0..K-1``. ``UNSUPPORTED`` rejects a
    non-empty categorical modeled group.
    """

    RAW_VALUES = "raw_values"
    FINITE_STATE_CODES = "finite_state_codes"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ColumnSpec:
    """Semantic declaration of one modeled raw column.

    Parameters
    ----------
    name:
        Exact column label in :attr:`TabularDataset.frame`.
    kind:
        Dataset semantics used by the shared codec. Adapters must not infer or
        replace it from pandas dtype or observed cardinality.
    ordered_values:
        Optional complete ordinal domain in semantic order. It is valid only
        for categorical columns. ``None`` means that a categorical column is
        nominal. Numeric discrete columns are always ordered by ascending
        numeric support and therefore cannot override that order here.
    """

    name: str
    kind: ColumnKind
    ordered_values: tuple[object, ...] | None = None


@dataclass(frozen=True)
class TabularDataset:
    """One raw dataset and all semantics required by the benchmark.

    Parameters
    ----------
    name:
        Stable human-readable label for artifacts and logs. Shared code must
        never branch on it.
    frame:
        Raw modeled table. It may additionally contain the one declared
        identifier, but no undeclared columns.
    columns:
        Every modeled column in canonical generated-table order, including the
        target when one exists.
    target:
        Optional modeled column used only as a label by utility evaluation and
        native APIs. Declaring it does not remove it from the table.
    task:
        Classification or regression semantics for ``target``. It is present
        exactly when ``target`` is present.
    identifier:
        Optional raw identifier excluded from all modeled columns. The runner,
        not an adapter, owns any identifier generation after sampling.
    """

    name: str
    frame: pd.DataFrame
    columns: tuple[ColumnSpec, ...]
    target: str | None = None
    task: TaskType | None = None
    identifier: str | None = None

    @property
    def column_order(self) -> tuple[str, ...]:
        """Return canonical modeled output order, including target."""

        return tuple(column.name for column in self.columns)

    def columns_of_kind(self, kind: ColumnKind) -> tuple[str, ...]:
        """Return modeled column names of ``kind`` in canonical order."""

        return tuple(column.name for column in self.columns if column.kind is kind)

    @property
    def continuous_columns(self) -> tuple[str, ...]:
        """Return semantically continuous modeled columns."""

        return self.columns_of_kind(ColumnKind.CONTINUOUS)

    @property
    def discrete_columns(self) -> tuple[str, ...]:
        """Return semantically numeric discrete modeled columns."""

        return self.columns_of_kind(ColumnKind.DISCRETE)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        """Return semantically categorical modeled columns."""

        return self.columns_of_kind(ColumnKind.CATEGORICAL)

    def column(self, name: str) -> ColumnSpec:
        """Return the declaration for ``name`` or fail with dataset context."""

        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"Dataset {self.name!r} has no modeled column {name!r}.")


@dataclass(frozen=True)
class InputSpec:
    """Only the semantic prepared views requested by one model family.

    Native layout, dtype, target extraction, device, missing handling, and
    model hyperparameters deliberately do not belong to this contract.

    Parameters
    ----------
    continuous_view:
        Representation the shared codec must prepare for every continuous
        modeled column.
    discrete_view:
        Representation the shared codec must prepare for every numeric
        discrete modeled column.
    categorical_view:
        Representation the shared codec must prepare for every categorical
        modeled column, including a categorical target.
    """

    continuous_view: ContinuousView
    discrete_view: DiscreteView
    categorical_view: CategoricalView


@dataclass(frozen=True)
class StateColumn:
    """Prepared finite-state domain for one named column.

    Parameters
    ----------
    cardinality:
        Number of train-observed states. Valid prepared codes are integers in
        ``[0, cardinality)``.
    ordered:
        Whether neighbouring prepared codes have semantic adjacency. This is
        true for numeric discrete columns and explicitly ordinal categories,
        not for encoder-assigned nominal category codes.
    """

    cardinality: int
    ordered: bool


@dataclass(frozen=True)
class PreparedSchema:
    """Semantic schema attached to every canonical prepared table.

    ``state_columns`` is copied into an immutable mapping so codec state cannot
    be changed through a dictionary retained by the caller. Transform
    parameters and reversible codebooks remain private codec state.

    Parameters
    ----------
    column_order:
        Exact prepared table order expected from adapter input and output. It
        includes target and excludes raw identifier.
    continuous_columns, discrete_columns, categorical_columns:
        A disjoint semantic partition of ``column_order``. These tuples describe
        meaning, not native tensor blocks.
    target_col:
        Optional target label retained in the prepared frame. It does not imply
        a shared ``X``/``y`` split.
    task_type:
        Classification or regression semantics used by evaluation and present
        exactly when ``target_col`` is present.
    state_columns:
        Named metadata only for columns represented by finite-state codes.
        Cardinalities and order are never inferred from array position.
    """

    column_order: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    discrete_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    target_col: str | None
    task_type: TaskType | None
    state_columns: Mapping[str, StateColumn] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Snapshot named state metadata without performing full validation."""

        object.__setattr__(
            self,
            "state_columns",
            MappingProxyType(dict(self.state_columns)),
        )


@dataclass(frozen=True)
class PreparedTable:
    """Canonical adapter input and output.

    Parameters
    ----------
    frame:
        Prepared modeled columns in exactly ``schema.column_order``. It includes
        target, excludes identifier, and is treated as read-only by recipients.
    schema:
        Named semantic metadata used by validators and adapters. An adapter
        returns the same schema object with every sample.
    """

    frame: pd.DataFrame
    schema: PreparedSchema
