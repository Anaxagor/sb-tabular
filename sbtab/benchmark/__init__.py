"""Unified, model-independent benchmark data contracts.

The package is intentionally independent of the legacy ``sbtab.data``,
``sbtab.transforms``, and ``sbtab.experiments`` orchestration paths.
"""

from __future__ import annotations

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
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_input_spec,
    validate_prepared_table,
    validate_tabular_dataset,
)

__all__ = [
    "CategoricalView",
    "ColumnKind",
    "ColumnSpec",
    "ContinuousView",
    "ContractViolation",
    "DiscreteView",
    "InputSpec",
    "PreparedSchema",
    "PreparedTable",
    "StateColumn",
    "TabularDataset",
    "TaskType",
    "validate_input_spec",
    "validate_prepared_table",
    "validate_tabular_dataset",
]
