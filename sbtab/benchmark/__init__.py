"""Unified, model-independent benchmark data boundary.

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
from sbtab.benchmark.missing import (
    ClassCount,
    MissingPolicy,
    MissingPolicyResult,
    MissingReport,
    MissingValuesError,
    apply_missing_policy,
)
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_input_spec,
    validate_prepared_table,
    validate_tabular_dataset,
)

__all__ = [
    "CategoricalView",
    "ClassCount",
    "ColumnKind",
    "ColumnSpec",
    "ContinuousView",
    "ContractViolation",
    "DiscreteView",
    "InputSpec",
    "MissingPolicy",
    "MissingPolicyResult",
    "MissingReport",
    "MissingValuesError",
    "PreparedSchema",
    "PreparedTable",
    "StateColumn",
    "TabularDataset",
    "TaskType",
    "apply_missing_policy",
    "validate_input_spec",
    "validate_prepared_table",
    "validate_tabular_dataset",
]
