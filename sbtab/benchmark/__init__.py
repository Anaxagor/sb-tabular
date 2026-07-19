"""Unified, model-independent benchmark contracts and orchestration.

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
from sbtab.benchmark.codec import ModelCodec, compile_codec
from sbtab.benchmark.missing import (
    ClassCount,
    MissingPolicy,
    MissingPolicyResult,
    MissingReport,
    MissingValuesError,
    apply_missing_policy,
)
from sbtab.benchmark.splitting import (
    FoldSplit,
    KFoldConfig,
    SplitConfig,
    StratifiedKFoldConfig,
    make_splits,
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
    "ClassCount",
    "DiscreteView",
    "FoldSplit",
    "InputSpec",
    "KFoldConfig",
    "MissingPolicy",
    "MissingPolicyResult",
    "MissingReport",
    "MissingValuesError",
    "ModelCodec",
    "PreparedSchema",
    "PreparedTable",
    "SplitConfig",
    "StateColumn",
    "StratifiedKFoldConfig",
    "TabularDataset",
    "TaskType",
    "apply_missing_policy",
    "compile_codec",
    "make_splits",
    "validate_input_spec",
    "validate_prepared_table",
    "validate_tabular_dataset",
]
