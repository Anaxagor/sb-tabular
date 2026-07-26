"""Unified, model-independent benchmark contracts and orchestration.

The package is intentionally independent of the legacy ``sbtab.data``,
``sbtab.transforms``, and ``sbtab.experiments`` orchestration paths.
"""

from __future__ import annotations

from sbtab.benchmark.adapter import (
    ModelAdapter,
    RunContext,
    validate_adapter_definition,
    validate_sample_request,
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
    HoldoutConfig,
    HoldoutSplit,
    KFoldConfig,
    SplitConfig,
    StratifiedHoldoutConfig,
    StratifiedKFoldConfig,
    make_holdout,
    make_splits,
)
from sbtab.benchmark.runner import (
    BenchmarkConfig,
    CrossValidationResult,
    FoldResult,
    HoldoutResult,
    HoldoutRunConfig,
    run_cross_validation,
    run_holdout_trial,
)
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_input_spec,
    validate_prepared_table,
    validate_tabular_dataset,
)

__all__ = [
    "CategoricalView",
    "BenchmarkConfig",
    "ColumnKind",
    "ColumnSpec",
    "ContinuousView",
    "ContractViolation",
    "CrossValidationResult",
    "ClassCount",
    "DiscreteView",
    "FoldSplit",
    "FoldResult",
    "HoldoutConfig",
    "HoldoutResult",
    "HoldoutRunConfig",
    "HoldoutSplit",
    "InputSpec",
    "KFoldConfig",
    "MissingPolicy",
    "MissingPolicyResult",
    "MissingReport",
    "MissingValuesError",
    "ModelAdapter",
    "ModelCodec",
    "PreparedSchema",
    "PreparedTable",
    "RunContext",
    "SplitConfig",
    "StateColumn",
    "StratifiedHoldoutConfig",
    "StratifiedKFoldConfig",
    "TabularDataset",
    "TaskType",
    "apply_missing_policy",
    "compile_codec",
    "make_holdout",
    "make_splits",
    "run_cross_validation",
    "run_holdout_trial",
    "validate_adapter_definition",
    "validate_input_spec",
    "validate_prepared_table",
    "validate_sample_request",
    "validate_tabular_dataset",
]
