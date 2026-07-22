"""Tests for the common model-adapter runtime boundary."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sbtab.benchmark import (
    CategoricalView,
    ContinuousView,
    ContractViolation,
    DiscreteView,
    InputSpec,
    ModelAdapter,
    PreparedSchema,
    PreparedTable,
    RunContext,
    validate_adapter_definition,
    validate_sample_request,
)


class _StubAdapter:
    name = "stub"
    input_spec = InputSpec(
        continuous_view=ContinuousView.RAW,
        discrete_view=DiscreteView.RAW_VALUES,
        categorical_view=CategoricalView.RAW_VALUES,
    )

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        self.schema = train.schema

    def sample(self, n: int, seed: int) -> PreparedTable:
        return PreparedTable(
            frame=pd.DataFrame({"value": [0.0] * n}),
            schema=self.schema,
        )


class AdapterContractTests(unittest.TestCase):
    """Verify typed context and structural adapter metadata."""

    def test_valid_structural_adapter_satisfies_runtime_protocol(self) -> None:
        adapter = _StubAdapter()

        self.assertIsInstance(adapter, ModelAdapter)
        validate_adapter_definition(adapter)

    def test_adapter_definition_rejects_invalid_metadata(self) -> None:
        adapter = _StubAdapter()
        adapter.name = " "

        with self.assertRaisesRegex(ContractViolation, "non-empty"):
            validate_adapter_definition(adapter)

    def test_runtime_definition_rejects_non_callable_protocol_members(self) -> None:
        adapter = _StubAdapter()
        adapter.fit = 1  # type: ignore[method-assign]
        adapter.sample = 2  # type: ignore[method-assign]

        self.assertIsInstance(adapter, ModelAdapter)
        with self.assertRaisesRegex(ContractViolation, "callable methods"):
            validate_adapter_definition(adapter)

    def test_run_context_accepts_controls_without_touching_artifact_path(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            artifact_dir = Path(temporary_dir) / "not-created" / "fold-0"

            context = RunContext(
                run_id="pilot",
                fold_id=0,
                seed=42,
                device="cpu",
                artifact_dir=artifact_dir,
            )

            self.assertEqual(context.artifact_dir, artifact_dir)
            self.assertFalse(artifact_dir.exists())

    def test_run_context_rejects_boolean_fold_and_out_of_range_seed(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "fold_id"):
            RunContext("pilot", True, 42, "cpu", Path("artifacts"))
        with self.assertRaisesRegex(ContractViolation, "range"):
            RunContext("pilot", 0, 2**32, "cpu", Path("artifacts"))

    def test_sample_request_requires_positive_explicit_values(self) -> None:
        validate_sample_request(n=1, seed=0)
        with self.assertRaisesRegex(ContractViolation, "n must be an integer"):
            validate_sample_request(n=True, seed=0)
        with self.assertRaisesRegex(ContractViolation, "positive"):
            validate_sample_request(n=0, seed=0)
        with self.assertRaisesRegex(ContractViolation, "positive"):
            validate_sample_request(n=-1, seed=0)
        with self.assertRaisesRegex(ContractViolation, "seed must be an integer"):
            validate_sample_request(n=1, seed=True)

    def test_stub_preserves_schema_identity_after_fit(self) -> None:
        schema = PreparedSchema(
            column_order=("value",),
            continuous_columns=("value",),
            discrete_columns=(),
            categorical_columns=(),
            target_col=None,
            task_type=None,
        )
        train = PreparedTable(
            frame=pd.DataFrame({"value": [1.0]}),
            schema=schema,
        )
        adapter = _StubAdapter()
        adapter.fit(
            train,
            RunContext("pilot", 0, 42, "cpu", Path("artifacts")),
        )

        sample = adapter.sample(n=2, seed=7)

        self.assertIs(sample.schema, schema)


if __name__ == "__main__":
    unittest.main()
