"""Boundary tests for canonical/MSBM translation with real Torch tensors."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from sbtab.benchmark import (
    CategoricalView,
    ContinuousView,
    ContractViolation,
    DiscreteView,
    PreparedSchema,
    PreparedTable,
    RunContext,
    StateColumn,
    TaskType,
)
from sbtab.benchmark.adapters import MSBMAdapter, MSBMCompatibilityError
from sbtab.benchmark.adapters import msbm as msbm_module
from sbtab.solvers.msbm import MixedSBMConfig


def _sample(n_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    del seed
    numeric = torch.arange(n_samples * 2, dtype=torch.float32).reshape(
        n_samples,
        2,
    )
    state = torch.column_stack(
        (
            torch.arange(n_samples, dtype=torch.int64) % 3,
            torch.arange(n_samples, dtype=torch.int64) % 2,
        )
    )
    return numeric, state


def _mixed_table() -> PreparedTable:
    schema = PreparedSchema(
        column_order=("amount", "count", "label", "duration"),
        continuous_columns=("amount", "duration"),
        discrete_columns=("count",),
        categorical_columns=("label",),
        target_col="label",
        task_type=TaskType.CLASSIFICATION,
        state_columns={
            "count": StateColumn(cardinality=3, ordered=True),
            "label": StateColumn(cardinality=2, ordered=False),
        },
    )
    return PreparedTable(
        frame=pd.DataFrame(
            {
                "amount": [1.0, 2.0, 3.0],
                "count": pd.Series([0, 2, 1], dtype="int64"),
                "label": pd.Series([1, 0, 1], dtype="int64"),
                "duration": [10.0, 20.0, 30.0],
            }
        ),
        schema=schema,
    )


def _context() -> RunContext:
    return RunContext(
        run_id="msbm-test",
        fold_id=0,
        seed=42,
        device="cpu",
        artifact_dir=Path("unused-test-artifacts"),
    )


class MSBMAdapterTests(unittest.TestCase):
    """Verify data mapping, lifecycle, and compatibility failures."""

    def _fit(
        self,
        table: PreparedTable | None = None,
    ) -> tuple[MSBMAdapter, MagicMock, MagicMock]:
        adapter = MSBMAdapter()
        with patch.object(
            msbm_module,
            "MixedSBMSolver",
            autospec=True,
        ) as solver_class:
            solver = solver_class.return_value
            solver.sample.side_effect = _sample
            adapter.fit(table or _mixed_table(), _context())
        return adapter, solver, solver_class

    def test_declares_only_the_approved_semantic_views(self) -> None:
        spec = MSBMAdapter().input_spec

        self.assertEqual(spec.continuous_view, ContinuousView.STANDARD)
        self.assertEqual(spec.discrete_view, DiscreteView.FINITE_STATE_CODES)
        self.assertEqual(
            spec.categorical_view,
            CategoricalView.FINITE_STATE_CODES,
        )

    def test_fit_maps_canonical_blocks_metadata_context_and_target(self) -> None:
        adapter, solver, solver_class = self._fit()
        constructor_args = solver_class.call_args.kwargs
        is_ordered = constructor_args["is_ordered"]
        config = constructor_args["cfg"]
        train_num, train_cat = solver.fit.call_args.args

        self.assertEqual(adapter.name, "msbm")
        self.assertEqual(constructor_args["continuous_dim"], 2)
        self.assertEqual(constructor_args["cardinalities"], [3, 2])
        self.assertEqual(is_ordered.tolist(), [True, False])
        self.assertEqual(is_ordered.dtype, torch.bool)
        self.assertIsInstance(config, MixedSBMConfig)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.seed, 42)
        np.testing.assert_array_equal(
            train_num.detach().cpu().numpy(),
            np.array(
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            train_cat.detach().cpu().numpy(),
            np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64),
        )
        self.assertEqual(train_num.dtype, torch.float32)
        self.assertEqual(train_cat.dtype, torch.int64)
        self.assertEqual(train_num.device.type, "cpu")
        self.assertEqual(train_cat.device.type, "cpu")
        self.assertTrue(train_num.numpy().flags.writeable)
        self.assertTrue(train_cat.numpy().flags.writeable)

    def test_sample_reassembles_canonical_order_and_same_schema(self) -> None:
        table = _mixed_table()
        adapter, solver, _ = self._fit(table)

        sample = adapter.sample(n=2, seed=11)

        self.assertEqual(tuple(sample.frame.columns), table.schema.column_order)
        self.assertIs(sample.schema, table.schema)
        self.assertEqual(sample.frame["amount"].tolist(), [0.0, 2.0])
        self.assertEqual(sample.frame["duration"].tolist(), [1.0, 3.0])
        self.assertEqual(sample.frame["count"].tolist(), [0, 1])
        self.assertEqual(sample.frame["label"].tolist(), [0, 1])
        solver.sample.assert_called_once_with(n_samples=2, seed=11)

    def test_zero_row_sample_bypasses_native_solver(self) -> None:
        table = _mixed_table()
        adapter, solver, _ = self._fit(table)

        sample = adapter.sample(n=0, seed=11)

        self.assertEqual(len(sample.frame), 0)
        self.assertEqual(tuple(sample.frame.columns), table.schema.column_order)
        self.assertIs(sample.schema, table.schema)
        solver.sample.assert_not_called()

    def test_invalid_native_shape_and_dtype_fail_before_assembly(self) -> None:
        adapter, solver, _ = self._fit()
        solver.sample.side_effect = None
        valid_state = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
        solver.sample.return_value = (
            torch.zeros((2, 1), dtype=torch.float32),
            valid_state,
        )

        with self.assertRaisesRegex(ContractViolation, "shape"):
            adapter.sample(n=2, seed=11)

        solver.sample.return_value = (
            torch.zeros((1, 2), dtype=torch.float32),
            valid_state,
        )
        with self.assertRaisesRegex(ContractViolation, "shape"):
            adapter.sample(n=2, seed=11)

        solver.sample.return_value = (
            torch.zeros((2, 2), dtype=torch.float64),
            valid_state,
        )
        with self.assertRaisesRegex(ContractViolation, "dtype"):
            adapter.sample(n=2, seed=11)

        solver.sample.return_value = (
            torch.zeros((2, 2), dtype=torch.float32),
            valid_state.to(dtype=torch.int32),
        )
        with self.assertRaisesRegex(ContractViolation, "dtype"):
            adapter.sample(n=2, seed=11)

    def test_invalid_native_state_is_rejected_without_repair(self) -> None:
        adapter, solver, _ = self._fit()
        solver.sample.side_effect = None
        solver.sample.return_value = (
            torch.zeros((1, 2), dtype=torch.float32),
            torch.tensor([[3, 0]], dtype=torch.int64),
        )

        with self.assertRaisesRegex(ContractViolation, "invalid codes"):
            adapter.sample(n=1, seed=11)

    def test_malformed_prepared_train_fails_before_solver_construction(self) -> None:
        table = _mixed_table()
        frame = table.frame.copy()
        frame.loc[0, "count"] = 3

        with patch.object(msbm_module, "MixedSBMSolver") as solver_class:
            with self.assertRaisesRegex(ContractViolation, "invalid codes"):
                MSBMAdapter().fit(
                    PreparedTable(frame=frame, schema=table.schema),
                    _context(),
                )
        solver_class.assert_not_called()

    def test_non_finite_native_continuous_output_is_rejected(self) -> None:
        adapter, solver, _ = self._fit()
        solver.sample.side_effect = None
        solver.sample.return_value = (
            torch.tensor([[torch.inf, 0.0]], dtype=torch.float32),
            torch.tensor([[0, 0]], dtype=torch.int64),
        )

        with self.assertRaisesRegex(ContractViolation, "non-finite"):
            adapter.sample(n=1, seed=11)

    def test_current_solver_requires_both_native_blocks(self) -> None:
        state_only = PreparedTable(
            frame=pd.DataFrame({"state": pd.Series([0, 1], dtype="int64")}),
            schema=PreparedSchema(
                column_order=("state",),
                continuous_columns=(),
                discrete_columns=("state",),
                categorical_columns=(),
                target_col=None,
                task_type=None,
                state_columns={
                    "state": StateColumn(cardinality=2, ordered=True),
                },
            ),
        )

        with self.assertRaisesRegex(MSBMCompatibilityError, "continuous"):
            MSBMAdapter().fit(state_only, _context())

        continuous_only = PreparedTable(
            frame=pd.DataFrame({"value": [0.0, 1.0]}),
            schema=PreparedSchema(
                column_order=("value",),
                continuous_columns=("value",),
                discrete_columns=(),
                categorical_columns=(),
                target_col=None,
                task_type=None,
            ),
        )
        with self.assertRaisesRegex(MSBMCompatibilityError, "finite-state"):
            MSBMAdapter().fit(continuous_only, _context())

    def test_singleton_nominal_state_is_rejected_with_column_evidence(self) -> None:
        table = _mixed_table()
        schema = PreparedSchema(
            column_order=table.schema.column_order,
            continuous_columns=table.schema.continuous_columns,
            discrete_columns=table.schema.discrete_columns,
            categorical_columns=table.schema.categorical_columns,
            target_col=table.schema.target_col,
            task_type=table.schema.task_type,
            state_columns={
                "count": table.schema.state_columns["count"],
                "label": StateColumn(cardinality=1, ordered=False),
            },
        )
        frame = table.frame.copy()
        frame["label"] = pd.Series([0, 0, 0], dtype="int64")

        with self.assertRaisesRegex(MSBMCompatibilityError, "label"):
            MSBMAdapter().fit(
                PreparedTable(frame=frame, schema=schema),
                _context(),
            )

    def test_float32_overflow_is_rejected_before_solver_construction(self) -> None:
        table = _mixed_table()
        frame = table.frame.copy()
        frame.loc[0, "amount"] = np.finfo(np.float64).max

        with patch.object(msbm_module, "MixedSBMSolver") as solver_class:
            with self.assertRaisesRegex(
                MSBMCompatibilityError,
                "cannot be represented",
            ):
                MSBMAdapter().fit(
                    PreparedTable(frame=frame, schema=table.schema),
                    _context(),
                )
        solver_class.assert_not_called()

    def test_train_state_metadata_must_describe_observed_dense_codes(self) -> None:
        table = _mixed_table()
        sparse_train = PreparedTable(
            frame=table.frame.iloc[:2].copy(),
            schema=table.schema,
        )

        with patch.object(msbm_module, "MixedSBMSolver") as solver_class:
            with self.assertRaisesRegex(
                MSBMCompatibilityError,
                "train-observed cardinality",
            ):
                MSBMAdapter().fit(sparse_train, _context())
        solver_class.assert_not_called()

    def test_uint64_state_overflow_is_rejected_before_native_cast(self) -> None:
        int64_overflow = 2**63
        schema = PreparedSchema(
            column_order=("value", "state"),
            continuous_columns=("value",),
            discrete_columns=("state",),
            categorical_columns=(),
            target_col=None,
            task_type=None,
            state_columns={
                "state": StateColumn(
                    cardinality=int64_overflow + 1,
                    ordered=True,
                ),
            },
        )
        table = PreparedTable(
            frame=pd.DataFrame(
                {
                    "value": [0.0, 1.0],
                    "state": pd.Series(
                        [int64_overflow, 0],
                        dtype="uint64",
                    ),
                }
            ),
            schema=schema,
        )

        with patch.object(msbm_module, "MixedSBMSolver") as solver_class:
            with self.assertRaisesRegex(MSBMCompatibilityError, "torch.int64"):
                MSBMAdapter().fit(table, _context())
        solver_class.assert_not_called()

    def test_lifecycle_rejects_sample_before_fit_and_second_fit(self) -> None:
        adapter, _, _ = self._fit()

        with self.assertRaisesRegex(ContractViolation, "only once"):
            adapter.fit(_mixed_table(), _context())
        with self.assertRaisesRegex(ContractViolation, "before sample"):
            MSBMAdapter().sample(n=1, seed=11)


if __name__ == "__main__":
    unittest.main()
