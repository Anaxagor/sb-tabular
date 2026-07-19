"""Boundary tests for canonical/native MSBM translation without Torch."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

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
from sbtab.benchmark.adapters import (
    MSBMAdapter,
    MSBMCompatibilityError,
    MSBMDependencyError,
)
from sbtab.benchmark.adapters import msbm as msbm_module


class _FakeTensor:
    def __init__(self, array: np.ndarray, dtype: object, device: object) -> None:
        self.array = array
        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array.copy()


class _FakeTorch:
    float32 = "torch.float32"
    int64 = "torch.int64"
    bool = "torch.bool"

    def device(self, value: str) -> str:
        return value

    def as_tensor(
        self,
        data: np.ndarray,
        *,
        dtype: object,
        device: object,
    ) -> _FakeTensor:
        numpy_dtype = {
            self.float32: np.float32,
            self.int64: np.int64,
            self.bool: np.bool_,
        }[dtype]
        return _FakeTensor(
            np.asarray(data, dtype=numpy_dtype),
            dtype=dtype,
            device=device,
        )


@dataclass
class _FakeConfig:
    device: str
    seed: int
    fb_sequence: tuple[str, ...] = ("b", "f", "b", "f", "b")
    cat_emb_dim: int = 16
    hidden_dim: int = 512
    time_dim: int = 128
    n_layers: int = 5
    num_steps: int = 100
    batch_size: int = 2
    epochs_per_direction: int = 5


class _FakeSolver:
    instances: list[_FakeSolver] = []

    def __init__(
        self,
        continuous_dim: int,
        cardinalities: list[int],
        is_ordered: _FakeTensor,
        cfg: _FakeConfig,
    ) -> None:
        self.continuous_dim = continuous_dim
        self.cardinalities = cardinalities
        self.is_ordered = is_ordered
        self.cfg = cfg
        self.train_num: _FakeTensor | None = None
        self.train_cat: _FakeTensor | None = None
        self.sample_calls: list[tuple[int, int]] = []
        self.numeric_shape_override: tuple[int, int] | None = None
        self.numeric_dtype_override: object | None = None
        self.numeric_array_override: np.ndarray | None = None
        self.state_dtype_override: object | None = None
        self.state_array_override: np.ndarray | None = None
        self.__class__.instances.append(self)

    def fit(
        self,
        train_num: _FakeTensor,
        train_cat: _FakeTensor,
    ) -> _FakeSolver:
        self.train_num = train_num
        self.train_cat = train_cat
        return self

    def sample(
        self,
        n_samples: int,
        seed: int,
    ) -> tuple[_FakeTensor, _FakeTensor]:
        self.sample_calls.append((n_samples, seed))
        numeric_shape = self.numeric_shape_override or (
            n_samples,
            self.continuous_dim,
        )
        numeric = np.arange(
            int(np.prod(numeric_shape)),
            dtype=np.float32,
        ).reshape(numeric_shape)
        if self.numeric_array_override is not None:
            numeric = self.numeric_array_override
        state = np.column_stack(
            [
                np.arange(n_samples, dtype=np.int64) % cardinality
                for cardinality in self.cardinalities
            ]
        )
        if self.state_array_override is not None:
            state = self.state_array_override
        return (
            _FakeTensor(
                numeric,
                self.numeric_dtype_override or _FakeTorch.float32,
                self.cfg.device,
            ),
            _FakeTensor(
                state,
                self.state_dtype_override or _FakeTorch.int64,
                self.cfg.device,
            ),
        )


def _bindings(
    **config_overrides: object,
) -> msbm_module._NativeBindings:
    torch = _FakeTorch()

    def config_factory(*, device: str, seed: int) -> _FakeConfig:
        config = _FakeConfig(device=device, seed=seed)
        for name, value in config_overrides.items():
            setattr(config, name, value)
        return config

    return msbm_module._NativeBindings(
        torch=torch,
        config_factory=config_factory,
        solver_factory=_FakeSolver,
    )


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
    """Verify exact block mapping, lifecycle, and compatibility failures."""

    def setUp(self) -> None:
        _FakeSolver.instances.clear()

    def _fit(
        self,
        table: PreparedTable | None = None,
        **config_overrides: object,
    ) -> tuple[MSBMAdapter, _FakeSolver]:
        adapter = MSBMAdapter()
        with patch.object(
            msbm_module,
            "_load_native_msbm",
            return_value=_bindings(**config_overrides),
        ):
            adapter.fit(table or _mixed_table(), _context())
        return adapter, _FakeSolver.instances[-1]

    def test_declares_only_the_approved_semantic_views(self) -> None:
        spec = MSBMAdapter().input_spec

        self.assertEqual(spec.continuous_view, ContinuousView.STANDARD)
        self.assertEqual(spec.discrete_view, DiscreteView.FINITE_STATE_CODES)
        self.assertEqual(
            spec.categorical_view,
            CategoricalView.FINITE_STATE_CODES,
        )

    def test_fit_maps_canonical_blocks_metadata_context_and_target(self) -> None:
        adapter, solver = self._fit()

        self.assertEqual(adapter.name, "msbm")
        self.assertEqual(solver.continuous_dim, 2)
        self.assertEqual(solver.cardinalities, [3, 2])
        self.assertEqual(solver.is_ordered.array.tolist(), [True, False])
        self.assertEqual(solver.is_ordered.dtype, _FakeTorch.bool)
        self.assertEqual(solver.cfg.device, "cpu")
        self.assertEqual(solver.cfg.seed, 42)
        self.assertIsNotNone(solver.train_num)
        self.assertIsNotNone(solver.train_cat)
        if solver.train_num is None or solver.train_cat is None:
            self.fail("Fake native solver did not receive both train tensors.")
        np.testing.assert_array_equal(
            solver.train_num.array,
            np.array(
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            solver.train_cat.array,
            np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64),
        )
        self.assertEqual(solver.train_num.dtype, _FakeTorch.float32)
        self.assertEqual(solver.train_cat.dtype, _FakeTorch.int64)
        self.assertEqual(solver.train_num.device, "cpu")
        self.assertEqual(solver.train_cat.device, "cpu")

    def test_sample_reassembles_canonical_order_and_same_schema(self) -> None:
        table = _mixed_table()
        adapter, solver = self._fit(table)

        sample = adapter.sample(n=2, seed=11)

        self.assertEqual(tuple(sample.frame.columns), table.schema.column_order)
        self.assertIs(sample.schema, table.schema)
        self.assertEqual(sample.frame["amount"].tolist(), [0.0, 2.0])
        self.assertEqual(sample.frame["duration"].tolist(), [1.0, 3.0])
        self.assertEqual(sample.frame["count"].tolist(), [0, 1])
        self.assertEqual(sample.frame["label"].tolist(), [0, 1])
        self.assertEqual(solver.sample_calls, [(2, 11)])

    def test_zero_row_sample_bypasses_native_solver(self) -> None:
        table = _mixed_table()
        adapter, solver = self._fit(table)

        sample = adapter.sample(n=0, seed=11)

        self.assertEqual(len(sample.frame), 0)
        self.assertEqual(tuple(sample.frame.columns), table.schema.column_order)
        self.assertIs(sample.schema, table.schema)
        self.assertEqual(solver.sample_calls, [])

    def test_invalid_native_shape_and_dtype_fail_before_assembly(self) -> None:
        adapter, solver = self._fit()
        solver.numeric_shape_override = (2, 1)

        with self.assertRaisesRegex(ContractViolation, "shape"):
            adapter.sample(n=2, seed=11)

        solver.numeric_shape_override = (1, 2)
        with self.assertRaisesRegex(ContractViolation, "shape"):
            adapter.sample(n=2, seed=11)

        solver.numeric_shape_override = None
        solver.numeric_dtype_override = "torch.float64"
        with self.assertRaisesRegex(ContractViolation, "dtype"):
            adapter.sample(n=2, seed=11)

        solver.numeric_dtype_override = None
        solver.state_dtype_override = "torch.int32"
        with self.assertRaisesRegex(ContractViolation, "dtype"):
            adapter.sample(n=2, seed=11)

    def test_invalid_native_state_is_rejected_without_repair(self) -> None:
        adapter, solver = self._fit()
        solver.state_array_override = np.array([[3, 0]], dtype=np.int64)

        with self.assertRaisesRegex(ContractViolation, "invalid codes"):
            adapter.sample(n=1, seed=11)

    def test_malformed_prepared_train_fails_before_native_loader(self) -> None:
        table = _mixed_table()
        frame = table.frame.copy()
        frame.loc[0, "count"] = 3

        with patch.object(msbm_module, "_load_native_msbm") as native_loader:
            with self.assertRaisesRegex(ContractViolation, "invalid codes"):
                MSBMAdapter().fit(
                    PreparedTable(frame=frame, schema=table.schema),
                    _context(),
                )
        native_loader.assert_not_called()

    def test_non_finite_native_continuous_output_is_rejected(self) -> None:
        adapter, solver = self._fit()
        solver.numeric_array_override = np.array([[np.inf, 0.0]], dtype=np.float32)

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
        adapter = MSBMAdapter()

        with self.assertRaisesRegex(MSBMCompatibilityError, "continuous"):
            adapter.fit(state_only, _context())

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

    def test_drop_last_zero_step_fit_is_rejected_not_reconfigured(self) -> None:
        adapter = MSBMAdapter()
        with patch.object(
            msbm_module,
            "_load_native_msbm",
            return_value=_bindings(batch_size=4),
        ):
            with self.assertRaisesRegex(MSBMCompatibilityError, "zero optimizer"):
                adapter.fit(_mixed_table(), _context())
        self.assertEqual(_FakeSolver.instances, [])

    def test_float32_overflow_is_rejected_before_loading_native_runtime(self) -> None:
        table = _mixed_table()
        frame = table.frame.copy()
        frame.loc[0, "amount"] = np.finfo(np.float64).max

        with patch.object(msbm_module, "_load_native_msbm") as native_loader:
            with self.assertRaisesRegex(
                MSBMCompatibilityError,
                "cannot be represented",
            ):
                MSBMAdapter().fit(
                    PreparedTable(frame=frame, schema=table.schema),
                    _context(),
                )
        native_loader.assert_not_called()

    def test_train_state_metadata_must_describe_observed_dense_codes(self) -> None:
        table = _mixed_table()
        sparse_train = PreparedTable(
            frame=table.frame.iloc[:2].copy(),
            schema=table.schema,
        )

        with patch.object(msbm_module, "_load_native_msbm") as native_loader:
            with self.assertRaisesRegex(
                MSBMCompatibilityError,
                "train-observed cardinality",
            ):
                MSBMAdapter().fit(sparse_train, _context())
        native_loader.assert_not_called()

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

        with patch.object(msbm_module, "_load_native_msbm") as native_loader:
            with self.assertRaisesRegex(MSBMCompatibilityError, "torch.int64"):
                MSBMAdapter().fit(table, _context())
        native_loader.assert_not_called()

    def test_invalid_native_config_fails_before_solver_construction(self) -> None:
        adapter = MSBMAdapter()
        with patch.object(
            msbm_module,
            "_load_native_msbm",
            return_value=_bindings(time_dim=3),
        ):
            with self.assertRaisesRegex(MSBMCompatibilityError, "must be even"):
                adapter.fit(_mixed_table(), _context())
        self.assertEqual(_FakeSolver.instances, [])

        with patch.object(
            msbm_module,
            "_load_native_msbm",
            return_value=_bindings(device="cuda:0"),
        ):
            with self.assertRaisesRegex(MSBMCompatibilityError, "RunContext"):
                MSBMAdapter().fit(_mixed_table(), _context())
        self.assertEqual(_FakeSolver.instances, [])

        with patch.object(
            msbm_module,
            "_load_native_msbm",
            return_value=_bindings(fb_sequence=("f",)),
        ):
            with self.assertRaisesRegex(MSBMCompatibilityError, "backward"):
                MSBMAdapter().fit(_mixed_table(), _context())
        self.assertEqual(_FakeSolver.instances, [])

    def test_missing_native_dependency_has_an_explicit_error(self) -> None:
        adapter = MSBMAdapter()
        with patch.object(
            msbm_module,
            "_load_native_msbm",
            side_effect=MSBMDependencyError("torch unavailable"),
        ):
            with self.assertRaisesRegex(MSBMDependencyError, "torch unavailable"):
                adapter.fit(_mixed_table(), _context())

    def test_lifecycle_rejects_sample_before_fit_and_second_fit(self) -> None:
        adapter, _ = self._fit()

        with self.assertRaisesRegex(ContractViolation, "only once"):
            adapter.fit(_mixed_table(), _context())
        with self.assertRaisesRegex(ContractViolation, "before sample"):
            MSBMAdapter().sample(n=1, seed=11)


if __name__ == "__main__":
    unittest.main()
