"""Tests for the thin canonical/TabDDPM data translation."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from sbtab.baselines.tabddpm.native import TabDDPMConfig, TabDDPMSolver
from sbtab.benchmark import (
    CategoricalView,
    ContinuousView,
    ContractViolation,
    ColumnKind,
    ColumnSpec,
    DiscreteView,
    HoldoutConfig,
    HoldoutRunConfig,
    PreparedSchema,
    PreparedTable,
    RunContext,
    StateColumn,
    TaskType,
    TabularDataset,
    compile_codec,
    run_holdout_trial,
)
from sbtab.benchmark.adapters import TabDDPMAdapter
from sbtab.benchmark.adapters import tabddpm as tabddpm_module


def _mixed_table() -> PreparedTable:
    """Return interleaved semantic blocks with a categorical target."""

    schema = PreparedSchema(
        column_order=("amount", "count", "label", "duration"),
        continuous_columns=("amount", "duration"),
        discrete_columns=("count",),
        categorical_columns=("label",),
        target_col="label",
        task_type=TaskType.CLASSIFICATION,
        state_columns={
            "label": StateColumn(cardinality=2, ordered=False),
        },
    )
    return PreparedTable(
        frame=pd.DataFrame(
            {
                "amount": [1.0, 2.0, 3.0],
                "count": pd.Series([10, 40, 20], dtype="int64"),
                "label": pd.Series([1, 0, 1], dtype="int64"),
                "duration": [10.0, 20.0, 30.0],
            }
        ),
        schema=schema,
    )


def _context() -> RunContext:
    """Return deterministic CPU controls for adapter boundary tests."""

    return RunContext(
        run_id="tabddpm-test",
        fold_id=0,
        seed=42,
        device="cpu",
        artifact_dir=Path("unused-test-artifacts"),
    )


def _lightweight_native_config() -> TabDDPMConfig:
    """Return a one-update profile for real boundary characterization."""

    return TabDDPMConfig(
        steps=1,
        num_timesteps=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        d_layers=[4],
        dropout=0.0,
        scheduler="cosine",
        ema_decay=0.9,
        use_ema_for_sampling=False,
        device="cpu",
        seed=42,
    )


def _fit_with_mocked_training(
    table: PreparedTable | None = None,
) -> tuple[TabDDPMAdapter, MagicMock, MagicMock]:
    """Fit through a mocked native seam and return captured objects."""

    adapter = TabDDPMAdapter()
    with patch.object(
        tabddpm_module,
        "TabDDPMSolver",
        autospec=True,
    ) as solver_class:
        solver = solver_class.return_value
        adapter.fit(table or _mixed_table(), _context())
    return adapter, solver, solver_class


class TabDDPMAdapterTests(unittest.TestCase):
    """Verify only the adapter-owned canonical/native mapping."""

    def test_declares_the_approved_semantic_views(self) -> None:
        spec = TabDDPMAdapter().input_spec

        self.assertEqual(
            spec.continuous_view,
            ContinuousView.STANDARD,
        )
        self.assertEqual(spec.discrete_view, DiscreteView.RAW_VALUES)
        self.assertEqual(
            spec.categorical_view,
            CategoricalView.FINITE_STATE_CODES,
        )

    def test_fit_maps_named_blocks_dtypes_and_cardinalities(self) -> None:
        adapter, solver, solver_class = _fit_with_mocked_training()
        constructor_args = solver_class.call_args.kwargs
        config = constructor_args["cfg"]
        train_num, train_state = solver.fit.call_args.args

        self.assertEqual(adapter.name, "tabddpm")
        self.assertEqual(constructor_args["num_numerical_features"], 3)
        self.assertEqual(constructor_args["cardinalities"], [2])
        self.assertIsInstance(config, TabDDPMConfig)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.seed, 42)
        torch.testing.assert_close(
            train_num,
            torch.tensor(
                [[1.0, 10.0, 10.0], [2.0, 40.0, 20.0], [3.0, 20.0, 30.0]],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            train_state,
            torch.tensor(
                [[1], [0], [1]],
                dtype=torch.int64,
            ),
        )

    def test_fixed_config_and_typed_ema_choice_reach_native_solver(self) -> None:
        fixed_config = TabDDPMConfig(
            steps=7,
            num_timesteps=90,
            batch_size=11,
            lr=0.012,
            weight_decay=0.034,
            d_layers=[13],
            dropout=0.25,
            gaussian_loss_type="kl",
            scheduler="linear",
            ema_decay=0.8,
            use_ema_for_sampling=False,
            device="cuda:7",
            seed=999,
        )
        adapter = TabDDPMAdapter(fixed_config)
        with patch.object(
            tabddpm_module,
            "TabDDPMSolver",
            autospec=True,
        ) as solver_class:
            adapter.fit(_mixed_table(), _context())

        received = solver_class.call_args.kwargs["cfg"]
        self.assertIsInstance(received, TabDDPMConfig)
        self.assertIsNot(received, fixed_config)
        self.assertEqual(received.steps, 7)
        self.assertEqual(received.num_timesteps, 90)
        self.assertEqual(received.batch_size, 11)
        self.assertEqual(received.lr, 0.012)
        self.assertEqual(received.weight_decay, 0.034)
        self.assertEqual(received.d_layers, [13])
        self.assertEqual(received.dropout, 0.25)
        self.assertEqual(received.gaussian_loss_type, "kl")
        self.assertEqual(received.scheduler, "linear")
        self.assertEqual(received.ema_decay, 0.8)
        self.assertFalse(received.use_ema_for_sampling)
        self.assertEqual(received.device, "cpu")
        self.assertEqual(received.seed, 42)
        self.assertEqual(fixed_config.device, "cuda:7")
        self.assertEqual(fixed_config.seed, 999)

    def test_fit_preserves_empty_native_blocks(self) -> None:
        cases = (
            PreparedTable(
                frame=pd.DataFrame({"value": [1.0, 2.0]}),
                schema=PreparedSchema(
                    column_order=("value",),
                    continuous_columns=("value",),
                    discrete_columns=(),
                    categorical_columns=(),
                    target_col=None,
                    task_type=None,
                ),
            ),
            PreparedTable(
                frame=pd.DataFrame(
                    {"state": pd.Series([0, 1], dtype="int64")}
                ),
                schema=PreparedSchema(
                    column_order=("state",),
                    continuous_columns=(),
                    discrete_columns=("state",),
                    categorical_columns=(),
                    target_col=None,
                    task_type=None,
                ),
            ),
            PreparedTable(
                frame=pd.DataFrame({"label": pd.Series([0, 1], dtype="int64")}),
                schema=PreparedSchema(
                    column_order=("label",),
                    continuous_columns=(),
                    discrete_columns=(),
                    categorical_columns=("label",),
                    target_col="label",
                    task_type=TaskType.CLASSIFICATION,
                    state_columns={
                        "label": StateColumn(cardinality=2, ordered=False),
                    },
                ),
            ),
        )

        for table in cases:
            with self.subTest(column_order=table.schema.column_order):
                _, solver, solver_class = _fit_with_mocked_training(table)
                train_num, train_state = solver.fit.call_args.args
                constructor = solver_class.call_args.kwargs
                self.assertEqual(
                    tuple(train_num.shape),
                    (
                        len(table.frame),
                        len(table.schema.continuous_columns)
                        + len(table.schema.discrete_columns),
                    ),
                )
                self.assertEqual(
                    tuple(train_state.shape),
                    (len(table.frame), len(table.schema.categorical_columns)),
                )
                self.assertEqual(
                    constructor["cardinalities"],
                    [
                        table.schema.state_columns[name].cardinality
                        for name in table.schema.categorical_columns
                    ],
                )

    def test_sample_restores_canonical_order_target_and_schema(self) -> None:
        table = _mixed_table()
        adapter, solver, _ = _fit_with_mocked_training(table)
        solver.sample.return_value = (
            torch.tensor([[0.5, 40.0, 2.5], [1.5, 10.0, 3.5]]),
            torch.tensor([[1], [0]], dtype=torch.int64),
        )

        sample = adapter.sample(n=2, seed=11)

        self.assertEqual(tuple(sample.frame.columns), table.schema.column_order)
        self.assertIs(sample.schema, table.schema)
        self.assertEqual(sample.frame["amount"].tolist(), [0.5, 1.5])
        self.assertEqual(sample.frame["count"].tolist(), [40, 10])
        self.assertEqual(sample.frame["label"].tolist(), [1, 0])
        self.assertEqual(sample.frame["duration"].tolist(), [2.5, 3.5])
        self.assertEqual(sample.frame["label"].dtype, np.dtype(np.int64))
        solver.sample.assert_called_once_with(n_samples=2, seed=11)

    def test_real_adapter_matches_direct_native_boundary(self) -> None:
        table = _mixed_table()
        config = _lightweight_native_config()
        context = _context()
        continuous_names = ("amount", "count", "duration")
        state_names = ("label",)
        direct_solver = TabDDPMSolver(
            num_numerical_features=len(continuous_names),
            cardinalities=[
                table.schema.state_columns[name].cardinality
                for name in state_names
            ],
            cfg=config,
        )
        direct_solver.fit(
            torch.tensor(
                table.frame.loc[:, continuous_names].to_numpy(),
                dtype=torch.float32,
            ),
            torch.tensor(
                table.frame.loc[:, state_names].to_numpy(),
                dtype=torch.int64,
            ),
        )
        adapter = TabDDPMAdapter(config)
        adapter.fit(table, context)

        direct_num, direct_state = direct_solver.sample(
            n_samples=4,
            seed=91,
            use_ema=False,
        )
        first = adapter.sample(n=4, seed=91)
        repeated = adapter.sample(n=4, seed=91)
        expected = pd.concat(
            (
                pd.DataFrame(
                    direct_num.numpy(),
                    columns=continuous_names,
                ),
                pd.DataFrame(
                    direct_state.numpy(),
                    columns=state_names,
                ),
            ),
            axis=1,
        ).loc[:, table.schema.column_order]
        expected["count"] = np.rint(expected["count"])

        pd.testing.assert_frame_equal(first.frame, expected)
        pd.testing.assert_frame_equal(repeated.frame, expected)
        self.assertIs(first.schema, table.schema)

    def test_discrete_sampling_rounds_ties_without_clipping_or_code_mapping(self) -> None:
        adapter, solver, _ = _fit_with_mocked_training()
        discrete = [-2.5, -1.5, 0.5, 1.5, 3.7, 99.6]
        solver.sample.return_value = (
            torch.tensor([[0.25, value, 0.75] for value in discrete]),
            torch.zeros((6, 1), dtype=torch.int64),
        )

        sample = adapter.sample(n=6, seed=1)

        self.assertEqual(sample.frame["count"].tolist(), [-2, -2, 0, 2, 4, 100])
        self.assertEqual(sample.frame["amount"].tolist(), [0.25] * 6)
        self.assertEqual(sample.frame["duration"].tolist(), [0.75] * 6)
        self.assertEqual(sample.frame["label"].tolist(), [0] * 6)

    def test_fractional_discrete_train_is_rejected_before_native_fit(self) -> None:
        table = _mixed_table()
        table.frame["count"] = [0.0, 0.5, 1.5]
        with patch.object(tabddpm_module, "TabDDPMSolver") as solver_class:
            with self.assertRaisesRegex(ContractViolation, "count.*fractional"):
                TabDDPMAdapter().fit(table, _context())
            solver_class.assert_not_called()

    def test_real_holdout_decodes_integer_gaussian_samples(self) -> None:
        dataset = TabularDataset(
            name="gaussian-discrete-holdout",
            frame=pd.DataFrame({
                "amount": np.arange(20, dtype=float),
                "count": [10, 40] * 10,
                "label": ["a", "b"] * 10,
            }),
            columns=(
                ColumnSpec("amount", ColumnKind.CONTINUOUS),
                ColumnSpec("count", ColumnKind.DISCRETE),
                ColumnSpec("label", ColumnKind.CATEGORICAL),
            ),
        )
        result = run_holdout_trial(
            dataset,
            lambda: TabDDPMAdapter(_lightweight_native_config()),
            HoldoutRunConfig(split=HoldoutConfig(), device="cpu"),
        )
        self.assertEqual(tuple(result.synthetic_raw.columns), dataset.column_order)
        self.assertEqual(len(result.synthetic_raw), len(result.validation_raw))
        values = result.synthetic_raw["count"].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values, np.rint(values))
        self.assertTrue(set(result.synthetic_raw["label"]) <= {"a", "b"})

    def test_discrete_target_uses_gaussian_block_and_rounding(self) -> None:
        source = _mixed_table()
        table = PreparedTable(
            frame=source.frame,
            schema=replace(
                source.schema, target_col="count", task_type=TaskType.REGRESSION,
            ),
        )
        adapter, solver, solver_class = _fit_with_mocked_training(table)
        self.assertEqual(solver_class.call_args.kwargs["cardinalities"], [2])
        torch.testing.assert_close(
            solver.fit.call_args.args[0][:, 1],
            torch.tensor([10.0, 40.0, 20.0]),
        )
        solver.sample.return_value = (
            torch.tensor([[0.25, 3.7, 0.75]]),
            torch.tensor([[1]]),
        )
        sample = adapter.sample(1, seed=1)
        self.assertEqual(sample.frame["count"].tolist(), [4.0])
        self.assertEqual(sample.schema.target_col, "count")

    def test_nonfinite_discrete_sample_is_rejected_by_shared_codec(self) -> None:
        dataset = TabularDataset(
            name="integer-data",
            frame=pd.DataFrame({"count": [0, 1]}),
            columns=(ColumnSpec("count", ColumnKind.DISCRETE),),
        )
        adapter = TabDDPMAdapter()
        codec = compile_codec(dataset, adapter.input_spec)
        table = codec.fit_transform(dataset.frame)
        with patch.object(tabddpm_module, "TabDDPMSolver") as solver_class:
            solver_class.return_value.sample.return_value = (
                torch.tensor([[float("inf")]]),
                torch.empty((1, 0), dtype=torch.int64),
            )
            adapter.fit(table, _context())
            sample = adapter.sample(1, seed=1)
        with self.assertRaises(ContractViolation):
            codec.inverse_transform(sample)

    def test_sample_before_fit_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "before sample"):
            TabDDPMAdapter().sample(n=1, seed=11)


if __name__ == "__main__":
    unittest.main()
