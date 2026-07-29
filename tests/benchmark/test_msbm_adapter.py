"""Tests for the thin canonical/MSBM data translation."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from sbtab.benchmark.adapters import MSBMAdapter
from sbtab.benchmark.adapters import msbm as msbm_module
from sbtab.solvers.msbm import (
    CategoricalLossNormalization,
    MixedSBMConfig,
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


def _fit_with_mocked_training(
    table: PreparedTable | None = None,
) -> tuple[MSBMAdapter, MagicMock, MagicMock]:
    adapter = MSBMAdapter()
    with patch.object(
        msbm_module,
        "MixedSBMSolver",
        autospec=True,
    ) as solver_class:
        solver = solver_class.return_value
        adapter.fit(table or _mixed_table(), _context())
    return adapter, solver, solver_class


class MSBMAdapterTests(unittest.TestCase):
    """Verify only the adapter-owned mapping and lifecycle boundary."""

    def test_declares_the_approved_semantic_views(self) -> None:
        spec = MSBMAdapter().input_spec

        self.assertEqual(spec.continuous_view, ContinuousView.STANDARD)
        self.assertEqual(spec.discrete_view, DiscreteView.FINITE_STATE_CODES)
        self.assertEqual(
            spec.categorical_view,
            CategoricalView.FINITE_STATE_CODES,
        )

    def test_fit_maps_named_blocks_and_metadata_to_native_msbm(self) -> None:
        adapter, solver, solver_class = _fit_with_mocked_training()
        constructor_args = solver_class.call_args.kwargs
        config = constructor_args["cfg"]
        train_num, train_cat = solver.fit.call_args.args

        self.assertEqual(adapter.name, "msbm")
        self.assertEqual(constructor_args["continuous_dim"], 2)
        self.assertEqual(constructor_args["cardinalities"], [3, 2])
        self.assertEqual(constructor_args["is_ordered"].tolist(), [True, False])
        self.assertIsInstance(config, MixedSBMConfig)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.seed, 42)
        torch.testing.assert_close(
            train_num,
            torch.tensor(
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            train_cat,
            torch.tensor(
                [[0, 1], [2, 0], [1, 1]],
                dtype=torch.int64,
            ),
        )

    def test_fixed_native_config_is_preserved_except_for_fold_context(self) -> None:
        fixed_config = MixedSBMConfig(
            fb_sequence=("b",),
            hidden_dim=37,
            num_steps=9,
            alpha=0.798,
            categorical_loss_normalization=(
                CategoricalLossNormalization.NONE
            ),
            batch_size=11,
            device="cuda:7",
            seed=999,
        )
        adapter = MSBMAdapter(fixed_config)
        with patch.object(
            msbm_module,
            "MixedSBMSolver",
            autospec=True,
        ) as solver_class:
            adapter.fit(_mixed_table(), _context())

        received = solver_class.call_args.kwargs["cfg"]
        self.assertIsInstance(received, MixedSBMConfig)
        self.assertIsNot(received, fixed_config)
        self.assertEqual(received.fb_sequence, ("b",))
        self.assertEqual(received.hidden_dim, 37)
        self.assertEqual(received.num_steps, 9)
        self.assertEqual(received.alpha, 0.798)
        self.assertIs(
            received.categorical_loss_normalization,
            CategoricalLossNormalization.NONE,
        )
        self.assertEqual(received.batch_size, 11)
        self.assertEqual(received.device, "cpu")
        self.assertEqual(received.seed, 42)

    def test_sample_restores_canonical_order_and_schema(self) -> None:
        table = _mixed_table()
        adapter, solver, _ = _fit_with_mocked_training(table)
        solver.sample.return_value = (
            torch.tensor([[0.5, 2.5], [1.5, 3.5]], dtype=torch.float32),
            torch.tensor([[2, 1], [0, 0]], dtype=torch.int64),
        )

        sample = adapter.sample(n=2, seed=11)

        self.assertEqual(tuple(sample.frame.columns), table.schema.column_order)
        self.assertIs(sample.schema, table.schema)
        self.assertEqual(sample.frame["amount"].tolist(), [0.5, 1.5])
        self.assertEqual(sample.frame["count"].tolist(), [2, 0])
        self.assertEqual(sample.frame["label"].tolist(), [1, 0])
        self.assertEqual(sample.frame["duration"].tolist(), [2.5, 3.5])
        solver.sample.assert_called_once_with(n_samples=2, seed=11)

    def test_sample_before_fit_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "before sample"):
            MSBMAdapter().sample(n=1, seed=11)


if __name__ == "__main__":
    unittest.main()
