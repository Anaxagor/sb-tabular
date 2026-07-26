"""Tests for the minimal fixed-configuration cross-validation runner."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from sbtab.benchmark import (
    BenchmarkConfig,
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    ContractViolation,
    DiscreteView,
    HoldoutRunConfig,
    InputSpec,
    KFoldConfig,
    MissingPolicy,
    MissingValuesError,
    PreparedTable,
    RunContext,
    StratifiedHoldoutConfig,
    StratifiedKFoldConfig,
    TabularDataset,
    TaskType,
    run_cross_validation,
    run_holdout_trial,
)


def _dataset(*, with_missing: bool) -> TabularDataset:
    amount = [float(index) for index in range(18)]
    if with_missing:
        amount[0] = float("nan")
    return TabularDataset(
        name="runner-mixed",
        frame=pd.DataFrame(
            {
                "row_id": [
                    None if index == 5 else f"id-{index}" for index in range(18)
                ],
                "amount": amount,
                "count": [index % 3 for index in range(18)],
                "token": [f"row-{index}" for index in range(18)],
                "label": ["no", "yes"] * 9,
            }
        ),
        columns=(
            ColumnSpec("amount", ColumnKind.CONTINUOUS),
            ColumnSpec("count", ColumnKind.DISCRETE),
            ColumnSpec("token", ColumnKind.CATEGORICAL),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
        identifier="row_id",
    )


class _EchoAdapter:
    """Return the prepared train table to expose runner boundary behavior."""

    input_spec = InputSpec(
        continuous_view=ContinuousView.STANDARD,
        discrete_view=DiscreteView.FINITE_STATE_CODES,
        categorical_view=CategoricalView.FINITE_STATE_CODES,
    )

    def __init__(
        self,
        *,
        short_sample: bool = False,
        invalid_state: bool = False,
        name: str = "echo",
    ) -> None:
        self.name = name
        self.short_sample = short_sample
        self.invalid_state = invalid_state
        self.train: PreparedTable | None = None
        self.context: RunContext | None = None
        self.sample_request: tuple[int, int] | None = None

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        self.train = train
        self.context = context

    def sample(self, n: int, seed: int) -> PreparedTable:
        if self.train is None:
            raise RuntimeError("adapter was not fitted")
        self.sample_request = (n, seed)
        frame = self.train.frame.iloc[:n].copy()
        if self.short_sample:
            frame = frame.iloc[:-1].copy()
        if self.invalid_state:
            state = self.train.schema.state_columns["label"]
            frame.loc[frame.index[0], "label"] = state.cardinality
        return PreparedTable(frame=frame, schema=self.train.schema)


class CrossValidationRunnerTests(unittest.TestCase):
    """Verify orchestration order without introducing evaluation behavior."""

    def test_runner_applies_policy_once_and_returns_decoded_fold_tables(
        self,
    ) -> None:
        created: list[_EchoAdapter] = []

        def factory() -> _EchoAdapter:
            adapter = _EchoAdapter()
            created.append(adapter)
            return adapter

        config = BenchmarkConfig(
            split=StratifiedKFoldConfig(n_splits=3, seed=42),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="runner-test",
            training_seed=42,
            sample_seed=10_042,
            device="cpu",
            artifact_dir=Path("unused-runner-artifacts"),
        )

        result = run_cross_validation(_dataset(with_missing=True), factory, config)

        self.assertEqual(result.adapter_name, "echo")
        self.assertIs(result.config, config)
        self.assertEqual(result.missing_report.rows_before, 18)
        self.assertEqual(result.missing_report.rows_after, 17)
        self.assertEqual(result.missing_report.dropped_count, 1)
        self.assertEqual(result.missing_report.missing_by_column["amount"], 1)
        self.assertEqual(len(result.dataset.frame), 17)
        self.assertTrue(result.dataset.frame["row_id"].isna().any())
        self.assertEqual(len(result.folds), 3)
        self.assertEqual(len(created), 3)
        self.assertEqual(len({id(adapter) for adapter in created}), 3)
        self.assertEqual(
            len({id(adapter.train.schema) for adapter in created if adapter.train}),
            3,
        )

        all_test_positions: list[int] = []
        for fold, adapter in zip(result.folds, created):
            all_test_positions.extend(fold.split.test_positions)
            self.assertNotIn("row_id", fold.train_raw.columns)
            self.assertNotIn("row_id", fold.test_raw.columns)
            self.assertNotIn("row_id", fold.synthetic_raw.columns)
            self.assertEqual(
                tuple(fold.train_raw.columns),
                result.dataset.column_order,
            )
            self.assertEqual(len(fold.synthetic_raw), len(fold.train_raw))
            pd.testing.assert_frame_equal(
                fold.synthetic_raw,
                fold.train_raw,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertGreaterEqual(fold.fit_seconds, 0.0)
            self.assertGreaterEqual(fold.sample_seconds, 0.0)

            self.assertIsNotNone(adapter.train)
            self.assertIsNotNone(adapter.context)
            assert adapter.train is not None
            assert adapter.context is not None
            self.assertAlmostEqual(adapter.train.frame["amount"].mean(), 0.0)
            self.assertEqual(adapter.context.fold_id, fold.split.fold_id)
            self.assertEqual(
                adapter.context.seed,
                config.training_seed + fold.split.fold_id,
            )
            self.assertEqual(
                adapter.context.artifact_dir,
                config.artifact_dir / f"fold-{fold.split.fold_id}",
            )
            self.assertEqual(
                adapter.sample_request,
                (
                    len(fold.train_raw),
                    config.sample_seed + fold.split.fold_id,
                ),
            )

        self.assertEqual(sorted(all_test_positions), list(range(17)))
        self.assertFalse(config.artifact_dir.exists())

    def test_default_missing_policy_stops_before_adapter_construction(self) -> None:
        factory_calls = 0

        def factory() -> _EchoAdapter:
            nonlocal factory_calls
            factory_calls += 1
            return _EchoAdapter()

        with self.assertRaises(MissingValuesError):
            run_cross_validation(
                _dataset(with_missing=True),
                factory,
                BenchmarkConfig(split=KFoldConfig(n_splits=2, seed=42)),
            )

        self.assertEqual(factory_calls, 0)

    def test_holdout_trial_samples_validation_size_without_using_validation(
        self,
    ) -> None:
        created: list[_EchoAdapter] = []

        def factory() -> _EchoAdapter:
            adapter = _EchoAdapter()
            created.append(adapter)
            return adapter

        config = HoldoutRunConfig(
            split=StratifiedHoldoutConfig(),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="holdout-test",
            training_seed=5,
            sample_seed=105,
            artifact_dir=Path("unused-holdout-artifacts"),
        )

        result = run_holdout_trial(
            _dataset(with_missing=True),
            factory,
            config,
        )

        self.assertEqual(result.adapter_name, "echo")
        self.assertEqual(result.missing_report.rows_before, 18)
        self.assertEqual(result.missing_report.rows_after, 17)
        self.assertEqual(len(result.train_raw), 13)
        self.assertEqual(len(result.validation_raw), 4)
        self.assertEqual(len(result.synthetic_raw), 4)
        self.assertEqual(
            sorted(
                result.split.train_positions
                + result.split.validation_positions
            ),
            list(range(17)),
        )
        self.assertEqual(created[0].sample_request, (4, 105))
        self.assertIsNotNone(created[0].train)
        assert created[0].train is not None
        self.assertAlmostEqual(created[0].train.frame["amount"].mean(), 0.0)
        self.assertNotIn("row_id", result.validation_raw.columns)
        self.assertEqual(
            tuple(result.synthetic_raw.columns),
            result.dataset.column_order,
        )

    def test_runner_rejects_returned_row_count_before_decoding(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "row count"):
            run_cross_validation(
                _dataset(with_missing=False),
                lambda: _EchoAdapter(short_sample=True),
                BenchmarkConfig(split=KFoldConfig(n_splits=2, seed=42)),
            )

    def test_runner_rejects_invalid_generated_state_before_decoding(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "invalid codes"):
            run_cross_validation(
                _dataset(with_missing=False),
                lambda: _EchoAdapter(invalid_state=True),
                BenchmarkConfig(split=KFoldConfig(n_splits=2, seed=42)),
            )

    def test_runner_requires_factory_to_return_a_fresh_adapter(self) -> None:
        adapter = _EchoAdapter()

        with self.assertRaisesRegex(ContractViolation, "fresh adapter"):
            run_cross_validation(
                _dataset(with_missing=False),
                lambda: adapter,
                BenchmarkConfig(split=KFoldConfig(n_splits=2, seed=42)),
            )

    def test_config_rejects_boolean_base_seed(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "training_seed"):
            BenchmarkConfig(
                split=KFoldConfig(),
                training_seed=True,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
