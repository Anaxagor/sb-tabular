"""Tests for create-only cross-validation artifact handoff."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sbtab.benchmark import (
    BenchmarkConfig,
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    ContractViolation,
    DiscreteView,
    InputSpec,
    KFoldConfig,
    MissingPolicy,
    PreparedTable,
    RunContext,
    TabularDataset,
    run_cross_validation,
    write_cross_validation_artifacts,
)


class _EchoAdapter:
    name = "artifact-echo"
    input_spec = InputSpec(
        continuous_view=ContinuousView.RAW,
        discrete_view=DiscreteView.FINITE_STATE_CODES,
        categorical_view=CategoricalView.FINITE_STATE_CODES,
    )

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        self.train = train

    def sample(self, n: int, seed: int) -> PreparedTable:
        return PreparedTable(
            frame=self.train.frame.iloc[:n].copy(),
            schema=self.train.schema,
        )


def _result():
    dataset = TabularDataset(
        name="artifact-dataset",
        frame=pd.DataFrame(
            {
                "row_id": [f"id-{index}" for index in range(8)],
                "value": [float(index) for index in range(8)],
                "group": ["a", "b"] * 4,
            }
        ),
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("group", ColumnKind.CATEGORICAL),
        ),
        identifier="row_id",
    )
    return run_cross_validation(
        dataset,
        _EchoAdapter,
        BenchmarkConfig(
            split=KFoldConfig(n_splits=2, seed=42),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="artifact-test",
            artifact_dir=Path("logical-artifact-root"),
        ),
    )


class CrossValidationArtifactTests(unittest.TestCase):
    """Verify manifest completeness, stored tables, and no-overwrite policy."""

    def test_writer_stores_real_once_and_one_synthetic_table_per_fold(
        self,
    ) -> None:
        result = _result()
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"

            manifest_path = write_cross_validation_artifacts(
                result,
                output_dir,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["artifact_type"],
                "cross_validation_generation",
            )
            self.assertEqual(manifest["artifact_version"], 1)
            self.assertEqual(manifest["adapter_name"], "artifact-echo")
            self.assertEqual(manifest["dataset"]["rows"], 8)
            self.assertEqual(manifest["dataset"]["identifier"], "row_id")
            self.assertEqual(manifest["config"]["split"]["type"], "KFoldConfig")
            self.assertEqual(manifest["config"]["training_seed"], 42)
            self.assertEqual(manifest["missing_report"]["rows_after"], 8)

            stored_real = pd.read_csv(output_dir / manifest["real_path"])
            self.assertEqual(tuple(stored_real.columns), tuple(result.dataset.frame))
            self.assertEqual(len(stored_real), 8)
            self.assertEqual(len(manifest["folds"]), 2)
            for fold_entry, fold in zip(manifest["folds"], result.folds):
                stored_synthetic = pd.read_csv(
                    output_dir / fold_entry["synthetic_path"]
                )
                self.assertEqual(len(stored_synthetic), len(fold.synthetic_raw))
                self.assertEqual(
                    tuple(stored_synthetic.columns),
                    result.dataset.column_order,
                )
                self.assertEqual(
                    fold_entry["train_positions"],
                    list(fold.split.train_positions),
                )
                self.assertEqual(
                    fold_entry["test_positions"],
                    list(fold.split.test_positions),
                )

    def test_writer_refuses_to_overwrite_existing_directory(self) -> None:
        result = _result()
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "existing"
            output_dir.mkdir()

            with self.assertRaisesRegex(ContractViolation, "already exists"):
                write_cross_validation_artifacts(result, output_dir)


if __name__ == "__main__":
    unittest.main()
