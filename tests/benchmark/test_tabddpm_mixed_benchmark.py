"""Orchestration tests for the fourteen-dataset TabDDPM entrypoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import optuna
import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark import (
    ColumnKind,
    ColumnSpec,
    ContractViolation,
    TabularDataset,
    TaskType,
)
from sbtab.benchmark.pilots.tabddpm_mixed_benchmark import (
    TabDDPMMixedBenchmarkConfig,
    _run_one_dataset,
    _pickle_dataset_loader,
    run_tabddpm_mixed_benchmark,
)
from sbtab.benchmark.datasets import (
    ONLINE_SHOPPERS_COLUMNS,
    make_mixed_dataset,
)
from sbtab.benchmark.splitting import (
    HoldoutConfig,
    KFoldConfig,
    StratifiedHoldoutConfig,
    StratifiedKFoldConfig,
)


def _dataset(key: str, task: TaskType) -> TabularDataset:
    target_kind = (
        ColumnKind.CATEGORICAL
        if task is TaskType.CLASSIFICATION
        else ColumnKind.CONTINUOUS
    )
    return TabularDataset(
        name=key,
        frame=pd.DataFrame({"feature": [0.0, 1.0], "target": [0, 1]}),
        columns=(
            ColumnSpec("feature", ColumnKind.CONTINUOUS),
            ColumnSpec("target", target_kind),
        ),
        target="target",
        task=task,
    )


def _fake_dataset_runner(
    dataset: TabularDataset,
    config: TabDDPMMixedBenchmarkConfig,
) -> Path:
    """Write the smallest valid completed-dataset artifact boundary."""

    root = config.output_dir / dataset.name
    root.mkdir()
    artifact_digests: dict[str, tuple[str, str]] = {}
    for name in (
        "tuning_manifest",
        "generation_manifest",
        "evaluation_manifest",
    ):
        path = root / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        artifact_digests[name] = (
            path.name,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    summary = {
        "dataset_key": dataset.name,
        "published_name": dataset.name.replace("_", " ").title(),
        "task": dataset.task.value,
        "rows_after_complete_case": len(dataset.frame),
        "columns": len(dataset.columns),
        "tuning": {
            "best_rerank_score": 0.25,
            "completed_phase_a_trials": config.target_complete_trials,
        },
        "continuous": None,
        "discrete": None,
        "categorical": None,
        "final_mean_exact_state_js": None,
        "utility": None,
        "runtime_seconds": {
            "fit": {"mean": 1.0, "std": 0.0},
            "sample": {"mean": 0.5, "std": 0.0},
        },
    }
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest_path = root / "dataset-manifest.json"
    manifest_payload: dict[str, object] = {
        "artifact_type": "tabddpm_mixed_dataset_result",
        "artifact_version": 1,
        "status": "complete",
        "dataset_key": dataset.name,
        "summary": "summary.json",
        "summary_sha256": digest,
    }
    for name, (filename, artifact_digest) in artifact_digests.items():
        manifest_payload[name] = filename
        manifest_payload[f"{name}_sha256"] = artifact_digest
    manifest_path.write_text(
        json.dumps(
            manifest_payload,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _online_shoppers_frame() -> pd.DataFrame:
    rows = 20
    values: dict[str, list[object]] = {}
    for column in ONLINE_SHOPPERS_COLUMNS:
        if column.kind is ColumnKind.CONTINUOUS:
            values[column.name] = [float(index) for index in range(rows)]
        elif column.kind is ColumnKind.DISCRETE:
            values[column.name] = [index % 3 for index in range(rows)]
        elif column.name in {"Weekend", "Revenue"}:
            values[column.name] = [bool(index % 2) for index in range(rows)]
        else:
            values[column.name] = [
                "a" if index % 2 == 0 else "b" for index in range(rows)
            ]
    return pd.DataFrame(values)


def _tiny_config(trial: optuna.Trial) -> TabDDPMConfig:
    width = trial.suggest_categorical("width", [4])
    return TabDDPMConfig(
        steps=1,
        num_timesteps=2,
        batch_size=4,
        d_layers=[width],
        weight_decay=0.0,
        use_ema_for_sampling=False,
        show_progress=False,
    )


class TabDDPMMixedBenchmarkTests(unittest.TestCase):
    """Check split policy, collection artifacts, and strict resume behavior."""

    def test_checked_in_snapshot_contains_every_declared_dataset(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        loader = _pickle_dataset_loader(
            repository_root / "sbtab/data/datasets/datasets_mixed.pkl"
        )
        expected_rows = {
            "adult": 48_842,
            "credit_approval": 690,
            "online_shoppers": 12_330,
            "eucalyptus": 736,
            "forest_fires": 517,
            "insurance": 1_338,
            "house_sales": 21_613,
            "cardiovascular_disease": 70_000,
            "churn_modelling": 10_000,
            "auto_mpg": 398,
            "diamonds": 53_940,
            "real_estate": 414,
            "stroke_prediction": 5_110,
            "palmer_penguins": 344,
        }

        observed_rows = {
            key: len(loader(key).frame) for key in expected_rows
        }

        self.assertEqual(observed_rows, expected_rows)

    def test_real_solver_completes_one_dataset_collection(self) -> None:
        dataset = make_mixed_dataset(
            "online_shoppers",
            _online_shoppers_frame(),
        )
        with TemporaryDirectory() as temporary_dir:
            config = TabDDPMMixedBenchmarkConfig(
                output_dir=Path(temporary_dir) / "run",
                dataset_keys=("online_shoppers",),
                target_complete_trials=1,
                max_total_trials=1,
                rerank_candidates=1,
                rerank_seed_pairs=1,
                show_native_progress=False,
                dataset_source="test-fixture",
            )

            def run_dataset(
                value: TabularDataset,
                run_config: TabDDPMMixedBenchmarkConfig,
            ) -> Path:
                return _run_one_dataset(
                    value,
                    run_config,
                    suggest_config=_tiny_config,
                )

            with patch(
                "sbtab.benchmark.adapters.tabddpm_tuning.RERANK_STEPS",
                1,
            ):
                result = run_tabddpm_mixed_benchmark(
                    config,
                    dataset_loader=lambda key: dataset,
                    dataset_runner=run_dataset,
                )

            summary = json.loads(
                result.summary_json_path.read_text(encoding="utf-8")
            )[0]
            self.assertEqual(summary["dataset_key"], "online_shoppers")
            self.assertEqual(summary["utility"]["metric"], "macro_f1")
            self.assertIsNotNone(summary["continuous"]["mmd_rbf"])
            self.assertTrue(
                (
                    config.output_dir
                    / "online_shoppers/generation/manifest.json"
                ).is_file()
            )

    def test_task_controls_the_shared_splitter_not_the_adapter(self) -> None:
        from sbtab.benchmark.pilots.tabddpm_mixed_benchmark import (
            _final_split,
            _holdout_split,
        )

        classification = _dataset("adult", TaskType.CLASSIFICATION)
        regression = _dataset("forest_fires", TaskType.REGRESSION)

        self.assertIsInstance(_holdout_split(classification), StratifiedHoldoutConfig)
        self.assertIsInstance(_final_split(classification), StratifiedKFoldConfig)
        self.assertIsInstance(_holdout_split(regression), HoldoutConfig)
        self.assertIsInstance(_final_split(regression), KFoldConfig)

    def test_collection_writes_summaries_and_skips_completed_resume(self) -> None:
        keys = ("adult", "forest_fires")
        calls: list[str] = []

        def load(key: str) -> TabularDataset:
            calls.append(key)
            task = (
                TaskType.CLASSIFICATION
                if key == "adult"
                else TaskType.REGRESSION
            )
            return _dataset(key, task)

        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            config = TabDDPMMixedBenchmarkConfig(
                output_dir=output_dir,
                dataset_keys=keys,
                target_complete_trials=1,
                max_total_trials=1,
                rerank_candidates=1,
                rerank_seed_pairs=1,
                show_native_progress=False,
            )
            result = run_tabddpm_mixed_benchmark(
                config,
                dataset_loader=load,
                dataset_runner=_fake_dataset_runner,
            )

            self.assertEqual(calls, list(keys))
            self.assertTrue(result.manifest_path.is_file())
            self.assertEqual(len(pd.read_csv(result.summary_csv_path)), 2)
            progress = json.loads(
                (output_dir / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["completed"], list(keys))
            markdown = result.summary_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Adult", markdown)
            self.assertIn("Forest Fires", markdown)

            calls.clear()
            resumed = TabDDPMMixedBenchmarkConfig(
                output_dir=output_dir,
                dataset_keys=keys,
                target_complete_trials=1,
                max_total_trials=1,
                rerank_candidates=1,
                rerank_seed_pairs=1,
                resume=True,
                show_native_progress=False,
            )
            resumed_result = run_tabddpm_mixed_benchmark(
                resumed,
                dataset_loader=load,
                dataset_runner=_fake_dataset_runner,
            )

            self.assertEqual(calls, [])
            self.assertEqual(resumed_result.manifest_path, result.manifest_path)

    def test_resume_rejects_changed_protocol_controls(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            initial = TabDDPMMixedBenchmarkConfig(
                output_dir=output_dir,
                dataset_keys=("adult",),
                target_complete_trials=1,
                max_total_trials=1,
                rerank_candidates=1,
                rerank_seed_pairs=1,
                show_native_progress=False,
            )
            run_tabddpm_mixed_benchmark(
                initial,
                dataset_loader=lambda key: _dataset(
                    key, TaskType.CLASSIFICATION
                ),
                dataset_runner=_fake_dataset_runner,
            )
            changed = TabDDPMMixedBenchmarkConfig(
                output_dir=output_dir,
                dataset_keys=("adult",),
                target_complete_trials=2,
                max_total_trials=2,
                rerank_candidates=1,
                rerank_seed_pairs=1,
                resume=True,
                show_native_progress=False,
            )

            with self.assertRaisesRegex(ContractViolation, "Refusing to resume"):
                run_tabddpm_mixed_benchmark(changed)

    def test_config_rejects_unknown_and_duplicate_datasets(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "Unknown"):
            TabDDPMMixedBenchmarkConfig(
                output_dir=Path("run"),
                dataset_keys=("not-a-dataset",),
            )
        with self.assertRaisesRegex(ContractViolation, "duplicates"):
            TabDDPMMixedBenchmarkConfig(
                output_dir=Path("run"),
                dataset_keys=("adult", "adult"),
            )


if __name__ == "__main__":
    unittest.main()
