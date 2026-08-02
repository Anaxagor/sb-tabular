"""Tests for model-owned MSBM Optuna orchestration."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna
import pandas as pd

from sbtab.benchmark import (
    BenchmarkConfig,
    ColumnKind,
    ColumnSpec,
    ContractViolation,
    HoldoutRunConfig,
    MissingPolicy,
    StratifiedHoldoutConfig,
    StratifiedKFoldConfig,
    TabularDataset,
    TaskType,
    run_cross_validation,
)
from sbtab.benchmark.adapters.msbm import MSBMAdapter
from sbtab.benchmark.adapters.msbm_tuning import (
    MSBMTuningConfig,
    msbm_config_from_payload,
    msbm_config_payload,
    suggest_msbm_config,
    tune_msbm,
    write_msbm_tuning_artifacts,
)
from sbtab.solvers.msbm import (
    CategoricalLossNormalization,
    MixedSBMConfig,
)


def _dataset() -> TabularDataset:
    return TabularDataset(
        name="msbm-tuning-smoke",
        frame=pd.DataFrame(
            {
                "value": [float(index) for index in range(16)],
                "label": ["no", "yes"] * 8,
            }
        ),
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
    )


def _lightweight_config(trial: optuna.Trial) -> MixedSBMConfig:
    hidden_dim = trial.suggest_categorical("hidden_dim", [4, 8])
    return MixedSBMConfig(
        fb_sequence=("b",),
        cat_emb_dim=2,
        hidden_dim=hidden_dim,
        time_dim=4,
        n_layers=1,
        dropout=0.0,
        num_steps=2,
        batch_size=2,
        epochs_per_direction=1,
        device="cpu",
        seed=0,
    )


class MSBMTuningTests(unittest.TestCase):
    """Verify search translation and complete tuning-to-final data lifecycle."""

    def test_default_search_builds_current_native_config(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "imf_len": 3,
                "cat_emb_dim": 8,
                "hidden_dim": 128,
                "time_dim": 32,
                "n_layers": 2,
                "dropout": 0.1,
                "num_steps": 20,
                "sigma": 0.1,
                "lambda_num": 0.8,
                "lambda_cat": 0.2,
                "lr": 1e-4,
                "batch_size": 128,
                "epochs_per_direction": 5,
                "grad_clip": 1.0,
            }
        )

        config = suggest_msbm_config(trial)

        self.assertIsInstance(config, MixedSBMConfig)
        self.assertEqual(config.fb_sequence, ("b", "f", "b"))
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.lambda_num, 0.8)
        self.assertEqual(config.lambda_cat, 0.2)
        self.assertEqual(config.alpha, 0.01)
        self.assertIs(
            config.categorical_loss_normalization,
            CategoricalLossNormalization.BY_NUM_COLUMNS,
        )
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.seed, 0)

    def test_config_payload_round_trips_new_and_previous_artifacts(self) -> None:
        config = MixedSBMConfig(
            fb_sequence=("b",),
            alpha=0.798,
            categorical_loss_normalization=(
                CategoricalLossNormalization.NONE
            ),
        )

        restored = msbm_config_from_payload(msbm_config_payload(config))
        previous_payload = msbm_config_payload(config)
        previous_payload.pop("alpha")
        previous_payload.pop("categorical_loss_normalization")
        restored_previous = msbm_config_from_payload(previous_payload)

        self.assertEqual(restored, config)
        self.assertEqual(restored_previous.alpha, 0.01)
        self.assertIs(
            restored_previous.categorical_loss_normalization,
            CategoricalLossNormalization.BY_NUM_COLUMNS,
        )

    def test_lightweight_study_selects_config_used_by_final_folds(self) -> None:
        dataset = _dataset()
        tuning = tune_msbm(
            dataset,
            MSBMTuningConfig(
                run=HoldoutRunConfig(
                    split=StratifiedHoldoutConfig(
                        validation_fraction=0.25,
                        seed=5,
                    ),
                    missing_policy=MissingPolicy.COMPLETE_CASE,
                    run_id="msbm-tuning-test",
                    training_seed=7,
                    sample_seed=107,
                    artifact_dir=Path("unused-msbm-tuning-artifacts"),
                ),
                n_trials=2,
                sampler_seed=5,
            ),
            suggest_config=_lightweight_config,
        )

        self.assertEqual(len(tuning.study.trials), 2)
        self.assertIs(tuning.config.run.missing_policy, MissingPolicy.COMPLETE_CASE)
        self.assertIs(tuning.dataset, dataset)
        self.assertTrue(math.isfinite(tuning.best_score))
        self.assertIn(tuning.best_config.hidden_dim, (4, 8))
        self.assertEqual(tuning.study.user_attrs["objective_version"], 2)
        for trial in tuning.study.trials:
            self.assertEqual(trial.state, optuna.trial.TrialState.COMPLETE)
            self.assertIn("native_config", trial.user_attrs)
            self.assertIn(
                "mean_standardized_wasserstein",
                trial.user_attrs,
            )
            self.assertIn("mean_jensen_shannon", trial.user_attrs)
            self.assertIn("column_scores", trial.user_attrs)

        final = run_cross_validation(
            dataset,
            lambda: MSBMAdapter(tuning.best_config),
            BenchmarkConfig(
                split=StratifiedKFoldConfig(n_splits=2, seed=42),
                missing_policy=MissingPolicy.COMPLETE_CASE,
                run_id="msbm-final-test",
                artifact_dir=Path("unused-msbm-final-artifacts"),
            ),
        )
        self.assertEqual(final.adapter_name, "msbm")
        self.assertEqual(len(final.folds), 2)
        for fold in final.folds:
            self.assertEqual(len(fold.synthetic_raw), len(fold.train_raw))
            self.assertEqual(
                tuple(fold.synthetic_raw.columns),
                dataset.column_order,
            )

        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "tuning"
            manifest_path = write_msbm_tuning_artifacts(
                tuning,
                output_dir,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            best_config = json.loads(
                (output_dir / "best-config.json").read_text(encoding="utf-8")
            )
            trials = json.loads(
                (output_dir / "trials.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["artifact_type"], "msbm_tuning")
            self.assertEqual(manifest["artifact_version"], 3)
            self.assertEqual(manifest["dataset"]["name"], dataset.name)
            self.assertEqual(manifest["missing_report"]["rows_before"], 16)
            self.assertEqual(manifest["missing_report"]["rows_after"], 16)
            self.assertEqual(manifest["study"]["requested_trials"], 2)
            self.assertEqual(manifest["study"]["completed_trials"], 2)
            self.assertEqual(manifest["study"]["objective_version"], 2)
            self.assertFalse(manifest["study"]["storage_configured"])
            self.assertNotIn("storage", manifest["study"])
            self.assertEqual(best_config["hidden_dim"], tuning.best_config.hidden_dim)
            self.assertEqual(len(trials), 2)
            self.assertIn("column_scores", trials[0]["user_attrs"])

            with self.assertRaisesRegex(ContractViolation, "already exists"):
                write_msbm_tuning_artifacts(tuning, output_dir)

    def test_refuses_to_resume_trials_from_an_unversioned_objective(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            storage = f"sqlite:///{temporary_dir}/legacy-study.sqlite3"
            study = optuna.create_study(
                study_name="legacy-objective",
                storage=storage,
            )
            study.add_trial(
                optuna.trial.create_trial(
                    value=1.0,
                    params={},
                    distributions={},
                )
            )

            with self.assertRaisesRegex(
                ContractViolation,
                "obsolete raw-scale Wasserstein",
            ):
                tune_msbm(
                    _dataset(),
                    MSBMTuningConfig(
                        run=HoldoutRunConfig(
                            split=StratifiedHoldoutConfig(
                                validation_fraction=0.25,
                                seed=5,
                            )
                        ),
                        n_trials=1,
                        study_name="legacy-objective",
                        storage=storage,
                        load_if_exists=True,
                    ),
                    suggest_config=_lightweight_config,
                )


if __name__ == "__main__":
    unittest.main()
