"""Tests for model-owned MSBM Optuna orchestration."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import optuna
import pandas as pd

from sbtab.benchmark import (
    BenchmarkConfig,
    ColumnKind,
    ColumnSpec,
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
    suggest_msbm_config,
    tune_msbm,
)
from sbtab.solvers.msbm import MixedSBMConfig


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
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.seed, 0)

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
        self.assertTrue(math.isfinite(tuning.best_score))
        self.assertIn(tuning.best_config.hidden_dim, (4, 8))
        for trial in tuning.study.trials:
            self.assertEqual(trial.state, optuna.trial.TrialState.COMPLETE)
            self.assertIn("native_config", trial.user_attrs)
            self.assertIn("mean_wasserstein", trial.user_attrs)
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


if __name__ == "__main__":
    unittest.main()
