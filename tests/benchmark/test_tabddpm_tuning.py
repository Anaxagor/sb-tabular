"""Tests for resumable staged TabDDPM model-owned tuning."""

from __future__ import annotations

import json
import math
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
    HoldoutRunConfig,
    MissingPolicy,
    StratifiedHoldoutConfig,
    TabularDataset,
    TaskType,
)
from sbtab.benchmark.adapters.tabddpm_tuning import (
    PHASE_A_STEPS,
    TabDDPMTuningConfig,
    suggest_tabddpm_config,
    tabddpm_config_from_payload,
    tabddpm_config_payload,
    tune_tabddpm,
    write_tabddpm_tuning_artifacts,
)


def _dataset() -> TabularDataset:
    frame = pd.DataFrame(
        {
            "value": [float(index) for index in range(20)],
            "state": [index % 3 for index in range(20)],
            "target": ["no", "yes"] * 10,
        }
    )
    return TabularDataset(
        name="tabddpm-tuning-smoke",
        frame=frame,
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("state", ColumnKind.DISCRETE),
            ColumnSpec("target", ColumnKind.CATEGORICAL),
        ),
        target="target",
        task=TaskType.CLASSIFICATION,
    )


def _tiny_config(trial: optuna.Trial) -> TabDDPMConfig:
    width = trial.suggest_categorical("width", [4, 8])
    return TabDDPMConfig(
        steps=1,
        num_timesteps=2,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        d_layers=[width],
        dropout=0.0,
        ema_decay=0.9,
        device="cpu",
        seed=0,
        use_ema_for_sampling=False,
        show_progress=False,
    )


def _run_config(root: Path, *, target: int, resume: bool) -> TabDDPMTuningConfig:
    return TabDDPMTuningConfig(
        run=HoldoutRunConfig(
            split=StratifiedHoldoutConfig(validation_fraction=0.2, seed=5),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="tabddpm-tuning-test",
            training_seed=7,
            sample_seed=107,
            device="cpu",
            artifact_dir=root / "runtime",
        ),
        target_complete_trials=target,
        max_total_trials=target + 2,
        sampler_seed=5,
        study_name="tabddpm-test",
        storage="sqlite:///" + str((root / "study.sqlite3").resolve()),
        load_if_exists=resume,
        rerank_candidates=1,
        rerank_seed_pairs=1,
        show_native_progress=False,
        live_state_dir=root / "live",
    )


class TabDDPMTuningTests(unittest.TestCase):
    """Verify search translation, resume guard, rerank, and artifacts."""

    def test_default_search_builds_reviewed_native_config(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "architecture_profile": "small_3",
                "num_timesteps": 100,
                "batch_size": 512,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "use_ema_for_sampling": True,
            }
        )

        config = suggest_tabddpm_config(trial)

        self.assertEqual(config.steps, PHASE_A_STEPS)
        self.assertEqual(config.d_layers, [128, 256, 128])
        self.assertEqual(config.num_timesteps, 100)
        self.assertEqual(config.batch_size, 512)
        self.assertEqual(config.gaussian_loss_type, "mse")
        self.assertEqual(config.scheduler, "cosine")
        self.assertTrue(config.use_ema_for_sampling)
        self.assertFalse(config.show_progress)

    def test_config_payload_round_trips(self) -> None:
        original = _tiny_config(optuna.trial.FixedTrial({"width": 4}))

        restored = tabddpm_config_from_payload(tabddpm_config_payload(original))

        self.assertEqual(restored, original)

    def test_study_resumes_to_successful_trial_target_and_writes_evidence(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with patch("sbtab.benchmark.adapters.tabddpm_tuning.RERANK_STEPS", 1):
                first = tune_tabddpm(
                    _dataset(),
                    _run_config(root, target=1, resume=False),
                    suggest_config=_tiny_config,
                )
                first_trial_count = len(first.study.trials)
                resumed = tune_tabddpm(
                    _dataset(),
                    _run_config(root, target=2, resume=True),
                    suggest_config=_tiny_config,
                )

            self.assertEqual(first_trial_count, 1)
            self.assertEqual(len(resumed.study.trials), 2)
            self.assertTrue(math.isfinite(resumed.best_score))
            self.assertEqual(resumed.best_config.steps, 1)
            self.assertEqual(len(resumed.candidates), 1)
            self.assertEqual(len(resumed.candidates[0].runs), 1)

            manifest_path = write_tabddpm_tuning_artifacts(resumed, root / "tuning")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rerank = json.loads(
                (root / "tuning" / "rerank.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["artifact_type"], "tabddpm_tuning")
            self.assertEqual(manifest["completed_phase_a_trials"], 2)
            self.assertEqual(len(manifest["fingerprint"]), 64)
            self.assertEqual(len(rerank), 1)

    def test_resume_rejects_changed_protocol_fingerprint(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with patch("sbtab.benchmark.adapters.tabddpm_tuning.RERANK_STEPS", 1):
                tune_tabddpm(
                    _dataset(),
                    _run_config(root, target=1, resume=False),
                    suggest_config=_tiny_config,
                )
                changed = _run_config(root, target=1, resume=True)
                changed_run = HoldoutRunConfig(
                    split=changed.run.split,
                    missing_policy=changed.run.missing_policy,
                    run_id=changed.run.run_id,
                    training_seed=8,
                    sample_seed=changed.run.sample_seed,
                    device=changed.run.device,
                    artifact_dir=changed.run.artifact_dir,
                )
                changed = TabDDPMTuningConfig(
                    **{
                        **changed.__dict__,
                        "run": changed_run,
                    }
                )

                with self.assertRaisesRegex(ContractViolation, "Refusing to mix"):
                    tune_tabddpm(_dataset(), changed, suggest_config=_tiny_config)


if __name__ == "__main__":
    unittest.main()
