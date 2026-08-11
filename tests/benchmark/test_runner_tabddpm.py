"""Real TabDDPM smoke through the complete benchmark generation path."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark import (
    BenchmarkConfig,
    ColumnKind,
    ColumnSpec,
    KFoldConfig,
    MissingPolicy,
    StratifiedKFoldConfig,
    TabularDataset,
    TaskType,
    run_cross_validation,
)
from sbtab.benchmark.adapters import TabDDPMAdapter


def _dataset() -> TabularDataset:
    """Return mixed semantics that exercise both native diffusion blocks."""

    return TabularDataset(
        name="tabddpm-runner-smoke",
        frame=pd.DataFrame(
            {
                "value": [float(index) for index in range(12)],
                "count": [0, 1, 2, 0, 1, 2] * 2,
                "label": ["no", "yes"] * 6,
            }
        ),
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("count", ColumnKind.DISCRETE),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
    )


def _lightweight_native_config() -> TabDDPMConfig:
    """Return a one-update CPU profile for boundary characterization only."""

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
        use_ema_for_sampling=True,
        device="cpu",
        seed=0,
    )


def _regression_dataset() -> TabularDataset:
    """Return a pure Gaussian table whose target remains a modeled column."""

    return TabularDataset(
        name="tabddpm-regression-target-smoke",
        frame=pd.DataFrame(
            {
                "feature": [float(index) for index in range(8)],
                "target": [float(index * index) for index in range(8)],
            }
        ),
        columns=(
            ColumnSpec("feature", ColumnKind.CONTINUOUS),
            ColumnSpec("target", ColumnKind.CONTINUOUS),
        ),
        target="target",
        task=TaskType.REGRESSION,
    )


class TabDDPMRunnerSmokeTests(unittest.TestCase):
    """Exercise fold-local codec, real native sampling, and raw decoding."""

    def test_real_tabddpm_decodes_finite_states_without_output_repair(
        self,
    ) -> None:
        config = BenchmarkConfig(
            split=StratifiedKFoldConfig(n_splits=2, seed=42),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="tabddpm-runner-smoke",
            training_seed=42,
            sample_seed=10_042,
            device="cpu",
            artifact_dir=Path("unused-tabddpm-runner-artifacts"),
        )
        native_config = _lightweight_native_config()

        result = run_cross_validation(
            _dataset(),
            lambda: TabDDPMAdapter(native_config),
            config,
        )

        self.assertEqual(result.adapter_name, "tabddpm")
        self.assertEqual(len(result.folds), 2)
        for fold in result.folds:
            self.assertEqual(len(fold.train_raw), 6)
            self.assertEqual(len(fold.test_raw), 6)
            self.assertEqual(len(fold.synthetic_raw), 6)
            self.assertEqual(
                tuple(fold.synthetic_raw.columns),
                ("value", "count", "label"),
            )
            self.assertTrue(
                np.isfinite(fold.synthetic_raw["value"].to_numpy()).all()
            )
            self.assertTrue(
                set(fold.synthetic_raw["count"]).issubset({0, 1, 2})
            )
            self.assertTrue(
                set(fold.synthetic_raw["label"]).issubset({"no", "yes"})
            )

    def test_continuous_target_stays_in_generated_gaussian_table(self) -> None:
        config = BenchmarkConfig(
            split=KFoldConfig(n_splits=2, seed=17),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="tabddpm-regression-target-smoke",
            training_seed=17,
            sample_seed=10_017,
            device="cpu",
            artifact_dir=Path("unused-tabddpm-regression-artifacts"),
        )

        result = run_cross_validation(
            _regression_dataset(),
            lambda: TabDDPMAdapter(_lightweight_native_config()),
            config,
        )

        self.assertEqual(len(result.folds), 2)
        for fold in result.folds:
            self.assertEqual(
                tuple(fold.synthetic_raw.columns),
                ("feature", "target"),
            )
            self.assertTrue(
                np.isfinite(fold.synthetic_raw.to_numpy()).all()
            )


if __name__ == "__main__":
    unittest.main()
