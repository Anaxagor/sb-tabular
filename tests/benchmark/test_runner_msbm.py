"""Real native MSBM smoke through the complete pre-evaluation runner path."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sbtab.benchmark import (
    BenchmarkConfig,
    ColumnKind,
    ColumnSpec,
    MissingPolicy,
    StratifiedKFoldConfig,
    TabularDataset,
    TaskType,
    run_cross_validation,
)
from sbtab.benchmark.adapters import MSBMAdapter
from sbtab.solvers.msbm import MixedSBMConfig


def _dataset() -> TabularDataset:
    return TabularDataset(
        name="msbm-runner-smoke",
        frame=pd.DataFrame(
            {
                "value": [float(index) for index in range(12)],
                "label": ["no", "yes"] * 6,
            }
        ),
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
    )


def _lightweight_native_config(*, device: str, seed: int) -> MixedSBMConfig:
    return MixedSBMConfig(
        fb_sequence=("b",),
        cat_emb_dim=2,
        hidden_dim=4,
        time_dim=4,
        n_layers=1,
        dropout=0.0,
        num_steps=2,
        batch_size=2,
        epochs_per_direction=1,
        device=device,
        seed=seed,
    )


class MSBMRunnerSmokeTests(unittest.TestCase):
    """Exercise real native construction and sampling for every runner fold."""

    def test_real_msbm_completes_two_folds_and_decodes_raw_schema(self) -> None:
        config = BenchmarkConfig(
            split=StratifiedKFoldConfig(n_splits=2, seed=42),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="msbm-runner-smoke",
            artifact_dir=Path("unused-msbm-runner-artifacts"),
        )

        native_config = _lightweight_native_config(device="cpu", seed=0)
        result = run_cross_validation(
            _dataset(),
            lambda: MSBMAdapter(native_config),
            config,
        )

        self.assertEqual(result.adapter_name, "msbm")
        self.assertEqual(len(result.folds), 2)
        for fold in result.folds:
            self.assertEqual(len(fold.train_raw), 6)
            self.assertEqual(len(fold.test_raw), 6)
            self.assertEqual(len(fold.synthetic_raw), 6)
            self.assertEqual(
                tuple(fold.synthetic_raw.columns),
                ("value", "label"),
            )
            self.assertTrue(
                np.isfinite(fold.synthetic_raw["value"].to_numpy()).all()
            )
            self.assertTrue(
                set(fold.synthetic_raw["label"]).issubset({"no", "yes"})
            )


if __name__ == "__main__":
    unittest.main()
