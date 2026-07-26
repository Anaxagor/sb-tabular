"""Lightweight real-solver smoke for the Online Shoppers pilot entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import optuna
import pandas as pd

from sbtab.benchmark.datasets import ONLINE_SHOPPERS_COLUMNS
from sbtab.benchmark.pilots.msbm_online_shoppers import (
    MSBMOnlineShoppersPilotConfig,
    run_msbm_online_shoppers_pilot,
)
from sbtab.solvers.msbm import MixedSBMConfig


def _frame() -> pd.DataFrame:
    rows = 20
    values: dict[str, list[object]] = {}
    for column in ONLINE_SHOPPERS_COLUMNS:
        if column.kind.value == "continuous":
            values[column.name] = [
                float(index + offset / 10.0)
                for index, offset in zip(range(rows), range(rows))
            ]
        elif column.kind.value == "discrete":
            values[column.name] = [index % 3 for index in range(rows)]
        elif column.name in {"Weekend", "Revenue"}:
            values[column.name] = [bool(index % 2) for index in range(rows)]
        else:
            values[column.name] = [
                "state-a" if index % 2 == 0 else "state-b"
                for index in range(rows)
            ]
    return pd.DataFrame(values)


def _lightweight_config(trial: optuna.Trial) -> MixedSBMConfig:
    hidden_dim = trial.suggest_categorical("hidden_dim", [4])
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


class MSBMOnlineShoppersPilotTests(unittest.TestCase):
    """Verify fixed protocol, real native calls, and complete artifact handoff."""

    def test_lightweight_pilot_tunes_runs_five_folds_and_writes_manifests(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "pilot"

            result = run_msbm_online_shoppers_pilot(
                _frame(),
                MSBMOnlineShoppersPilotConfig(
                    output_dir=output_dir,
                    n_trials=1,
                    device="cpu",
                ),
                suggest_config=_lightweight_config,
            )

            pilot_manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(pilot_manifest["status"], "complete")
            self.assertEqual(pilot_manifest["uci_id"], 468)
            self.assertEqual(len(result.tuning.study.trials), 1)
            self.assertEqual(len(result.final.folds), 5)
            self.assertEqual(result.final.config.split.n_splits, 5)
            self.assertEqual(result.final.config.split.seed, 42)
            self.assertEqual(
                result.tuning.config.run.split.validation_fraction,
                0.2,
            )
            self.assertEqual(result.tuning.config.run.split.seed, 5)
            self.assertTrue(
                (output_dir / pilot_manifest["tuning_manifest"]).is_file()
            )
            self.assertTrue(
                (output_dir / pilot_manifest["generation_manifest"]).is_file()
            )
            self.assertTrue(
                (output_dir / pilot_manifest["evaluation_manifest"]).is_file()
            )
            self.assertEqual(len(result.evaluation.folds), 5)
            self.assertEqual(
                result.evaluation.summary.utility.metric.value,
                "macro_f1",
            )
            for fold in result.final.folds:
                self.assertEqual(len(fold.train_raw), 16)
                self.assertEqual(len(fold.test_raw), 4)
                self.assertEqual(len(fold.synthetic_raw), 16)


if __name__ == "__main__":
    unittest.main()
