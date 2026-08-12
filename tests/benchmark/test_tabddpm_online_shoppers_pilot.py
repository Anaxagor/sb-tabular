"""Lightweight real-solver smoke for the complete TabDDPM pilot."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import optuna
import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark.datasets import ONLINE_SHOPPERS_COLUMNS
from sbtab.benchmark.pilots.tabddpm_online_shoppers import (
    TabDDPMOnlineShoppersPilotConfig,
    run_tabddpm_online_shoppers_pilot,
)


def _frame() -> pd.DataFrame:
    rows = 20
    values: dict[str, list[object]] = {}
    for column in ONLINE_SHOPPERS_COLUMNS:
        if column.kind.value == "continuous":
            values[column.name] = [float(index) for index in range(rows)]
        elif column.kind.value == "discrete":
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


class TabDDPMOnlineShoppersPilotTests(unittest.TestCase):
    """Verify tuning, frozen five-fold generation, and all final artifacts."""

    def test_lightweight_complete_pilot_writes_mmd_and_tstr(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "pilot"
            config = TabDDPMOnlineShoppersPilotConfig(
                output_dir=output_dir,
                target_complete_trials=1,
                max_total_trials=2,
                rerank_candidates=1,
                rerank_seed_pairs=1,
                show_native_progress=False,
            )
            with patch("sbtab.benchmark.adapters.tabddpm_tuning.RERANK_STEPS", 1):
                result = run_tabddpm_online_shoppers_pilot(
                    _frame(), config, suggest_config=_tiny_config
                )

            pilot = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(
                (output_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pilot["status"], "complete")
            self.assertEqual(len(pilot["runtime_seconds"]["fit"]["folds"]), 5)
            self.assertGreaterEqual(pilot["runtime_seconds"]["fit"]["mean"], 0.0)
            self.assertEqual(len(result.final.folds), 5)
            self.assertIn("mmd_rbf", metrics["summary"]["continuous"])
            self.assertEqual(metrics["summary"]["utility"]["metric"], "macro_f1")
            self.assertTrue((output_dir / "study.sqlite3").is_file())
            self.assertTrue((output_dir / "tuning" / "rerank.json").is_file())


if __name__ == "__main__":
    unittest.main()
