"""Real lightweight holdout score for the TabDDPM Online Shoppers CLI."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark.datasets import ONLINE_SHOPPERS_COLUMNS
from sbtab.benchmark.pilots.tabddpm_online_shoppers_score import (
    TabDDPMOnlineShoppersScoreConfig,
    run_tabddpm_online_shoppers_score,
)
from sbtab.benchmark.validation import ContractViolation


def _frame() -> pd.DataFrame:
    """Build a complete small table with every canonical UCI column."""

    row_count = 20
    values: dict[str, list[object]] = {}
    for column in ONLINE_SHOPPERS_COLUMNS:
        if column.kind.value == "continuous":
            values[column.name] = [
                float(index + offset / 10.0)
                for index, offset in zip(range(row_count), range(row_count))
            ]
        elif column.kind.value == "discrete":
            values[column.name] = [index % 3 for index in range(row_count)]
        elif column.name in {"Weekend", "Revenue"}:
            values[column.name] = [
                bool(index % 2) for index in range(row_count)
            ]
        else:
            values[column.name] = [
                "state-a" if index % 2 == 0 else "state-b"
                for index in range(row_count)
            ]
    return pd.DataFrame(values)


def _native_config() -> TabDDPMConfig:
    """Use one optimizer update and two diffusion steps for the smoke test."""

    return TabDDPMConfig(
        steps=1,
        num_timesteps=2,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        d_layers=[4],
        dropout=0.0,
        scheduler="cosine",
        ema_decay=0.9,
        use_ema_for_sampling=False,
        device="cpu",
        seed=0,
    )


class TabDDPMOnlineShoppersScoreTests(unittest.TestCase):
    """Verify real fit/sample, decoded score, and create-only evidence."""

    def test_score_run_writes_complete_reproducible_artifact(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_json = Path(temporary_dir) / "score.json"
            config = TabDDPMOnlineShoppersScoreConfig(
                output_json=output_json,
                native_config=_native_config(),
                device="cpu",
            )

            result = run_tabddpm_online_shoppers_score(_frame(), config)
            payload = json.loads(output_json.read_text(encoding="utf-8"))

            self.assertTrue(np.isfinite(result.score.total))
            self.assertEqual(len(result.score.columns), 18)
            self.assertEqual(len(result.holdout.train_raw), 16)
            self.assertEqual(len(result.holdout.validation_raw), 4)
            self.assertEqual(len(result.holdout.synthetic_raw), 4)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["objective"]["direction"], "minimize")
            self.assertEqual(payload["objective"]["total"], result.score.total)
            self.assertEqual(payload["native_config"]["device"], "cpu")
            self.assertEqual(payload["native_config"]["seed"], 42)
            self.assertEqual(payload["protocol"]["split_seed"], 5)
            self.assertEqual(payload["rows"]["train"], 16)

            with self.assertRaisesRegex(ContractViolation, "already exists"):
                run_tabddpm_online_shoppers_score(_frame(), config)


if __name__ == "__main__":
    unittest.main()
