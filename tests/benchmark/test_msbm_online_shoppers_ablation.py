"""Integration tests for the fixed-config MSBM 2x2 ablation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from sbtab.benchmark.adapters.msbm_tuning import msbm_config_payload
from sbtab.benchmark.datasets import ONLINE_SHOPPERS_COLUMNS
from sbtab.benchmark.pilots.msbm_online_shoppers_ablation import (
    MSBMOnlineShoppersAblationConfig,
    run_msbm_online_shoppers_ablation,
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


class MSBMOnlineShoppersAblationTests(unittest.TestCase):
    """Verify the four cells change only the two declared factors."""

    def test_runs_full_factorial_with_frozen_base_config(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            base_config_path = root / "best-config.json"
            base_config = MixedSBMConfig(
                fb_sequence=("b",),
                cat_emb_dim=2,
                hidden_dim=4,
                time_dim=4,
                n_layers=1,
                dropout=0.0,
                num_steps=2,
                batch_size=2,
                epochs_per_direction=1,
                device="cpu",
                seed=0,
            )
            base_config_path.write_text(
                json.dumps(msbm_config_payload(base_config)),
                encoding="utf-8",
            )

            result = run_msbm_online_shoppers_ablation(
                _frame(),
                MSBMOnlineShoppersAblationConfig(
                    output_dir=root / "ablation",
                    base_config_path=base_config_path,
                    device="cpu",
                ),
            )
            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["design"]["type"], "full_factorial_2x2")
            self.assertEqual(len(result.variants), 4)
            self.assertEqual(
                {
                    (
                        variant.variant.alpha,
                        variant.variant.divides_categorical_loss_by_columns,
                    )
                    for variant in result.variants
                },
                {
                    (0.01, False),
                    (0.01, True),
                    (0.798, False),
                    (0.798, True),
                },
            )
            for variant in result.variants:
                self.assertEqual(variant.native_config.hidden_dim, 4)
                self.assertEqual(variant.native_config.num_steps, 2)
                self.assertEqual(len(variant.generation.folds), 5)
                self.assertEqual(len(variant.evaluation.folds), 5)
                self.assertTrue(variant.manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
