"""End-to-end smoke for the non-Optuna TabDDPM comparison report."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark.datasets import ONLINE_SHOPPERS_COLUMNS
from sbtab.benchmark.pilots.tabddpm_online_shoppers_fixed import (
    TabDDPMFixedReportConfig,
    run_tabddpm_online_shoppers_fixed_report,
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
                "state-a" if index % 2 == 0 else "state-b" for index in range(rows)
            ]
    return pd.DataFrame(values)


def _tiny_config() -> TabDDPMConfig:
    return TabDDPMConfig(
        steps=1,
        num_timesteps=2,
        batch_size=4,
        d_layers=[4],
        weight_decay=0.0,
        use_ema_for_sampling=False,
        show_progress=False,
    )


class TabDDPMOnlineShoppersFixedReportTests(unittest.TestCase):
    """Verify the fixed protocol, table conventions, and complete evidence."""

    def test_writes_preliminary_table_and_full_final_metrics(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "fixed"

            result = run_tabddpm_online_shoppers_fixed_report(
                _frame(),
                TabDDPMFixedReportConfig(
                    output_dir=output_dir,
                    native_config=_tiny_config(),
                    device="cpu",
                ),
            )

            comparison = json.loads(
                result.comparison_metrics_path.read_text(encoding="utf-8")
            )
            report = result.report_path.read_text(encoding="utf-8")
            full_metrics = json.loads(
                (output_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertFalse(comparison["optuna_used"])
            self.assertEqual(len(comparison["folds"]), 5)
            self.assertIsNotNone(comparison["summary"]["f1_real_minus_synth_percent"])
            self.assertIsNone(comparison["summary"]["r2_real_minus_synth_percent"])
            self.assertIn("Optuna was not used", report)
            self.assertIn("% F1_real - F1_synth", report)
            self.assertIn("mmd_rbf", full_metrics["summary"]["continuous"])
            self.assertEqual(len(result.generation.folds), 5)
            self.assertTrue(result.manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
