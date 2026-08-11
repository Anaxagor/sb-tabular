"""Fixed-configuration TabDDPM holdout score for UCI Online Shoppers.

This human-invoked pilot runs the shared 80/20 stratified tuning holdout once,
using a caller-selected native :class:`TabDDPMConfig`. It prints and stores the
model-independent tuning objective over decoded raw tables:

``mean train-standardized Wasserstein + mean Jensen--Shannon``.

The command does not perform Optuna search or final K-fold evaluation. Its
score is evidence for one exact native configuration and is minimized.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path

import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark.adapters import TabDDPMAdapter
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.datasets import (
    ONLINE_SHOPPERS_TARGET,
    ONLINE_SHOPPERS_UCI_ID,
    fetch_online_shoppers_frame,
    make_online_shoppers_dataset,
)
from sbtab.benchmark.missing import MissingPolicy
from sbtab.benchmark.runner import (
    HoldoutResult,
    HoldoutRunConfig,
    run_holdout_trial,
)
from sbtab.benchmark.splitting import StratifiedHoldoutConfig
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation import TuningScore, evaluate_tuning_score


TABDDPM_SCORE_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class TabDDPMOnlineShoppersScoreConfig:
    """Controls for scoring one fixed TabDDPM configuration.

    Parameters
    ----------
    output_json:
        New create-only JSON artifact. Existing files are never overwritten.
    native_config:
        Complete model-owned TabDDPM configuration. The shared run context
        overrides only its ``device`` and training ``seed`` for this holdout.
    device:
        Torch device used by the adapter for training and sampling. It is kept
        outside the native configuration because it is a run-level control.
    """

    output_json: Path
    native_config: TabDDPMConfig = field(default_factory=TabDDPMConfig)
    device: str = "cpu"


@dataclass(frozen=True)
class TabDDPMOnlineShoppersScoreResult:
    """Decoded holdout evidence and the persisted composite score."""

    dataset: TabularDataset
    holdout: HoldoutResult
    score: TuningScore
    artifact_path: Path


def _score_payload(
    result: TabDDPMOnlineShoppersScoreResult,
    effective_config: TabDDPMConfig,
) -> dict[str, object]:
    """Build the stable JSON evidence for one completed score run."""

    score = result.score
    holdout = result.holdout
    return {
        "artifact_type": "tabddpm_online_shoppers_holdout_score",
        "artifact_version": TABDDPM_SCORE_ARTIFACT_VERSION,
        "status": "complete",
        "dataset": result.dataset.name,
        "uci_id": ONLINE_SHOPPERS_UCI_ID,
        "target": ONLINE_SHOPPERS_TARGET,
        "objective": {
            "direction": "minimize",
            "total": score.total,
            "mean_standardized_wasserstein": score.mean_wasserstein,
            "mean_jensen_shannon": score.mean_jensen_shannon,
            "columns": [
                {
                    "column": column.column,
                    "kind": column.kind.value,
                    "metric": column.metric.value,
                    "value": column.value,
                    "reference_scale": column.reference_scale,
                }
                for column in score.columns
            ],
        },
        "protocol": {
            "validation_fraction": 0.2,
            "split_seed": 5,
            "training_seed": 42,
            "sample_seed": 10_042,
            "missing_policy": MissingPolicy.COMPLETE_CASE.value,
        },
        "rows": {
            "source": result.holdout.missing_report.rows_before,
            "after_missing_policy": result.holdout.missing_report.rows_after,
            "train": len(holdout.train_raw),
            "validation": len(holdout.validation_raw),
            "synthetic": len(holdout.synthetic_raw),
        },
        "runtime_seconds": {
            "fit": holdout.fit_seconds,
            "sample": holdout.sample_seconds,
        },
        "native_config": asdict(effective_config),
    }


def run_tabddpm_online_shoppers_score(
    frame: pd.DataFrame,
    config: TabDDPMOnlineShoppersScoreConfig,
) -> TabDDPMOnlineShoppersScoreResult:
    """Fit once on the reference holdout and persist its decoded score."""

    if config.output_json.exists():
        raise ContractViolation(
            f"Score artifact already exists: {config.output_json}."
        )
    dataset = make_online_shoppers_dataset(frame)
    holdout = run_holdout_trial(
        dataset,
        lambda: TabDDPMAdapter(config.native_config),
        HoldoutRunConfig(
            split=StratifiedHoldoutConfig(
                validation_fraction=0.2,
                seed=5,
            ),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="tabddpm-online-shoppers-score",
            training_seed=42,
            sample_seed=10_042,
            device=config.device,
            artifact_dir=config.output_json.parent / "runtime",
        ),
    )
    score = evaluate_tuning_score(
        holdout.dataset,
        holdout.train_raw,
        holdout.validation_raw,
        holdout.synthetic_raw,
    )
    result = TabDDPMOnlineShoppersScoreResult(
        dataset=holdout.dataset,
        holdout=holdout,
        score=score,
        artifact_path=config.output_json,
    )
    effective_config = replace(
        config.native_config,
        device=config.device,
        seed=42,
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        with config.output_json.open("x", encoding="utf-8") as stream:
            json.dump(
                _score_payload(result, effective_config),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
    except FileExistsError as error:
        raise ContractViolation(
            f"Score artifact already exists: {config.output_json}."
        ) from error
    return result


def _parse_layers(value: str) -> list[int]:
    """Parse a comma-separated positive MLP layout for ``argparse``."""

    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--d-layers must contain comma-separated integers."
        ) from error
    if not layers or any(width <= 0 for width in layers):
        raise argparse.ArgumentTypeError(
            "--d-layers must contain positive comma-separated integers."
        )
    return layers


def main() -> None:
    """Score a CSV or the canonical downloaded UCI 468 table."""

    parser = argparse.ArgumentParser(
        description="Calculate one fixed-config TabDDPM holdout score.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--num-timesteps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--d-layers",
        type=_parse_layers,
        default=[256, 512, 512, 256],
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "linear"),
        default="cosine",
    )
    parser.add_argument(
        "--gaussian-loss-type",
        choices=("mse", "kl"),
        default="mse",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    arguments = parser.parse_args()

    frame = (
        pd.read_csv(arguments.csv)
        if arguments.csv is not None
        else fetch_online_shoppers_frame()
    )
    result = run_tabddpm_online_shoppers_score(
        frame,
        TabDDPMOnlineShoppersScoreConfig(
            output_json=arguments.output_json,
            device=arguments.device,
            native_config=TabDDPMConfig(
                steps=arguments.steps,
                num_timesteps=arguments.num_timesteps,
                batch_size=arguments.batch_size,
                lr=arguments.lr,
                weight_decay=arguments.weight_decay,
                d_layers=arguments.d_layers,
                dropout=arguments.dropout,
                gaussian_loss_type=arguments.gaussian_loss_type,
                scheduler=arguments.scheduler,
                ema_decay=arguments.ema_decay,
                use_ema_for_sampling=not arguments.no_ema,
                device=arguments.device,
                seed=42,
            ),
        ),
    )
    print(f"total_score={result.score.total:.10f}")
    print(
        "mean_standardized_wasserstein="
        f"{result.score.mean_wasserstein:.10f}"
    )
    print(
        "mean_jensen_shannon="
        f"{result.score.mean_jensen_shannon:.10f}"
    )
    print(result.artifact_path)


if __name__ == "__main__":
    main()
