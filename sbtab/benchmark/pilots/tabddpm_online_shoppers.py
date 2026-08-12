"""Staged TabDDPM tuning and final Online Shoppers benchmark evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import fmean, pstdev

import pandas as pd
import optuna

from sbtab.baselines.tabddpm.native import TabDDPMConfig

from sbtab.benchmark.adapters.tabddpm import TabDDPMAdapter
from sbtab.benchmark.adapters.tabddpm_tuning import (
    TabDDPMTuningConfig,
    TabDDPMTuningResult,
    tune_tabddpm,
    write_tabddpm_tuning_artifacts,
)
from sbtab.benchmark.artifacts import write_cross_validation_artifacts
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.datasets import (
    ONLINE_SHOPPERS_TARGET,
    ONLINE_SHOPPERS_UCI_ID,
    fetch_online_shoppers_frame,
    make_online_shoppers_dataset,
)
from sbtab.benchmark.missing import MissingPolicy
from sbtab.benchmark.runner import (
    BenchmarkConfig,
    CrossValidationResult,
    HoldoutRunConfig,
    run_cross_validation,
)
from sbtab.benchmark.splitting import (
    StratifiedHoldoutConfig,
    StratifiedKFoldConfig,
)
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation import (
    CrossValidationEvaluation,
    evaluate_cross_validation,
    write_evaluation_artifacts,
)

TABDDPM_ONLINE_SHOPPERS_PILOT_VERSION = 1


@dataclass(frozen=True)
class TabDDPMOnlineShoppersPilotConfig:
    """Human-owned controls for the staged end-to-end experiment.

    ``resume`` reuses a compatible SQLite Phase-A study and completed rerank
    seed runs. Native training itself has no checkpoint, so interruption during
    one fit repeats that fit; final five-fold generation also restarts as a
    complete stage.
    """

    output_dir: Path
    target_complete_trials: int = 30
    max_total_trials: int = 45
    device: str = "cpu"
    timeout_seconds: float | None = None
    study_name: str = "tabddpm-online-shoppers-phase-a-v1"
    storage: str | None = None
    resume: bool = False
    rerank_candidates: int = 3
    rerank_seed_pairs: int = 2
    include_wide_profile: bool = False
    show_native_progress: bool = True


@dataclass(frozen=True)
class TabDDPMOnlineShoppersPilotResult:
    """Completed tuning, generation, evaluation, and linked manifest."""

    dataset: TabularDataset
    tuning: TabDDPMTuningResult
    final: CrossValidationResult
    evaluation: CrossValidationEvaluation
    manifest_path: Path


def _prepare_live_root(config: TabDDPMOnlineShoppersPilotConfig) -> None:
    if config.resume:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if (config.output_dir / "pilot-manifest.json").exists():
            raise ContractViolation(f"Pilot is already complete: {config.output_dir}.")
        if (config.output_dir / "generation").exists():
            raise ContractViolation(
                "Resume after final-generation artifact creation is not "
                "supported; preserve the root and inspect that stage."
            )
        tuning_dir = config.output_dir / "tuning"
        if tuning_dir.exists() and not (tuning_dir / "manifest.json").is_file():
            raise ContractViolation(
                "The tuning artifact directory is incomplete. Preserve it for "
                "diagnosis and resume into a new output root."
            )
        return
    try:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractViolation(
            f"Pilot output directory already exists: {config.output_dir}."
        ) from error


def run_tabddpm_online_shoppers_pilot(
    frame: pd.DataFrame,
    config: TabDDPMOnlineShoppersPilotConfig,
    *,
    suggest_config: Callable[[optuna.Trial], TabDDPMConfig] | None = None,
) -> TabDDPMOnlineShoppersPilotResult:
    """Tune, freeze one 30k-step config, and calculate every final metric."""

    _prepare_live_root(config)
    dataset = make_online_shoppers_dataset(frame)
    default_storage = "sqlite:///" + str(
        (config.output_dir / "study.sqlite3").resolve()
    )
    tuning = tune_tabddpm(
        dataset,
        TabDDPMTuningConfig(
            run=HoldoutRunConfig(
                split=StratifiedHoldoutConfig(
                    validation_fraction=0.2,
                    seed=5,
                ),
                missing_policy=MissingPolicy.COMPLETE_CASE,
                run_id="tabddpm-online-shoppers",
                training_seed=42,
                sample_seed=10_042,
                device=config.device,
                artifact_dir=config.output_dir / "runtime" / "tuning",
            ),
            target_complete_trials=config.target_complete_trials,
            max_total_trials=config.max_total_trials,
            sampler_seed=5,
            timeout_seconds=config.timeout_seconds,
            study_name=config.study_name,
            storage=config.storage or default_storage,
            load_if_exists=config.resume,
            rerank_candidates=config.rerank_candidates,
            rerank_seed_pairs=config.rerank_seed_pairs,
            include_wide_profile=config.include_wide_profile,
            show_native_progress=config.show_native_progress,
            live_state_dir=config.output_dir / "live",
        ),
        suggest_config=suggest_config,
    )
    tuning_dir = config.output_dir / "tuning"
    tuning_manifest = (
        tuning_dir / "manifest.json"
        if tuning_dir.exists()
        else write_tabddpm_tuning_artifacts(tuning, tuning_dir)
    )

    final = run_cross_validation(
        dataset,
        lambda: TabDDPMAdapter(tuning.best_config),
        BenchmarkConfig(
            split=StratifiedKFoldConfig(n_splits=5, seed=42),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="tabddpm-online-shoppers-final",
            training_seed=42,
            sample_seed=10_042,
            device=config.device,
            artifact_dir=config.output_dir / "runtime" / "final",
        ),
    )
    generation_manifest = write_cross_validation_artifacts(
        final, config.output_dir / "generation"
    )
    evaluation = evaluate_cross_validation(final)
    evaluation_manifest = write_evaluation_artifacts(
        evaluation,
        config.output_dir / "evaluation",
        generation_manifest=generation_manifest,
    )
    fit_times = [fold.fit_seconds for fold in final.folds]
    sample_times = [fold.sample_seconds for fold in final.folds]
    manifest = {
        "artifact_type": "tabddpm_online_shoppers_pilot",
        "artifact_version": TABDDPM_ONLINE_SHOPPERS_PILOT_VERSION,
        "status": "complete",
        "dataset": dataset.name,
        "uci_id": ONLINE_SHOPPERS_UCI_ID,
        "target": ONLINE_SHOPPERS_TARGET,
        "best_rerank_score": tuning.best_score,
        "runtime_seconds": {
            "fit": {
                "mean": fmean(fit_times),
                "std": pstdev(fit_times),
                "folds": fit_times,
            },
            "sample": {
                "mean": fmean(sample_times),
                "std": pstdev(sample_times),
                "folds": sample_times,
            },
        },
        "tuning_manifest": str(tuning_manifest.relative_to(config.output_dir)),
        "generation_manifest": str(generation_manifest.relative_to(config.output_dir)),
        "evaluation_manifest": str(evaluation_manifest.relative_to(config.output_dir)),
    }
    path = config.output_dir / "pilot-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return TabDDPMOnlineShoppersPilotResult(
        dataset=dataset,
        tuning=tuning,
        final=final,
        evaluation=evaluation,
        manifest_path=path,
    )


def main() -> None:
    """Run the staged experiment from UCI 468 or an equivalent raw CSV."""

    parser = argparse.ArgumentParser(
        description="Tune and evaluate TabDDPM on Online Shoppers.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-complete-trials", type=int, default=30)
    parser.add_argument("--max-total-trials", type=int, default=45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--study-name",
        default="tabddpm-online-shoppers-phase-a-v1",
    )
    parser.add_argument("--storage", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerank-candidates", type=int, default=3)
    parser.add_argument("--rerank-seed-pairs", type=int, default=2)
    parser.add_argument("--include-wide-profile", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    arguments = parser.parse_args()

    frame = (
        pd.read_csv(arguments.csv)
        if arguments.csv is not None
        else fetch_online_shoppers_frame()
    )
    result = run_tabddpm_online_shoppers_pilot(
        frame,
        TabDDPMOnlineShoppersPilotConfig(
            output_dir=arguments.output_dir,
            target_complete_trials=arguments.target_complete_trials,
            max_total_trials=arguments.max_total_trials,
            device=arguments.device,
            timeout_seconds=arguments.timeout_seconds,
            study_name=arguments.study_name,
            storage=arguments.storage,
            resume=arguments.resume,
            rerank_candidates=arguments.rerank_candidates,
            rerank_seed_pairs=arguments.rerank_seed_pairs,
            include_wide_profile=arguments.include_wide_profile,
            show_native_progress=not arguments.no_progress,
        ),
    )
    print(result.manifest_path)


if __name__ == "__main__":
    main()
