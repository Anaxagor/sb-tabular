"""End-to-end MSBM pilot for UCI Online Shoppers.

The pilot performs the approved model-owned tuning holdout, freezes the best
native ``MixedSBMConfig``, runs final five-fold generation, and writes
create-only tuning, generation, and final-evaluation artifacts. Quality/TSTR
remains a model-independent stage over decoded fold results.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

import optuna
import pandas as pd

from sbtab.benchmark.adapters.msbm import MSBMAdapter
from sbtab.benchmark.adapters.msbm_tuning import (
    MSBMTuningConfig,
    MSBMTuningResult,
    suggest_msbm_config,
    tune_msbm,
    write_msbm_tuning_artifacts,
)
from sbtab.benchmark.artifacts import write_cross_validation_artifacts
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.datasets import (
    ONLINE_SHOPPERS_TARGET,
    ONLINE_SHOPPERS_UCI_ID,
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
from sbtab.solvers.msbm import MixedSBMConfig


MSBM_ONLINE_SHOPPERS_PILOT_VERSION = 1


@dataclass(frozen=True)
class MSBMOnlineShoppersPilotConfig:
    """Human-selected controls for one complete pilot invocation.

    Parameters
    ----------
    output_dir:
        New create-only root for tuning, generation, and pilot manifests.
    n_trials:
        Number of model-owned MSBM Optuna trials.
    device:
        Native device forwarded to all holdout and final-fold adapters.
    timeout_seconds:
        Optional limit for the Optuna optimization call.
    study_name:
        Stable Optuna label. It does not alter benchmark behavior.
    storage:
        Optional Optuna storage URI. It is used but never copied into artifacts.
    load_if_exists:
        Whether Optuna may resume ``study_name`` in configured storage.
    """

    output_dir: Path
    n_trials: int = 50
    device: str = "cpu"
    timeout_seconds: float | None = None
    study_name: str = "msbm-online-shoppers-uci-468"
    storage: str | None = None
    load_if_exists: bool = False


@dataclass(frozen=True)
class MSBMOnlineShoppersPilotResult:
    """Completed pilot outputs retained for evaluation and review.

    Parameters
    ----------
    dataset:
        Validated canonical Online Shoppers declaration used by both phases.
    tuning:
        Model-owned Optuna study and selected native config.
    final:
        Five-fold raw/synthetic generation result using the selected config.
    evaluation:
        Model-independent statistical quality and TSTR results for all folds.
    manifest_path:
        Root manifest written last after all three child artifact sets succeed.
    """

    dataset: TabularDataset
    tuning: MSBMTuningResult
    final: CrossValidationResult
    evaluation: CrossValidationEvaluation
    manifest_path: Path


def fetch_online_shoppers_frame() -> pd.DataFrame:
    """Download and assemble the canonical raw UCI 468 frame.

    ``ucimlrepo`` is imported lazily so importing benchmark modules does not
    perform network work or require the optional acquisition dependency.
    """

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as error:
        raise RuntimeError(
            "Fetching UCI 468 requires the optional ucimlrepo package. "
            "Install it or pass --csv."
        ) from error

    repository = fetch_ucirepo(id=ONLINE_SHOPPERS_UCI_ID)
    features = repository.data.features.copy().reset_index(drop=True)
    targets = repository.data.targets
    if targets is None:
        raise ContractViolation(
            f"UCI {ONLINE_SHOPPERS_UCI_ID} returned no target table."
        )
    if isinstance(targets, pd.Series):
        target_frame = targets.to_frame()
    elif isinstance(targets, pd.DataFrame):
        target_frame = targets.copy()
    else:
        target_frame = pd.DataFrame(targets)
    target_frame = target_frame.reset_index(drop=True)
    if ONLINE_SHOPPERS_TARGET not in target_frame.columns:
        raise ContractViolation(
            f"UCI {ONLINE_SHOPPERS_UCI_ID} target table lacks "
            f"{ONLINE_SHOPPERS_TARGET!r}."
        )
    if ONLINE_SHOPPERS_TARGET in features.columns:
        raise ContractViolation(
            f"UCI features unexpectedly contain target "
            f"{ONLINE_SHOPPERS_TARGET!r}."
        )
    return pd.concat(
        (features, target_frame[[ONLINE_SHOPPERS_TARGET]]),
        axis=1,
    )


def run_msbm_online_shoppers_pilot(
    frame: pd.DataFrame,
    config: MSBMOnlineShoppersPilotConfig,
    *,
    suggest_config: Callable[[optuna.Trial], MixedSBMConfig] = (
        suggest_msbm_config
    ),
) -> MSBMOnlineShoppersPilotResult:
    """Tune MSBM, run final five-fold generation, and write local artifacts."""

    dataset = make_online_shoppers_dataset(frame)
    try:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractViolation(
            f"Pilot output directory already exists: {config.output_dir}."
        ) from error

    tuning = tune_msbm(
        dataset,
        MSBMTuningConfig(
            run=HoldoutRunConfig(
                split=StratifiedHoldoutConfig(
                    validation_fraction=0.2,
                    seed=5,
                ),
                missing_policy=MissingPolicy.COMPLETE_CASE,
                run_id="msbm-online-shoppers-tuning",
                training_seed=42,
                sample_seed=10_042,
                device=config.device,
                artifact_dir=config.output_dir / "runtime" / "tuning",
            ),
            n_trials=config.n_trials,
            sampler_seed=5,
            timeout_seconds=config.timeout_seconds,
            study_name=config.study_name,
            storage=config.storage,
            load_if_exists=config.load_if_exists,
        ),
        suggest_config=suggest_config,
    )
    tuning_manifest = write_msbm_tuning_artifacts(
        tuning,
        config.output_dir / "tuning",
    )

    final = run_cross_validation(
        dataset,
        lambda: MSBMAdapter(tuning.best_config),
        BenchmarkConfig(
            split=StratifiedKFoldConfig(n_splits=5, seed=42),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="msbm-online-shoppers-final",
            training_seed=42,
            sample_seed=10_042,
            device=config.device,
            artifact_dir=config.output_dir / "runtime" / "final",
        ),
    )
    generation_manifest = write_cross_validation_artifacts(
        final,
        config.output_dir / "generation",
    )
    evaluation = evaluate_cross_validation(final)
    evaluation_manifest = write_evaluation_artifacts(
        evaluation,
        config.output_dir / "evaluation",
        generation_manifest=generation_manifest,
    )

    manifest = {
        "artifact_type": "msbm_online_shoppers_pilot",
        "artifact_version": MSBM_ONLINE_SHOPPERS_PILOT_VERSION,
        "status": "complete",
        "dataset": dataset.name,
        "uci_id": ONLINE_SHOPPERS_UCI_ID,
        "target": ONLINE_SHOPPERS_TARGET,
        "best_score": tuning.best_score,
        "best_trial": tuning.study.best_trial.number,
        "tuning_manifest": str(
            tuning_manifest.relative_to(config.output_dir)
        ),
        "generation_manifest": str(
            generation_manifest.relative_to(config.output_dir)
        ),
        "evaluation_manifest": str(
            evaluation_manifest.relative_to(config.output_dir)
        ),
    }
    manifest_path = config.output_dir / "pilot-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return MSBMOnlineShoppersPilotResult(
        dataset=dataset,
        tuning=tuning,
        final=final,
        evaluation=evaluation,
        manifest_path=manifest_path,
    )


def main() -> None:
    """Run the human-owned pilot from a CSV or the canonical UCI source."""

    parser = argparse.ArgumentParser(
        description="Tune and run the MSBM Online Shoppers pilot.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional raw UCI 468 CSV; omitted means fetch through ucimlrepo.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--study-name",
        type=str,
        default="msbm-online-shoppers-uci-468",
    )
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--load-if-exists", action="store_true")
    arguments = parser.parse_args()

    frame = (
        pd.read_csv(arguments.csv)
        if arguments.csv is not None
        else fetch_online_shoppers_frame()
    )
    result = run_msbm_online_shoppers_pilot(
        frame,
        MSBMOnlineShoppersPilotConfig(
            output_dir=arguments.output_dir,
            n_trials=arguments.n_trials,
            device=arguments.device,
            timeout_seconds=arguments.timeout_seconds,
            study_name=arguments.study_name,
            storage=arguments.storage,
            load_if_exists=arguments.load_if_exists,
        ),
    )
    print(result.manifest_path)


if __name__ == "__main__":
    main()
