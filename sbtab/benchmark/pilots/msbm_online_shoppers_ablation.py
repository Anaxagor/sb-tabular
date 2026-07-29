"""Fixed-config 2x2 MSBM ablation for UCI Online Shoppers.

The experiment freezes every field from an existing MSBM tuning artifact and
changes only categorical reference ``alpha`` and optional division of the
categorical loss by the number of state columns. Every variant uses the same
five folds, training seeds, sample seeds, raw evaluation, and artifact format.
No Optuna search runs inside the ablation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from sbtab.benchmark.adapters.msbm import MSBMAdapter
from sbtab.benchmark.adapters.msbm_tuning import (
    msbm_config_from_payload,
    msbm_config_payload,
)
from sbtab.benchmark.artifacts import write_cross_validation_artifacts
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.datasets import (
    ONLINE_SHOPPERS_TARGET,
    ONLINE_SHOPPERS_UCI_ID,
    make_online_shoppers_dataset,
)
from sbtab.benchmark.missing import MissingPolicy
from sbtab.benchmark.pilots.msbm_online_shoppers import (
    fetch_online_shoppers_frame,
)
from sbtab.benchmark.runner import (
    BenchmarkConfig,
    CrossValidationResult,
    run_cross_validation,
)
from sbtab.benchmark.splitting import StratifiedKFoldConfig
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation import (
    CrossValidationEvaluation,
    evaluate_cross_validation,
    write_evaluation_artifacts,
)
from sbtab.solvers.msbm import (
    CategoricalLossNormalization,
    MixedSBMConfig,
)


MSBM_ABLATION_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class MSBMAblationVariant:
    """One cell of the predeclared categorical-mechanics factorial design.

    Parameters
    ----------
    name:
        Stable artifact label containing no inferred dataset semantics.
    alpha:
        Categorical-reference transition parameter passed to every fold.
    categorical_loss_normalization:
        Whether native mixed loss divides categorical CSBM loss by the number
        of finite-state columns.
    """

    name: str
    alpha: float
    categorical_loss_normalization: CategoricalLossNormalization

    @property
    def divides_categorical_loss_by_columns(self) -> bool:
        """Return the human-readable on/off factor value."""

        return (
            self.categorical_loss_normalization
            is CategoricalLossNormalization.BY_NUM_COLUMNS
        )


MSBM_ABLATION_VARIANTS = (
    MSBMAblationVariant(
        name="alpha-0p01-loss-raw",
        alpha=0.01,
        categorical_loss_normalization=CategoricalLossNormalization.NONE,
    ),
    MSBMAblationVariant(
        name="alpha-0p01-loss-div-c",
        alpha=0.01,
        categorical_loss_normalization=(
            CategoricalLossNormalization.BY_NUM_COLUMNS
        ),
    ),
    MSBMAblationVariant(
        name="alpha-0p798-loss-raw",
        alpha=0.798,
        categorical_loss_normalization=CategoricalLossNormalization.NONE,
    ),
    MSBMAblationVariant(
        name="alpha-0p798-loss-div-c",
        alpha=0.798,
        categorical_loss_normalization=(
            CategoricalLossNormalization.BY_NUM_COLUMNS
        ),
    ),
)


@dataclass(frozen=True)
class MSBMOnlineShoppersAblationConfig:
    """Human-selected inputs for one create-only 2x2 ablation run.

    Parameters
    ----------
    output_dir:
        New root for four linked generation/evaluation artifact sets.
    base_config_path:
        Frozen ``best-config.json`` from an already completed MSBM tuning run.
        The ablation does not retune or read the associated Optuna database.
    device:
        Native device shared by all variants and folds.
    """

    output_dir: Path
    base_config_path: Path
    device: str = "cpu"


@dataclass(frozen=True)
class MSBMAblationVariantResult:
    """Completed generation, evaluation, and manifest for one variant."""

    variant: MSBMAblationVariant
    native_config: MixedSBMConfig
    generation: CrossValidationResult
    evaluation: CrossValidationEvaluation
    manifest_path: Path


@dataclass(frozen=True)
class MSBMOnlineShoppersAblationResult:
    """All four completed variants and the root comparison manifest."""

    dataset: TabularDataset
    variants: tuple[MSBMAblationVariantResult, ...]
    manifest_path: Path


def _load_base_config(path: Path) -> tuple[MixedSBMConfig, str]:
    if not path.is_file():
        raise ContractViolation(f"Base MSBM config does not exist: {path}.")
    payload_bytes = path.read_bytes()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ContractViolation(
            f"Base MSBM config is not valid JSON: {path}."
        ) from error
    return (
        msbm_config_from_payload(payload),
        hashlib.sha256(payload_bytes).hexdigest(),
    )


def _variant_payload(variant: MSBMAblationVariant) -> dict[str, object]:
    return {
        "name": variant.name,
        "alpha": variant.alpha,
        "categorical_loss_normalization": (
            variant.categorical_loss_normalization.value
        ),
        "divide_categorical_loss_by_columns": (
            variant.divides_categorical_loss_by_columns
        ),
    }


def run_msbm_online_shoppers_ablation(
    frame: pd.DataFrame,
    config: MSBMOnlineShoppersAblationConfig,
) -> MSBMOnlineShoppersAblationResult:
    """Run all four frozen-config cells under one identical benchmark protocol."""

    dataset = make_online_shoppers_dataset(frame)
    base_config, base_config_sha256 = _load_base_config(
        config.base_config_path
    )
    try:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractViolation(
            f"Ablation output directory already exists: {config.output_dir}."
        ) from error

    results: list[MSBMAblationVariantResult] = []
    for variant in MSBM_ABLATION_VARIANTS:
        variant_dir = config.output_dir / variant.name
        variant_dir.mkdir()
        native_config = replace(
            base_config,
            alpha=variant.alpha,
            categorical_loss_normalization=(
                variant.categorical_loss_normalization
            ),
        )
        generation = run_cross_validation(
            dataset,
            lambda native_config=native_config: MSBMAdapter(native_config),
            BenchmarkConfig(
                split=StratifiedKFoldConfig(n_splits=5, seed=42),
                missing_policy=MissingPolicy.COMPLETE_CASE,
                run_id=f"msbm-online-shoppers-ablation-{variant.name}",
                training_seed=42,
                sample_seed=10_042,
                device=config.device,
                artifact_dir=variant_dir / "runtime",
            ),
        )
        generation_manifest = write_cross_validation_artifacts(
            generation,
            variant_dir / "generation",
        )
        evaluation = evaluate_cross_validation(generation)
        evaluation_manifest = write_evaluation_artifacts(
            evaluation,
            variant_dir / "evaluation",
            generation_manifest=generation_manifest,
        )

        native_config_path = variant_dir / "native-config.json"
        native_config_path.write_text(
            json.dumps(
                msbm_config_payload(native_config),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        variant_manifest = {
            "artifact_type": "msbm_ablation_variant",
            "artifact_version": MSBM_ABLATION_ARTIFACT_VERSION,
            "status": "complete",
            "dataset": dataset.name,
            "variant": _variant_payload(variant),
            "native_config": native_config_path.name,
            "generation_manifest": os.path.relpath(
                generation_manifest,
                start=variant_dir,
            ),
            "evaluation_manifest": os.path.relpath(
                evaluation_manifest,
                start=variant_dir,
            ),
            "summary": asdict(evaluation.summary),
        }
        variant_manifest_path = variant_dir / "manifest.json"
        variant_manifest_path.write_text(
            json.dumps(
                variant_manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        results.append(
            MSBMAblationVariantResult(
                variant=variant,
                native_config=native_config,
                generation=generation,
                evaluation=evaluation,
                manifest_path=variant_manifest_path,
            )
        )

    manifest = {
        "artifact_type": "msbm_online_shoppers_ablation",
        "artifact_version": MSBM_ABLATION_ARTIFACT_VERSION,
        "status": "complete",
        "dataset": dataset.name,
        "uci_id": ONLINE_SHOPPERS_UCI_ID,
        "target": ONLINE_SHOPPERS_TARGET,
        "design": {
            "type": "full_factorial_2x2",
            "alpha": [0.01, 0.798],
            "divide_categorical_loss_by_columns": [False, True],
            "frozen_fields": (
                "all remaining MixedSBMConfig fields from base_config"
            ),
        },
        "protocol": {
            "split": "StratifiedKFoldConfig",
            "n_splits": 5,
            "split_seed": 42,
            "training_seed": 42,
            "sample_seed": 10_042,
            "missing_policy": MissingPolicy.COMPLETE_CASE.value,
            "device": config.device,
        },
        "base_config": {
            "path": os.path.relpath(
                config.base_config_path,
                start=config.output_dir,
            ),
            "sha256": base_config_sha256,
            "payload": msbm_config_payload(base_config),
        },
        "variants": [
            {
                **_variant_payload(result.variant),
                "manifest": os.path.relpath(
                    result.manifest_path,
                    start=config.output_dir,
                ),
                "summary": asdict(result.evaluation.summary),
            }
            for result in results
        ],
    }
    manifest_path = config.output_dir / "ablation-manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return MSBMOnlineShoppersAblationResult(
        dataset=dataset,
        variants=tuple(results),
        manifest_path=manifest_path,
    )


def main() -> None:
    """Run the human-owned ablation from a CSV or canonical UCI source."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-config 2x2 MSBM Online Shoppers ablation."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional raw UCI 468 CSV; omitted means fetch through ucimlrepo.",
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    arguments = parser.parse_args()

    frame = (
        pd.read_csv(arguments.csv)
        if arguments.csv is not None
        else fetch_online_shoppers_frame()
    )
    result = run_msbm_online_shoppers_ablation(
        frame,
        MSBMOnlineShoppersAblationConfig(
            output_dir=arguments.output_dir,
            base_config_path=arguments.base_config,
            device=arguments.device,
        ),
    )
    print(result.manifest_path)


if __name__ == "__main__":
    main()
