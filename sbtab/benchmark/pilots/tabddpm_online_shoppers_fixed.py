"""Fixed-config TabDDPM five-fold report for UCI Online Shoppers.

This entrypoint is intentionally independent of Optuna. It provides an honest
preliminary benchmark row for one explicitly recorded native configuration
while the model-owned hyperparameter study runs separately.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from statistics import fmean, pstdev

import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark.adapters.tabddpm import TabDDPMAdapter
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
    run_cross_validation,
)
from sbtab.benchmark.splitting import StratifiedKFoldConfig
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation import (
    CrossValidationEvaluation,
    UtilityMetric,
    evaluate_cross_validation,
    evaluate_tuning_score,
    write_evaluation_artifacts,
)

TABDDPM_FIXED_REPORT_VERSION = 1


def default_fixed_config() -> TabDDPMConfig:
    """Return the already characterized non-Optuna MPS configuration."""

    return TabDDPMConfig(
        steps=10_000,
        num_timesteps=100,
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-4,
        d_layers=[128, 256, 128],
        dropout=0.0,
        gaussian_loss_type="mse",
        scheduler="cosine",
        ema_decay=0.999,
        use_ema_for_sampling=True,
        show_progress=True,
    )


@dataclass(frozen=True)
class TabDDPMFixedReportConfig:
    """Controls for one preliminary fixed-configuration benchmark report.

    Parameters
    ----------
    output_dir:
        New create-only root containing generation, full evaluation, the
        comparison-table payload, and a human-readable Markdown row.
    native_config:
        Exact model-owned configuration used unchanged for every final fold;
        only fold device and seed are supplied by the common runner.
    device:
        Native Torch device label, normally ``mps`` or ``cuda``.
    """

    output_dir: Path
    native_config: TabDDPMConfig = field(default_factory=default_fixed_config)
    device: str = "cpu"


@dataclass(frozen=True)
class MetricSummary:
    """Population mean/std for one scalar over all final folds."""

    mean: float
    std: float


@dataclass(frozen=True)
class TabDDPMFixedReportResult:
    """Complete fixed-config generation, evaluation, and report locations."""

    dataset: TabularDataset
    generation: CrossValidationResult
    evaluation: CrossValidationEvaluation
    comparison_metrics_path: Path
    report_path: Path
    manifest_path: Path


def _summary(values: list[float]) -> MetricSummary:
    if not values:
        raise ContractViolation("Cannot summarize an empty metric sequence.")
    return MetricSummary(mean=fmean(values), std=pstdev(values))


def _format_summary(summary: MetricSummary | None) -> str:
    if summary is None:
        return "—"
    return f"{summary.mean:.4f} ± {summary.std:.4f}"


def _comparison_payload(
    generation: CrossValidationResult,
    evaluation: CrossValidationEvaluation,
    native_config: TabDDPMConfig,
) -> dict[str, object]:
    """Build metrics matching the columns of the historical comparison table."""

    standardized_wd: list[float] = []
    mean_kl: list[float] = []
    correlation: list[float] = []
    mmd: list[float] = []
    f1_degradation: list[float] = []
    r2_degradation: list[float] = []
    fold_payloads: list[dict[str, object]] = []

    for generation_fold, evaluation_fold in zip(
        generation.folds, evaluation.folds, strict=True
    ):
        tuning_score = evaluate_tuning_score(
            generation.dataset,
            generation_fold.train_raw,
            generation_fold.test_raw,
            generation_fold.synthetic_raw,
        )
        continuous = evaluation_fold.quality.continuous
        if tuning_score.mean_wasserstein is None or continuous is None:
            raise ContractViolation(
                "Online Shoppers comparison requires continuous columns."
            )
        if continuous.pearson_frobenius is None:
            raise ContractViolation(
                "Online Shoppers comparison requires Pearson correlation."
            )
        standardized_wd.append(tuning_score.mean_wasserstein)
        mean_kl.append(continuous.mean_kl)
        correlation.append(continuous.pearson_frobenius)
        mmd.append(continuous.mmd_rbf)

        utility = evaluation_fold.utility
        if utility is None:
            raise ContractViolation("Online Shoppers comparison requires TSTR utility.")
        degradation = (
            None
            if utility.relative_change_percent is None
            else -utility.relative_change_percent
        )
        if utility.metric is UtilityMetric.MACRO_F1:
            if degradation is None:
                raise ContractViolation("F1 degradation is undefined.")
            f1_degradation.append(degradation)
        elif utility.metric is UtilityMetric.R2:
            if degradation is None:
                raise ContractViolation("R2 degradation is undefined.")
            r2_degradation.append(degradation)

        fold_payloads.append(
            {
                "fold_id": evaluation_fold.fold_id,
                "mean_kl": continuous.mean_kl,
                "mean_wd_train_standardized": tuning_score.mean_wasserstein,
                "mean_wd_raw_units": continuous.mean_wasserstein,
                "corr_distance_pearson": continuous.pearson_frobenius,
                "f1_real_minus_synth_percent": (
                    degradation if utility.metric is UtilityMetric.MACRO_F1 else None
                ),
                "r2_real_minus_synth_percent": (
                    degradation if utility.metric is UtilityMetric.R2 else None
                ),
                "mmd_rbf": continuous.mmd_rbf,
                "f1_or_r2_real": utility.real_score,
                "f1_or_r2_synthetic": utility.synthetic_score,
                "fit_seconds": generation_fold.fit_seconds,
                "sample_seconds": generation_fold.sample_seconds,
            }
        )

    summaries = {
        "mean_kl": _summary(mean_kl),
        "mean_wd_train_standardized": _summary(standardized_wd),
        "corr_distance_pearson": _summary(correlation),
        "f1_real_minus_synth_percent": (
            _summary(f1_degradation) if f1_degradation else None
        ),
        "r2_real_minus_synth_percent": (
            _summary(r2_degradation) if r2_degradation else None
        ),
        "mmd_rbf": _summary(mmd),
    }
    return {
        "artifact_type": "tabddpm_fixed_comparison_metrics",
        "artifact_version": TABDDPM_FIXED_REPORT_VERSION,
        "status": "preliminary_fixed_config",
        "optuna_used": False,
        "note": (
            "Preliminary fixed-config result; model-owned Optuna tuning runs "
            "separately."
        ),
        "dataset": generation.dataset.name,
        "uci_id": ONLINE_SHOPPERS_UCI_ID,
        "model": "TabDDPM",
        "native_config": asdict(native_config),
        "protocol": {
            "split": "StratifiedKFold",
            "n_splits": 5,
            "split_seed": 42,
            "training_seeds": [42 + index for index in range(5)],
            "sample_seeds": [10_042 + index for index in range(5)],
            "missing_policy": MissingPolicy.COMPLETE_CASE.value,
            "mean_wd_space": "train_standardized_continuous",
            "mean_kl": "continuous_50_bin_kl_real_to_synthetic",
            "corr_distance": "continuous_pearson_frobenius",
            "mmd": "train_standardized_continuous_rbf_biased_squared",
            "utility_sign": "100 * (real - synthetic) / abs(real)",
        },
        "summary": {
            name: asdict(value) if value is not None else None
            for name, value in summaries.items()
        },
        "folds": fold_payloads,
    }


def _markdown_report(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)

    def read(name: str) -> MetricSummary | None:
        value = summary[name]
        return MetricSummary(**value) if isinstance(value, dict) else None

    row = " | ".join(
        (
            "Online Shoppers Purchasing Intention Dataset",
            "TabDDPM (fixed config; Optuna running)",
            _format_summary(read("mean_kl")),
            _format_summary(read("mean_wd_train_standardized")),
            _format_summary(read("corr_distance_pearson")),
            _format_summary(read("f1_real_minus_synth_percent")),
            _format_summary(read("r2_real_minus_synth_percent")),
            _format_summary(read("mmd_rbf")),
        )
    )
    return "\n".join(
        (
            "# Preliminary TabDDPM result",
            "",
            "> Fixed configuration; Optuna was not used for this row and is "
            "running separately.",
            "",
            "| Dataset | Generative model | Mean KL | Mean WD | Corr distance "
            "| % F1_real - F1_synth | % R2_real - R2_synth | MMD |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {row} |",
            "",
            "`Mean WD` is calculated in train-standardized continuous space, "
            "matching the historical comparison. Full raw-unit WD and all "
            "discrete/categorical metrics are retained in `evaluation/metrics.json`.",
            "",
        )
    )


def run_tabddpm_online_shoppers_fixed_report(
    frame: pd.DataFrame,
    config: TabDDPMFixedReportConfig,
) -> TabDDPMFixedReportResult:
    """Run fixed five-fold generation and write a shareable preliminary row."""

    try:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractViolation(
            f"Fixed-report output directory already exists: {config.output_dir}."
        ) from error

    dataset = make_online_shoppers_dataset(frame)
    generation = run_cross_validation(
        dataset,
        lambda: TabDDPMAdapter(config.native_config),
        BenchmarkConfig(
            split=StratifiedKFoldConfig(n_splits=5, seed=42),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id="tabddpm-online-shoppers-fixed",
            training_seed=42,
            sample_seed=10_042,
            device=config.device,
            artifact_dir=config.output_dir / "runtime",
        ),
    )
    generation_manifest = write_cross_validation_artifacts(
        generation, config.output_dir / "generation"
    )
    evaluation = evaluate_cross_validation(generation)
    evaluation_manifest = write_evaluation_artifacts(
        evaluation,
        config.output_dir / "evaluation",
        generation_manifest=generation_manifest,
    )
    effective_config = replace(
        config.native_config,
        device=config.device,
        seed=42,
    )
    comparison = _comparison_payload(generation, evaluation, effective_config)
    comparison_metrics_path = config.output_dir / "comparison-metrics.json"
    comparison_metrics_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path = config.output_dir / "report.md"
    report_path.write_text(_markdown_report(comparison), encoding="utf-8")
    manifest = {
        "artifact_type": "tabddpm_online_shoppers_fixed_report",
        "artifact_version": TABDDPM_FIXED_REPORT_VERSION,
        "status": "complete_preliminary_fixed_config",
        "optuna_used": False,
        "dataset": dataset.name,
        "uci_id": ONLINE_SHOPPERS_UCI_ID,
        "target": ONLINE_SHOPPERS_TARGET,
        "generation_manifest": str(generation_manifest.relative_to(config.output_dir)),
        "evaluation_manifest": str(evaluation_manifest.relative_to(config.output_dir)),
        "comparison_metrics": comparison_metrics_path.name,
        "report": report_path.name,
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return TabDDPMFixedReportResult(
        dataset=dataset,
        generation=generation,
        evaluation=evaluation,
        comparison_metrics_path=comparison_metrics_path,
        report_path=report_path,
        manifest_path=manifest_path,
    )


def main() -> None:
    """Run the fixed report from UCI 468 or an equivalent raw CSV."""

    parser = argparse.ArgumentParser(
        description="Run a non-Optuna five-fold TabDDPM preliminary report.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    arguments = parser.parse_args()

    native = default_fixed_config()
    native.steps = arguments.steps
    native.num_timesteps = arguments.num_timesteps
    native.batch_size = arguments.batch_size
    native.lr = arguments.lr
    native.weight_decay = arguments.weight_decay
    native.use_ema_for_sampling = not arguments.no_ema
    native.show_progress = not arguments.no_progress
    frame = (
        pd.read_csv(arguments.csv)
        if arguments.csv is not None
        else fetch_online_shoppers_frame()
    )
    result = run_tabddpm_online_shoppers_fixed_report(
        frame,
        TabDDPMFixedReportConfig(
            output_dir=arguments.output_dir,
            native_config=native,
            device=arguments.device,
        ),
    )
    print(result.report_path)


if __name__ == "__main__":
    main()
