"""Tune and evaluate TabDDPM on the fourteen published mixed datasets.

Each dataset owns an independent Optuna study, high-budget candidate rerank,
frozen-configuration five-fold generation run, and complete final evaluation.
Completed dataset roots are immutable.  ``--resume`` skips them and resumes an
interrupted Optuna/rerank stage at trial boundaries for the current dataset.

The entrypoint uses only shared benchmark and evaluation components.  It does
not preprocess data, calculate a private metric variant, or branch inside the
TabDDPM adapter based on dataset identity.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev

import optuna
import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark.adapters.tabddpm import TabDDPMAdapter
from sbtab.benchmark.adapters.tabddpm_tuning import (
    TABDDPM_TUNING_PROTOCOL_VERSION,
    TabDDPMTuningConfig,
    tune_tabddpm,
    write_tabddpm_tuning_artifacts,
)
from sbtab.benchmark.artifacts import write_cross_validation_artifacts
from sbtab.benchmark.contracts import TabularDataset, TaskType
from sbtab.benchmark.datasets import (
    MIXED_DATASET_KEYS,
    fetch_mixed_dataset,
    make_mixed_dataset,
)
from sbtab.benchmark.missing import MissingPolicy
from sbtab.benchmark.runner import (
    BenchmarkConfig,
    CrossValidationResult,
    HoldoutRunConfig,
    run_cross_validation,
)
from sbtab.benchmark.splitting import (
    HoldoutConfig,
    KFoldConfig,
    StratifiedHoldoutConfig,
    StratifiedKFoldConfig,
)
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation import (
    CrossValidationEvaluation,
    ScalarSummary,
    UtilityMetric,
    evaluate_cross_validation,
    evaluate_tuning_score,
    write_evaluation_artifacts,
)


TABDDPM_MIXED_BENCHMARK_VERSION = 3
_PUBLISHED_NAME_BY_KEY: Mapping[str, str] = {
    "adult": "Adult",
    "credit_approval": "Credit Approval",
    "online_shoppers": "Online Shoppers",
    "eucalyptus": "Eucalyptus",
    "forest_fires": "Forest Fires",
    "insurance": "Insurance",
    "house_sales": "House Sales",
    "cardiovascular_disease": "Cardiovascular Disease",
    "churn_modelling": "Churn Modelling",
    "auto_mpg": "Auto MPG",
    "diamonds": "Diamonds",
    "real_estate": "Real Estate",
    "stroke_prediction": "Stroke Prediction",
    "palmer_penguins": "Palmer Penguins",
}


@dataclass(frozen=True)
class TabDDPMMixedBenchmarkConfig:
    """Operational controls for the complete multi-dataset experiment.

    Parameters
    ----------
    output_dir:
        Create-only collection root. With ``resume=True``, it must contain a
        compatible ``run-spec.json`` written by an earlier invocation.
    dataset_keys:
        Ordered, unique subset of :data:`MIXED_DATASET_KEYS`. The default is
        the complete published collection.
    target_complete_trials, max_total_trials:
        Successful Phase-A trial target and per-invocation total-trial safety
        ceiling for each independent dataset study. The target is immutable;
        the ceiling may be increased on resume after failures are reviewed.
    device:
        Native Torch device label used by tuning, rerank, and final folds.
    timeout_seconds_per_dataset:
        Optional Phase-A wall-clock limit for each dataset in this invocation.
        Reaching it leaves resumable study state and fails explicitly before
        final evaluation.
    rerank_candidates, rerank_seed_pairs:
        Number of distinct Phase-A leaders and seed pairs evaluated at the
        tuner's fixed high-budget rerank stage.
    include_wide_profile:
        Opt in to the calibrated high-memory architecture profile.
    resume:
        Reuse compatible Optuna/rerank state and skip dataset roots whose
        final manifest already exists. Native training cannot resume midway
        through one fit.
    show_native_progress:
        Enable native fit/sample progress bars.
    dataset_source, dataset_source_sha256:
        Auditable acquisition label and optional digest of one immutable
        collection snapshot. CLI pickle runs always populate both fields.
    """

    output_dir: Path
    dataset_keys: tuple[str, ...] = MIXED_DATASET_KEYS
    target_complete_trials: int = 30
    max_total_trials: int = 45
    device: str = "cpu"
    timeout_seconds_per_dataset: float | None = None
    rerank_candidates: int = 3
    rerank_seed_pairs: int = 2
    include_wide_profile: bool = False
    resume: bool = False
    show_native_progress: bool = True
    dataset_source: str = "caller-provided"
    dataset_source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise ContractViolation("output_dir must be pathlib.Path.")
        if not isinstance(self.dataset_keys, tuple) or not self.dataset_keys:
            raise ContractViolation("dataset_keys must be a non-empty tuple.")
        if len(set(self.dataset_keys)) != len(self.dataset_keys):
            raise ContractViolation("dataset_keys must not contain duplicates.")
        unknown = tuple(
            key for key in self.dataset_keys if key not in MIXED_DATASET_KEYS
        )
        if unknown:
            raise ContractViolation(f"Unknown mixed dataset keys: {unknown!r}.")
        for name in (
            "target_complete_trials",
            "max_total_trials",
            "rerank_candidates",
            "rerank_seed_pairs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractViolation(f"{name} must be a positive integer.")
        if self.max_total_trials < self.target_complete_trials:
            raise ContractViolation(
                "max_total_trials cannot be below target_complete_trials."
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ContractViolation("device must be a non-empty string.")
        if not isinstance(self.dataset_source, str) or not self.dataset_source:
            raise ContractViolation("dataset_source must be a non-empty string.")
        if self.dataset_source_sha256 is not None and (
            not isinstance(self.dataset_source_sha256, str)
            or len(self.dataset_source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.dataset_source_sha256
            )
        ):
            raise ContractViolation(
                "dataset_source_sha256 must be a lowercase SHA-256 or None."
            )
        if (
            self.timeout_seconds_per_dataset is not None
            and self.timeout_seconds_per_dataset <= 0
        ):
            raise ContractViolation(
                "timeout_seconds_per_dataset must be positive or None."
            )
        for name in ("include_wide_profile", "resume", "show_native_progress"):
            if not isinstance(getattr(self, name), bool):
                raise ContractViolation(f"{name} must be bool.")


@dataclass(frozen=True)
class TabDDPMMixedBenchmarkResult:
    """Locations of the completed collection manifest and summary tables."""

    manifest_path: Path
    summary_json_path: Path
    summary_csv_path: Path
    summary_markdown_path: Path


def _json_write_atomic(path: Path, payload: object) -> None:
    """Write JSON atomically so interruption cannot create a valid-looking file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_spec(config: TabDDPMMixedBenchmarkConfig) -> dict[str, object]:
    """Return immutable collection controls checked on every resume."""

    return {
        "artifact_type": "tabddpm_mixed_benchmark_run_spec",
        "artifact_version": TABDDPM_MIXED_BENCHMARK_VERSION,
        "model": "tabddpm",
        "dataset_keys": list(config.dataset_keys),
        "target_complete_trials": config.target_complete_trials,
        "device": config.device,
        "rerank_candidates": config.rerank_candidates,
        "rerank_seed_pairs": config.rerank_seed_pairs,
        "include_wide_profile": config.include_wide_profile,
        "dataset_source": config.dataset_source,
        "dataset_source_sha256": config.dataset_source_sha256,
        "protocol": {
            "tuning_protocol_version": TABDDPM_TUNING_PROTOCOL_VERSION,
            "missing_policy": MissingPolicy.COMPLETE_CASE.value,
            "tuning_split": "80/20; seed=5; task-stratified classification",
            "final_split": "5-fold; seed=42; task-stratified classification",
            "training_seed": 42,
            "sample_seed": 10_042,
            "continuous_preprocessing": "train-fold standard; ddof=0",
            "discrete_preprocessing": "raw numeric values; Gaussian diffusion",
            "discrete_output": "np.rint; ties to even; no clipping or support projection",
            "categorical_preprocessing": "train-fold finite-state codes",
        },
    }


def _prepare_collection_root(config: TabDDPMMixedBenchmarkConfig) -> Path:
    """Create or validate the immutable collection run specification."""

    spec_path = config.output_dir / "run-spec.json"
    expected = _run_spec(config)
    if config.resume:
        if not config.output_dir.is_dir() or not spec_path.is_file():
            raise ContractViolation(
                "Resume requires an existing root with run-spec.json: "
                f"{config.output_dir}."
            )
        observed = json.loads(spec_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ContractViolation(
                "Refusing to resume with dataset, target, device, rerank, or "
                "protocol controls different from run-spec.json."
            )
        return spec_path

    try:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractViolation(
            f"Benchmark output directory already exists: {config.output_dir}."
        ) from error
    _json_write_atomic(spec_path, expected)
    return spec_path


def _holdout_split(
    dataset: TabularDataset,
) -> HoldoutConfig | StratifiedHoldoutConfig:
    if dataset.task is TaskType.CLASSIFICATION:
        return StratifiedHoldoutConfig(validation_fraction=0.2, seed=5)
    return HoldoutConfig(validation_fraction=0.2, seed=5)


def _final_split(
    dataset: TabularDataset,
) -> KFoldConfig | StratifiedKFoldConfig:
    if dataset.task is TaskType.CLASSIFICATION:
        return StratifiedKFoldConfig(n_splits=5, seed=42)
    return KFoldConfig(n_splits=5, seed=42)


def _scalar_payload(value: ScalarSummary | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"mean": value.mean, "std": value.std}


def _values_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ContractViolation("Cannot summarize an empty metric sequence.")
    return {"mean": fmean(values), "std": pstdev(values)}


def _dataset_summary(
    dataset: TabularDataset,
    tuning_best_score: float,
    completed_trials: int,
    generation: CrossValidationResult,
    evaluation: CrossValidationEvaluation,
) -> dict[str, object]:
    """Build one machine-readable row with every requested mixed metric."""

    summary = evaluation.summary
    continuous = summary.continuous
    discrete = summary.discrete
    categorical = summary.categorical
    utility = summary.utility

    standardized_wd: list[float] = []
    final_js: list[float] = []
    for generation_fold in generation.folds:
        score = evaluate_tuning_score(
            generation.dataset,
            generation_fold.train_raw,
            generation_fold.test_raw,
            generation_fold.synthetic_raw,
        )
        if score.mean_wasserstein is not None:
            standardized_wd.append(score.mean_wasserstein)
        if score.mean_jensen_shannon is not None:
            final_js.append(score.mean_jensen_shannon)

    f1_degradation = None
    r2_degradation = None
    if utility is not None and utility.relative_change_percent is not None:
        degradation = {
            "mean": -utility.relative_change_percent.mean,
            "std": utility.relative_change_percent.std,
        }
        if utility.metric is UtilityMetric.MACRO_F1:
            f1_degradation = degradation
        elif utility.metric is UtilityMetric.R2:
            r2_degradation = degradation

    return {
        "dataset_key": dataset.name,
        "published_name": _PUBLISHED_NAME_BY_KEY[dataset.name],
        "task": dataset.task.value if dataset.task is not None else None,
        "rows_after_complete_case": len(generation.dataset.frame),
        "columns": len(dataset.columns),
        "tuning": {
            "best_rerank_score": tuning_best_score,
            "completed_phase_a_trials": completed_trials,
            "objective": "mean_train_standardized_wd + mean_exact_state_js",
        },
        "continuous": (
            {
                "mean_wd_raw_units": _scalar_payload(
                    continuous.mean_wasserstein
                ),
                "mean_wd_train_standardized": (
                    _values_summary(standardized_wd)
                    if standardized_wd
                    else None
                ),
                "mean_kl_50_bins": _scalar_payload(continuous.mean_kl),
                "corr_distance_pearson": _scalar_payload(
                    continuous.pearson_frobenius
                ),
                "mmd_rbf": _scalar_payload(continuous.mmd_rbf),
            }
            if continuous is not None
            else None
        ),
        "discrete": (
            {
                "mean_kl": _scalar_payload(discrete.mean_kl),
                "corr_distance_spearman": _scalar_payload(
                    discrete.spearman_frobenius
                ),
            }
            if discrete is not None
            else None
        ),
        "categorical": (
            {
                "mean_kl": _scalar_payload(categorical.mean_kl),
                "nmi_distance": _scalar_payload(categorical.nmi_frobenius),
            }
            if categorical is not None
            else None
        ),
        "final_mean_exact_state_js": (
            _values_summary(final_js) if final_js else None
        ),
        "utility": (
            {
                "metric": utility.metric.value,
                "real_score": _scalar_payload(utility.real_score),
                "synthetic_score": _scalar_payload(utility.synthetic_score),
                "f1_real_minus_synth_percent": f1_degradation,
                "r2_real_minus_synth_percent": r2_degradation,
            }
            if utility is not None
            else None
        ),
        "runtime_seconds": {
            "fit": _values_summary(
                [fold.fit_seconds for fold in generation.folds]
            ),
            "sample": _values_summary(
                [fold.sample_seconds for fold in generation.folds]
            ),
        },
    }


def _prepare_dataset_root(root: Path, *, resume: bool) -> None:
    """Create one live root or validate the supported resume boundary."""

    if not root.exists():
        root.mkdir()
        return
    if not resume:
        raise ContractViolation(f"Dataset output directory already exists: {root}.")
    if (root / "dataset-manifest.json").exists():
        return
    if (root / "generation").exists() or (root / "evaluation").exists():
        raise ContractViolation(
            "Resume after final generation artifact creation is not supported "
            f"for {root.name!r}; preserve this evidence and use a new root."
        )
    tuning_dir = root / "tuning"
    if tuning_dir.exists() and not (tuning_dir / "manifest.json").is_file():
        raise ContractViolation(
            f"Incomplete tuning artifact directory for {root.name!r}. Preserve "
            "it for diagnosis and use a new output root."
        )


def _run_one_dataset(
    dataset: TabularDataset,
    config: TabDDPMMixedBenchmarkConfig,
    *,
    suggest_config: Callable[[optuna.Trial], TabDDPMConfig] | None = None,
) -> Path:
    """Run independent tuning, final generation, and evaluation for a dataset."""

    key = dataset.name
    root = config.output_dir / key
    _prepare_dataset_root(root, resume=config.resume)
    complete_manifest = root / "dataset-manifest.json"
    if complete_manifest.is_file():
        return complete_manifest

    study_path = (root / "study.sqlite3").resolve()
    tuning = tune_tabddpm(
        dataset,
        TabDDPMTuningConfig(
            run=HoldoutRunConfig(
                split=_holdout_split(dataset),
                missing_policy=MissingPolicy.COMPLETE_CASE,
                run_id=f"tabddpm-{key}-tuning",
                training_seed=42,
                sample_seed=10_042,
                device=config.device,
                artifact_dir=root / "runtime" / "tuning",
            ),
            target_complete_trials=config.target_complete_trials,
            max_total_trials=config.max_total_trials,
            sampler_seed=5,
            timeout_seconds=config.timeout_seconds_per_dataset,
            study_name=f"tabddpm-{key}-phase-a-v3",
            storage="sqlite:///" + str(study_path),
            load_if_exists=config.resume,
            rerank_candidates=config.rerank_candidates,
            rerank_seed_pairs=config.rerank_seed_pairs,
            include_wide_profile=config.include_wide_profile,
            show_native_progress=config.show_native_progress,
            live_state_dir=root / "live",
        ),
        suggest_config=suggest_config,
    )
    tuning_dir = root / "tuning"
    tuning_manifest = (
        tuning_dir / "manifest.json"
        if tuning_dir.exists()
        else write_tabddpm_tuning_artifacts(tuning, tuning_dir)
    )

    generation = run_cross_validation(
        dataset,
        lambda: TabDDPMAdapter(tuning.best_config),
        BenchmarkConfig(
            split=_final_split(dataset),
            missing_policy=MissingPolicy.COMPLETE_CASE,
            run_id=f"tabddpm-{key}-final",
            training_seed=42,
            sample_seed=10_042,
            device=config.device,
            artifact_dir=root / "runtime" / "final",
        ),
    )
    generation_manifest = write_cross_validation_artifacts(
        generation,
        root / "generation",
    )
    evaluation = evaluate_cross_validation(generation)
    evaluation_manifest = write_evaluation_artifacts(
        evaluation,
        root / "evaluation",
        generation_manifest=generation_manifest,
    )
    completed_trials = sum(
        trial.state is optuna.trial.TrialState.COMPLETE
        for trial in tuning.study.trials
    )
    summary = _dataset_summary(
        generation.dataset,
        tuning.best_score,
        completed_trials,
        generation,
        evaluation,
    )
    summary_path = root / "summary.json"
    _json_write_atomic(summary_path, summary)
    manifest = {
        "artifact_type": "tabddpm_mixed_dataset_result",
        "artifact_version": TABDDPM_MIXED_BENCHMARK_VERSION,
        "status": "complete",
        "dataset_key": key,
        "tuning_manifest": str(tuning_manifest.relative_to(root)),
        "generation_manifest": str(generation_manifest.relative_to(root)),
        "evaluation_manifest": str(evaluation_manifest.relative_to(root)),
        "evaluation_manifest_sha256": _sha256(evaluation_manifest),
        "summary": str(summary_path.relative_to(root)),
        "summary_sha256": _sha256(summary_path),
        "generation_manifest_sha256": _sha256(generation_manifest),
        "tuning_manifest_sha256": _sha256(tuning_manifest),
    }
    _json_write_atomic(complete_manifest, manifest)
    return complete_manifest


def _read_dataset_summary(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_type") != "tabddpm_mixed_dataset_result"
        or manifest.get("status") != "complete"
    ):
        raise ContractViolation(f"Invalid dataset manifest: {manifest_path}.")
    artifact_paths: dict[str, Path] = {}
    for name in (
        "tuning_manifest",
        "generation_manifest",
        "evaluation_manifest",
        "summary",
    ):
        relative = manifest.get(name)
        if not isinstance(relative, str):
            raise ContractViolation(
                f"Dataset manifest lacks {name}: {manifest_path}."
            )
        path = manifest_path.parent / relative
        if not path.is_file() or _sha256(path) != manifest.get(f"{name}_sha256"):
            raise ContractViolation(f"Dataset artifact digest mismatch: {path}.")
        artifact_paths[name] = path
    summary_path = artifact_paths["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ContractViolation(f"Dataset summary must be an object: {summary_path}.")
    if summary.get("dataset_key") != manifest_path.parent.name:
        raise ContractViolation(
            f"Dataset summary identity mismatch: {summary_path}."
        )
    return summary


def _completed_collection_result(
    config: TabDDPMMixedBenchmarkConfig,
    manifest_path: Path,
) -> TabDDPMMixedBenchmarkResult:
    """Validate a completed root before treating resume as a no-op."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_type") != "tabddpm_mixed_benchmark"
        or manifest.get("status") != "complete"
    ):
        raise ContractViolation(f"Invalid collection manifest: {manifest_path}.")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ContractViolation("Completed collection manifest lacks datasets.")
    observed_keys: list[str] = []
    for item in datasets:
        if not isinstance(item, Mapping):
            raise ContractViolation("Collection dataset entry must be an object.")
        key = item.get("key")
        relative = item.get("manifest")
        digest = item.get("manifest_sha256")
        if not isinstance(key, str) or not isinstance(relative, str):
            raise ContractViolation("Collection dataset entry is incomplete.")
        dataset_manifest = config.output_dir / relative
        if not dataset_manifest.is_file() or _sha256(dataset_manifest) != digest:
            raise ContractViolation(
                f"Completed dataset manifest digest mismatch: {dataset_manifest}."
            )
        _read_dataset_summary(dataset_manifest)
        observed_keys.append(key)
    if tuple(observed_keys) != config.dataset_keys:
        raise ContractViolation(
            "Completed collection dataset order differs from run-spec.json."
        )

    result_paths: dict[str, Path] = {}
    for name in ("summary_json", "summary_csv", "summary_markdown"):
        relative = manifest.get(name)
        digest = manifest.get(f"{name}_sha256")
        if not isinstance(relative, str):
            raise ContractViolation(f"Collection manifest lacks {name}.")
        path = config.output_dir / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ContractViolation(f"Collection artifact digest mismatch: {path}.")
        result_paths[name] = path
    return TabDDPMMixedBenchmarkResult(
        manifest_path=manifest_path,
        summary_json_path=result_paths["summary_json"],
        summary_csv_path=result_paths["summary_csv"],
        summary_markdown_path=result_paths["summary_markdown"],
    )


def _nested_scalar(
    row: Mapping[str, object],
    group: str,
    metric: str,
    statistic: str,
) -> float | None:
    group_value = row.get(group)
    if not isinstance(group_value, Mapping):
        return None
    metric_value = group_value.get(metric)
    if not isinstance(metric_value, Mapping):
        return None
    value = metric_value.get(statistic)
    return float(value) if isinstance(value, (int, float)) else None


def _flat_summary_row(row: Mapping[str, object]) -> dict[str, object]:
    """Flatten one nested summary into stable machine-readable CSV columns."""

    utility = row.get("utility")
    utility_mapping = utility if isinstance(utility, Mapping) else {}
    runtime = row.get("runtime_seconds")
    runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
    tuning = row.get("tuning")
    tuning_mapping = tuning if isinstance(tuning, Mapping) else {}

    result: dict[str, object] = {
        "dataset_key": row["dataset_key"],
        "dataset": row["published_name"],
        "task": row["task"],
        "rows": row["rows_after_complete_case"],
        "columns": row["columns"],
        "optuna_best_score": tuning_mapping.get("best_rerank_score"),
        "optuna_complete_trials": tuning_mapping.get(
            "completed_phase_a_trials"
        ),
    }
    metric_paths = {
        "continuous_mean_wd_raw": ("continuous", "mean_wd_raw_units"),
        "continuous_mean_wd_standardized": (
            "continuous",
            "mean_wd_train_standardized",
        ),
        "continuous_mean_kl_50_bins": ("continuous", "mean_kl_50_bins"),
        "continuous_corr_pearson": (
            "continuous",
            "corr_distance_pearson",
        ),
        "mmd_rbf": ("continuous", "mmd_rbf"),
        "discrete_mean_kl": ("discrete", "mean_kl"),
        "discrete_corr_spearman": (
            "discrete",
            "corr_distance_spearman",
        ),
        "categorical_mean_kl": ("categorical", "mean_kl"),
        "categorical_nmi_distance": ("categorical", "nmi_distance"),
    }
    for output_name, (group, metric) in metric_paths.items():
        result[f"{output_name}_mean"] = _nested_scalar(
            row, group, metric, "mean"
        )
        result[f"{output_name}_std"] = _nested_scalar(
            row, group, metric, "std"
        )
    for name in (
        "f1_real_minus_synth_percent",
        "r2_real_minus_synth_percent",
    ):
        value = utility_mapping.get(name)
        mapping = value if isinstance(value, Mapping) else {}
        result[f"{name}_mean"] = mapping.get("mean")
        result[f"{name}_std"] = mapping.get("std")
    for name in ("fit", "sample"):
        value = runtime_mapping.get(name)
        mapping = value if isinstance(value, Mapping) else {}
        result[f"{name}_seconds_mean"] = mapping.get("mean")
        result[f"{name}_seconds_std"] = mapping.get("std")
    return result


def _format_scalar(mean: object, std: object) -> str:
    if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
        return "—"
    return f"{float(mean):.4f} ± {float(std):.4f}"


def _markdown_summary(rows: list[dict[str, object]]) -> str:
    columns = (
        ("Cont. KL", "continuous_mean_kl_50_bins"),
        ("Cont. WD", "continuous_mean_wd_raw"),
        ("Pearson", "continuous_corr_pearson"),
        ("Disc. KL", "discrete_mean_kl"),
        ("Spearman", "discrete_corr_spearman"),
        ("Cat. KL", "categorical_mean_kl"),
        ("NMI dist.", "categorical_nmi_distance"),
        ("F1 gap %", "f1_real_minus_synth_percent"),
        ("R2 gap %", "r2_real_minus_synth_percent"),
        ("MMD", "mmd_rbf"),
        ("Fit s", "fit_seconds"),
    )
    header = "| Dataset | " + " | ".join(label for label, _ in columns) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in columns) + " |"
    lines = [
        "# TabDDPM mixed benchmark",
        "",
        "All cells are five-fold population mean ± standard deviation.",
        "Continuous WD is reported in decoded raw units; the JSON/CSV also "
        "contains train-standardized WD used by the tuning objective.",
        "",
        header,
        divider,
    ]
    for row in rows:
        values = [str(row["dataset"])]
        for _, key in columns:
            values.append(
                _format_scalar(row.get(f"{key}_mean"), row.get(f"{key}_std"))
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_progress(
    config: TabDDPMMixedBenchmarkConfig,
    manifests: list[Path],
) -> Path:
    completed = [path.parent.name for path in manifests]
    payload = {
        "artifact_type": "tabddpm_mixed_benchmark_progress",
        "artifact_version": TABDDPM_MIXED_BENCHMARK_VERSION,
        "completed": completed,
        "pending": [key for key in config.dataset_keys if key not in completed],
        "dataset_manifests": [
            str(path.relative_to(config.output_dir)) for path in manifests
        ],
    }
    path = config.output_dir / "progress.json"
    _json_write_atomic(path, payload)
    return path


def run_tabddpm_mixed_benchmark(
    config: TabDDPMMixedBenchmarkConfig,
    *,
    dataset_loader: Callable[[str], TabularDataset] = fetch_mixed_dataset,
    dataset_runner: Callable[
        [TabularDataset, TabDDPMMixedBenchmarkConfig], Path
    ]
    | None = None,
) -> TabDDPMMixedBenchmarkResult:
    """Run or resume the configured ordered mixed-dataset collection."""

    if not isinstance(config, TabDDPMMixedBenchmarkConfig):
        raise ContractViolation("config must be TabDDPMMixedBenchmarkConfig.")
    if not callable(dataset_loader):
        raise ContractViolation("dataset_loader must be callable.")
    if dataset_runner is not None and not callable(dataset_runner):
        raise ContractViolation("dataset_runner must be callable or None.")

    spec_path = _prepare_collection_root(config)
    existing_manifest = config.output_dir / "manifest.json"
    if existing_manifest.is_file():
        if not config.resume:
            raise ContractViolation("Completed benchmark root is immutable.")
        return _completed_collection_result(config, existing_manifest)

    manifests: list[Path] = []
    runner = dataset_runner or _run_one_dataset
    for index, key in enumerate(config.dataset_keys, start=1):
        complete_manifest = config.output_dir / key / "dataset-manifest.json"
        if complete_manifest.is_file():
            manifests.append(complete_manifest)
            _write_progress(config, manifests)
            print(f"[{index}/{len(config.dataset_keys)}] {key}: already complete")
            continue
        print(f"[{index}/{len(config.dataset_keys)}] {key}: acquiring")
        dataset = dataset_loader(key)
        if dataset.name != key:
            raise ContractViolation(
                f"Dataset loader returned name {dataset.name!r} for key {key!r}."
            )
        print(f"[{index}/{len(config.dataset_keys)}] {key}: tuning and evaluation")
        manifest = runner(dataset, config)
        manifests.append(manifest)
        _write_progress(config, manifests)
        print(f"[{index}/{len(config.dataset_keys)}] {key}: complete")

    nested_rows = [_read_dataset_summary(path) for path in manifests]
    flat_rows = [_flat_summary_row(row) for row in nested_rows]
    summary_json_path = config.output_dir / "summary.json"
    _json_write_atomic(summary_json_path, nested_rows)
    summary_csv_path = config.output_dir / "summary.csv"
    pd.DataFrame(flat_rows).to_csv(summary_csv_path, index=False)
    summary_markdown_path = config.output_dir / "summary.md"
    summary_markdown_path.write_text(
        _markdown_summary(flat_rows),
        encoding="utf-8",
    )
    manifest = {
        "artifact_type": "tabddpm_mixed_benchmark",
        "artifact_version": TABDDPM_MIXED_BENCHMARK_VERSION,
        "status": "complete",
        "run_spec": str(spec_path.relative_to(config.output_dir)),
        "run_spec_sha256": _sha256(spec_path),
        "datasets": [
            {
                "key": path.parent.name,
                "manifest": str(path.relative_to(config.output_dir)),
                "manifest_sha256": _sha256(path),
            }
            for path in manifests
        ],
        "summary_json": summary_json_path.name,
        "summary_json_sha256": _sha256(summary_json_path),
        "summary_csv": summary_csv_path.name,
        "summary_csv_sha256": _sha256(summary_csv_path),
        "summary_markdown": summary_markdown_path.name,
        "summary_markdown_sha256": _sha256(summary_markdown_path),
    }
    manifest_path = config.output_dir / "manifest.json"
    _json_write_atomic(manifest_path, manifest)
    return TabDDPMMixedBenchmarkResult(
        manifest_path=manifest_path,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        summary_markdown_path=summary_markdown_path,
    )


def _pickle_dataset_loader(path: Path) -> Callable[[str], TabularDataset]:
    """Load the explicit opt-in upstream pickle once, ignoring all attrs."""

    payload = pd.read_pickle(path)
    if not isinstance(payload, Mapping):
        raise ContractViolation("Dataset pickle root must be a mapping.")

    def load(key: str) -> TabularDataset:
        published_name = _PUBLISHED_NAME_BY_KEY[key]
        frame = payload.get(published_name)
        if not isinstance(frame, pd.DataFrame):
            raise ContractViolation(
                f"Dataset pickle lacks DataFrame {published_name!r}."
            )
        # Attrs contain legacy inferred metadata and are deliberately ignored.
        return make_mixed_dataset(key, frame)

    return load


def main() -> None:
    """Parse CLI controls and run the complete or selected dataset collection."""

    parser = argparse.ArgumentParser(
        description="Tune and evaluate TabDDPM on mixed benchmark datasets.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=MIXED_DATASET_KEYS,
        default=list(MIXED_DATASET_KEYS),
        help="Ordered subset; defaults to all fourteen datasets.",
    )
    parser.add_argument("--target-complete-trials", type=int, default=30)
    parser.add_argument("--max-total-trials", type=int, default=45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-seconds-per-dataset", type=float, default=None)
    parser.add_argument("--rerank-candidates", type=int, default=3)
    parser.add_argument("--rerank-seed-pairs", type=int, default=2)
    parser.add_argument("--include-wide-profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--dataset-pickle",
        type=Path,
        default=None,
        help=(
            "Optional trusted upstream datasets_mixed.pkl. If omitted, data "
            "is fetched from UCI/OpenML/Kaggle. Never load an untrusted pickle."
        ),
    )
    arguments = parser.parse_args()

    if arguments.dataset_pickle is not None:
        dataset_pickle = arguments.dataset_pickle.resolve()
        if not dataset_pickle.is_file():
            raise FileNotFoundError(dataset_pickle)
        loader = _pickle_dataset_loader(dataset_pickle)
        dataset_source = "trusted-pickle"
        dataset_source_sha256 = _sha256(dataset_pickle)
    else:
        loader = fetch_mixed_dataset
        dataset_source = "live-uci-openml-kaggle-acquisition"
        dataset_source_sha256 = None

    config = TabDDPMMixedBenchmarkConfig(
        output_dir=arguments.output_dir,
        dataset_keys=tuple(arguments.datasets),
        target_complete_trials=arguments.target_complete_trials,
        max_total_trials=arguments.max_total_trials,
        device=arguments.device,
        timeout_seconds_per_dataset=arguments.timeout_seconds_per_dataset,
        rerank_candidates=arguments.rerank_candidates,
        rerank_seed_pairs=arguments.rerank_seed_pairs,
        include_wide_profile=arguments.include_wide_profile,
        resume=arguments.resume,
        show_native_progress=not arguments.no_progress,
        dataset_source=dataset_source,
        dataset_source_sha256=dataset_source_sha256,
    )
    result = run_tabddpm_mixed_benchmark(config, dataset_loader=loader)
    print(result.manifest_path)


if __name__ == "__main__":
    main()
