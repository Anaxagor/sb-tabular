"""Model-owned staged Optuna selection for the native TabDDPM solver.

Phase A compares architecture and optimizer choices at one fixed 10,000-step
budget on the common benchmark holdout.  Phase B reruns the best distinct
configurations at 30,000 steps on two explicit model/sample seed pairs.  Only
the shared raw-space tuning score selects configurations; report metrics remain
reserved for final cross-validation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from time import monotonic

import optuna
import pandas as pd

from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.baselines.tabddpm.utils import FoundNANsError
from sbtab.benchmark.adapters.tabddpm import TabDDPMAdapter
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.missing import MissingReport, apply_missing_policy
from sbtab.benchmark.runner import HoldoutRunConfig, run_holdout_trial
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation import evaluate_tuning_score

TABDDPM_TUNING_ARTIFACT_VERSION = 1
TABDDPM_TUNING_OBJECTIVE_VERSION = 1
TABDDPM_SEARCH_SPACE_VERSION = 1
TABDDPM_RERANK_VERSION = 1
PHASE_A_STEPS = 10_000
RERANK_STEPS = 30_000
ARCHITECTURE_PROFILES: Mapping[str, tuple[int, ...]] = {
    "small_3": (128, 256, 128),
    "default_4": (256, 512, 512, 256),
    "medium_6": (256, 512, 512, 512, 512, 256),
    "wide_4": (512, 1024, 1024, 512),
}


class TabDDPMNumericalTrialError(RuntimeError):
    """Catchable Optuna signal for a native non-finite sampling trajectory.

    The imported native ``FoundNANsError`` inherits directly from
    ``BaseException`` and therefore bypasses Optuna's normal failed-trial
    handling.  The tuning boundary translates only that known numerical
    condition; contract, integration, and unexpected model errors still stop
    the study immediately.
    """


@dataclass(frozen=True)
class TabDDPMTuningConfig:
    """Controls for one resumable Phase-A study and deterministic reranking.

    Parameters
    ----------
    run:
        Common stratified holdout, missing policy, device, and base seeds.
    target_complete_trials:
        Desired total number of successful Phase-A trials in the study. On
        resume this is a target, not a count of additional trials.
    max_total_trials:
        Safety ceiling including failed trials. It prevents an invalid search
        space from retrying indefinitely.
    sampler_seed:
        Seed for Optuna TPE proposals. Native training/sample seeds come from
        ``run`` and are identical across Phase-A trials.
    timeout_seconds:
        Optional wall-clock budget for this invocation only.
    study_name, storage, load_if_exists:
        Optuna persistence controls. A SQLite storage allows restart at trial
        boundaries; the model currently has no mid-fit checkpoint seam.
    rerank_candidates, rerank_seed_pairs:
        Number of distinct Phase-A leaders and independent native seed pairs
        evaluated at :data:`RERANK_STEPS` before freezing final configuration.
    show_native_progress:
        Operational tqdm output for long native fits and samples.
    """

    run: HoldoutRunConfig
    target_complete_trials: int = 30
    max_total_trials: int = 45
    sampler_seed: int = 5
    timeout_seconds: float | None = None
    study_name: str = "tabddpm-online-shoppers-phase-a-v1"
    storage: str | None = None
    load_if_exists: bool = False
    rerank_candidates: int = 3
    rerank_seed_pairs: int = 2
    include_wide_profile: bool = False
    show_native_progress: bool = True
    live_state_dir: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run, HoldoutRunConfig):
            raise ContractViolation("TabDDPMTuningConfig.run must be HoldoutRunConfig.")
        for name in (
            "target_complete_trials",
            "max_total_trials",
            "sampler_seed",
            "rerank_candidates",
            "rerank_seed_pairs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractViolation(
                    f"TabDDPMTuningConfig.{name} must be an integer."
                )
        if self.target_complete_trials < 1:
            raise ContractViolation("target_complete_trials must be positive.")
        if self.max_total_trials < self.target_complete_trials:
            raise ContractViolation(
                "max_total_trials cannot be below target_complete_trials."
            )
        if not 0 <= self.sampler_seed < 2**32:
            raise ContractViolation("sampler_seed must be in [0, 2**32).")
        if self.rerank_candidates < 1 or self.rerank_seed_pairs < 1:
            raise ContractViolation(
                "rerank candidate and seed counts must be positive."
            )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ContractViolation("timeout_seconds must be positive or None.")
        if not isinstance(self.include_wide_profile, bool):
            raise ContractViolation("include_wide_profile must be bool.")
        if not isinstance(self.show_native_progress, bool):
            raise ContractViolation("show_native_progress must be bool.")
        if self.live_state_dir is not None and not isinstance(
            self.live_state_dir, Path
        ):
            raise ContractViolation("live_state_dir must be pathlib.Path or None.")


@dataclass(frozen=True)
class TabDDPMRerankRun:
    """One high-budget score for a Phase-A candidate and explicit seed pair."""

    candidate_rank: int
    phase_a_trial: int
    training_seed: int
    sample_seed: int
    score: float
    mean_standardized_wasserstein: float | None
    mean_jensen_shannon: float | None
    fit_seconds: float
    sample_seconds: float


@dataclass(frozen=True)
class TabDDPMCandidateResult:
    """High-budget mean/std evidence for one distinct Phase-A candidate."""

    candidate_rank: int
    phase_a_trial: int
    config: TabDDPMConfig
    runs: tuple[TabDDPMRerankRun, ...]
    mean_score: float
    std_score: float


@dataclass(frozen=True)
class TabDDPMTuningResult:
    """Completed Phase A, Phase B evidence, and one frozen native config."""

    study: optuna.Study
    config: TabDDPMTuningConfig
    dataset: TabularDataset
    candidates: tuple[TabDDPMCandidateResult, ...]
    best_config: TabDDPMConfig
    best_score: float
    fingerprint: str


def suggest_tabddpm_config(
    trial: optuna.Trial,
    *,
    include_wide_profile: bool = False,
) -> TabDDPMConfig:
    """Suggest the reviewed compute-bounded Phase-A search space.

    The 2.5M-parameter wide profile is opt-in because it is roughly eleven
    times larger than the already measured MPS pilot network. Calibrate it on
    the actual accelerator before enabling it for a multi-hour study.
    """

    profiles = (
        list(ARCHITECTURE_PROFILES)
        if include_wide_profile
        else [
            "small_3",
            "default_4",
            "medium_6",
        ]
    )
    profile = trial.suggest_categorical("architecture_profile", profiles)
    return TabDDPMConfig(
        steps=PHASE_A_STEPS,
        n_epochs=None,
        num_timesteps=trial.suggest_categorical("num_timesteps", [100, 1000]),
        batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4]),
        d_layers=list(ARCHITECTURE_PROFILES[profile]),
        dropout=0.0,
        gaussian_loss_type="mse",
        scheduler="cosine",
        ema_decay=0.999,
        device="cpu",
        seed=0,
        use_ema_for_sampling=trial.suggest_categorical(
            "use_ema_for_sampling", [False, True]
        ),
        show_progress=False,
    )


def tabddpm_config_payload(config: TabDDPMConfig) -> dict[str, object]:
    """Serialize one complete native configuration to JSON-compatible data."""

    return asdict(config)


def tabddpm_config_from_payload(payload: object) -> TabDDPMConfig:
    """Reconstruct and validate the complete config stored on an Optuna trial."""

    if not isinstance(payload, Mapping):
        raise ContractViolation("TabDDPM trial has no native_config mapping.")
    try:
        config = TabDDPMConfig(**dict(payload))
    except TypeError as error:
        raise ContractViolation(
            "Stored native_config does not match current TabDDPMConfig."
        ) from error
    if not isinstance(config.d_layers, list) or not all(
        isinstance(width, int) and not isinstance(width, bool) and width > 0
        for width in config.d_layers
    ):
        raise ContractViolation("Stored TabDDPM d_layers are invalid.")
    return config


def _missing_report_payload(report: MissingReport) -> dict[str, object]:
    def counts_payload(counts):
        if counts is None:
            return None
        return [
            {"label_repr": repr(item.label), "count": item.count} for item in counts
        ]

    return {
        "policy": report.policy.value,
        "rows_before": report.rows_before,
        "rows_after": report.rows_after,
        "dropped_count": report.dropped_count,
        "dropped_fraction": report.dropped_fraction,
        "missing_by_column": dict(report.missing_by_column),
        "class_counts_before": counts_payload(report.class_counts_before),
        "class_counts_after": counts_payload(report.class_counts_after),
    }


def _study_fingerprint(dataset: TabularDataset, config: TabDDPMTuningConfig) -> str:
    """Hash data content and every semantic/protocol choice shared by trials."""

    filtered = apply_missing_policy(dataset, config.run.missing_policy).dataset
    frame_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            filtered.frame.loc[:, list(filtered.column_order)], index=False
        )
        .to_numpy()
        .tobytes()
    ).hexdigest()
    spec = TabDDPMAdapter().input_spec
    payload = {
        "objective_version": TABDDPM_TUNING_OBJECTIVE_VERSION,
        "search_space_version": TABDDPM_SEARCH_SPACE_VERSION,
        "phase_a_steps": PHASE_A_STEPS,
        "rerank_version": TABDDPM_RERANK_VERSION,
        "rerank_steps": RERANK_STEPS,
        "frame_sha256": frame_hash,
        "dataset": {
            "name": filtered.name,
            "columns": [
                {
                    "name": column.name,
                    "kind": column.kind.value,
                    "ordered_values": (
                        [repr(value) for value in column.ordered_values]
                        if column.ordered_values is not None
                        else None
                    ),
                }
                for column in filtered.columns
            ],
            "target": filtered.target,
            "task": filtered.task.value if filtered.task is not None else None,
        },
        "input_spec": {
            "continuous_view": spec.continuous_view.value,
            "discrete_view": spec.discrete_view.value,
            "categorical_view": spec.categorical_view.value,
        },
        "holdout": {
            "type": type(config.run.split).__name__,
            **asdict(config.run.split),
        },
        "missing_policy": config.run.missing_policy.value,
        "training_seed": config.run.training_seed,
        "sample_seed": config.run.sample_seed,
        "device": config.run.device,
        "include_wide_profile": config.include_wide_profile,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_or_initialize_study(
    study: optuna.Study,
    fingerprint: str,
) -> None:
    stored = study.user_attrs.get("fingerprint")
    if stored is None:
        if study.trials:
            raise ContractViolation(
                "Cannot resume an unversioned TabDDPM study; start a new study."
            )
        study.set_user_attr("fingerprint", fingerprint)
        study.set_user_attr("objective_version", TABDDPM_TUNING_OBJECTIVE_VERSION)
        study.set_user_attr("search_space_version", TABDDPM_SEARCH_SPACE_VERSION)
    elif stored != fingerprint:
        raise ContractViolation(
            "Refusing to mix TabDDPM trials from a different dataset, semantic "
            "view, split, seed, device, objective, or search space."
        )


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE
    ]


def _phase_a(
    dataset: TabularDataset,
    config: TabDDPMTuningConfig,
    suggest_config: Callable[[optuna.Trial], TabDDPMConfig] | None,
) -> tuple[optuna.Study, str]:
    sampler = optuna.samplers.TPESampler(
        seed=config.sampler_seed,
        n_startup_trials=min(10, config.target_complete_trials),
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=config.load_if_exists,
    )
    fingerprint = _study_fingerprint(dataset, config)
    _validate_or_initialize_study(study, fingerprint)
    running_trials = [
        trial.number
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.RUNNING
    ]
    if running_trials:
        if not config.load_if_exists:
            raise ContractViolation(
                "A new TabDDPM study unexpectedly contains running trials."
            )
        for trial_number in running_trials:
            study.tell(trial_number, state=optuna.trial.TrialState.FAIL)
        study.set_user_attr("recovered_running_trials", running_trials)
    started = monotonic()

    def objective(trial: optuna.Trial) -> float:
        native_config = (
            suggest_config(trial)
            if suggest_config is not None
            else suggest_tabddpm_config(
                trial,
                include_wide_profile=config.include_wide_profile,
            )
        )
        if not isinstance(native_config, TabDDPMConfig):
            raise ContractViolation(
                "suggest_config must return the real TabDDPMConfig type."
            )
        native_config.show_progress = config.show_native_progress
        trial.set_user_attr("native_config", tabddpm_config_payload(native_config))
        trial_run = replace(
            config.run,
            run_id=f"{config.run.run_id}-trial-{trial.number}",
            artifact_dir=config.run.artifact_dir / f"trial-{trial.number}",
        )
        try:
            holdout = run_holdout_trial(
                dataset, lambda: TabDDPMAdapter(native_config), trial_run
            )
        except FoundNANsError as error:
            trial.set_user_attr(
                "failure",
                {
                    "type": "non_finite_sampling_trajectory",
                    "message": str(error),
                },
            )
            raise TabDDPMNumericalTrialError(str(error)) from error
        score = evaluate_tuning_score(
            holdout.dataset,
            holdout.train_raw,
            holdout.validation_raw,
            holdout.synthetic_raw,
        )
        trial.set_user_attr("mean_standardized_wasserstein", score.mean_wasserstein)
        trial.set_user_attr("mean_jensen_shannon", score.mean_jensen_shannon)
        trial.set_user_attr("fit_seconds", holdout.fit_seconds)
        trial.set_user_attr("sample_seconds", holdout.sample_seconds)
        trial.set_user_attr(
            "missing_report", _missing_report_payload(holdout.missing_report)
        )
        trial.set_user_attr(
            "column_scores",
            {
                item.column: {
                    "kind": item.kind.value,
                    "metric": item.metric.value,
                    "value": item.value,
                    "reference_scale": item.reference_scale,
                }
                for item in score.columns
            },
        )
        return score.total

    while len(_completed_trials(study)) < config.target_complete_trials:
        if len(study.trials) >= config.max_total_trials:
            break
        remaining_timeout = None
        if config.timeout_seconds is not None:
            remaining_timeout = config.timeout_seconds - (monotonic() - started)
            if remaining_timeout <= 0:
                break
        study.optimize(
            objective,
            n_trials=1,
            timeout=remaining_timeout,
            gc_after_trial=True,
            show_progress_bar=False,
            catch=(TabDDPMNumericalTrialError,),
        )

    complete = len(_completed_trials(study))
    if complete < config.target_complete_trials:
        raise ContractViolation(
            "TabDDPM Phase A stopped before its successful-trial target: "
            f"complete={complete}, target={config.target_complete_trials}, "
            f"total={len(study.trials)}. Resume the same SQLite study."
        )
    return study, fingerprint


def _distinct_phase_a_leaders(
    study: optuna.Study,
    count: int,
) -> tuple[tuple[optuna.trial.FrozenTrial, TabDDPMConfig], ...]:
    leaders: list[tuple[optuna.trial.FrozenTrial, TabDDPMConfig]] = []
    seen: set[str] = set()
    for trial in sorted(_completed_trials(study), key=lambda item: float(item.value)):
        native = tabddpm_config_from_payload(trial.user_attrs.get("native_config"))
        key = json.dumps(tabddpm_config_payload(native), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        leaders.append((trial, native))
        if len(leaders) == count:
            break
    if not leaders:
        raise ContractViolation("TabDDPM study has no complete candidate to rerank.")
    return tuple(leaders)


def _rerank(
    dataset: TabularDataset,
    config: TabDDPMTuningConfig,
    study: optuna.Study,
) -> tuple[TabDDPMCandidateResult, ...]:
    leaders = _distinct_phase_a_leaders(study, config.rerank_candidates)
    cache_path = (
        config.live_state_dir / "rerank-live.json"
        if config.live_state_dir is not None
        else None
    )
    cached_runs: dict[tuple[int, int, int], TabDDPMRerankRun] = {}
    if cache_path is not None and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != _study_fingerprint(dataset, config):
            raise ContractViolation("Rerank live state has a different fingerprint.")
        for item in payload.get("runs", []):
            run = TabDDPMRerankRun(**item)
            cached_runs[(run.phase_a_trial, run.training_seed, run.sample_seed)] = run

    results = []
    for rank, (trial, phase_a_config) in enumerate(leaders, start=1):
        native = replace(
            phase_a_config,
            steps=RERANK_STEPS,
            show_progress=config.show_native_progress,
        )
        runs = []
        for seed_offset in range(config.rerank_seed_pairs):
            training_seed = config.run.training_seed + seed_offset
            sample_seed = config.run.sample_seed + seed_offset
            cache_key = (trial.number, training_seed, sample_seed)
            if cache_key in cached_runs:
                runs.append(replace(cached_runs[cache_key], candidate_rank=rank))
                continue
            rerank_run = replace(
                config.run,
                run_id=f"{config.run.run_id}-rerank-{rank}-seed-{seed_offset}",
                training_seed=training_seed,
                sample_seed=sample_seed,
                artifact_dir=(
                    config.run.artifact_dir
                    / "rerank"
                    / f"candidate-{rank}"
                    / f"seed-{seed_offset}"
                ),
            )
            holdout = run_holdout_trial(
                dataset, lambda native=native: TabDDPMAdapter(native), rerank_run
            )
            score = evaluate_tuning_score(
                holdout.dataset,
                holdout.train_raw,
                holdout.validation_raw,
                holdout.synthetic_raw,
            )
            run = TabDDPMRerankRun(
                candidate_rank=rank,
                phase_a_trial=trial.number,
                training_seed=training_seed,
                sample_seed=sample_seed,
                score=score.total,
                mean_standardized_wasserstein=score.mean_wasserstein,
                mean_jensen_shannon=score.mean_jensen_shannon,
                fit_seconds=holdout.fit_seconds,
                sample_seconds=holdout.sample_seconds,
            )
            runs.append(run)
            cached_runs[cache_key] = run
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                live_payload = {
                    "artifact_type": "tabddpm_rerank_live",
                    "artifact_version": TABDDPM_RERANK_VERSION,
                    "fingerprint": _study_fingerprint(dataset, config),
                    "runs": [asdict(item) for item in cached_runs.values()],
                }
                temporary_path = cache_path.with_suffix(".tmp")
                temporary_path.write_text(
                    json.dumps(live_payload, indent=2, sort_keys=True, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(cache_path)
        values = [run.score for run in runs]
        results.append(
            TabDDPMCandidateResult(
                candidate_rank=rank,
                phase_a_trial=trial.number,
                config=native,
                runs=tuple(runs),
                mean_score=float(sum(values) / len(values)),
                std_score=float(
                    math.sqrt(
                        sum(
                            (value - sum(values) / len(values)) ** 2 for value in values
                        )
                        / len(values)
                    )
                ),
            )
        )
    return tuple(results)


def tune_tabddpm(
    dataset: TabularDataset,
    config: TabDDPMTuningConfig,
    *,
    suggest_config: Callable[[optuna.Trial], TabDDPMConfig] | None = None,
) -> TabDDPMTuningResult:
    """Run resumable Phase A and high-budget two-seed candidate reranking."""

    if not isinstance(dataset, TabularDataset):
        raise ContractViolation("dataset must be TabularDataset.")
    if not isinstance(config, TabDDPMTuningConfig):
        raise ContractViolation("config must be TabDDPMTuningConfig.")
    study, fingerprint = _phase_a(dataset, config, suggest_config)
    candidates = _rerank(dataset, config, study)
    best = min(candidates, key=lambda item: (item.mean_score, item.std_score))
    return TabDDPMTuningResult(
        study=study,
        config=config,
        dataset=dataset,
        candidates=candidates,
        best_config=best.config,
        best_score=best.mean_score,
        fingerprint=fingerprint,
    )


def write_tabddpm_tuning_artifacts(
    result: TabDDPMTuningResult,
    output_dir: Path,
) -> Path:
    """Write complete Phase-A and rerank evidence to a create-only directory."""

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractViolation(
            f"Artifact directory already exists: {output_dir}."
        ) from error
    (output_dir / "best-config.json").write_text(
        json.dumps(tabddpm_config_payload(result.best_config), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in result.study.trials
    ]
    (output_dir / "phase-a-trials.json").write_text(
        json.dumps(trials, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rerank = [
        {
            "candidate_rank": item.candidate_rank,
            "phase_a_trial": item.phase_a_trial,
            "config": tabddpm_config_payload(item.config),
            "mean_score": item.mean_score,
            "std_score": item.std_score,
            "runs": [asdict(run) for run in item.runs],
        }
        for item in result.candidates
    ]
    (output_dir / "rerank.json").write_text(
        json.dumps(rerank, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact_type": "tabddpm_tuning",
        "artifact_version": TABDDPM_TUNING_ARTIFACT_VERSION,
        "status": "complete",
        "dataset": result.dataset.name,
        "fingerprint": result.fingerprint,
        "objective_version": TABDDPM_TUNING_OBJECTIVE_VERSION,
        "search_space_version": TABDDPM_SEARCH_SPACE_VERSION,
        "rerank_version": TABDDPM_RERANK_VERSION,
        "phase_a_steps": PHASE_A_STEPS,
        "rerank_steps": RERANK_STEPS,
        "completed_phase_a_trials": len(_completed_trials(result.study)),
        "total_phase_a_trials": len(result.study.trials),
        "best_rerank_score": result.best_score,
        "storage_configured": result.config.storage is not None,
        "files": {
            "best_config": "best-config.json",
            "phase_a_trials": "phase-a-trials.json",
            "rerank": "rerank.json",
        },
    }
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
