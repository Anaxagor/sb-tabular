"""Model-owned Optuna study for selecting one fixed MSBM configuration.

This module owns only MSBM hyperparameters and Optuna lifecycle. It delegates
the reference split, fold-local codec, fit/sample orchestration, raw decoding,
and semantic objective to shared benchmark components. Neither
``MSBMAdapter.fit`` nor the native solver imports Optuna.

The default search space is provisional pending model-owner review. It includes
only fields consumed by the current ``MixedSBMConfig`` implementation; notably,
the unused native ``eps`` field is not tuned.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import optuna

from sbtab.benchmark.adapters.msbm import MSBMAdapter
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.runner import (
    HoldoutRunConfig,
    run_holdout_trial,
)
from sbtab.benchmark.missing import MissingReport
from sbtab.benchmark.validation import ContractViolation
from sbtab.bridge.reference import InvalidCategoricalProbabilitiesError
from sbtab.evaluation import evaluate_tuning_score
from sbtab.solvers.msbm import (
    CategoricalLossNormalization,
    MixedSBMConfig,
)


MSBM_TUNING_ARTIFACT_VERSION = 3
MSBM_TUNING_OBJECTIVE_VERSION = 2


@dataclass(frozen=True)
class MSBMTuningConfig:
    """Controls for one model-owned MSBM study.

    Parameters
    ----------
    run:
        Common reference holdout and runtime controls reused for every trial.
        Trial-specific paths and labels are derived without changing split or
        random seeds.
    n_trials:
        Number of new Optuna trials requested by this call.
    sampler_seed:
        Seed for Optuna's TPE parameter proposal sequence. It does not replace
        model training or sampling seeds in ``run``.
    timeout_seconds:
        Optional wall-clock limit passed to ``Study.optimize``. ``None`` means
        no timeout.
    study_name, storage, load_if_exists:
        Native Optuna persistence controls. ``storage=None`` creates an
        in-memory study. Remote storage configuration is supplied by the human
        caller and is not inferred by benchmark code.
    """

    run: HoldoutRunConfig
    n_trials: int = 50
    sampler_seed: int = 5
    timeout_seconds: float | None = None
    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = False

    def __post_init__(self) -> None:
        """Reject impossible study controls before creating Optuna state."""

        if not isinstance(self.run, HoldoutRunConfig):
            raise ContractViolation(
                "MSBMTuningConfig.run must be HoldoutRunConfig."
            )
        if isinstance(self.n_trials, bool) or not isinstance(self.n_trials, int):
            raise ContractViolation("MSBMTuningConfig.n_trials must be an integer.")
        if self.n_trials < 1:
            raise ContractViolation("MSBMTuningConfig.n_trials must be positive.")
        if isinstance(self.sampler_seed, bool) or not isinstance(
            self.sampler_seed,
            int,
        ):
            raise ContractViolation(
                "MSBMTuningConfig.sampler_seed must be an integer."
            )
        if not 0 <= self.sampler_seed < 2**32:
            raise ContractViolation(
                "MSBMTuningConfig.sampler_seed must be in [0, 2**32)."
            )
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds,
                (int, float),
            ):
                raise ContractViolation(
                    "MSBMTuningConfig.timeout_seconds must be a number or None."
                )
            if self.timeout_seconds <= 0:
                raise ContractViolation(
                    "MSBMTuningConfig.timeout_seconds must be positive."
                )


@dataclass(frozen=True)
class MSBMTuningResult:
    """Completed study and the reconstructed native configuration it selected.

    Parameters
    ----------
    study:
        Native Optuna study containing trial states, parameters, and evidence.
    config:
        Study, holdout, seed, device, and persistence controls used by the run.
    dataset:
        Raw declared dataset used by every trial. It is retained for artifact
        schema metadata and treated as read-only.
    best_config:
        Complete ``MixedSBMConfig`` recorded by the best completed trial.
        Final K-fold adapters receive this configuration directly.
    best_score:
        Minimized common tuning objective for ``best_config``.
    """

    study: optuna.Study
    config: MSBMTuningConfig
    dataset: TabularDataset
    best_config: MixedSBMConfig
    best_score: float


def suggest_msbm_config(trial: optuna.Trial) -> MixedSBMConfig:
    """Suggest the provisional search space for the current native solver.

    The alternating sequence always begins and ends with a backward stage so
    native sampling has a backward snapshot. ``device`` and ``seed`` are
    placeholders overwritten by ``RunContext`` inside each adapter instance.
    """

    imf_len = trial.suggest_int("imf_len", 3, 9, step=2)
    fb_sequence = tuple(
        "b" if index % 2 == 0 else "f" for index in range(imf_len)
    )
    return MixedSBMConfig(
        fb_sequence=fb_sequence,
        cat_emb_dim=trial.suggest_int("cat_emb_dim", 8, 32),
        hidden_dim=trial.suggest_categorical(
            "hidden_dim",
            [128, 256, 512],
        ),
        time_dim=trial.suggest_int("time_dim", 32, 128, step=32),
        n_layers=trial.suggest_int("n_layers", 2, 6),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        num_steps=trial.suggest_int("num_steps", 20, 100, step=10),
        sigma=trial.suggest_float("sigma", 0.01, 1.0, log=True),
        lambda_num=trial.suggest_float("lambda_num", 0.1, 1.0),
        lambda_cat=trial.suggest_float("lambda_cat", 0.1, 1.0),
        lr=trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        batch_size=trial.suggest_categorical(
            "batch_size",
            [128, 256, 512],
        ),
        epochs_per_direction=trial.suggest_int(
            "epochs_per_direction",
            5,
            20,
        ),
        grad_clip=trial.suggest_float("grad_clip", 0.1, 1.0),
        device="cpu",
        seed=0,
    )


def msbm_config_payload(config: MixedSBMConfig) -> dict[str, object]:
    """Serialize one complete native config into a JSON-compatible mapping."""

    payload = asdict(config)
    payload["fb_sequence"] = list(config.fb_sequence)
    payload["categorical_loss_normalization"] = (
        config.categorical_loss_normalization.value
    )
    return payload


def msbm_config_from_payload(payload: object) -> MixedSBMConfig:
    """Reconstruct a native config, including older artifacts with new defaults."""

    if not isinstance(payload, Mapping):
        raise ContractViolation(
            "Best MSBM trial has no reconstructable native_config artifact."
        )
    values = dict(payload)
    sequence = values.get("fb_sequence")
    if not isinstance(sequence, (list, tuple)):
        raise ContractViolation(
            "Best MSBM trial native_config has invalid fb_sequence."
        )
    values["fb_sequence"] = tuple(sequence)
    normalization = values.get("categorical_loss_normalization")
    if normalization is not None:
        try:
            values["categorical_loss_normalization"] = (
                CategoricalLossNormalization(normalization)
            )
        except ValueError as error:
            raise ContractViolation(
                "Best MSBM trial native_config has invalid categorical loss "
                f"normalization: {normalization!r}."
            ) from error
    try:
        return MixedSBMConfig(**values)
    except TypeError as error:
        raise ContractViolation(
            "Best MSBM trial native_config does not match current "
            "MixedSBMConfig."
        ) from error


def _missing_report_payload(report: MissingReport) -> dict[str, object]:
    def class_counts(counts):
        if counts is None:
            return None
        return [
            {
                "label_repr": repr(item.label),
                "label_type": type(item.label).__name__,
                "count": item.count,
            }
            for item in counts
        ]

    return {
        "policy": report.policy.value,
        "rows_before": report.rows_before,
        "rows_after": report.rows_after,
        "dropped_count": report.dropped_count,
        "dropped_fraction": report.dropped_fraction,
        "missing_by_column": dict(report.missing_by_column),
        "class_counts_before": class_counts(report.class_counts_before),
        "class_counts_after": class_counts(report.class_counts_after),
    }


def tune_msbm(
    dataset: TabularDataset,
    config: MSBMTuningConfig,
    *,
    suggest_config: Callable[[optuna.Trial], MixedSBMConfig] = (
        suggest_msbm_config
    ),
) -> MSBMTuningResult:
    """Run model-owned Optuna trials and return one frozen native config.

    ``suggest_config`` is injectable for narrow tests and explicitly reviewed
    alternative MSBM profiles. It must return the real native config type.
    Invalid categorical probabilities mark only their numerically unstable
    trial as failed. They are not converted to an infinite score or repaired.
    Other adapter and model exceptions remain visible to Optuna and the caller
    and stop the study.
    """

    if not isinstance(dataset, TabularDataset):
        raise ContractViolation("dataset must be TabularDataset.")
    if not isinstance(config, MSBMTuningConfig):
        raise ContractViolation("config must be MSBMTuningConfig.")
    if not callable(suggest_config):
        raise ContractViolation("suggest_config must be callable.")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.sampler_seed),
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=config.load_if_exists,
    )
    objective_version = study.user_attrs.get("objective_version")
    if objective_version is None:
        if study.trials:
            raise ContractViolation(
                "Cannot resume an MSBM study without objective_version: its "
                "existing trial values may use the obsolete raw-scale "
                "Wasserstein objective. Start a new study."
            )
        study.set_user_attr(
            "objective_version",
            MSBM_TUNING_OBJECTIVE_VERSION,
        )
    elif objective_version != MSBM_TUNING_OBJECTIVE_VERSION:
        raise ContractViolation(
            "Cannot resume an MSBM study with a different tuning objective: "
            f"stored={objective_version!r}, "
            f"required={MSBM_TUNING_OBJECTIVE_VERSION!r}."
        )

    def objective(trial: optuna.Trial) -> float:
        native_config = suggest_config(trial)
        if not isinstance(native_config, MixedSBMConfig):
            raise ContractViolation(
                "MSBM suggest_config must return MixedSBMConfig."
            )
        trial_run = replace(
            config.run,
            run_id=f"{config.run.run_id}-trial-{trial.number}",
            artifact_dir=config.run.artifact_dir / f"trial-{trial.number}",
        )
        trial.set_user_attr(
            "native_config",
            msbm_config_payload(native_config),
        )
        try:
            holdout = run_holdout_trial(
                dataset,
                lambda: MSBMAdapter(native_config),
                trial_run,
            )
        except InvalidCategoricalProbabilitiesError as error:
            trial.set_user_attr(
                "failure",
                {
                    "type": "invalid_categorical_probabilities",
                    "message": str(error),
                },
            )
            raise
        score = evaluate_tuning_score(
            holdout.dataset,
            holdout.train_raw,
            holdout.validation_raw,
            holdout.synthetic_raw,
        )
        trial.set_user_attr(
            "mean_standardized_wasserstein",
            score.mean_wasserstein,
        )
        trial.set_user_attr(
            "mean_jensen_shannon",
            score.mean_jensen_shannon,
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
        trial.set_user_attr("fit_seconds", holdout.fit_seconds)
        trial.set_user_attr("sample_seconds", holdout.sample_seconds)
        trial.set_user_attr(
            "missing_report",
            _missing_report_payload(holdout.missing_report),
        )
        return score.total

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout_seconds,
        gc_after_trial=True,
        show_progress_bar=False,
        catch=(InvalidCategoricalProbabilitiesError,),
    )
    if not any(
        trial.state is optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    ):
        raise ContractViolation(
            "MSBM tuning finished without a successful trial; inspect failed "
            "trial evidence before changing the search space."
        )
    best_trial = study.best_trial
    best_config = msbm_config_from_payload(
        best_trial.user_attrs.get("native_config")
    )
    return MSBMTuningResult(
        study=study,
        config=config,
        dataset=dataset,
        best_config=best_config,
        best_score=float(best_trial.value),
    )


def _run_payload(config: HoldoutRunConfig) -> dict[str, object]:
    return {
        "split": {
            "type": type(config.split).__name__,
            **asdict(config.split),
        },
        "missing_policy": config.missing_policy.value,
        "run_id": config.run_id,
        "training_seed": config.training_seed,
        "sample_seed": config.sample_seed,
        "device": config.device,
        "artifact_dir": str(config.artifact_dir),
    }


def write_msbm_tuning_artifacts(
    result: MSBMTuningResult,
    output_dir: Path,
) -> Path:
    """Create a local review directory for one completed MSBM study.

    The directory is create-only. It stores the complete best native config,
    every Optuna trial's parameters and user evidence, and a manifest written
    last. The storage URI is never persisted because it may contain
    credentials.
    """

    if not isinstance(result, MSBMTuningResult):
        raise ContractViolation("result must be MSBMTuningResult.")
    if not isinstance(output_dir, Path):
        raise ContractViolation("output_dir must be pathlib.Path.")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractViolation(
            f"Artifact directory already exists: {output_dir}."
        ) from error

    best_config_path = output_dir / "best-config.json"
    best_config_path.write_text(
        json.dumps(
            msbm_config_payload(result.best_config),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    trial_payloads = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in result.study.trials
    ]
    trials_path = output_dir / "trials.json"
    trials_path.write_text(
        json.dumps(
            trial_payloads,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    tuning_config = result.config
    manifest = {
        "artifact_type": "msbm_tuning",
        "artifact_version": MSBM_TUNING_ARTIFACT_VERSION,
        "dataset": {
            "name": result.dataset.name,
            "target": result.dataset.target,
            "task": (
                result.dataset.task.value
                if result.dataset.task is not None
                else None
            ),
            "columns": [
                {"name": column.name, "kind": column.kind.value}
                for column in result.dataset.columns
            ],
        },
        "study": {
            "name": result.study.study_name,
            "direction": result.study.direction.name,
            "objective_version": result.study.user_attrs.get(
                "objective_version"
            ),
            "sampler": type(result.study.sampler).__name__,
            "sampler_seed": tuning_config.sampler_seed,
            "requested_trials": tuning_config.n_trials,
            "completed_trials": sum(
                trial.state is optuna.trial.TrialState.COMPLETE
                for trial in result.study.trials
            ),
            "timeout_seconds": tuning_config.timeout_seconds,
            "storage_configured": tuning_config.storage is not None,
            "load_if_exists": tuning_config.load_if_exists,
        },
        "holdout_run": _run_payload(tuning_config.run),
        "missing_report": result.study.best_trial.user_attrs.get(
            "missing_report"
        ),
        "best_trial": result.study.best_trial.number,
        "best_score": result.best_score,
        "best_config_path": best_config_path.name,
        "trials_path": trials_path.name,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path
