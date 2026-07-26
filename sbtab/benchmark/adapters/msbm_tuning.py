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

import optuna

from sbtab.benchmark.adapters.msbm import MSBMAdapter
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.runner import (
    HoldoutRunConfig,
    run_holdout_trial,
)
from sbtab.benchmark.validation import ContractViolation
from sbtab.evaluation import evaluate_tuning_score
from sbtab.solvers.msbm import MixedSBMConfig


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
    best_config:
        Complete ``MixedSBMConfig`` recorded by the best completed trial.
        Final K-fold adapters receive this configuration directly.
    best_score:
        Minimized common tuning objective for ``best_config``.
    """

    study: optuna.Study
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


def _config_payload(config: MixedSBMConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["fb_sequence"] = list(config.fb_sequence)
    return payload


def _config_from_payload(payload: object) -> MixedSBMConfig:
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
    try:
        return MixedSBMConfig(**values)
    except TypeError as error:
        raise ContractViolation(
            "Best MSBM trial native_config does not match current "
            "MixedSBMConfig."
        ) from error


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
    Trial failures are not converted to an infinite score: precise adapter or
    model exceptions remain visible to Optuna and the caller.
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
        holdout = run_holdout_trial(
            dataset,
            lambda: MSBMAdapter(native_config),
            trial_run,
        )
        score = evaluate_tuning_score(
            holdout.dataset,
            holdout.validation_raw,
            holdout.synthetic_raw,
        )
        trial.set_user_attr("native_config", _config_payload(native_config))
        trial.set_user_attr("mean_wasserstein", score.mean_wasserstein)
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
                }
                for item in score.columns
            },
        )
        trial.set_user_attr("fit_seconds", holdout.fit_seconds)
        trial.set_user_attr("sample_seconds", holdout.sample_seconds)
        return score.total

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout_seconds,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    best_trial = study.best_trial
    best_config = _config_from_payload(
        best_trial.user_attrs.get("native_config")
    )
    return MSBMTuningResult(
        study=study,
        best_config=best_config,
        best_score=float(best_trial.value),
    )
