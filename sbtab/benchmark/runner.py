"""Final cross-validation orchestration for a fixed generator configuration.

The runner owns experiment order, not model behavior. It applies one missing
policy, creates common raw folds, constructs a fresh codec and adapter per
fold or holdout trial, and returns decoded tables ready for model-independent
evaluation. Hyperparameter search, metric calculation, and artifact
serialization remain outside shared orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from sbtab.benchmark.adapter import (
    ModelAdapter,
    RunContext,
    validate_adapter_definition,
    validate_sample_request,
)
from sbtab.benchmark.codec import compile_codec
from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.missing import (
    MissingPolicy,
    MissingReport,
    apply_missing_policy,
)
from sbtab.benchmark.splitting import (
    FoldSplit,
    HoldoutConfig,
    HoldoutSplit,
    SplitConfig,
    StratifiedHoldoutConfig,
    make_holdout,
    make_splits,
)
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_prepared_table,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Model-independent controls for one final cross-validation run.

    Parameters
    ----------
    split:
        Final K-fold strategy shared by every model in the comparison. Tuning
        holdout configuration does not belong here.
    missing_policy:
        One global policy applied across modeled columns before folds are
        created. ``ERROR`` is the safe default; the official v1 comparison
        explicitly selects ``COMPLETE_CASE``.
    run_id:
        Stable label forwarded to each adapter context for logs and artifacts.
        Shared code and adapters must not branch on its value.
    training_seed:
        Base native initialization/training seed. Fold ``i`` receives
        ``training_seed + i``.
    sample_seed:
        Base generation seed. Fold ``i`` receives ``sample_seed + i``.
    device:
        Native execution-device label forwarded unchanged to each adapter.
    artifact_dir:
        Run-level artifact root. Each adapter receives its fold-specific child
        path. This runner does not create directories or serialize artifacts.
    """

    split: SplitConfig
    missing_policy: MissingPolicy = MissingPolicy.ERROR
    run_id: str = "benchmark"
    training_seed: int = 42
    sample_seed: int = 10_042
    device: str = "cpu"
    artifact_dir: Path = Path("artifacts")

    def __post_init__(self) -> None:
        """Reject invalid base controls before any fold starts training."""

        _validate_base_controls(
            owner="BenchmarkConfig",
            training_seed=self.training_seed,
            sample_seed=self.sample_seed,
            artifact_dir=self.artifact_dir,
        )


@dataclass(frozen=True)
class HoldoutRunConfig:
    """Model-independent controls for one fixed-configuration tuning trial.

    Parameters
    ----------
    split:
        Reference train/validation strategy. The approved defaults are 80/20
        with seed 5; classification uses ``StratifiedHoldoutConfig``.
    missing_policy:
        One global policy applied before the reference split. A model-owned
        tuner cannot override it per trial.
    run_id:
        Trial label forwarded to the adapter context. Shared code must not
        interpret model or hyperparameter semantics from it.
    training_seed, sample_seed:
        Explicit native seeds for this trial's single fit and sample calls.
    device:
        Native execution-device label forwarded unchanged to the adapter.
    artifact_dir:
        Trial artifact root assigned to the adapter. This runner does not
        create or serialize it.
    """

    split: HoldoutConfig | StratifiedHoldoutConfig
    missing_policy: MissingPolicy = MissingPolicy.ERROR
    run_id: str = "tuning"
    training_seed: int = 42
    sample_seed: int = 10_042
    device: str = "cpu"
    artifact_dir: Path = Path("artifacts")

    def __post_init__(self) -> None:
        """Reject invalid base controls before the trial starts training."""

        _validate_base_controls(
            owner="HoldoutRunConfig",
            training_seed=self.training_seed,
            sample_seed=self.sample_seed,
            artifact_dir=self.artifact_dir,
        )


@dataclass(frozen=True)
class FoldResult:
    """Decoded data and native timings produced by one final benchmark fold.

    Parameters
    ----------
    split:
        Positional membership of this fold in the post-policy dataset.
    train_raw, test_raw:
        Raw modeled columns in canonical order. The identifier is excluded.
        ``test_raw`` bypassed both the model codec and adapter.
    synthetic_raw:
        Decoded generated table in the same modeled-column order. Its row count
        equals ``len(train_raw)`` under the final comparison protocol.
    fit_seconds, sample_seconds:
        Wall-clock durations of adapter ``fit`` and ``sample`` calls. They do
        not include missing handling, splitting, codec fitting, decoding, or
        future evaluation.
    """

    split: FoldSplit
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    synthetic_raw: pd.DataFrame
    fit_seconds: float
    sample_seconds: float


@dataclass(frozen=True)
class CrossValidationResult:
    """Complete pre-evaluation output for one adapter and common fold set.

    Parameters
    ----------
    adapter_name:
        Stable name reported by every fresh adapter instance.
    config:
        Controls used to construct all fold contexts and sample requests.
    dataset:
        Post-policy dataset to which every stored position refers. It retains a
        declared identifier for auditing even though fold tables exclude it.
    missing_report:
        Auditable effect of the one global missing-policy application.
    folds:
        Fold outputs in increasing ``fold_id`` order.
    """

    adapter_name: str
    config: BenchmarkConfig
    dataset: TabularDataset
    missing_report: MissingReport
    folds: tuple[FoldResult, ...]


@dataclass(frozen=True)
class HoldoutResult:
    """Decoded pre-evaluation output of one fixed-configuration tuning trial.

    Parameters
    ----------
    adapter_name:
        Stable model-family name reported by the fresh adapter.
    config:
        Reference split and runtime controls used for this trial.
    dataset, missing_report:
        Post-policy dataset and evidence from the one global filtering pass.
    split:
        Positional train/validation membership in ``dataset``.
    train_raw, validation_raw:
        Raw modeled tables. Validation bypasses codec and adapter entirely.
    synthetic_raw:
        Decoded generated table with exactly ``len(validation_raw)`` rows.
    fit_seconds, sample_seconds:
        Wall-clock native adapter durations, excluding shared preprocessing,
        decoding, and future objective calculation.
    """

    adapter_name: str
    config: HoldoutRunConfig
    dataset: TabularDataset
    missing_report: MissingReport
    split: HoldoutSplit
    train_raw: pd.DataFrame
    validation_raw: pd.DataFrame
    synthetic_raw: pd.DataFrame
    fit_seconds: float
    sample_seconds: float


def _validate_base_controls(
    *,
    owner: str,
    training_seed: int,
    sample_seed: int,
    artifact_dir: Path,
) -> None:
    for field_name, value in (
        ("training_seed", training_seed),
        ("sample_seed", sample_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractViolation(f"{owner}.{field_name} must be an integer.")
        if not 0 <= value < 2**32:
            raise ContractViolation(f"{owner}.{field_name} must be in [0, 2**32).")
    if not isinstance(artifact_dir, Path):
        raise ContractViolation(f"{owner}.artifact_dir must be pathlib.Path.")


def _modeled_partition(
    dataset: TabularDataset,
    positions: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = dataset.frame.iloc[list(positions)].copy()
    modeled = source.loc[:, list(dataset.column_order)].reset_index(drop=True)
    return source, modeled


def _fit_sample_decode(
    *,
    dataset: TabularDataset,
    adapter: ModelAdapter,
    train_positions: tuple[int, ...],
    context: RunContext,
    n_samples: int,
    sample_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    train_source, train_raw = _modeled_partition(dataset, train_positions)
    codec = compile_codec(dataset, adapter.input_spec)
    train_prepared = codec.fit_transform(train_source)

    fit_started = perf_counter()
    adapter.fit(train_prepared, context)
    fit_seconds = perf_counter() - fit_started

    validate_sample_request(n_samples, sample_seed)
    sample_started = perf_counter()
    sample_prepared = adapter.sample(n_samples, sample_seed)
    sample_seconds = perf_counter() - sample_started
    validate_prepared_table(sample_prepared, expected_rows=n_samples)
    synthetic_raw = codec.inverse_transform(sample_prepared)
    return train_raw, synthetic_raw, fit_seconds, sample_seconds


def run_cross_validation(
    dataset: TabularDataset,
    adapter_factory: Callable[[], ModelAdapter],
    config: BenchmarkConfig,
) -> CrossValidationResult:
    """Fit, sample, and decode every final fold for one fixed adapter config.

    ``adapter_factory`` must return a new unfitted instance on every call.
    Generated row count is the train-fold size. Held-out raw rows are copied
    into the result for later evaluation but never passed to the codec or
    adapter.
    """

    if not callable(adapter_factory):
        raise ContractViolation("adapter_factory must be callable.")
    if not isinstance(config, BenchmarkConfig):
        raise ContractViolation("config must be BenchmarkConfig.")

    missing_result = apply_missing_policy(dataset, config.missing_policy)
    run_dataset = missing_result.dataset
    splits = make_splits(run_dataset, config.split)

    fold_results: list[FoldResult] = []
    adapters: list[ModelAdapter] = []
    adapter_name: str | None = None
    for split in splits:
        adapter = adapter_factory()
        validate_adapter_definition(adapter)
        if any(adapter is previous for previous in adapters):
            raise ContractViolation(
                "adapter_factory must return a fresh adapter instance per fold."
            )
        adapters.append(adapter)
        if adapter_name is None:
            adapter_name = adapter.name
        elif adapter.name != adapter_name:
            raise ContractViolation(
                "adapter_factory returned inconsistent model names: "
                f"{adapter_name!r} and {adapter.name!r}."
            )

        _, test_raw = _modeled_partition(
            run_dataset,
            split.test_positions,
        )
        context = RunContext(
            run_id=config.run_id,
            fold_id=split.fold_id,
            seed=config.training_seed + split.fold_id,
            device=config.device,
            artifact_dir=config.artifact_dir / f"fold-{split.fold_id}",
        )
        train_raw, synthetic_raw, fit_seconds, sample_seconds = (
            _fit_sample_decode(
                dataset=run_dataset,
                adapter=adapter,
                train_positions=split.train_positions,
                context=context,
                n_samples=len(split.train_positions),
                sample_seed=config.sample_seed + split.fold_id,
            )
        )

        fold_results.append(
            FoldResult(
                split=split,
                train_raw=train_raw,
                test_raw=test_raw,
                synthetic_raw=synthetic_raw,
                fit_seconds=fit_seconds,
                sample_seconds=sample_seconds,
            )
        )

    if adapter_name is None:
        raise ContractViolation("Cross-validation produced no folds.")
    return CrossValidationResult(
        adapter_name=adapter_name,
        config=config,
        dataset=run_dataset,
        missing_report=missing_result.report,
        folds=tuple(fold_results),
    )


def run_holdout_trial(
    dataset: TabularDataset,
    adapter_factory: Callable[[], ModelAdapter],
    config: HoldoutRunConfig,
) -> HoldoutResult:
    """Execute one reference holdout trial for an already fixed model config.

    The function has no Optuna dependency and knows no model hyperparameters.
    A model-owned tuner creates ``adapter_factory`` for each candidate config
    and evaluates the returned raw validation and synthetic tables.
    """

    if not callable(adapter_factory):
        raise ContractViolation("adapter_factory must be callable.")
    if not isinstance(config, HoldoutRunConfig):
        raise ContractViolation("config must be HoldoutRunConfig.")

    missing_result = apply_missing_policy(dataset, config.missing_policy)
    run_dataset = missing_result.dataset
    split = make_holdout(run_dataset, config.split)
    adapter = adapter_factory()
    validate_adapter_definition(adapter)

    _, validation_raw = _modeled_partition(
        run_dataset,
        split.validation_positions,
    )
    context = RunContext(
        run_id=config.run_id,
        fold_id=0,
        seed=config.training_seed,
        device=config.device,
        artifact_dir=config.artifact_dir,
    )
    train_raw, synthetic_raw, fit_seconds, sample_seconds = _fit_sample_decode(
        dataset=run_dataset,
        adapter=adapter,
        train_positions=split.train_positions,
        context=context,
        n_samples=len(validation_raw),
        sample_seed=config.sample_seed,
    )
    return HoldoutResult(
        adapter_name=adapter.name,
        config=config,
        dataset=run_dataset,
        missing_report=missing_result.report,
        split=split,
        train_raw=train_raw,
        validation_raw=validation_raw,
        synthetic_raw=synthetic_raw,
        fit_seconds=fit_seconds,
        sample_seconds=sample_seconds,
    )
