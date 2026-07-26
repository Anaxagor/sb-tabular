"""Reproducible model-independent split strategies for raw benchmark rows.

Splitters operate on positional row indices after the global missing policy.
They never fit preprocessing and never inspect a model or adapter. Classification
stratification uses the declared raw target solely to construct common
partitions; the generator still receives only its train partition.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sbtab.benchmark.contracts import ColumnKind, TabularDataset, TaskType
from sbtab.benchmark.validation import ContractViolation, validate_tabular_dataset


@dataclass(frozen=True)
class HoldoutConfig:
    """Shuffled train/validation strategy for model-family tuning.

    Parameters
    ----------
    validation_fraction:
        Fraction of post-policy rows assigned to validation. It must be
        strictly between zero and one. The reference protocol uses ``0.2``.
    seed:
        Seed used only to choose validation membership. The reference tuning
        protocol uses ``5``.
    """

    validation_fraction: float = 0.2
    seed: int = 5


@dataclass(frozen=True)
class StratifiedHoldoutConfig:
    """Target-stratified train/validation strategy for classification.

    The split keeps every observed target class in both partitions. It is the
    reference tuning strategy for classification datasets.

    Parameters
    ----------
    validation_fraction:
        Fraction of post-policy rows assigned to validation. It must be
        strictly between zero and one and leave room for every class in both
        partitions.
    seed:
        Seed used only to choose validation membership. The reference tuning
        protocol uses ``5``.
    """

    validation_fraction: float = 0.2
    seed: int = 5


@dataclass(frozen=True)
class KFoldConfig:
    """Deterministic shuffled K-fold strategy for non-stratified comparisons.

    Parameters
    ----------
    n_splits:
        Number of non-overlapping held-out folds. It must be at least two and no
        greater than the number of post-policy rows.
    seed:
        Seed used only for the common row permutation before assigning folds.
    """

    n_splits: int = 5
    seed: int = 42


@dataclass(frozen=True)
class StratifiedKFoldConfig:
    """Deterministic target-stratified K-fold strategy for classification.

    Every target class must contain at least ``n_splits`` post-policy rows so
    each held-out fold can represent every class.
    """

    n_splits: int = 5
    seed: int = 42


SplitConfig = KFoldConfig | StratifiedKFoldConfig


@dataclass(frozen=True)
class FoldSplit:
    """Immutable positional train/test partition for one benchmark fold."""

    fold_id: int
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]


@dataclass(frozen=True)
class HoldoutSplit:
    """Immutable positional train/validation partition used for tuning.

    Parameters
    ----------
    train_positions:
        Positions into the post-policy raw frame used to fit one trial.
    validation_positions:
        Disjoint positions used only by the tuning evaluator. Generator code
        must not fit or infer on these rows.
    """

    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractViolation("split seed must be an integer.")
    if not 0 <= seed < 2**32:
        raise ContractViolation("split seed must be in the range [0, 2**32).")


def _validate_kfold_config(config: SplitConfig, n_rows: int) -> None:
    if isinstance(config.n_splits, bool) or not isinstance(config.n_splits, int):
        raise ContractViolation("split n_splits must be an integer.")
    if config.n_splits < 2:
        raise ContractViolation("split n_splits must be at least two.")
    if config.n_splits > n_rows:
        raise ContractViolation(
            f"split n_splits={config.n_splits} exceeds row count {n_rows}."
        )
    _validate_seed(config.seed)


def _validate_holdout_config(
    config: HoldoutConfig | StratifiedHoldoutConfig,
    n_rows: int,
) -> tuple[int, int]:
    fraction = config.validation_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ContractViolation("validation_fraction must be a real number.")
    if not 0.0 < float(fraction) < 1.0:
        raise ContractViolation(
            "validation_fraction must be strictly between zero and one."
        )
    _validate_seed(config.seed)

    validation_rows = int(np.ceil(n_rows * float(fraction)))
    train_rows = n_rows - validation_rows
    if train_rows == 0:
        raise ContractViolation(
            "holdout split must leave at least one post-policy training row."
        )
    return train_rows, validation_rows


def _validate_ready_for_split(dataset: TabularDataset) -> None:
    missing = {
        name: int(dataset.frame[name].isna().sum())
        for name in dataset.column_order
        if dataset.frame[name].isna().any()
    }
    if missing:
        raise ContractViolation(
            "Dataset still contains modeled missing values before split: "
            f"{missing!r}. Apply one benchmark MissingPolicy first."
        )


def _folds_from_sklearn(
    raw_splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[FoldSplit, ...]:
    folds: list[FoldSplit] = []
    for fold_id, (train_positions, test_positions) in enumerate(raw_splits):
        folds.append(
            FoldSplit(
                fold_id=fold_id,
                train_positions=tuple(int(value) for value in train_positions),
                test_positions=tuple(int(value) for value in test_positions),
            )
        )
    return tuple(folds)


def _classification_target(
    dataset: TabularDataset,
) -> tuple[np.ndarray, pd.Series]:
    if dataset.task is not TaskType.CLASSIFICATION or dataset.target is None:
        raise ContractViolation(
            "Stratified split requires a declared classification target."
        )
    if dataset.column(dataset.target).kind is ColumnKind.CONTINUOUS:
        raise ContractViolation(
            "Stratified classification target must be discrete or categorical, "
            f"got continuous target {dataset.target!r}."
        )

    target = dataset.frame[dataset.target]
    class_counts = target.value_counts(dropna=False, sort=False)
    observed_class_count = sum(int(count) > 0 for count in class_counts)
    if observed_class_count < 2:
        raise ContractViolation(
            "Stratified classification requires at least two observed target "
            f"classes after the missing policy, got {observed_class_count}."
        )

    target_codes, _ = pd.factorize(target, sort=False)
    return target_codes, class_counts


def make_holdout(
    dataset: TabularDataset,
    config: HoldoutConfig | StratifiedHoldoutConfig,
) -> HoldoutSplit:
    """Create the deterministic train/validation split used for tuning.

    The caller applies the global missing policy before this function.
    ``StratifiedHoldoutConfig`` additionally requires a finite-state
    classification target and keeps every observed class in both partitions.
    Returned values are positions into the post-policy frame, never pandas
    index labels.
    """

    validate_tabular_dataset(dataset)
    if not isinstance(config, (HoldoutConfig, StratifiedHoldoutConfig)):
        raise ContractViolation(
            "config must be HoldoutConfig or StratifiedHoldoutConfig, got "
            f"{type(config).__name__}."
        )
    _validate_ready_for_split(dataset)
    train_rows, validation_rows = _validate_holdout_config(
        config,
        len(dataset.frame),
    )

    positions = np.arange(len(dataset.frame))
    target_codes: np.ndarray | None = None
    if isinstance(config, StratifiedHoldoutConfig):
        target_codes, class_counts = _classification_target(dataset)
        rare_classes = {
            label: int(count)
            for label, count in class_counts.items()
            if 0 < int(count) < 2
        }
        if rare_classes:
            raise ContractViolation(
                "Every classification target value must have at least two rows "
                f"for stratified holdout; rare={rare_classes!r}."
            )
        observed_classes = sum(int(count) > 0 for count in class_counts)
        if train_rows < observed_classes or validation_rows < observed_classes:
            raise ContractViolation(
                "Stratified holdout must allocate at least one row per class to "
                "both partitions; "
                f"classes={observed_classes}, train_rows={train_rows}, "
                f"validation_rows={validation_rows}."
            )

    train_positions, validation_positions = train_test_split(
        positions,
        test_size=float(config.validation_fraction),
        random_state=config.seed,
        shuffle=True,
        stratify=target_codes,
    )
    return HoldoutSplit(
        train_positions=tuple(sorted(int(value) for value in train_positions)),
        validation_positions=tuple(
            sorted(int(value) for value in validation_positions)
        ),
    )


def make_splits(
    dataset: TabularDataset,
    config: SplitConfig,
) -> tuple[FoldSplit, ...]:
    """Create common deterministic folds after the global missing policy.

    ``StratifiedKFoldConfig`` requires a declared classification target and
    preserves target proportions in every held-out fold. ``KFoldConfig`` does
    not inspect target values and is appropriate for regression or explicitly
    non-stratified studies.
    """

    validate_tabular_dataset(dataset)
    if not isinstance(config, (KFoldConfig, StratifiedKFoldConfig)):
        raise ContractViolation(
            "config must be KFoldConfig or StratifiedKFoldConfig, got "
            f"{type(config).__name__}."
        )
    _validate_ready_for_split(dataset)
    _validate_kfold_config(config, len(dataset.frame))

    positions = tuple(range(len(dataset.frame)))
    if isinstance(config, KFoldConfig):
        splitter = KFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.seed,
        )
        return _folds_from_sklearn(splitter.split(positions))

    target_codes, class_counts = _classification_target(dataset)
    rare_classes = {
        label: int(count)
        for label, count in class_counts.items()
        if 0 < int(count) < config.n_splits
    }
    if rare_classes:
        raise ContractViolation(
            "Every classification target value must have at least n_splits "
            f"rows; n_splits={config.n_splits}, rare={rare_classes!r}."
        )

    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    return _folds_from_sklearn(splitter.split(positions, target_codes))
