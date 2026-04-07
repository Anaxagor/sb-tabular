from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class SplitConfigKFold:
    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class KFoldSplit:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class SplitConfigHoldout:
    val_size: float = 0.2
    shuffle: bool = True
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class HoldoutSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_kfold_splits(n_samples: int, cfg: SplitConfigKFold) -> List[KFoldSplit]:
    if cfg.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_samples < cfg.n_splits:
        raise ValueError("n_samples must be greater than or equal to n_splits.")

    indices = np.arange(n_samples)
    if cfg.shuffle:
        rng = np.random.default_rng(cfg.random_state)
        indices = rng.permutation(indices)

    fold_sizes = np.full(cfg.n_splits, n_samples // cfg.n_splits, dtype=int)
    fold_sizes[: n_samples % cfg.n_splits] += 1

    splits: List[KFoldSplit] = []
    current = 0
    for fold_id, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append(KFoldSplit(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx))
        current = stop
    return splits


def make_holdout_split(n_samples: int, cfg: SplitConfigHoldout) -> HoldoutSplit:
    if not 0.0 < cfg.val_size < 1.0:
        raise ValueError("val_size must be in the open interval (0, 1).")
    if n_samples < 2:
        raise ValueError("At least 2 samples are required to create a holdout split.")

    indices = np.arange(n_samples)
    if cfg.shuffle:
        rng = np.random.default_rng(cfg.random_state)
        indices = rng.permutation(indices)

    n_val = int(round(n_samples * cfg.val_size))
    n_val = min(max(n_val, 1), n_samples - 1)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return HoldoutSplit(train_idx=train_idx, val_idx=val_idx)
