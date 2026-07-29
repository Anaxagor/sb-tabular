"""Typed native configuration for the mixed-state bridge model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sbtab.bridge.losses import CategoricalLossNormalization

FB = Literal["f", "b"]


@dataclass
class MixedSBMConfig:
    """Control native MSBM mathematics, optimization, and runtime.

    ``alpha`` controls categorical reference transitions. The categorical-loss
    normalization is explicit because dividing the already reduced CSBM loss
    by the number of state columns changes its scale relative to numeric loss.
    ``device`` and ``seed`` are replaced by the benchmark fold context.
    """

    fb_sequence: tuple[FB, ...] = ("b", "f", "b", "f", "b")

    cat_emb_dim: int = 16
    hidden_dim: int = 512
    time_dim: int = 128
    n_layers: int = 5
    dropout: float = 0.1

    num_steps: int = 100
    sigma: float = 0.1
    alpha: float = 0.01
    lambda_num: float = 0.8
    lambda_cat: float = 0.2
    categorical_loss_normalization: CategoricalLossNormalization = (
        CategoricalLossNormalization.BY_NUM_COLUMNS
    )
    eps: float = 1e-3

    lr: float = 1e-4
    batch_size: int = 256
    epochs_per_direction: int = 5
    grad_clip: float | None = 1.0

    device: str = "cpu"
    seed: int = 42
