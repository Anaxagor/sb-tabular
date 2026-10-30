from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .time_embedding import SinusoidalTimeEmbedding, SinusoidalTimeEmbeddingConfig


@dataclass
class TimeMLPConfig:
    in_dim: int
    hidden_dim: int = 256
    n_layers: int = 4
    out_dim: Optional[int] = None  # if None -> in_dim
    dropout: float = 0.0
    time_emb: SinusoidalTimeEmbeddingConfig = field(
        default_factory=lambda: SinusoidalTimeEmbeddingConfig(dim=64)
    )


class TimeConditionedMLP(nn.Module):
    """
    Time-conditioned MLP: takes x (B,D) and t (B,1) and predicts vector (B,D).

    """

    def __init__(self, cfg: TimeMLPConfig):
        super().__init__()
        out_dim = cfg.in_dim if cfg.out_dim is None else cfg.out_dim
        self.in_dim = cfg.in_dim
        self.out_dim = out_dim

        self.time_emb = SinusoidalTimeEmbedding(cfg.time_emb)

        layers = []
        d0 = cfg.in_dim + cfg.time_emb.dim
        for i in range(cfg.n_layers):
            di = d0 if i == 0 else cfg.hidden_dim
            layers.append(nn.Linear(di, cfg.hidden_dim))
            layers.append(nn.SiLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(cfg.hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (B, D)")
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("t must have shape (B, 1)")
        te = self.time_emb(t)
        h = torch.cat([x, te], dim=1)
        return self.net(h)


@dataclass
class StepMLPConfig:
    in_dim: int
    num_steps: int
    hidden_dim: int = 128
    n_layers: int = 3
    out_dim: Optional[int] = None  # if None -> in_dim
    dropout: float = 0.0


class StepIndexedMLP(nn.Module):
    """
    Genuinely discrete-time field: ONE independent plain MLP per step index
    k = 0..num_steps-1. There is no time input and no parameter sharing between
    steps; ``forward(x, k)`` evaluates the k-th network on a batch whose rows all
    belong to step k.
    """

    def __init__(self, cfg: StepMLPConfig):
        super().__init__()
        if cfg.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if cfg.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        out_dim = cfg.in_dim if cfg.out_dim is None else cfg.out_dim
        self.in_dim = cfg.in_dim
        self.out_dim = out_dim
        self.num_steps = int(cfg.num_steps)

        def _one() -> nn.Sequential:
            layers = []
            for i in range(cfg.n_layers):
                layers.append(nn.Linear(cfg.in_dim if i == 0 else cfg.hidden_dim, cfg.hidden_dim))
                layers.append(nn.SiLU())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
            layers.append(nn.Linear(cfg.hidden_dim, out_dim))
            return nn.Sequential(*layers)

        self.steps = nn.ModuleList([_one() for _ in range(self.num_steps)])

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (B, D)")
        k = int(k)
        if not 0 <= k < self.num_steps:
            raise IndexError(f"step index {k} outside 0..{self.num_steps - 1}")
        return self.steps[k](x)
