
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SinusoidalTimeEmbeddingConfig:
    dim: int = 64
    max_period: float = 10_000.0
    learnable_scale: bool = False


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for a scalar time t.

    Input:
      t: (B, 1) float tensor
    Output:
      emb: (B, dim)
    """
    def __init__(self, cfg: SinusoidalTimeEmbeddingConfig):
        super().__init__()
        if cfg.dim % 2 != 0:
            raise ValueError("Time embedding dim must be even.")
        self.dim = cfg.dim
        self.max_period = float(cfg.max_period)
        if cfg.learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scale", torch.tensor(1.0), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("t must have shape (B, 1)")
        t = t * self.scale

        half = self.dim // 2
        device = t.device
        dtype = t.dtype

        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=device, dtype=dtype))
            * torch.arange(0, half, device=device, dtype=dtype)
            / half
        )  # (half,)
        args = t * freqs.view(1, -1)  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B, dim)
        return emb
