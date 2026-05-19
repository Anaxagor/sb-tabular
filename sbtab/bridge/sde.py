from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


FieldFn = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
# signature: field(x, t, step_idx) -> drift-like tensor of same shape as x


@dataclass(frozen=True)
class EulerMaruyama:
    """
    Basic Euler-Maruyama integrator for discrete steps with per-step gamma.

    Update:
      x_{k+1} = x_k + a(x_k, t_k) * gamma_k + sqrt(gamma_k) * eps_k

    Notes:
      - Works across torch versions by avoiding randn_like(generator=...).
    """
    noise: bool = True
    sigma: float = 1.0

    def step(
        self,
        x: torch.Tensor,
        drift: torch.Tensor,
        gamma: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # gamma is scalar tensor or broadcastable
        if self.noise:
            # randn_like(..., generator=...) is not supported in some torch builds
            eps = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            return x + drift * gamma + self.sigma * torch.sqrt(gamma) * eps
        return x + drift * gamma
