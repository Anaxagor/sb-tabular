from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LightSBM(nn.Module):

    def __init__(
        self,
        dim: int,
        n_potentials: int = 50,
        epsilon: float = 0.1,
        is_diagonal: bool = True,
        sampling_batch_size: int = 512,
        S_diagonal_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_potentials = n_potentials
        self.epsilon = epsilon
        self.is_diagonal = is_diagonal
        self.sampling_batch_size = sampling_batch_size

        self.r = nn.Parameter(torch.randn(n_potentials, dim))
        self.log_a = nn.Parameter(torch.zeros(n_potentials))

        if is_diagonal:
            self.log_s = nn.Parameter(
                torch.full((n_potentials, dim), math.log(S_diagonal_init))
            )
        else:
            self.L = nn.Parameter(
                torch.eye(dim).unsqueeze(0).expand(n_potentials, -1, -1).clone()
            )

    def init_r_by_samples(self, samples: torch.Tensor) -> None:
        with torch.no_grad():
            k = min(self.n_potentials, samples.shape[0])
            self.r[:k].copy_(samples[:k])

    def _kernel_weights(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.r.unsqueeze(0)

        if self.is_diagonal:
            s = torch.exp(self.log_s)
            mahal = (diff ** 2 * s.unsqueeze(0)).sum(-1)
        else:
            P = self.L @ self.L.transpose(-1, -2)
            Pd = torch.einsum("bkd,kde->bke", diff, P)
            mahal = (diff * Pd).sum(-1)

        log_w = self.log_a.unsqueeze(0) - 0.5 * mahal
        return torch.softmax(log_w, dim=-1)

    def _predict_x1(self, x_t: torch.Tensor) -> torch.Tensor:
        w = self._kernel_weights(x_t)
        return (w.unsqueeze(-1) * self.r.unsqueeze(0)).sum(1)

    def get_drift(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x1_pred = self._predict_x1(x_t)
        t_safe = t.clamp(max=1.0 - 1e-4).unsqueeze(-1)
        return (x1_pred - x_t) / (1.0 - t_safe)

    @torch.no_grad()
    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        n_steps = 100
        dt = 1.0 / n_steps
        B = x_0.shape[0]
        device = x_0.device
        x = x_0.clone()

        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device, dtype=x.dtype)
            x = x + self.get_drift(x, t) * dt

        return x

    @torch.no_grad()
    def sample_euler_maruyama(
        self,
        x_0: torch.Tensor,
        n_steps: int = 100,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        B, D = x_0.shape
        device = x_0.device
        dt = 1.0 / n_steps
        sqrt_eps_dt = math.sqrt(self.epsilon * dt)

        traj = [x_0]
        x = x_0.clone()

        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device, dtype=x.dtype)
            drift = self.get_drift(x, t)
            noise = torch.randn(B, D, device=device, dtype=x.dtype, generator=generator)
            x = x + drift * dt + sqrt_eps_dt * noise
            traj.append(x)

        return torch.stack(traj, dim=1)
