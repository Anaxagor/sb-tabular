from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sbtab.bridge.reference import GaussianReference
from sbtab.solvers.light_sb.config import LightSBConfig
from sbtab.models.sb.light_sb import LightSBM


class LightSBSolver:

    def __init__(self, dim: int, cfg: LightSBConfig):
        self.dim = int(dim)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        self.reference = GaussianReference(dim=self.dim, device=self.device)

        self.model = LightSBM(
            dim=self.dim,
            n_potentials=cfg.n_potentials,
            epsilon=cfg.epsilon,
            is_diagonal=cfg.is_diagonal,
            sampling_batch_size=cfg.sampling_batch_size,
            S_diagonal_init=cfg.S_diagonal_init,
        ).to(self.device)

        self._fitted = False

    def _as_tensor(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            arr = x.to_numpy(dtype=np.float32, copy=True)
            return torch.from_numpy(arr).to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        raise TypeError(f"Unsupported type: {type(x)}")

    def _sample_data_batch(
        self,
        x_data: torch.Tensor,
        batch_size: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        N = x_data.shape[0]
        idx = torch.randint(0, N, (batch_size,), generator=gen, device=self.device)
        return x_data[idx]

    def fit(self, train: pd.DataFrame | np.ndarray | torch.Tensor) -> "LightSBSolver":
        x_data = self._as_tensor(train)
        if x_data.ndim != 2 or x_data.shape[1] != self.dim:
            raise ValueError(
                f"Expected train shape (N, {self.dim}), got {tuple(x_data.shape)}"
            )

        cfg = self.cfg

        if cfg.init_r_from_data:
            N = x_data.shape[0]
            n_pot = cfg.n_potentials
            if N >= n_pot:
                perm = torch.randperm(N, device=self.device)[:n_pot]
                init_samples = x_data[perm].detach().clone()
            else:
                reps = (n_pot + N - 1) // N
                repeated = x_data.repeat(reps, 1)[:n_pot]
                noise = torch.randn_like(repeated) * 0.01
                init_samples = (repeated + noise).detach().clone()
            self.model.init_r_by_samples(init_samples)

        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        gen = torch.Generator(device=str(self.device))
        gen.manual_seed(int(cfg.seed) + 1)

        self.model.train()

        for _ in range(int(cfg.max_iter)):
            x_1 = self._sample_data_batch(x_data, cfg.batch_size, gen)
            x_0 = torch.randn(
                (cfg.batch_size, self.dim),
                device=self.device,
                dtype=torch.float32,
                generator=gen,
            )

            t = (
                torch.rand((cfg.batch_size, 1), device=self.device, generator=gen)
                * (1.0 - cfg.safe_t)
            )

            noise = torch.randn(x_0.shape, device=self.device, dtype=torch.float32, generator=gen)
            x_t = x_1 * t + x_0 * (1.0 - t) + torch.sqrt(cfg.epsilon * t * (1.0 - t)) * noise

            t_sq = t.squeeze(1)
            predicted_drift = self.model.get_drift(x_t, t_sq)
            target_drift = (x_1 - x_t) / (1.0 - t)

            loss = F.mse_loss(predicted_drift, target_drift)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.model.eval()
        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive.")

        x_0 = self.reference.sample(n=n, seed=seed).to(self.device)

        if not self.cfg.use_sde_sampling:
            result = self.model(x_0)
        else:
            gen = None
            if seed is not None:
                gen = torch.Generator(device=str(self.device))
                gen.manual_seed(int(seed))
            trajectory = self.model.sample_euler_maruyama(
                x_0, n_steps=self.cfg.n_euler_steps, generator=gen
            )
            result = trajectory[:, -1, :]

        return result.detach().cpu().numpy()
