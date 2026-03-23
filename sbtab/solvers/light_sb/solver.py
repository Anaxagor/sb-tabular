from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from sbtab.bridge.reference import GaussianReference
from sbtab.models.sb.light_sb import LightSBPotential
from sbtab.solvers.light_sb.config import LightSBConfig


class LightSBSolver:
    """
    Proper LightSB solver for the repository.

    - Training uses the empirical objective from the LightSB paper / official code:
        mean(log C_theta(x0)) - mean(log v_theta(x1))
    - Fast sampling uses the learned conditional plan π_theta(x1 | x0).
    - Optional SDE sampling uses the exact drift with Euler-Maruyama.
    """

    def __init__(self, dim: int, cfg: LightSBConfig):
        self.dim = int(dim)
        self.cfg = cfg

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(cfg.seed))

        self.reference = GaussianReference(dim=self.dim, device=self.device)
        self.model = LightSBPotential(dim=self.dim, cfg=cfg.potential).to(self.device)

        self._columns: Optional[list[str]] = None
        self._fitted = False

        # Dedicated generator for Gaussian reference batches during training/sampling.
        self._ref_gen = torch.Generator(device=str(self.device))
        self._ref_gen.manual_seed(int(cfg.seed) + 1)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def _as_tensor(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            self._columns = list(x.columns)
            arr = x.to_numpy(dtype=np.float32, copy=True)
            return torch.from_numpy(arr).to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        raise TypeError(f"Unsupported type: {type(x)}")

    def _sample_reference_batch(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        gen = self._ref_gen if generator is None else generator
        x = torch.randn(
            (int(n), self.dim),
            device=self.device,
            dtype=torch.float32,
            generator=gen,
        )
        # GaussianReference currently uses mean/std scalars, so we match its distribution.
        return x * float(self.reference.std) + float(self.reference.mean)

    def _sample_data_batch(
        self,
        x_data: torch.Tensor,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        N = x_data.shape[0]
        idx = torch.randint(0, N, (int(batch_size),), generator=generator, device=self.device)
        return x_data[idx]

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame | np.ndarray | torch.Tensor) -> "LightSBSolver":
        x_data = self._as_tensor(train)
        if x_data.ndim != 2 or x_data.shape[1] != self.dim:
            raise ValueError(f"Expected train shape (N, {self.dim}), got {tuple(x_data.shape)}")
        if torch.isnan(x_data).any():
            raise ValueError("Input contains NaNs. Apply preprocessing first.")

        # Optional init of centers r_k by data samples (reference code / paper appendix).
        if self.cfg.init_r_from_data:
            N = x_data.shape[0]
            K = self.cfg.potential.n_potentials
            if N >= K:
                perm = torch.randperm(N, device=self.device)[:K]
                init_samples = x_data[perm].detach().clone()
            else:
                reps = (K + N - 1) // N
                repeated = x_data.repeat(reps, 1)[:K]
                noise = 0.01 * torch.randn_like(repeated)
                init_samples = (repeated + noise).detach().clone()
            self.model.init_r_by_samples(init_samples)

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        data_gen = torch.Generator(device=str(self.device))
        data_gen.manual_seed(int(self.cfg.seed) + 2)

        self.model.train()
        for it in range(int(self.cfg.max_iter)):
            x1 = self._sample_data_batch(x_data, self.cfg.batch_size, data_gen)
            x0 = self._sample_reference_batch(self.cfg.batch_size)

            loss = self.model.get_log_C(x0).mean() - self.model.get_log_potential(x1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip))
            opt.step()

            if self.cfg.verbose_every > 0 and ((it + 1) % self.cfg.verbose_every == 0 or it == 0):
                print(f"[LightSB] iter={it+1}/{self.cfg.max_iter} loss={float(loss.detach().cpu()):.6f}")

        self.model.eval()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def transport(
        self,
        x0: pd.DataFrame | np.ndarray | torch.Tensor,
        seed: Optional[int] = None,
        use_sde_sampling: Optional[bool] = None,
        n_euler_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Transport given input x0 to x1.

        - if use_sde_sampling=False: sample from conditional plan π_theta(.|x0)
        - if use_sde_sampling=True:  simulate the associated bridge process with EM
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transport().")

        if use_sde_sampling is None:
            use_sde_sampling = self.cfg.use_sde_sampling
        if n_euler_steps is None:
            n_euler_steps = self.cfg.n_euler_steps

        x0_t = self._as_tensor(x0)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=str(self.device))
            generator.manual_seed(int(seed))
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))

        if not use_sde_sampling:
            x1 = self.model(x0_t)
        else:
            traj = self.model.sample_euler_maruyama(x0_t, n_steps=int(n_euler_steps), generator=generator)
            x1 = traj[:, -1, :]

        return x1.detach().cpu().numpy()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        use_sde_sampling: Optional[bool] = None,
        n_euler_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate n samples from the target side by starting from the Gaussian reference.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=str(self.device))
            generator.manual_seed(int(seed))
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))

        x0 = self._sample_reference_batch(int(n), generator=generator)
        return self.transport(
            x0,
            seed=seed,
            use_sde_sampling=use_sde_sampling,
            n_euler_steps=n_euler_steps,
        )

    def sample_df(
        self,
        n: int,
        seed: Optional[int] = None,
        use_sde_sampling: Optional[bool] = None,
        n_euler_steps: Optional[int] = None,
    ) -> pd.DataFrame:
        arr = self.sample(
            n=n,
            seed=seed,
            use_sde_sampling=use_sde_sampling,
            n_euler_steps=n_euler_steps,
        )
        if self._columns is None:
            return pd.DataFrame(arr)
        return pd.DataFrame(arr, columns=self._columns)

    @torch.no_grad()
    def sample_paths(
        self,
        n: int,
        seed: Optional[int] = None,
        n_euler_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample full Euler-Maruyama trajectories starting from the Gaussian reference.

        Returns:
            array of shape (n, n_steps + 1, dim)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample_paths().")
        if n <= 0:
            raise ValueError("n must be positive")
        if n_euler_steps is None:
            n_euler_steps = self.cfg.n_euler_steps

        generator = None
        if seed is not None:
            generator = torch.Generator(device=str(self.device))
            generator.manual_seed(int(seed))
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))

        x0 = self._sample_reference_batch(int(n), generator=generator)
        traj = self.model.sample_euler_maruyama(x0, n_steps=int(n_euler_steps), generator=generator)
        return traj.detach().cpu().numpy()
