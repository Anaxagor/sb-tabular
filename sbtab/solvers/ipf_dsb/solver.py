
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sbtab.bridge.timegrid import TimeGrid
from sbtab.bridge.reference import GaussianReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.models.neural.mlp import TimeConditionedMLP, TimeMLPConfig
from sbtab.models.neural.trainer import NeuralTrainer, NeuralTrainerConfig
from sbtab.bridge.losses import RegressionLoss


@dataclass
class IPFDSBConfig:
    """
    A practical, tabular-friendly adaptation of the DSB-IPF solver.

    Key ideas (matching the other repo):
      - maintain two models: forward f and backward b
      - alternate training of b and f (IPF iterations)
      - each training phase uses caches built by simulating trajectories using the opposite model
      - regress a "residual" target: x_prev - x_next  (mean-match style target)

    This implementation is intentionally minimal and cleanly compatible with sb-tabular.
    """
    ipf_iters: int = 6

    # time discretization
    num_steps: int = 20
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: Literal["linear", "geom"] = "geom"

    # training
    batch_size: int = 512
    cache_batches: int = 200  # number of batches to generate per IPF phase
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs_per_phase: int = 1
    grad_clip: Optional[float] = 1.0

    # sampler
    noise: bool = True

    # parametrization
    # "mean_map": model predicts residual (x_prev - x_next) / gamma  style (we use direct residual here)
    # For simplicity we keep one variant aligned with cache target defined below.
    device: str = "cpu"
    seed: int = 42


class IPFDSBSolver:
    """
    Solver implementing IPF + DSB-style training for fully continuous tabular data.

    Public API (minimal):
      - fit(train_df)   : train on real data in transformed space
      - sample(n)       : generate synthetic samples in transformed space

    Integration note:
      - In sb-tabular experiments you should pass *already transformed* train data
        (after DropMissingRows + StandardScaler) from the DataModule.
    """

    def __init__(self, dim: int, cfg: IPFDSBConfig):
        self.dim = int(dim)
        self.cfg = cfg

        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        self.device = torch.device(cfg.device)

        self.timegrid = TimeGrid(
            num_steps=cfg.num_steps,
            gamma_min=cfg.gamma_min,
            gamma_max=cfg.gamma_max,
            schedule=cfg.schedule,
            device=self.device,
            dtype=torch.float32,
        )
        self.integrator = EulerMaruyama(noise=cfg.noise)
        self.reference = GaussianReference(dim=self.dim, device=self.device)

        # Models (f and b): time-conditioned MLPs
        mlp_cfg = TimeMLPConfig(in_dim=self.dim)
        self.net_f = TimeConditionedMLP(mlp_cfg).to(self.device)
        self.net_b = TimeConditionedMLP(mlp_cfg).to(self.device)

        self.loss = RegressionLoss(kind="mse", reduction="mean")

        self.trainer = NeuralTrainer(
            NeuralTrainerConfig(
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                max_epochs=cfg.epochs_per_phase,
                grad_clip=cfg.grad_clip,
                device=cfg.device,
            ),
            loss=self.loss,
        )

        self._fitted = False

    # ------------------------ Core helpers ------------------------

    def _as_tensor(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            arr = x.to_numpy(dtype=np.float32, copy=True)
            return torch.from_numpy(arr).to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        raise TypeError(f"Unsupported type: {type(x)}")

    def _predict(self, net: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return net(x, t)

    @torch.no_grad()
    def _simulate_one_step(
        self,
        x: torch.Tensor,
        k: int,
        net: torch.nn.Module,
        direction: Literal["forward", "backward"],
        gen: Optional[torch.Generator],
    ) -> torch.Tensor:
        """
        One discrete update that mirrors the other repo's "mean_match" spirit:
          x_next = x + net(x, t_k) * gamma_k + sqrt(gamma_k)*eps

        We keep direction mainly for bookkeeping; the step size is always gamma[k].
        """
        g = self.timegrid.gammas()
        t = self.timegrid.times()

        tk = t[k].expand(x.shape[0], 1)
        drift = self._predict(net, x, tk)
        return self.integrator.step(x, drift=drift, gamma=g[k], generator=gen)

    @torch.no_grad()
    def _make_cache(
        self,
        init_x: torch.Tensor,
        net_opposite: torch.nn.Module,
        direction: Literal["forward", "backward"],
        cache_batches: int,
        batch_size: int,
        seed: int,
    ) -> TensorDataset:
        """
        Build a cache dataset of (x_current, t_k, target_residual).

        Target choice (simple & stable):
          target = x_prev - x_next
        so the model trained on (x_next, t_k) can learn to predict a correction.
        This matches the other repo's cacheloader idea `out = t_old - t_new`.
        """
        g = self.timegrid.gammas()
        t = self.timegrid.times()
        K = self.timegrid.num_steps

        gen = torch.Generator(device=str(self.device))
        gen.manual_seed(int(seed))

        xs = []
        ts = []
        ys = []

        # We generate cache_batches * batch_size samples in total.
        # Each sample also chooses a random step k, to avoid storing full trajectories.
        # (This is tabular-friendly and avoids huge memory.)
        N = cache_batches * batch_size

        # Repeat / sample init_x to size N
        if init_x.shape[0] >= N:
            base = init_x[torch.randperm(init_x.shape[0], generator=gen)[:N]]
        else:
            reps = (N + init_x.shape[0] - 1) // init_x.shape[0]
            base = init_x.repeat((reps, 1))[:N]
            base = base[torch.randperm(base.shape[0], generator=gen)]

        # Random step indices
        k_idx = torch.randint(low=0, high=K, size=(N,), generator=gen, device=self.device)

        # For each step, we do one-step transition using the opposite model.
        # We keep it simple and local (no full-path simulation).
        x = base
        for k in range(K):
            mask = (k_idx == k)
            if not mask.any():
                continue

            x_k = x[mask]

            # direction handling: for "backward" cache, we often want to simulate using net_f;
            # for "forward" cache, simulate using net_b.
            # Here, direction is passed for semantic clarity only.
            x_next = self._simulate_one_step(x_k, k=k, net=net_opposite, direction=direction, gen=gen)

            # target residual: x_prev - x_next
            target = x_k - x_next

            xs.append(x_next)  # regress from the *new* point (like cacheloader uses "new")
            ts.append(t[k].expand(x_next.shape[0], 1))
            ys.append(target)

        X = torch.cat(xs, dim=0)
        T = torch.cat(ts, dim=0)
        Y = torch.cat(ys, dim=0)

        # Shuffle cache
        perm = torch.randperm(X.shape[0], generator=gen, device=self.device)
        X, T, Y = X[perm], T[perm], Y[perm]
        return TensorDataset(X, T, Y)

    def _train_phase(self, net_to_train: torch.nn.Module, cache: TensorDataset) -> None:
        loader = DataLoader(cache, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
        self.trainer.fit(net_to_train, loader, predict_fn=self._predict)

    # ------------------------ Public API ------------------------

    def fit(self, train: pd.DataFrame | np.ndarray | torch.Tensor) -> "IPFDSBSolver":
        """
        Train IPF iterations on real (data) distribution at t=0.

        We interpret:
          - data samples ~ P0 (real)
          - Gaussian samples ~ PT (prior)
        and learn two maps/fields that transport between them.
        """
        x_data = self._as_tensor(train)
        if x_data.ndim != 2 or x_data.shape[1] != self.dim:
            raise ValueError(f"Expected train shape (N,{self.dim}), got {tuple(x_data.shape)}")

        # Endpoint samples
        # In the DSB repo: b is trained first (n==1) using special init,
        # but for tabular we can start with a generic cache from the opposite net.
        for it in range(self.cfg.ipf_iters):
            # Phase 1: train backward model b using caches built by simulating from data with net_f
            cache_b = self._make_cache(
                init_x=x_data,
                net_opposite=self.net_f,
                direction="forward",
                cache_batches=self.cfg.cache_batches,
                batch_size=self.cfg.batch_size,
                seed=self.cfg.seed + 1000 * it + 1,
            )
            self._train_phase(self.net_b, cache_b)

            # Phase 2: train forward model f using caches built by simulating from Gaussian with net_b
            x_prior = self.reference.sample(n=self.cfg.cache_batches * self.cfg.batch_size, seed=self.cfg.seed + 1000 * it + 2)
            cache_f = self._make_cache(
                init_x=x_prior,
                net_opposite=self.net_b,
                direction="backward",
                cache_batches=self.cfg.cache_batches,
                batch_size=self.cfg.batch_size,
                seed=self.cfg.seed + 1000 * it + 3,
            )
            self._train_phase(self.net_f, cache_f)

        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic samples (in transformed space).

        We start from Gaussian and run "backward" steps using net_b.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")

        x = self.reference.sample(n=n, seed=seed)
        K = self.timegrid.num_steps

        gen = None
        if seed is not None:
            gen = torch.Generator(device=str(self.device))
            gen.manual_seed(int(seed))

        # Backward sweep: k = K-1..0
        for k in range(K - 1, -1, -1):
            x = self._simulate_one_step(x, k=k, net=self.net_b, direction="backward", gen=gen)

        return x.detach().cpu().numpy()
