
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sbtab.bridge.timegrid import TimeGrid
from sbtab.bridge.reference import GaussianReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.models.neural.mlp import TimeConditionedMLP, TimeMLPConfig
from sbtab.models.neural.time_embedding import SinusoidalTimeEmbeddingConfig
from sbtab.models.neural.trainer import NeuralTrainer, NeuralTrainerConfig
from sbtab.bridge.losses import RegressionLoss


@dataclass
class IPFDSBConfig:
    """
    Discrete-time joint IPF-DSB (MLP fields on a finite time grid).

    Algorithmically aligned with
    ``sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb``:
    two time-conditioned MLPs, alternating IPF phases, caches built by
    one-step Euler–Maruyama transitions with per-step ``gamma[k]``.
    """

    ipf_iters: int = 6

    num_steps: int = 20
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: Literal["linear", "geom"] = "geom"
    alpha_ou: float = 1.0

    batch_size: int = 512
    cache_batches: int = 200
    steps_per_phase: Optional[int] = None
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs_per_phase: int = 1
    grad_clip: Optional[float] = 1.0

    hidden_units: int = 256
    time_features: int = 64

    noise: bool = True

    device: str = "cpu"
    seed: int = 42


class IPFDSBSolver:
    """
    IPF + DSB-style training for fully continuous tabular features on a
    discrete-time grid (joint distribution, MLP parameterization).

    Public API:
      - ``fit(train)`` — train on real data in transformed space
      - ``sample(n)`` — generate synthetic samples in transformed space
    """

    def __init__(self, dim: int, cfg: IPFDSBConfig):
        self.dim = int(dim)
        self.cfg = cfg

        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        self.device = torch.device(cfg.device)

        gamma_min = cfg.gamma_min * cfg.alpha_ou
        gamma_max = cfg.gamma_max * cfg.alpha_ou

        self.timegrid = TimeGrid(
            num_steps=cfg.num_steps,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            schedule=cfg.schedule,
            device=self.device,
            dtype=torch.float32,
        )
        self.integrator = EulerMaruyama(noise=cfg.noise)
        self.reference = GaussianReference(dim=self.dim, device=self.device)

        te_dim = cfg.time_features if cfg.time_features % 2 == 0 else cfg.time_features + 1
        mlp_cfg = TimeMLPConfig(
            in_dim=self.dim,
            hidden_dim=cfg.hidden_units,
            time_emb=SinusoidalTimeEmbeddingConfig(dim=te_dim),
        )
        self.net_f = TimeConditionedMLP(mlp_cfg).to(self.device)
        self.net_b = TimeConditionedMLP(mlp_cfg).to(self.device)

        self.loss = RegressionLoss(kind="mse", reduction="mean")

        epochs = cfg.epochs_per_phase
        if cfg.steps_per_phase is not None and cfg.steps_per_phase > 0:
            epochs = max(1, cfg.steps_per_phase // cfg.cache_batches)

        self.trainer = NeuralTrainer(
            NeuralTrainerConfig(
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                max_epochs=epochs,
                grad_clip=cfg.grad_clip,
                device=cfg.device,
            ),
            loss=self.loss,
        )

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
        t = self.timegrid.times()
        K = self.timegrid.num_steps

        gen = torch.Generator(device=str(self.device))
        gen.manual_seed(int(seed))

        xs = []
        ts = []
        ys = []

        N = cache_batches * batch_size

        if init_x.shape[0] >= N:
            base = init_x[torch.randperm(init_x.shape[0], generator=gen)[:N]]
        else:
            reps = (N + init_x.shape[0] - 1) // init_x.shape[0]
            base = init_x.repeat((reps, 1))[:N]
            base = base[torch.randperm(base.shape[0], generator=gen)]

        k_idx = torch.randint(low=0, high=K, size=(N,), generator=gen, device=self.device)

        x = base
        for k in range(K):
            mask = (k_idx == k)
            if not mask.any():
                continue

            x_k = x[mask]

            x_next = self._simulate_one_step(x_k, k=k, net=net_opposite, direction=direction, gen=gen)

            target = x_k - x_next

            xs.append(x_next)
            ts.append(t[k].expand(x_next.shape[0], 1))
            ys.append(target)

        X = torch.cat(xs, dim=0)
        T = torch.cat(ts, dim=0)
        Y = torch.cat(ys, dim=0)

        perm = torch.randperm(X.shape[0], generator=gen, device=self.device)
        X, T, Y = X[perm], T[perm], Y[perm]
        return TensorDataset(X, T, Y)

    def _train_phase(self, net_to_train: torch.nn.Module, cache: TensorDataset) -> None:
        loader = DataLoader(cache, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
        self.trainer.fit(net_to_train, loader, predict_fn=self._predict)

    def fit(self, train: pd.DataFrame | np.ndarray | torch.Tensor) -> "IPFDSBSolver":
        x_data = self._as_tensor(train)
        if x_data.ndim != 2 or x_data.shape[1] != self.dim:
            raise ValueError(f"Expected train shape (N,{self.dim}), got {tuple(x_data.shape)}")

        for it in range(self.cfg.ipf_iters):
            cache_b = self._make_cache(
                init_x=x_data,
                net_opposite=self.net_f,
                direction="forward",
                cache_batches=self.cfg.cache_batches,
                batch_size=self.cfg.batch_size,
                seed=self.cfg.seed + 1000 * it + 1,
            )
            self._train_phase(self.net_b, cache_b)

            x_prior = self.reference.sample(
                n=self.cfg.cache_batches * self.cfg.batch_size, seed=self.cfg.seed + 1000 * it + 2
            )
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
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")

        x = self.reference.sample(n=n, seed=seed)
        K = self.timegrid.num_steps

        gen = None
        if seed is not None:
            gen = torch.Generator(device=str(self.device))
            gen.manual_seed(int(seed))

        for k in range(K - 1, -1, -1):
            x = self._simulate_one_step(x, k=k, net=self.net_b, direction="backward", gen=gen)

        return x.detach().cpu().numpy()
