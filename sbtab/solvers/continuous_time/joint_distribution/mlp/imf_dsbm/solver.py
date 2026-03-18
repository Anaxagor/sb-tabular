from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sbtab.bridge.reference import GaussianReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.losses import RegressionLoss
from sbtab.models.neural.mlp import TimeConditionedMLP, TimeMLPConfig
from sbtab.models.neural.trainer import NeuralTrainer, NeuralTrainerConfig


FB = Literal["f", "b"]


@dataclass
class IMFDSBMConfig:
    """
    IMF + DSBM configuration.

    This follows the structure of DSBM-Gaussian.py:
      - Two time-conditioned networks: f (forward), b (backward)
      - DSBM training tuples via noisy interpolation z_t between coupling endpoints (z0,z1)
      - IMF outer loop alternates training directions, and creates new couplings using the
        previously trained model to sample endpoints.

    Endpoint convention in sb-tabular:
      - x0 = data samples (in transformed space)
      - x1 = Gaussian prior samples (same size/dim)

    Generation:
      - start from x1 ~ N(0,I)
      - apply the latest trained backward model 'b' to obtain synthetic samples.
    """

    # IMF outer loop sequence. First should be "b" for the standard first coupling.
    fb_sequence: Tuple[FB, ...] = ("b", "f", "b", "f", "b")

    # DSBM SDE discretization in sampling (N steps in [0,1])
    num_steps: int = 1000

    # DSBM noise scale (sig in reference)
    sigma: float = 0.1

    # Avoid degenerate t close to 0 or 1 in training tuples
    eps: float = 1e-3

    # First coupling construction for the first iteration:
    #   "ref": z1 = z0 + sigma * N(0,I)  (reference coupling)
    #   "ind": z1 is a shuffled copy of the other endpoint
    first_coupling: Literal["ref", "ind"] = "ref"

    # Training control (per IMF direction)
    inner_iters: int = 2000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0

    # Loss
    loss_kind: str = "mse"      # passed to RegressionLoss(kind=...)
    loss_reduction: str = "mean"

    # Integrator
    noise: bool = True          # whether to inject noise during sampling

    # Device / seed
    device: str = "cpu"
    seed: int = 42


class _DSBMModel(nn.Module):
    """
    Holds the two nets (f and b) used by DSBM.
    """
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        cfg = TimeMLPConfig(in_dim=dim)
        self.net_f = TimeConditionedMLP(cfg).to(device)
        self.net_b = TimeConditionedMLP(cfg).to(device)

    def net(self, fb: FB) -> nn.Module:
        return self.net_f if fb == "f" else self.net_b


class IMFDSBMSolver:
    """
    IMF + DSBM solver integrated into sb-tabular.

    Public API:
      - fit(train_df_or_array)
      - sample(n, seed=None, steps=None) -> np.ndarray

    Expected input is in transformed space (after DropMissingRows + StandardScaler).
    """

    def __init__(self, dim: int, cfg: IMFDSBMConfig):
        self.dim = int(dim)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        self.reference = GaussianReference(dim=self.dim, device=self.device)
        self.integrator = EulerMaruyama(noise=cfg.noise)

        self.model = _DSBMModel(dim=self.dim, device=self.device)

        self.loss_fn = RegressionLoss(kind=cfg.loss_kind, reduction=cfg.loss_reduction)
        self.trainer_cfg = NeuralTrainerConfig(
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            max_epochs=1,                  # we do iterations manually (inner_iters)
            grad_clip=cfg.grad_clip,
            device=cfg.device,
        )
        self.trainer = NeuralTrainer(self.trainer_cfg, loss=self.loss_fn)

        # IMF snapshots: each element is {"fb": "b"/"f", "state": state_dict}
        self.snapshots: List[dict] = []

        self._fitted = False

    # ---------------------------
    # Utilities
    # ---------------------------

    def _as_tensor(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            return torch.from_numpy(x.to_numpy(dtype=np.float32, copy=True)).to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        raise TypeError(f"Unsupported type: {type(x)}")

    @staticmethod
    def _clone_state(model: nn.Module) -> dict:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _load_state_into(self, model: nn.Module, state: dict) -> None:
        model.load_state_dict(state, strict=True)

    # ---------------------------
    # DSBM-specific pieces
    # ---------------------------

    @torch.no_grad()
    def _dsbm_train_tuple(
        self,
        z_pairs: torch.Tensor,   # (B,2,D)
        fb: FB,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implements the DSBM training tuple formulas (from DSBM-Gaussian.py):

        Sample t ~ Uniform(eps, 1-eps)
        z_t = (1-t) z0 + t z1 + sigma * sqrt(t(1-t)) * noise

        Targets:
          if fb == "f":
            target = (z1 - z0) - sigma * sqrt(t/(1-t)) * noise
          if fb == "b":
            target = -(z1 - z0) - sigma * sqrt((1-t)/t) * noise

        Return: (z_t, t, target)
        """
        z0 = z_pairs[:, 0]
        z1 = z_pairs[:, 1]
        B = z0.shape[0]

        t = torch.rand((B, 1), device=self.device) * (1 - 2 * self.cfg.eps) + self.cfg.eps
        z_t = (1.0 - t) * z0 + t * z1

        noise = torch.randn((B, self.dim), device=self.device, dtype=z_t.dtype)
        z_t = z_t + self.cfg.sigma * torch.sqrt(t * (1.0 - t)) * noise

        delta = (z1 - z0)
        if fb == "f":
            target = delta - self.cfg.sigma * torch.sqrt(t / (1.0 - t)) * noise
        else:
            target = -delta - self.cfg.sigma * torch.sqrt((1.0 - t) / t) * noise

        return z_t, t, target

    @torch.no_grad()
    def _sample_sde(
        self,
        net: nn.Module,
        fb: FB,
        zstart: torch.Tensor,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sampling dynamics in DSBM-Gaussian.py is Euler stepping over t in [0,1]:

          z <- z + net(z,t) * dt + sigma * sqrt(dt) * eps

        Direction:
          - fb="f": t increases 0->1
          - fb="b": t decreases 1->0  (equivalent schedule used in the reference)

        We implement it using the existing EulerMaruyama integrator:
          x_{k+1} = x_k + drift * dt + sqrt(dt) * eps
        and we scale the noise by sigma.
        """
        N = int(self.cfg.num_steps if steps is None else steps)
        dt = torch.tensor(1.0 / N, device=self.device, dtype=zstart.dtype)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=str(self.device))
            gen.manual_seed(int(seed))

        z = zstart.detach().clone()
        B = z.shape[0]

        # Build time values as in reference
        # ts = i/N, and if fb="b" then use (1 - ts)
        for i in range(N):
            tau = float(i) / float(N)
            if fb == "b":
                tau = 1.0 - tau
            t = torch.full((B, 1), tau, device=self.device, dtype=z.dtype)

            drift = net(z, t)  # (B,D)

            # Euler step with dt, but noise is sigma*sqrt(dt)*eps
            # Our integrator noise term gives sqrt(dt)*eps, so we scale separately.
            if self.integrator.noise:
                eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=gen)
                z = z + drift * dt + self.cfg.sigma * torch.sqrt(dt) * eps
            else:
                z = z + drift * dt

        return z

    @torch.no_grad()
    def _generate_coupling(
        self,
        x_pairs: torch.Tensor,              # (N,2,D) endpoints (x0=data, x1=prior)
        prev_state: Optional[dict],
        prev_fb: Optional[FB],
        fb_to_train: FB,
        first_it: bool,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coupling construction for IMF.

        Matches the reference logic:

        First iteration (prev=None):
          - only allowed when fb_to_train == "b"
          - z0 = x0
          - z1 = x0 + sigma*noise  (first_coupling="ref") OR shuffled x1 ("ind")

        Later iterations:
          - sample zend by applying prev model (in direction prev_fb) to appropriate start
          - then define (z0,z1) depending on prev_fb
        """
        if prev_state is None:
            if not first_it:
                raise RuntimeError("prev_state is None but first_it=False")
            if fb_to_train != "b":
                raise RuntimeError("DSBM initialization expects first direction to be 'b'")

            z0 = x_pairs[:, 0]  # data
            if self.cfg.first_coupling == "ref":
                gen = torch.Generator(device=str(self.device))
                gen.manual_seed(int(seed))
                noise = torch.randn(z0.shape, device=self.device, dtype=z0.dtype, generator=gen)
                z1 = z0 + self.cfg.sigma * noise
            elif self.cfg.first_coupling == "ind":
                z1 = x_pairs[:, 1].clone()
                perm = torch.randperm(z1.shape[0], device=self.device)
                z1 = z1[perm]
            else:
                raise NotImplementedError(f"Unknown first_coupling: {self.cfg.first_coupling}")
            return z0, z1

        # Later iteration: build a temp model with prev_state loaded
        tmp = _DSBMModel(dim=self.dim, device=self.device)
        self._load_state_into(tmp, prev_state)

        if prev_fb is None:
            raise RuntimeError("prev_fb must be provided if prev_state is not None")

        # Decide start for prev sampling as in reference:
        # if prev_fb == "f": start from data (x0)
        # else: start from prior (x1)
        zstart = x_pairs[:, 0] if prev_fb == "f" else x_pairs[:, 1]

        zend = self._sample_sde(
            net=tmp.net(prev_fb),
            fb=prev_fb,
            zstart=zstart,
            steps=self.cfg.num_steps,
            seed=seed,
        )

        if prev_fb == "f":
            # forward moved data -> ... so pair (data, zend)
            z0, z1 = zstart, zend
        else:
            # backward moved prior -> ... so pair (zend, prior)
            z0, z1 = zend, zstart

        return z0, z1

    # ---------------------------
    # Training loop (IMF)
    # ---------------------------

    def _train_direction(
        self,
        fb: FB,
        z0: torch.Tensor,
        z1: torch.Tensor,
    ) -> None:
        """
        Train net_f or net_b for `inner_iters` steps on coupling endpoints (z0,z1).
        """
        dataset = TensorDataset(z0, z1)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        it = iter(loader)

        net = self.model.net(fb)
        net.train()

        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        for _ in range(int(self.cfg.inner_iters)):
            try:
                b0, b1 = next(it)
            except StopIteration:
                it = iter(loader)
                b0, b1 = next(it)

            z_pairs = torch.stack([b0, b1], dim=1)  # (B,2,D)
            z_t, t, target = self._dsbm_train_tuple(z_pairs, fb=fb)

            pred = net(z_t, t)
            loss = self.loss_fn(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(self.cfg.grad_clip))
            opt.step()

    def fit(self, train: pd.DataFrame | np.ndarray | torch.Tensor) -> "IMFDSBMSolver":
        """
        Train IMF sequence on transformed data.

        Endpoints:
          x0 = data
          x1 = Gaussian prior samples
        """
        x0 = self._as_tensor(train)
        if x0.ndim != 2 or x0.shape[1] != self.dim:
            raise ValueError(f"Expected train shape (N,{self.dim}), got {tuple(x0.shape)}")

        x1 = self.reference.sample(n=x0.shape[0], seed=self.cfg.seed + 999).to(self.device)

        x_pairs = torch.stack([x0, x1], dim=1)  # (N,2,D)

        prev_state = None
        prev_fb: Optional[FB] = None

        for it_idx, fb in enumerate(self.cfg.fb_sequence, start=1):
            first_it = (it_idx == 1)

            z0, z1 = self._generate_coupling(
                x_pairs=x_pairs,
                prev_state=prev_state,
                prev_fb=prev_fb,
                fb_to_train=fb,
                first_it=first_it,
                seed=self.cfg.seed + 10_000 * it_idx,
            )

            # Train chosen direction on this coupling
            self._train_direction(fb=fb, z0=z0, z1=z1)

            # Snapshot full model state after this direction
            snap_state = self._clone_state(self.model)
            self.snapshots.append({"fb": fb, "state": snap_state})

            prev_state = snap_state
            prev_fb = fb

        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n: int, seed: Optional[int] = None, steps: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic samples in transformed space:
          - start from Gaussian
          - apply latest backward ('b') model
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")

        # Find latest 'b' snapshot
        b_state = None
        for item in reversed(self.snapshots):
            if item["fb"] == "b":
                b_state = item["state"]
                break
        if b_state is None:
            raise RuntimeError("No backward ('b') model found. Ensure fb_sequence contains 'b'.")

        tmp = _DSBMModel(dim=self.dim, device=self.device)
        self._load_state_into(tmp, b_state)
        tmp.eval()

        zstart = self.reference.sample(n=n, seed=seed).to(self.device)

        z = self._sample_sde(
            net=tmp.net("b"),
            fb="b",
            zstart=zstart,
            steps=steps,
            seed=seed,
        )
        return z.detach().cpu().numpy()
