from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sbtab.bridge.losses import RegressionLoss
from sbtab.bridge.reference import GaussianReference
from sbtab.models.neural.mlp import TimeConditionedMLP, TimeMLPConfig
from sbtab.models.neural.time_embedding import SinusoidalTimeEmbeddingConfig


FB = Literal["f", "b"]

CANONICAL_ID = "dsbm_ct_joint_mlp"
CHECKPOINT_FORMAT = "sbtab.dsbm_ct_joint_mlp.v1"


@dataclass
class IMFDSBMConfig:
    """
    IMF + DSBM configuration (continuous time, joint MLP drift).

    Structure (DSBM-Gaussian.py):
      - two time-conditioned networks: f (forward) and b (backward)
      - bridge-matching tuples from the Brownian bridge between coupling endpoints
      - the IMF outer loop alternates directions; every stage after the first is
        trained on the coupling simulated with the latest OPPOSITE-direction model.

    Endpoint convention in sb-tabular:
      - x0 = data rows (transformed space), time t = 0
      - x1 = N(0, I) prior rows,            time t = 1

    Generation starts from x1 ~ N(0, I) and integrates the latest backward model.
    """

    # IMF stage sequence. Must be non-empty, contain only "f"/"b" and strictly
    # alternate: a stage is trained on the coupling produced by the opposite
    # direction. Either direction may come first; sample() needs at least one "b".
    fb_sequence: Tuple[FB, ...] = ("b", "f", "b", "f", "b")

    # Number of Euler-Maruyama steps on [0, 1] (couplings and generation).
    num_steps: int = 1000

    # Diffusion scale of the Brownian reference.
    sigma: float = 0.1

    # Training times are drawn from U(eps, 1 - eps). The sampler clamps the time
    # FED TO THE NETWORK into the same interval (see _sample_sde).
    eps: float = 1e-3

    # Coupling used by the first stage:
    #   "ind": independent coupling data (x) prior. This is the DSBM-IMF
    #          initialiser: its t=1 marginal is the N(0, I) prior generation
    #          starts from.
    #   "ref": z1 = z0 + sigma * N(0, I), the reference (IPF-like) coupling. Its
    #          t=1 marginal is data * N(0, sigma^2 I), NOT the prior, so the first
    #          backward model is trained away from where generation starts.
    first_coupling: Literal["ind", "ref"] = "ind"

    # Training control (per IMF stage)
    inner_iters: int = 2000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0

    # Loss
    loss_kind: str = "mse"      # passed to RegressionLoss(kind=...)
    loss_reduction: str = "mean"

    # noise=False selects the NOISELESS HEURISTIC sampler for generation only
    # (variant_id gets the "_noiseless_heuristic" suffix). Couplings simulated
    # during fit() always use noise.
    noise: bool = True

    # Device / seed
    device: str = "cpu"
    seed: int = 42

    # Network architecture (both directions)
    hidden_dim: int = 256
    n_layers: int = 4
    dropout: float = 0.0
    time_emb_dim: int = 64
    time_emb_max_period: float = 10_000.0
    time_emb_learnable_scale: bool = False


class _DSBMModel(nn.Module):
    """Holds the two drift networks (f and b) used by DSBM."""

    def __init__(self, dim: int, cfg: IMFDSBMConfig, device: torch.device):
        super().__init__()
        self.net_f = TimeConditionedMLP(self.mlp_config(dim, cfg)).to(device)
        self.net_b = TimeConditionedMLP(self.mlp_config(dim, cfg)).to(device)

    @staticmethod
    def mlp_config(dim: int, cfg: IMFDSBMConfig) -> TimeMLPConfig:
        return TimeMLPConfig(
            in_dim=int(dim),
            hidden_dim=int(cfg.hidden_dim),
            n_layers=int(cfg.n_layers),
            dropout=float(cfg.dropout),
            time_emb=SinusoidalTimeEmbeddingConfig(
                dim=int(cfg.time_emb_dim),
                max_period=float(cfg.time_emb_max_period),
                learnable_scale=bool(cfg.time_emb_learnable_scale),
            ),
        )

    def net(self, fb: FB) -> nn.Module:
        return self.net_f if fb == "f" else self.net_b


class IMFDSBMSolver:
    """
    IMF + DSBM solver, continuous time, joint MLP drift (registry id
    ``dsbm_ct_joint_mlp``).

    Public API:
      - fit(train) -> self
      - sample(n, seed=None, steps=None) -> np.ndarray
      - sample_df(n, seed=None, steps=None) -> pd.DataFrame
      - save_checkpoint(path) / load_checkpoint(path, device=None)
      - variant_id, n_updates, stage_log, snapshots

    Expected input is in transformed space (after DropMissingRows + StandardScaler).

    Sampler. Generation integrates the learned backward SDE with Euler-Maruyama,
        z <- z + b(z, t) dt + sigma sqrt(dt) eps.
    The drift is trained for the STOCHASTIC bridge and contains the score term.
    With ``cfg.noise=False`` the same drift is integrated without noise: that is a
    heuristic, NOT a probability-flow ODE, and it does not preserve the marginals.
    It is reported as ``variant_id == "dsbm_ct_joint_mlp_noiseless_heuristic"``.
    The flag only affects sample(); couplings simulated during fit() always use
    noise, so training does not depend on it.
    """

    canonical_id = CANONICAL_ID

    def __init__(self, dim: int, cfg: IMFDSBMConfig):
        self.dim = int(dim)
        self.cfg = cfg
        self._validate_config(cfg)
        self.device = torch.device(cfg.device)

        self.columns_: Optional[List[str]] = None
        self.reference = GaussianReference(dim=self.dim, device=self.device)
        self.loss_fn = RegressionLoss(kind=cfg.loss_kind, reduction=cfg.loss_reduction)
        self.model = self._new_model()

        # One entry per IMF stage: {"fb", "stage", "state"} with the state of the
        # network trained in that stage.
        self.snapshots: List[dict] = []
        self.stage_log: List[dict] = []
        self._fitted = False

    # ------------------------------------------------------------------ config
    @staticmethod
    def _validate_config(cfg: IMFDSBMConfig) -> None:
        seq = tuple(cfg.fb_sequence)
        if len(seq) == 0:
            raise ValueError("fb_sequence must not be empty")
        for d in seq:
            if d not in ("f", "b"):
                raise ValueError(f"Unknown direction in fb_sequence: {d!r} (allowed: 'f', 'b')")
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                raise ValueError(
                    "fb_sequence must strictly alternate: each stage is trained on the coupling "
                    f"of the opposite direction, got {seq}"
                )
        if cfg.first_coupling not in ("ind", "ref"):
            raise ValueError(f"Unknown first_coupling={cfg.first_coupling!r} (allowed: 'ind', 'ref')")
        if int(cfg.num_steps) < 1:
            raise ValueError("num_steps must be >= 1")
        if int(cfg.inner_iters) < 1:
            raise ValueError("inner_iters must be >= 1")
        if int(cfg.batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        if not float(cfg.sigma) >= 0.0:
            raise ValueError("sigma must be >= 0")
        if not 0.0 < float(cfg.eps) < 0.5:
            raise ValueError("eps must lie in (0, 0.5)")

    @property
    def variant_id(self) -> str:
        return self.canonical_id if self.cfg.noise else f"{self.canonical_id}_noiseless_heuristic"

    @property
    def n_updates(self) -> int:
        """Total number of optimizer updates over all IMF stages of the last fit()."""
        return int(sum(s["n_updates"] for s in self.stage_log))

    # ------------------------------------------------------------------ utilities
    def _new_model(self) -> _DSBMModel:
        # Weight init honours cfg.seed without touching the global RNG state.
        with self._forked_rng(int(self.cfg.seed)):
            return _DSBMModel(dim=self.dim, cfg=self.cfg, device=self.device)

    @contextlib.contextmanager
    def _forked_rng(self, seed: int):
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            yield

    def _make_generator(self, seed: Optional[int]) -> torch.Generator:
        """Seeded generator on the solver device; seed=None draws fresh entropy."""
        gen = torch.Generator(device=str(self.device))
        if seed is None:
            gen.seed()
        else:
            gen.manual_seed(int(seed))
        return gen

    def _as_tensor(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            return torch.from_numpy(x.to_numpy(dtype=np.float32, copy=True)).to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32, copy=True)).to(self.device)
        if isinstance(x, torch.Tensor):
            return x.detach().to(self.device, dtype=torch.float32).clone()
        raise TypeError(f"Unsupported type: {type(x)}")

    @staticmethod
    def _clone_state(model: nn.Module) -> dict:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ------------------------------------------------------------------ DSBM pieces
    @torch.no_grad()
    def _dsbm_train_tuple(
        self,
        z_pairs: torch.Tensor,   # (B,2,D)
        fb: FB,
        *,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DSBM training tuple (DSBM-Gaussian.py):

          t ~ Uniform(eps, 1-eps)
          z_t = (1-t) z0 + t z1 + sigma * sqrt(t(1-t)) * noise

          fb == "f": target = (z1 - z0) - sigma * sqrt(t/(1-t)) * noise   = (z1 - z_t)/(1-t)
          fb == "b": target = -(z1 - z0) - sigma * sqrt((1-t)/t) * noise  = (z0 - z_t)/t

        The same t and noise build z_t and the target. ``t`` (B,1) and ``noise``
        (B,D) may be supplied; otherwise they are drawn from ``generator``.

        Return: (z_t, t, target)
        """
        z0 = z_pairs[:, 0]
        z1 = z_pairs[:, 1]
        B = z0.shape[0]

        if t is None:
            u = torch.rand((B, 1), device=self.device, dtype=z0.dtype, generator=generator)
            t = u * (1 - 2 * self.cfg.eps) + self.cfg.eps
        if noise is None:
            noise = torch.randn((B, self.dim), device=self.device, dtype=z0.dtype, generator=generator)

        z_t = (1.0 - t) * z0 + t * z1
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
        generator: Optional[torch.Generator] = None,
        noise: bool = True,
    ) -> torch.Tensor:
        """
        Euler-Maruyama over t in [0,1] with N steps, dt = 1/N:

          z <- z + net(z, t) * dt + sigma * sqrt(dt) * eps

        Direction:
          - fb="f": state time i/N increases 0 -> 1
          - fb="b": state time 1 - i/N decreases 1 -> 0

        The network was trained on t in [eps, 1-eps] only, so the time fed to the
        network is clamped into that interval (this affects the first step, and
        every step with i/N < eps when 1/N < eps). The state update always uses
        the true dt.

        ``noise=False`` is the noiseless heuristic (see class docstring).
        """
        N = int(self.cfg.num_steps if steps is None else steps)
        if N < 1:
            raise ValueError("steps must be >= 1")
        dt = 1.0 / float(N)
        sqrt_dt = math.sqrt(dt)
        sigma = float(self.cfg.sigma)
        eps_t = float(self.cfg.eps)

        z = zstart.detach().clone()
        B = z.shape[0]

        for i in range(N):
            tau = float(i) / float(N)
            if fb == "b":
                tau = 1.0 - tau
            tau_net = min(max(tau, eps_t), 1.0 - eps_t)
            t = torch.full((B, 1), tau_net, device=z.device, dtype=z.dtype)

            drift = net(z, t)  # (B,D)
            z = z + drift * dt
            if noise:
                eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
                z = z + sigma * sqrt_dt * eps

        return z

    @torch.no_grad()
    def _generate_coupling(
        self,
        x0: torch.Tensor,            # (N,D) real data rows
        x1: torch.Tensor,            # (N,D) real prior rows
        prev_fb: Optional[FB],
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Coupling (z0, z1) for the next IMF stage and its source.

        First stage (prev_fb is None):
          "ind": z0 = data, z1 = permuted prior rows (independent coupling)
          "ref": z0 = data, z1 = data + sigma * noise (reference coupling)

        Later stages: simulate the latest model of the OPPOSITE direction, always
        with noise, anchored at the real rows it starts from:
          prev "f": z0 = data (real),   z1 = forward simulation from the data
          prev "b": z1 = prior (real),  z0 = backward simulation from the prior
        """
        gen = self._make_generator(seed)

        if prev_fb is None:
            z0 = x0
            if self.cfg.first_coupling == "ind":
                perm = torch.randperm(x1.shape[0], device=self.device, generator=gen)
                return z0, x1[perm], "independent"
            if self.cfg.first_coupling == "ref":
                noise = torch.randn(z0.shape, device=self.device, dtype=z0.dtype, generator=gen)
                return z0, z0 + self.cfg.sigma * noise, "reference"
            raise ValueError(f"Unknown first_coupling={self.cfg.first_coupling!r}")

        net = self.model.net(prev_fb)
        net.eval()
        zstart = x0 if prev_fb == "f" else x1
        zend = self._sample_sde(net=net, fb=prev_fb, zstart=zstart, generator=gen, noise=True)

        if prev_fb == "f":
            return zstart, zend, "forward_model"
        return zend, zstart, "backward_model"

    # ------------------------------------------------------------------ training
    def _train_direction(
        self,
        fb: FB,
        z0: torch.Tensor,
        z1: torch.Tensor,
        seed: int,
    ) -> Tuple[int, float]:
        """
        Train net_f or net_b for ``inner_iters`` optimizer updates on the coupling
        (z0, z1). Rows are visited in shuffled epochs and the last partial batch of
        an epoch is kept, so any N >= 1 trains. Returns (n_updates, last_loss).
        """
        n = int(z0.shape[0])
        bs = int(self.cfg.batch_size)

        net = self.model.net(fb)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        shuffle_gen = torch.Generator(device="cpu")
        shuffle_gen.manual_seed(int(seed))
        tuple_gen = self._make_generator(int(seed) + 1)

        n_updates = 0
        last_loss = float("nan")
        perm = torch.empty(0, dtype=torch.long)
        pos = 0

        # fork: dropout masks are reproducible and the global RNG is left untouched
        with self._forked_rng(int(seed) + 2):
            while n_updates < int(self.cfg.inner_iters):
                if pos >= perm.numel():
                    perm = torch.randperm(n, generator=shuffle_gen)
                    pos = 0
                idx = perm[pos:pos + bs].to(self.device)
                pos += bs

                z_pairs = torch.stack([z0[idx], z1[idx]], dim=1)  # (B,2,D)
                z_t, t, target = self._dsbm_train_tuple(z_pairs, fb=fb, generator=tuple_gen)

                pred = net(z_t, t)
                loss = self.loss_fn(pred, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss while training DSBM direction '{fb}'")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), float(self.cfg.grad_clip))
                opt.step()

                n_updates += 1
                last_loss = float(loss.detach().cpu())

        net.eval()
        return n_updates, last_loss

    def fit(self, train: pd.DataFrame | np.ndarray | torch.Tensor) -> "IMFDSBMSolver":
        """
        Train the IMF sequence on transformed data (x0 = data, x1 = Gaussian prior).

        Every call is a fresh fit: networks are re-initialised from cfg.seed and
        previous snapshots / stage log are discarded. The training rows are not
        retained on the solver.
        """
        if isinstance(train, pd.DataFrame):
            self.columns_ = list(train.columns)
        x0 = self._as_tensor(train)
        if x0.ndim != 2 or x0.shape[1] != self.dim:
            raise ValueError(f"Expected train shape (N,{self.dim}), got {tuple(x0.shape)}")
        if x0.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")
        if not torch.isfinite(x0).all():
            raise ValueError("fit expects finite numeric data (found NaN/inf)")

        self._fitted = False
        self.snapshots = []
        self.stage_log = []
        self.model = self._new_model()

        x1 = self.reference.sample(n=x0.shape[0], seed=int(self.cfg.seed) + 999).to(self.device)

        prev_fb: Optional[FB] = None
        for idx, fb in enumerate(self.cfg.fb_sequence):
            coupling_seed = int(self.cfg.seed) + 10_000 + idx
            train_seed = int(self.cfg.seed) + 20_000 + idx

            z0, z1, source = self._generate_coupling(x0=x0, x1=x1, prev_fb=prev_fb, seed=coupling_seed)
            n_updates, last_loss = self._train_direction(fb=fb, z0=z0, z1=z1, seed=train_seed)
            if n_updates <= 0:
                raise RuntimeError(f"IMF stage {idx} ('{fb}') performed no optimizer updates")

            self.snapshots.append({"fb": fb, "stage": idx, "state": self._clone_state(self.model.net(fb))})
            self.stage_log.append({
                "stage": idx,
                "direction": fb,
                "coupling_source": source,
                # which endpoint of the coupling consists of real (non-simulated) rows
                "anchored_endpoint": {"independent": "both", "reference": "data",
                                      "forward_model": "data", "backward_model": "prior"}[source],
                "n_updates": int(n_updates),
                "last_loss": float(last_loss),
                "coupling_seed": int(coupling_seed),
                "train_seed": int(train_seed),
            })
            prev_fb = fb

        self._fitted = True
        return self

    # ------------------------------------------------------------------ sampling
    def _has_direction(self, fb: FB) -> bool:
        return any(s["direction"] == fb for s in self.stage_log)

    @torch.no_grad()
    def sample(self, n: int, seed: Optional[int] = None, steps: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic rows in transformed space: start from the Gaussian prior
        and integrate the latest backward model.

        ONE generator drives the start sample and all SDE noise, so ``seed`` makes
        the call reproducible and the first Brownian increment is independent of
        the start sample. With seed=None fresh entropy is used.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        if not self._has_direction("b"):
            raise RuntimeError("No backward ('b') model found. Ensure fb_sequence contains 'b'.")
        if int(n) <= 0:
            raise ValueError("n must be positive")

        gen = self._make_generator(seed)
        net = self.model.net("b")
        net.eval()

        zstart = self.reference.sample(n=int(n), generator=gen).to(self.device)
        z = self._sample_sde(net=net, fb="b", zstart=zstart, steps=steps, generator=gen,
                             noise=bool(self.cfg.noise))
        return z.detach().cpu().numpy()

    def sample_df(self, n: int, seed: Optional[int] = None, steps: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame(self.sample(n, seed=seed, steps=steps), columns=self.columns_)

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        """Inference-complete state (plain containers + tensors); reload never refits."""
        return {
            "format": CHECKPOINT_FORMAT,
            "canonical_id": self.canonical_id,
            "variant_id": self.variant_id,
            "config": asdict(self.cfg),
            "dim": self.dim,
            "columns": None if self.columns_ is None else list(self.columns_),
            "orientation": {"x0": "data", "x1": "prior", "generation": "backward"},
            "sigma": float(self.cfg.sigma),
            "num_steps": int(self.cfg.num_steps),
            "time_clamp": [float(self.cfg.eps), 1.0 - float(self.cfg.eps)],
            "model_state": {
                "f": self._clone_state(self.model.net_f),
                "b": self._clone_state(self.model.net_b),
            },
            "snapshots": self.snapshots,
            "stage_log": self.stage_log,
            "fitted": bool(self._fitted),
        }

    def save_checkpoint(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, device: Optional[str] = None) -> "IMFDSBMSolver":
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported {CANONICAL_ID} checkpoint format: {state.get('format')!r}")
        cfg_dict = dict(state["config"])
        cfg_dict["fb_sequence"] = tuple(cfg_dict["fb_sequence"])
        if device is not None:
            cfg_dict["device"] = device
        solver = cls(dim=int(state["dim"]), cfg=IMFDSBMConfig(**cfg_dict))
        solver.columns_ = state["columns"]
        solver.model.net_f.load_state_dict(state["model_state"]["f"], strict=True)
        solver.model.net_b.load_state_dict(state["model_state"]["b"], strict=True)
        solver.model.eval()
        solver.snapshots = state["snapshots"]
        solver.stage_log = state["stage_log"]
        solver._fitted = bool(state["fitted"])
        return solver
