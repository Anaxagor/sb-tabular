"""
IPF / Diffusion Schrödinger Bridge (DSB) for fully continuous tabular data with
one TIME-CONDITIONED MLP per direction (registry id ``dsb_ct_joint_mlp``).

Reference: De Bortoli, Thornton, Heng, Doucet, "Diffusion Schrödinger Bridge with
Applications to Score-Based Generative Modeling" (NeurIPS 2021), Algorithm 1 with
the mean-matching regression targets of Proposition 3.

Setting
-------
Grid t_0 = 0 < ... < t_K = T with steps gamma_k = t_{k+1} - t_k. "Edge k" joins
the states X_k (at t_k) and X_{k+1} (at t_{k+1}) and uses gamma_k in BOTH
directions. Data lives at t_0, the N(0, I) prior at t_K.

Declared reference process (used to simulate IPF iteration 0)::

    dX = -alpha_ou * X dt + sigma dW   on [0, T],   T = sum_k gamma_k

discretised by Euler-Maruyama, X_{k+1} = X_k - gamma_k alpha_ou X_k + sigma sqrt(gamma_k) Z.
``alpha_ou = 0`` is the Brownian reference. With the defaults (alpha_ou = 1,
sigma = sqrt(2)) the stationary law of the reference is the N(0, I) prior, as in
the DSB paper. An untrained network is never used as a reference.

Units: the networks are DISPLACEMENTS (units of x, not drifts)
--------------------------------------------------------------
With mean maps F_k(x) = x + d_f(x, k) and B_k(x) = x + d_b(x, k) the chains are

    forward :  X_{k+1} = F_k(X_k)     + sigma sqrt(gamma_k) Z
    backward:  X_k     = B_k(X_{k+1}) + sigma sqrt(gamma_k) Z

so the sampler ADDS the network output; it is never multiplied by gamma_k. (For
the reference, d_f(x, k) = -gamma_k alpha_ou x.) The IPF half-steps regress

    backward update (FORWARD chain simulated from the data):
        d_b(X_{k+1}, k)  on  F_k(X_k) - F_k(X_{k+1})          [= B_k(X_{k+1}) - X_{k+1}]
    forward update (BACKWARD chain simulated from the prior):
        d_f(X_k, k)      on  B_k(X_{k+1}) - B_k(X_k)          [= F_k(X_k) - X_k]

Since X_{k+1} = F_k(X_k) + sigma sqrt(gamma_k) Z, the backward target equals
-gamma_k f_k(X_{k+1}) - sigma sqrt(gamma_k) Z, whose conditional mean given X_{k+1}
is gamma_k (-f_k + sigma^2 grad log p_{k+1})(X_{k+1}): the time-reversal drift times
gamma_k. A naive reverse increment X_k - X_{k+1} is NOT used.

Caches are full trajectories of the opposite process and are ALWAYS stochastic.

Time conditioning
-----------------
The sinusoidal embedding (max_period 1e4) cannot resolve times in [1e-4, ~0.2],
so the networks receive the rescaled clock ``time_scale * t / T`` (default
time_scale = 1000, i.e. a DDPM-like 0..1000 clock). The forward network on edge k
is labelled with t_k (time of its input X_k), the backward network with t_{k+1}
(time of its input X_{k+1}). Training, cache building and sampling all obtain the
label from the single method ``_displacement``.

The network is a function of continuous t, but its output is a per-step
displacement tied to the training grid: sampling must use the grid it was trained
on (the grid is part of the checkpoint).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from sbtab.bridge.losses import RegressionLoss
from sbtab.bridge.reference import GaussianReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.mlp import TimeConditionedMLP, TimeMLPConfig
from sbtab.models.neural.time_embedding import SinusoidalTimeEmbeddingConfig

CHECKPOINT_FORMAT = "sbtab.ipf_dsb_mlp.v1"

Direction = Literal["forward", "backward"]
Simulator = Literal["reference", "net_f", "net_b"]


@dataclass
class IPFDSBConfig:
    """
    Configuration of the MLP IPF-DSB solvers.

    Reference process: dX = -alpha_ou X dt + sigma dW on [0, T] with
    T = sum(gamma) (or ``horizon`` if given). NOTE: the default geometric schedule
    has a SHORT horizon (T ~= 0.046 for 20 steps): the reference then does not
    carry the data to the prior and many IPF iterations are needed. For data that
    is not already close to N(0, I) set ``horizon`` to ~2-3 (OU mixing time).

    Training cost. One epoch simulates one fresh cache with the frozen opposite
    process and makes one pass over it (partial batches are kept). A cache holds
    about ``cache_batches * batch_size`` regression rows, so an epoch is about
    ``cache_batches`` optimizer updates. ``steps_per_phase`` (if set) is the exact
    number of updates of a half-iteration (caches are refreshed as needed) and
    overrides ``epochs_per_phase``.
    """
    ipf_iters: int = 6

    # time discretisation
    num_steps: int = 20
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: Literal["linear", "geom", "uniform"] = "geom"
    horizon: Optional[float] = None  # rescale the steps so that sum(gamma) == horizon

    # reference process dX = -alpha_ou X dt + sigma dW (alpha_ou = 0: Brownian)
    alpha_ou: float = 1.0
    sigma: float = math.sqrt(2.0)

    # training
    batch_size: int = 512
    cache_batches: int = 200
    steps_per_phase: Optional[int] = None
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs_per_phase: int = 1
    grad_clip: Optional[float] = 1.0

    # MLP architecture
    hidden_units: int = 256
    time_features: int = 64
    n_layers: int = 4
    dropout: float = 0.0
    time_scale: float = 1000.0  # network clock = time_scale * t / T

    # sampler. False = deterministic mean chain, a HEURISTIC (not the DSB sampler);
    # caches stay stochastic regardless. See IPFDSBSolver.variant_id.
    noise: bool = True

    device: str = "cpu"
    seed: int = 42


@dataclass
class IPFCache:
    """
    Regression cache of one half-iteration, edge-major.

    x[k] : (M, D) network inputs of edge k. Backward training: X_{k+1} of the
           simulated FORWARD chain; forward training: X_k of the BACKWARD chain.
    y[k] : (M, D) displacement targets of edge k.
    path : (K + 1, M, D) simulated states, path[i] at grid index i (optional).
    """
    direction: str        # direction being trained
    simulated_with: str   # "reference" | "net_f" | "net_b"
    x: torch.Tensor
    y: torch.Tensor
    path: Optional[torch.Tensor] = None

    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0] * self.x.shape[1])


class IPFDSBSolver:
    """
    IPF-DSB with two TIME-CONDITIONED MLPs (``time_parameterization`` =
    "time_conditioned"): net_f / net_b map (x, clock) to the forward / backward
    displacement of the edge. See the module docstring for the algorithm, the
    declared reference, the units and the clock.

    Public API
      fit(train) -> self                  train: DataFrame | ndarray | Tensor (N, dim), transformed space
      sample(n, seed=None, batch_size=None) -> ndarray (n, dim)
      sample_paths(n, seed=None, batch_size=None) -> ndarray (n, K + 1, dim); [:, i] is the state at grid index i
      save_checkpoint(path) / load_checkpoint(path, device=None)
      describe() -> dict, variant_id, n_updates, stage_log
    """

    canonical_id = "dsb_ct_joint_mlp"
    time_parameterization = "time_conditioned"
    config_class = IPFDSBConfig

    def __init__(self, dim: int, cfg: IPFDSBConfig):
        self.dim = int(dim)
        self.cfg = cfg
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if cfg.ipf_iters < 1:
            raise ValueError("ipf_iters must be >= 1")
        if cfg.batch_size < 1 or cfg.cache_batches < 1:
            raise ValueError("batch_size and cache_batches must be >= 1")
        if cfg.steps_per_phase is None and cfg.epochs_per_phase < 1:
            raise ValueError("epochs_per_phase must be >= 1")
        if cfg.steps_per_phase is not None and cfg.steps_per_phase < 1:
            raise ValueError("steps_per_phase must be >= 1 (or None)")
        if not cfg.sigma > 0:
            raise ValueError("sigma must be positive")
        if cfg.alpha_ou < 0:
            raise ValueError("alpha_ou must be >= 0 (0 = Brownian reference)")
        if not cfg.time_scale > 0:
            raise ValueError("time_scale must be positive")

        self.device = torch.device(cfg.device)
        self.sigma = float(cfg.sigma)
        self.alpha_ou = float(cfg.alpha_ou)

        self._install_grid(TimeGrid(
            num_steps=cfg.num_steps,
            gamma_min=cfg.gamma_min,
            gamma_max=cfg.gamma_max,
            schedule=cfg.schedule,
            horizon=cfg.horizon,
            device=self.device,
            dtype=torch.float32,
        ))

        # Caches are ALWAYS stochastic; cfg.noise only affects the sampler.
        self._cache_integrator = EulerMaruyama(noise=True, sigma=self.sigma)
        self._sample_integrator = EulerMaruyama(noise=bool(cfg.noise), sigma=self.sigma)
        self.reference = GaussianReference(dim=self.dim, device=self.device)  # N(0, I) prior at t_K
        self.loss = RegressionLoss(kind="mse", reduction="mean")

        self._build_networks()
        self.stage_log: List[dict] = []
        self._edge_trace: List[Tuple[str, int]] = []
        self._fitted = False

    # ------------------------------------------------------------------ setup
    def _install_grid(self, timegrid: TimeGrid) -> None:
        self.timegrid = timegrid
        self.K = int(timegrid.num_steps)
        self._gamma = timegrid.dt()          # (K,)  gamma_k, shared by both directions of edge k
        grid = timegrid.grid()               # (K + 1,)
        scale = float(self.cfg.time_scale) / float(timegrid.T)
        # Rescaled network clock: time of the state the network is evaluated at.
        self._clock: Dict[str, torch.Tensor] = {
            "forward": grid[:-1] * scale,    # edge k, input X_k     at t_k
            "backward": grid[1:] * scale,    # edge k, input X_{k+1} at t_{k+1}
        }
        if self.alpha_ou * float(self._gamma.max()) >= 1.0:
            raise ValueError(
                "alpha_ou * max(gamma) must be < 1 for the Euler discretisation of the OU reference "
                f"(got {self.alpha_ou * float(self._gamma.max()):.3g})"
            )

    @contextmanager
    def _forked_rng(self, seed: int):
        """Seed the global RNG locally (network init) without clobbering the caller's RNG."""
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            yield

    def _build_network(self) -> nn.Module:
        cfg = self.cfg
        te_dim = cfg.time_features if cfg.time_features % 2 == 0 else cfg.time_features + 1
        return TimeConditionedMLP(TimeMLPConfig(
            in_dim=self.dim,
            hidden_dim=cfg.hidden_units,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            time_emb=SinusoidalTimeEmbeddingConfig(dim=te_dim),
        ))

    def _build_networks(self) -> None:
        with self._forked_rng(self.cfg.seed):
            self.net_f = self._build_network().to(self.device).eval()
            self.net_b = self._build_network().to(self.device).eval()

    def _generator(self, seed: int) -> torch.Generator:
        gen = torch.Generator(device=str(self.device))
        gen.manual_seed(int(seed))
        return gen

    def _as_tensor(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            arr = x.to_numpy(dtype=np.float32, copy=True)
            return torch.from_numpy(arr).to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        raise TypeError(f"Unsupported type: {type(x)}")

    # ------------------------------------------------------------------ fields
    def _displacement(self, net: nn.Module, direction: Direction, x: torch.Tensor, k) -> torch.Tensor:
        """
        Displacement d(x, k) of edge(s) k in units of x. ``k`` is an int (all rows
        on one edge) or a LongTensor (R,) of per-row edges. This is the ONLY place
        where the clock label is attached, for training and sampling alike.
        """
        clock = self._clock[direction]
        if isinstance(k, torch.Tensor):
            t = clock[k].unsqueeze(1)
        else:
            t = clock[int(k)].expand(x.shape[0], 1)
        return net(x, t)

    def _reference_drift(self, x: torch.Tensor) -> torch.Tensor:
        """Drift f(x) of the declared reference dX = f(X) dt + sigma dW."""
        return -self.alpha_ou * x

    def _mean_map(self, sim: Simulator, x: torch.Tensor, k: int) -> torch.Tensor:
        """F_k(x) for sim in {"reference", "net_f"}; B_k(x) for sim == "net_b"."""
        self._edge_trace.append((sim, int(k)))
        if sim == "reference":
            return x + self._gamma[k] * self._reference_drift(x)
        if sim == "net_f":
            return x + self._displacement(self.net_f, "forward", x, k)
        if sim == "net_b":
            return x + self._displacement(self.net_b, "backward", x, k)
        raise ValueError(f"unknown simulator: {sim}")

    def _assert_trace(self, sim: str, order: List[int], calls_per_edge: int) -> None:
        expected = [(sim, k) for k in order for _ in range(calls_per_edge)]
        if self._edge_trace != expected:
            raise AssertionError(f"edge sweep mismatch: expected {expected}, got {self._edge_trace}")

    # ------------------------------------------------------------------ caches
    def _n_trajectories(self) -> int:
        """Trajectories per cache: every trajectory yields K rows (one per edge)."""
        rows = self.cfg.cache_batches * self.cfg.batch_size
        return max(1, -(-rows // self.K))

    def _draw_starts(self, direction: Direction, x_data: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        M = self._n_trajectories()
        if direction == "forward":   # the forward net is trained on BACKWARD chains started at the prior
            return self.reference.sample(n=M, generator=gen)
        N = x_data.shape[0]
        reps = -(-M // N)
        idx = torch.cat([torch.randperm(N, generator=gen, device=self.device) for _ in range(reps)])[:M]
        return x_data[idx]

    @torch.no_grad()
    def _make_cache(
        self,
        direction: Direction,
        simulated_with: Simulator,
        x_start: torch.Tensor,
        gen: torch.Generator,
        keep_path: bool = False,
    ) -> IPFCache:
        """
        Simulate FULL trajectories of the opposite process and build the
        mean-matching regression rows of every edge.

        direction="backward": forward chain from the data (reference or net_f);
            edge k: input X_{k+1}, target F_k(X_k) - F_k(X_{k+1}).
        direction="forward": backward chain from the prior (net_b);
            edge k: input X_k,     target B_k(X_{k+1}) - B_k(X_k).

        In both cases: input = the NEW state of the step, target =
        mean_map(old) - mean_map(new), with the mean map, the noise scale
        sqrt(gamma_k) and the cache slot all indexed by the same edge k. The
        simulation noise is always on (independent of cfg.noise).
        """
        forward_sim = simulated_with in ("reference", "net_f")
        if (direction == "backward") != forward_sim:
            raise ValueError(
                f"the {direction} network must be trained on the opposite process, not on {simulated_with!r}"
            )
        K = self.K
        order = list(range(K)) if forward_sim else list(range(K - 1, -1, -1))
        self.net_f.eval()
        self.net_b.eval()
        self._edge_trace = []

        xs: List[Optional[torch.Tensor]] = [None] * K
        ys: List[Optional[torch.Tensor]] = [None] * K
        path: List[Optional[torch.Tensor]] = [None] * (K + 1)
        path[0 if forward_sim else K] = x_start
        zero = torch.zeros_like(x_start)

        x_old = x_start
        for k in order:
            mean_old = self._mean_map(simulated_with, x_old, k)
            x_new = self._cache_integrator.step(mean_old, drift=zero, gamma=self._gamma[k], generator=gen)
            mean_new = self._mean_map(simulated_with, x_new, k)
            xs[k] = x_new
            ys[k] = mean_old - mean_new
            path[k + 1 if forward_sim else k] = x_new
            x_old = x_new

        self._assert_trace(simulated_with, order, calls_per_edge=2)
        return IPFCache(
            direction=direction,
            simulated_with=simulated_with,
            x=torch.stack(xs, dim=0),
            y=torch.stack(ys, dim=0),
            path=torch.stack(path, dim=0) if keep_path else None,
        )

    # ------------------------------------------------------------------ training
    def _iter_batches(self, cache: IPFCache, gen: torch.Generator) -> Iterator[tuple]:
        """Shuffled minibatches of (x, edge index, target) rows pooled over edges; partial batches kept."""
        K, M, D = cache.x.shape
        X = cache.x.reshape(K * M, D)
        Y = cache.y.reshape(K * M, D)
        k_idx = torch.arange(K, device=self.device).repeat_interleave(M)
        perm = torch.randperm(K * M, generator=gen, device=self.device)
        bs = int(self.cfg.batch_size)
        for s in range(0, K * M, bs):
            idx = perm[s:s + bs]
            yield X[idx], k_idx[idx], Y[idx]

    def _batch_loss(self, net: nn.Module, direction: Direction, batch: tuple) -> torch.Tensor:
        xb, kb, yb = batch
        return self.loss(self._displacement(net, direction, xb, kb), yb)

    def _clip_gradients(self, net: nn.Module) -> None:
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(self.cfg.grad_clip))

    def _record_update(self, direction: Direction, batch: tuple) -> None:
        """Hook: called once per optimizer update."""

    def _check_phase(self, direction: Direction, entry: dict) -> None:
        """Hook: called once per half-iteration, after training."""

    def _train_phase(
        self,
        iteration: int,
        direction: Direction,
        simulated_with: Simulator,
        x_data: torch.Tensor,
        gen: torch.Generator,
    ) -> dict:
        net = self.net_b if direction == "backward" else self.net_f
        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        max_updates = self.cfg.steps_per_phase
        n_updates, epochs, rows, last_loss = 0, 0, 0, float("nan")

        done = False
        while not done:
            # A fresh cache per epoch: the opposite process is frozen, so this only
            # adds new trajectories (cf. the cache refresh of the DSB reference code).
            cache = self._make_cache(direction, simulated_with, self._draw_starts(direction, x_data, gen), gen)
            rows += cache.n_rows
            net.train()
            for batch in self._iter_batches(cache, gen):
                loss = self._batch_loss(net, direction, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                self._clip_gradients(net)
                opt.step()
                n_updates += 1
                last_loss = float(loss.detach())
                self._record_update(direction, batch)
                if max_updates is not None and n_updates >= max_updates:
                    done = True
                    break
            net.eval()
            epochs += 1
            if max_updates is None and epochs >= self.cfg.epochs_per_phase:
                done = True

        if n_updates <= 0:
            raise RuntimeError(f"IPF half-iteration ({direction}, iteration {iteration}) made no optimizer update")
        if not math.isfinite(last_loss):
            raise RuntimeError(f"non-finite loss in IPF half-iteration ({direction}, iteration {iteration})")
        entry = {
            "iteration": int(iteration),
            "trained": direction,
            "simulated_with": simulated_with,
            "n_updates": int(n_updates),
            "epochs": int(epochs),
            "cache_rows": int(rows),
            "last_loss": last_loss,
        }
        self._check_phase(direction, entry)
        return entry

    def fit(self, train: pd.DataFrame | np.ndarray | torch.Tensor) -> "IPFDSBSolver":
        """
        Run ``ipf_iters`` IPF iterations (backward half-step, then forward
        half-step). Data ~ P_0, prior N(0, I) ~ P_T. The first backward half-step
        simulates the DECLARED REFERENCE; all later ones simulate net_f. fit()
        always starts from freshly initialised networks and is deterministic given
        cfg.seed (on CPU); it does not touch the global RNG.
        """
        x_data = self._as_tensor(train)
        if x_data.ndim != 2 or x_data.shape[1] != self.dim:
            raise ValueError(f"Expected train shape (N,{self.dim}), got {tuple(x_data.shape)}")
        if x_data.shape[0] < 1:
            raise ValueError("train must contain at least one row")
        if not torch.isfinite(x_data).all():
            raise ValueError("train contains NaN/inf; apply preprocessing first")

        self._build_networks()
        self.stage_log = []
        self._fitted = False
        gen = self._generator(self.cfg.seed)

        # Dropout uses torch's global RNG; cache generators alone do not seed it.
        with self._forked_rng(self.cfg.seed + 104729):
            for it in range(self.cfg.ipf_iters):
                sim: Simulator = "reference" if it == 0 else "net_f"
                self.stage_log.append(self._train_phase(it, "backward", sim, x_data, gen))
                self.stage_log.append(self._train_phase(it, "forward", "net_b", x_data, gen))

        self._fitted = True
        return self

    @property
    def n_updates(self) -> int:
        return int(sum(s["n_updates"] for s in self.stage_log))

    # ------------------------------------------------------------------ sampling
    def _sampler_step(self, x: torch.Tensor, k: int, gen: Optional[torch.Generator]) -> torch.Tensor:
        """
        One backward step on edge k: x + d_b(x, k) [+ sigma sqrt(gamma_k) Z].
        The displacement is ADDED as is; it is not multiplied by gamma_k.
        """
        mean = self._mean_map("net_b", x, k)
        return self._sample_integrator.step(mean, drift=torch.zeros_like(mean), gamma=self._gamma[k], generator=gen)

    @torch.no_grad()
    def _sample_impl(self, n: int, seed: Optional[int], batch_size: Optional[int], return_paths: bool) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive")
        if seed is None:
            # draw the seed from the global RNG: unseeded calls differ, and follow torch.manual_seed
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        # ONE generator drives the prior draw and all path noise of every chunk.
        gen = self._generator(seed)
        self.net_b.eval()

        order = list(range(self.K - 1, -1, -1))
        bs = int(batch_size) if batch_size else int(n)
        outs = []
        for s in range(0, n, bs):
            m = min(bs, n - s)
            x = self.reference.sample(n=m, generator=gen)
            path = [x]
            self._edge_trace = []
            for k in order:
                x = self._sampler_step(x, k, gen)
                if return_paths:
                    path.append(x)
            self._assert_trace("net_b", order, calls_per_edge=1)
            # path was collected from t_K down to t_0; report it indexed by grid index
            outs.append(torch.stack(path[::-1], dim=1) if return_paths else x)
        return torch.cat(outs, dim=0).detach().cpu().numpy()

    def sample(self, n: int, seed: Optional[int] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Draw x_K ~ N(0, I) and run the backward chain k = K-1..0 with net_b.
        The result is a deterministic function of (n, seed, batch_size).
        """
        return self._sample_impl(n, seed, batch_size, return_paths=False)

    def sample_paths(self, n: int, seed: Optional[int] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """(n, K + 1, dim) backward paths; [:, K] is the prior draw, [:, 0] equals sample(n, seed, batch_size)."""
        return self._sample_impl(n, seed, batch_size, return_paths=True)

    # ------------------------------------------------------------------ metadata
    @property
    def variant_id(self) -> str:
        """Canonical id for the DSB sampler; '<id>_noiseless_heuristic' when cfg.noise is False."""
        return self.canonical_id if self.cfg.noise else f"{self.canonical_id}_noiseless_heuristic"

    def describe(self) -> dict:
        return {
            "solver_id": self.canonical_id,
            "variant_id": self.variant_id,
            "algorithm": "IPF-DSB, mean-matching (De Bortoli et al. 2021, Alg. 1 / Prop. 3)",
            "time_parameterization": self.time_parameterization,
            "network_output": "displacement d(x, k) = mean_map_k(x) - x, units of x; added by the sampler as is",
            "reference": {
                "kind": "ou" if self.alpha_ou > 0 else "brownian",
                "sde": "dX = -alpha_ou * X dt + sigma dW",
                "alpha_ou": self.alpha_ou,
                "sigma": self.sigma,
                "horizon_T": float(self.timegrid.T),
                "used_for": "simulation of IPF iteration 0",
            },
            "prior": "N(0, I) at t_K; data at t_0; generation runs the backward chain",
            "num_steps": self.K,
            "clock": self._describe_clock(),
            "cache_noise": True,
            "sampler_noise": bool(self.cfg.noise),
            "sampler_note": None if self.cfg.noise else
                "noise=False is a heuristic deterministic mean chain, not the DSB sampler",
        }

    def _describe_clock(self) -> Optional[dict]:
        return {
            "network_time_input": "time_scale * t / T",
            "time_scale": float(self.cfg.time_scale),
            "forward_label": "t_k (time of the input X_k)",
            "backward_label": "t_{k+1} (time of the input X_{k+1})",
        }

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        """Inference-complete state; reload never refits."""
        cpu = lambda sd: {k: v.detach().cpu().clone() for k, v in sd.items()}  # noqa: E731
        return {
            "format": CHECKPOINT_FORMAT,
            "solver_id": self.canonical_id,
            "variant_id": self.variant_id,
            "time_parameterization": self.time_parameterization,
            "dim": self.dim,
            "config": asdict(self.cfg),
            "timegrid": [float(v) for v in self.timegrid._grid64().tolist()],
            "sigma": self.sigma,
            "alpha_ou": self.alpha_ou,
            "net_f": cpu(self.net_f.state_dict()),
            "net_b": cpu(self.net_b.state_dict()),
            "stage_log": [dict(s) for s in self.stage_log],
            "fitted": bool(self._fitted),
        }

    def save_checkpoint(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, device: Optional[str] = None) -> "IPFDSBSolver":
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported IPF-DSB checkpoint format: {state.get('format')!r}")
        if state.get("solver_id") != cls.canonical_id:
            raise ValueError(f"checkpoint belongs to {state.get('solver_id')!r}, not {cls.canonical_id!r}")
        cfg_dict = dict(state["config"])
        if device is not None:
            cfg_dict["device"] = device
        solver = cls(int(state["dim"]), cls.config_class(**cfg_dict))
        if float(state["sigma"]) != solver.sigma or float(state["alpha_ou"]) != solver.alpha_ou:
            raise ValueError("checkpoint sigma/alpha_ou disagree with its config")
        # The stored grid is authoritative: the displacement networks are tied to it.
        solver._install_grid(TimeGrid.from_points(state["timegrid"], device=solver.device, dtype=torch.float32))
        solver.net_f.load_state_dict(state["net_f"])
        solver.net_b.load_state_dict(state["net_b"])
        solver.net_f.eval()
        solver.net_b.eval()
        solver.stage_log = [dict(s) for s in state["stage_log"]]
        solver._fitted = bool(state["fitted"])
        return solver
