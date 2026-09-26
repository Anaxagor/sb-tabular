from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from sbtab.bridge.reference import GaussianReference
from sbtab.models.neural.mlp_discrete_joint import (
    MLPTimeDiscretizedField,
    StepMLPJointConfig,
)


FB = Literal["f", "b"]

CANONICAL_ID = "dsbm_dt_joint_mlp"
CHECKPOINT_FORMAT = "sbtab.dsbm_dt_joint_mlp.v1"

# Feature modes that feed the chain start x0 to the drift model. In DSBM x0 is the
# regression answer at train time (target = (z0 - x_t)/t) and the prior start at
# sample time, and conditioning on it breaks the Markov projection.
_X0_FEATURE_MODES = ("x_x0", "x_x0_t")


@dataclass
class IMFDSBMDiscreteJointMLPConfig:
    # IMF stage sequence: non-empty, only "f"/"b", strictly alternating (a stage is
    # trained on the coupling of the opposite direction). sample() needs a "b".
    fb_sequence: Sequence[FB] = ("b", "f", "b", "f", "b")

    num_steps: int = 32
    sigma: float = 0.1
    # Unused: the state-time grids need no clipping (both targets are finite on
    # them). Kept so existing configs keep loading.
    eps: float = 1e-3

    # "ind": independent coupling data (x) prior -- the DSBM-IMF initialiser.
    # "ref": z1 = z0 + sigma * N(0, I), the reference (IPF-like) coupling; its t=1
    #        marginal is data * N(0, sigma^2 I), not the N(0, I) prior.
    first_coupling: Literal["ind", "ref"] = "ind"
    n_noise_per_pair: int = 1

    # noise=False selects the NOISELESS HEURISTIC sampler for generation only.
    # Couplings simulated during fit() always use noise.
    noise: bool = True
    seed: int = 42

    # feature_mode must be "x" or "x_t"; the x0 modes are rejected.
    field: StepMLPJointConfig = field(default_factory=StepMLPJointConfig)


class IMFDSBMDiscreteJointMLPSolver:
    """
    IMF + DSBM solver on a discrete time grid with one joint MLP per step and
    direction (registry id ``dsbm_dt_joint_mlp``).

    Endpoints: x0 = data rows (t=0), x1 = N(0, I) prior rows (t=1).

    Time grids. Model k always drives the edge between k/N and (k+1)/N, and it is
    trained at the time of the state it is APPLIED to:
        forward  model k: t_f[k] = k/N       (state at the left end of the edge)
        backward model k: t_b[k] = (k+1)/N   (state at the right end of the edge)
    The forward target is finite at t=0 and the backward target at t=1, so no
    clipping is needed. With these grids an Euler step of size 1/N with the exact
    bridge drift lands exactly on a point-mass endpoint.

    Sampler: x <- x + drift_k(x) dt + sigma sqrt(dt) eps. The drift is trained for
    the STOCHASTIC bridge and contains the score term. With ``cfg.noise=False`` the
    same drift is integrated without noise: that is a heuristic, NOT a
    probability-flow ODE, and it does not preserve the marginals. It is reported as
    ``variant_id == "dsbm_dt_joint_mlp_noiseless_heuristic"``. The flag only
    affects sample(); couplings simulated during fit() always use noise.
    """

    canonical_id = CANONICAL_ID

    def __init__(self, dim: int, cfg: IMFDSBMDiscreteJointMLPConfig):
        self.dim = int(dim)
        self.cfg = cfg
        self._validate_config(cfg)

        self.columns_: Optional[List[str]] = None
        self.t_grid_f, self.t_grid_b = self._make_t_grids(int(cfg.num_steps))

        self.reference = GaussianReference(dim=self.dim, device=torch.device("cpu"))

        self.field_f: Optional[MLPTimeDiscretizedField] = None
        self.field_b: Optional[MLPTimeDiscretizedField] = None

        self.stage_log: List[dict] = []
        self._fitted = False

    # ------------------------------------------------------------------ config
    @staticmethod
    def _validate_config(cfg: IMFDSBMDiscreteJointMLPConfig) -> None:
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
        if int(cfg.n_noise_per_pair) < 1:
            raise ValueError("n_noise_per_pair must be >= 1")
        if not float(cfg.sigma) >= 0.0:
            raise ValueError("sigma must be >= 0")
        mode = cfg.field.feature_mode
        if mode in _X0_FEATURE_MODES:
            raise ValueError(
                f"feature_mode={mode!r} is not valid for DSBM: x0 is the regression answer at train "
                "time and the chain start at sample time, and conditioning on it breaks the Markov "
                "projection. Use 'x' or 'x_t'."
            )
        if mode not in ("x", "x_t"):
            raise ValueError(f"Unknown feature_mode={mode!r} (allowed: 'x', 'x_t')")

    @property
    def variant_id(self) -> str:
        return self.canonical_id if self.cfg.noise else f"{self.canonical_id}_noiseless_heuristic"

    @property
    def n_updates(self) -> int:
        """Total number of optimizer updates over all IMF stages of the last fit()."""
        return int(sum(s["n_updates"] for s in self.stage_log))

    # ------------------------------------------------------------------ utilities
    @staticmethod
    def _make_t_grids(N: int) -> tuple[np.ndarray, np.ndarray]:
        """State-time grids: t_f[k] = k/N, t_b[k] = (k+1)/N, k = 0..N-1."""
        if N < 1:
            raise ValueError("num_steps must be >= 1")
        k = np.arange(N, dtype=np.float64)
        return k / N, (k + 1.0) / N

    def _t_grid(self, fb: FB) -> np.ndarray:
        return self.t_grid_f if fb == "f" else self.t_grid_b

    @staticmethod
    def _make_generator(seed: Optional[int]) -> torch.Generator:
        """Seeded CPU generator; seed=None draws fresh entropy."""
        gen = torch.Generator(device="cpu")
        if seed is None:
            gen.seed()
        else:
            gen.manual_seed(int(seed))
        return gen

    @staticmethod
    def _randn(shape, generator: torch.Generator) -> np.ndarray:
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).numpy()

    def _as_array(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            self.columns_ = list(x.columns)
            arr = x.to_numpy(dtype=np.float32, copy=True)
        else:
            arr = np.array(x, dtype=np.float32, copy=True)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N,{self.dim}), got {tuple(arr.shape)}")
        if arr.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")
        return arr

    def _sample_reference(self, n: int, generator: torch.Generator) -> np.ndarray:
        return self.reference.sample(n=n, generator=generator).detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ DSBM pieces
    def _dsbm_train_tuple(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        t: float,
        fb: FB,
        *,
        generator: Optional[torch.Generator] = None,
        noise: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bridge-matching tuple at a fixed time t, the SAME noise in x_t and target:

          x_t = (1-t) z0 + t z1 + sigma sqrt(t(1-t)) noise
          fb == "f": target = (z1 - z0) - sigma sqrt(t/(1-t)) noise   = (z1 - x_t)/(1-t),  t < 1
          fb == "b": target = -(z1 - z0) - sigma sqrt((1-t)/t) noise  = (z0 - x_t)/t,      t > 0
        """
        t = float(t)
        if fb == "f" and not 0.0 <= t < 1.0:
            raise ValueError("the forward target needs t in [0, 1)")
        if fb == "b" and not 0.0 < t <= 1.0:
            raise ValueError("the backward target needs t in (0, 1]")
        sigma = float(self.cfg.sigma)

        if noise is None:
            noise = self._randn(z0.shape, generator)
        noise = np.asarray(noise, dtype=np.float32)

        xt = (1.0 - t) * z0 + t * z1 + sigma * math.sqrt(t * (1.0 - t)) * noise

        delta = z1 - z0
        if fb == "f":
            target = delta - sigma * math.sqrt(t / (1.0 - t)) * noise
        else:
            target = -delta - sigma * math.sqrt((1.0 - t) / t) * noise

        return xt.astype(np.float32), target.astype(np.float32)

    def _build_step_batch(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        t: float,
        fb: FB,
        generator: torch.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        xt_list, y_list = [], []
        for _ in range(int(self.cfg.n_noise_per_pair)):
            xt, target = self._dsbm_train_tuple(z0, z1, t, fb, generator=generator)
            xt_list.append(xt)
            y_list.append(target)
        return np.concatenate(xt_list, axis=0), np.concatenate(y_list, axis=0)

    def _train_direction(self, fb: FB, z0: np.ndarray, z1: np.ndarray, seed: int) -> tuple[int, int]:
        """
        Fit fresh per-step MLPs for ``fb`` on the coupling, each at its state time.
        Returns (n_models_fitted, n_updates).
        """
        t_grid = self._t_grid(fb)
        step_field = MLPTimeDiscretizedField(dim=self.dim, t_grid=t_grid, cfg=self.cfg.field)
        gen = self._make_generator(seed)

        n_updates = 0
        for k, t in enumerate(t_grid):
            xt, target = self._build_step_batch(z0, z1, float(t), fb, gen)
            X_feat = step_field._build_features(xt, t=float(t))
            # per-step seed: weight init, shuffling and dropout honour cfg.seed
            # (stage seeds are consecutive integers; the stride keeps step seeds disjoint)
            n_k = step_field.fit_step(k, X_feat, target, seed=int(seed) * 1_000_003 + k)
            if n_k <= 0:
                raise RuntimeError(f"step model {k} ('{fb}') performed no optimizer updates")
            n_updates += int(n_k)

        if fb == "f":
            self.field_f = step_field
        else:
            self.field_b = step_field
        return len(t_grid), n_updates

    def _sample_with_direction(
        self,
        zstart: np.ndarray,
        direction: FB,
        generator: Optional[torch.Generator] = None,
        noise: bool = True,
    ) -> np.ndarray:
        """
        Euler-Maruyama over the N edges with dt = 1/N. Forward visits k = 0..N-1
        with the state at k/N; backward visits k = N-1..0 with the state at
        (k+1)/N -- exactly the time model k was trained at.

        ``noise=False`` is the noiseless heuristic (see class docstring).
        """
        step_field = self.field_f if direction == "f" else self.field_b
        if step_field is None:
            raise RuntimeError(f"Direction '{direction}' has not been trained.")

        N = int(self.cfg.num_steps)
        dt = 1.0 / float(N)
        sqrt_dt = math.sqrt(dt)
        sigma = float(self.cfg.sigma)
        if generator is None:
            generator = self._make_generator(None)

        x = np.array(zstart, dtype=np.float32, copy=True)
        step_indices = range(N) if direction == "f" else range(N - 1, -1, -1)

        for k in step_indices:
            drift = np.asarray(step_field.predict_step(k, x), dtype=np.float32).reshape(x.shape)
            x = x + drift * dt
            if noise and sigma != 0.0:
                x = x + sigma * sqrt_dt * self._randn(x.shape, generator)

        return x.astype(np.float32, copy=False)

    def _generate_coupling(
        self,
        x_data: np.ndarray,
        x_prior: np.ndarray,
        prev_fb: Optional[FB],
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Coupling (z0, z1) for the next IMF stage and its source. Later stages
        simulate the latest OPPOSITE-direction model, always with noise, anchored at
        the real rows it starts from (data after "f", prior after "b").
        """
        gen = self._make_generator(seed)

        if prev_fb is None:
            z0 = x_data.copy()
            if self.cfg.first_coupling == "ind":
                perm = torch.randperm(len(x_prior), generator=gen).numpy()
                return z0, x_prior[perm].copy(), "independent"
            if self.cfg.first_coupling == "ref":
                return z0, z0 + float(self.cfg.sigma) * self._randn(z0.shape, gen), "reference"
            raise ValueError(f"Unknown first_coupling={self.cfg.first_coupling!r}")

        if prev_fb == "f":
            zstart = x_data.copy()
            zend = self._sample_with_direction(zstart, "f", generator=gen, noise=True)
            return zstart, zend, "forward_model"

        zstart = x_prior.copy()
        zend = self._sample_with_direction(zstart, "b", generator=gen, noise=True)
        return zend, zstart, "backward_model"

    # ------------------------------------------------------------------ fit / sample
    def fit(self, train: pd.DataFrame | np.ndarray) -> "IMFDSBMDiscreteJointMLPSolver":
        """Fresh IMF fit. The training rows are not retained on the solver."""
        x_data = self._as_array(train)

        self._fitted = False
        self.field_f = None
        self.field_b = None
        self.stage_log = []

        x_prior = self._sample_reference(len(x_data), self._make_generator(int(self.cfg.seed) + 999))

        prev_fb: Optional[FB] = None
        for idx, fb in enumerate(self.cfg.fb_sequence):
            coupling_seed = int(self.cfg.seed) + 10_000 + idx
            train_seed = int(self.cfg.seed) + 20_000 + idx

            z0, z1, source = self._generate_coupling(x_data, x_prior, prev_fb, coupling_seed)
            n_models, n_updates = self._train_direction(fb, z0, z1, train_seed)

            self.stage_log.append({
                "stage": idx,
                "direction": fb,
                "coupling_source": source,
                # which endpoint of the coupling consists of real (non-simulated) rows
                "anchored_endpoint": {"independent": "both", "reference": "data",
                                      "forward_model": "data", "backward_model": "prior"}[source],
                "n_models_fitted": int(n_models),
                "n_updates": int(n_updates),
                "coupling_seed": int(coupling_seed),
                "train_seed": int(train_seed),
            })
            prev_fb = fb

        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Start from the Gaussian prior and run the latest backward chain. ONE
        generator drives the start sample and all path noise: the same seed
        reproduces the output exactly, seed=None uses fresh entropy.
        """
        if not self._fitted or self.field_b is None:
            raise RuntimeError("Call fit() before sample(); a backward ('b') model must be trained.")
        if int(n) <= 0:
            raise ValueError("n must be positive")

        gen = self._make_generator(seed)
        zstart = self._sample_reference(int(n), gen)
        return self._sample_with_direction(zstart, "b", generator=gen, noise=bool(self.cfg.noise))

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame(self.sample(n, seed=seed), columns=self.columns_)

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        """Inference-complete state (plain containers + tensors); never training rows."""
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
            "t_grid_f": [float(t) for t in self.t_grid_f],
            "t_grid_b": [float(t) for t in self.t_grid_b],
            "fields": {
                "f": None if self.field_f is None else self.field_f.state(),
                "b": None if self.field_b is None else self.field_b.state(),
            },
            "stage_log": self.stage_log,
            "fitted": bool(self._fitted),
        }

    def save_checkpoint(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, device: Optional[str] = None) -> "IMFDSBMDiscreteJointMLPSolver":
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported {CANONICAL_ID} checkpoint format: {state.get('format')!r}")
        cfg_dict = dict(state["config"])
        cfg_dict["fb_sequence"] = tuple(cfg_dict["fb_sequence"])
        field_dict = dict(cfg_dict["field"])
        if device is not None:
            field_dict["device"] = device
        cfg_dict["field"] = StepMLPJointConfig(**field_dict)
        solver = cls(dim=int(state["dim"]), cfg=IMFDSBMDiscreteJointMLPConfig(**cfg_dict))
        solver.columns_ = state["columns"]
        solver.t_grid_f = np.asarray(state["t_grid_f"], dtype=np.float64)
        solver.t_grid_b = np.asarray(state["t_grid_b"], dtype=np.float64)
        if state["fields"]["f"] is not None:
            solver.field_f = MLPTimeDiscretizedField.from_state(state["fields"]["f"], solver.cfg.field)
        if state["fields"]["b"] is not None:
            solver.field_b = MLPTimeDiscretizedField.from_state(state["fields"]["b"], solver.cfg.field)
        solver.stage_log = state["stage_log"]
        solver._fitted = bool(state["fitted"])
        return solver
