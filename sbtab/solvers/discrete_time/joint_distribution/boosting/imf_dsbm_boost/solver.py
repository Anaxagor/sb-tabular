from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from sbtab.bridge.reference import GaussianReference
from sbtab.models.boosted.catboost_discrete_joint import (
    CatBoostDiscreteFieldConfig,
    CatBoostTimeDiscretizedField,
)


FB = Literal["b", "f"]


@dataclass
class IMFDSBMBoostConfig:
    fb_sequence: Sequence[FB] = ("b", "f", "b", "f", "b")

    num_steps: int = 32
    sigma: float = 0.10
    eps: float = 1e-3

    first_coupling: Literal["ref", "ind"] = "ref"

    n_noise_per_pair: int = 1
    noise: bool = True
    seed: int = 0

    catboost: CatBoostDiscreteFieldConfig = field(default_factory=CatBoostDiscreteFieldConfig)


class IMFDSBMBoostSolver:
    """
    Joint (multivariate) boosted IMF+DSBM solver.

    Unified with the neural IMF+DSBM solver:
      - same IMF loop
      - same coupling construction
      - same DSBM targets
      - same reference process abstraction: GaussianReference
      - same final generation rule: Gaussian -> latest backward model
    """

    def __init__(self, dim: int, cfg: IMFDSBMBoostConfig):
        self.dim = int(dim)
        self.cfg = cfg

        self.columns_: Optional[list[str]] = None
        self.t_grid = self._make_t_grid(cfg.num_steps, cfg.eps)

        self.field_f: Optional[CatBoostTimeDiscretizedField] = None
        self.field_b: Optional[CatBoostTimeDiscretizedField] = None

        self._rng = np.random.default_rng(cfg.seed)
        self._torch_gen = torch.Generator(device="cpu")
        self._torch_gen.manual_seed(int(cfg.seed))

        self.reference = GaussianReference(dim=self.dim, device=torch.device("cpu"))

        self._x_data: Optional[np.ndarray] = None
        self._x_ref: Optional[np.ndarray] = None
        self._fitted: bool = False

    @staticmethod
    def _make_t_grid(N: int, eps: float) -> np.ndarray:
        if N <= 1:
            raise ValueError("num_steps must be > 1")
        t = (np.arange(N, dtype=np.float32) + 0.5) / float(N)
        return np.clip(t, eps, 1.0 - eps)

    def _as_array(self, train: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(train, pd.DataFrame):
            self.columns_ = list(train.columns)
            arr = train.to_numpy(dtype=np.float32, copy=True)
        else:
            arr = np.asarray(train, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {tuple(arr.shape)}")
        return arr

    def _sample_reference(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        x = self.reference.sample(n=n, seed=seed)
        return x.detach().cpu().numpy().astype(np.float32)

    def _dsbm_train_tuple(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        t: float,
        fb: FB,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma = float(self.cfg.sigma)

        epsn = self._rng.normal(size=z0.shape).astype(np.float32)

        xt = (1.0 - t) * z0 + t * z1
        xt = xt + sigma * np.sqrt(t * (1.0 - t)) * epsn

        delta = z1 - z0
        if fb == "f":
            target = delta - sigma * np.sqrt(t / (1.0 - t)) * epsn
        else:
            target = -delta - sigma * np.sqrt((1.0 - t) / t) * epsn

        return xt.astype(np.float32), target.astype(np.float32)

    def _build_step_batch(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        t: float,
        fb: FB,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reps = int(self.cfg.n_noise_per_pair)
        if reps < 1:
            raise ValueError("n_noise_per_pair must be >= 1")

        if reps == 1:
            xt, target = self._dsbm_train_tuple(z0, z1, t, fb)
            return xt, z0, target

        xt_list = []
        x0_list = []
        y_list = []
        for _ in range(reps):
            xt, target = self._dsbm_train_tuple(z0, z1, t, fb)
            xt_list.append(xt)
            x0_list.append(z0)
            y_list.append(target)

        xt_all = np.concatenate(xt_list, axis=0)
        x0_all = np.concatenate(x0_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        return xt_all, x0_all, y_all

    def _train_direction_on_coupling(
        self,
        fb: FB,
        z0: np.ndarray,
        z1: np.ndarray,
    ) -> None:
        field = CatBoostTimeDiscretizedField(
            dim=self.dim,
            t_grid=self.t_grid,
            cfg=self.cfg.catboost,
        )

        for k, t in enumerate(self.t_grid):
            xt, _, target = self._build_step_batch(z0, z1, float(t), fb)
            field.fit_step(k, xt, target)

        if fb == "f":
            self.field_f = field
        else:
            self.field_b = field

    def _sample_with_direction(
        self,
        zstart: np.ndarray,
        direction: FB,
    ) -> np.ndarray:
        field = self.field_f if direction == "f" else self.field_b
        if field is None:
            raise RuntimeError(f"Direction '{direction}' has not been trained yet.")

        N = int(self.cfg.num_steps)
        dt = 1.0 / float(N)
        sigma = float(self.cfg.sigma)

        x = zstart.astype(np.float32).copy()
        x0 = x.copy()

        step_indices = range(N) if direction == "f" else range(N - 1, -1, -1)

        for k in step_indices:
            drift = field.predict_step(k, x)
            x = x + drift * dt

            if self.cfg.noise and sigma != 0.0:
                x = x + sigma * np.sqrt(dt) * self._rng.normal(size=x.shape).astype(np.float32)

        return x

    def _make_first_coupling(self, fb_to_train: FB) -> tuple[np.ndarray, np.ndarray]:
        if fb_to_train != "b":
            raise RuntimeError("IMF+DSBM initialization expects the first direction to be 'b'.")

        if self._x_data is None or self._x_ref is None:
            raise RuntimeError("fit() has not initialized endpoints.")

        z0 = self._x_data.copy()

        if self.cfg.first_coupling == "ref":
            z1 = z0 + self.cfg.sigma * self._rng.normal(size=z0.shape).astype(np.float32)
        elif self.cfg.first_coupling == "ind":
            perm = self._rng.permutation(len(self._x_ref))
            z1 = self._x_ref[perm].copy()
        else:
            raise ValueError(f"Unknown first_coupling={self.cfg.first_coupling}")

        return z0, z1

    def _make_next_coupling(self, prev_fb: FB) -> tuple[np.ndarray, np.ndarray]:
        if self._x_data is None or self._x_ref is None:
            raise RuntimeError("fit() has not initialized endpoints.")

        if prev_fb == "f":
            zstart = self._x_data.copy()
            zend = self._sample_with_direction(zstart, "f")
            z0, z1 = zstart, zend
        else:
            zstart = self._x_ref.copy()
            zend = self._sample_with_direction(zstart, "b")
            z0, z1 = zend, zstart

        return z0, z1

    def fit(self, train: pd.DataFrame | np.ndarray) -> "IMFDSBMBoostSolver":
        x_data = self._as_array(train)

        self._x_data = x_data.astype(np.float32)
        self._x_ref = self._sample_reference(len(x_data), seed=self.cfg.seed + 999)

        prev_fb: Optional[FB] = None

        for it, fb in enumerate(self.cfg.fb_sequence):
            if it == 0:
                z0, z1 = self._make_first_coupling(fb)
            else:
                if prev_fb is None:
                    raise RuntimeError("Internal IMF state is inconsistent.")
                z0, z1 = self._make_next_coupling(prev_fb)

            self._train_direction_on_coupling(fb, z0, z1)
            prev_fb = fb

        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        if not self._fitted or self.field_b is None:
            raise RuntimeError("Call fit() before sample(); a backward field must be trained.")
        if n <= 0:
            raise ValueError("n must be positive")

        zstart = self._sample_reference(int(n), seed=seed)
        x = self._sample_with_direction(zstart, "b")
        return x

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        x = self.sample(n, seed)
        return pd.DataFrame(x, columns=self.columns_)