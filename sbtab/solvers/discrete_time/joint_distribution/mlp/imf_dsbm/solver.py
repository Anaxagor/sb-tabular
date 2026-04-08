from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from sbtab.bridge.reference import GaussianReference
from sbtab.models.neural.mlp_discrete_joint import (
    MLPTimeDiscretizedField,
    StepMLPJointConfig,
)


FB = Literal["f", "b"]


@dataclass
class IMFDSBMDiscreteJointMLPConfig:
    fb_sequence: Sequence[FB] = ("b", "f", "b", "f", "b")

    num_steps: int = 32
    sigma: float = 0.1
    eps: float = 1e-3

    first_coupling: Literal["ref", "ind"] = "ref"
    n_noise_per_pair: int = 1

    noise: bool = True
    seed: int = 42

    field: StepMLPJointConfig = field(default_factory=StepMLPJointConfig)


class IMFDSBMDiscreteJointMLPSolver:
    def __init__(self, dim: int, cfg: IMFDSBMDiscreteJointMLPConfig):
        self.dim = int(dim)
        self.cfg = cfg

        self.columns_: Optional[list[str]] = None
        self.t_grid = self._make_t_grid(cfg.num_steps, cfg.eps)

        self.reference = GaussianReference(dim=self.dim)

        self.field_f: Optional[MLPTimeDiscretizedField] = None
        self.field_b: Optional[MLPTimeDiscretizedField] = None

        self._rng = np.random.default_rng(cfg.seed)
        self._fitted = False

    @staticmethod
    def _make_t_grid(N: int, eps: float) -> np.ndarray:
        if N <= 1:
            raise ValueError("num_steps must be > 1")
        t = (np.arange(N, dtype=np.float32) + 0.5) / float(N)
        return np.clip(t, eps, 1.0 - eps)

    def _as_array(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            self.columns_ = list(x.columns)
            arr = x.to_numpy(dtype=np.float32, copy=True)
        else:
            arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N,{self.dim}), got {tuple(arr.shape)}")
        return arr

    def _sample_reference(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        return self.reference.sample(n=n, seed=seed).detach().cpu().numpy().astype(np.float32)

    def _dsbm_train_tuple(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        t: float,
        fb: FB,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma = float(self.cfg.sigma)

        noise = self._rng.normal(size=z0.shape).astype(np.float32)
        xt = (1.0 - t) * z0 + t * z1 + sigma * np.sqrt(t * (1.0 - t)) * noise

        delta = z1 - z0
        if fb == "f":
            target = delta - sigma * np.sqrt(t / (1.0 - t)) * noise
        else:
            target = -delta - sigma * np.sqrt((1.0 - t) / t) * noise

        return xt.astype(np.float32), target.astype(np.float32)

    def _build_step_batch(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        t: float,
        fb: FB,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reps = int(self.cfg.n_noise_per_pair)

        xt_list, x0_list, y_list = [], [], []
        for _ in range(reps):
            xt, target = self._dsbm_train_tuple(z0, z1, t, fb)
            xt_list.append(xt)
            x0_list.append(z0)
            y_list.append(target)

        return (
            np.concatenate(xt_list, axis=0),
            np.concatenate(x0_list, axis=0),
            np.concatenate(y_list, axis=0),
        )

    def _train_direction(self, fb: FB, z0: np.ndarray, z1: np.ndarray) -> None:
        field = MLPTimeDiscretizedField(dim=self.dim, t_grid=self.t_grid, cfg=self.cfg.field)

        for k, t in enumerate(self.t_grid):
            xt, x0_ctx, target = self._build_step_batch(z0, z1, float(t), fb)
            X_feat = field._build_features(xt, x0=x0_ctx, t=float(t))
            field.fit_step(k, X_feat, target)

        if fb == "f":
            self.field_f = field
        else:
            self.field_b = field

    def _sample_with_direction(self, zstart: np.ndarray, direction: FB) -> np.ndarray:
        field = self.field_f if direction == "f" else self.field_b
        if field is None:
            raise RuntimeError(f"Direction '{direction}' has not been trained.")

        dt = 1.0 / float(self.cfg.num_steps)
        sigma = float(self.cfg.sigma)

        x = zstart.astype(np.float32).copy()
        x0 = x.copy()

        step_indices = range(self.cfg.num_steps) if direction == "f" else range(self.cfg.num_steps - 1, -1, -1)

        for k in step_indices:
            drift = field.predict_step(k, x, x0=x0)
            x = x + drift * dt
            if self.cfg.noise:
                x = x + sigma * np.sqrt(dt) * self._rng.normal(size=x.shape).astype(np.float32)

        return x

    def _generate_coupling(
        self,
        x_pairs: np.ndarray,
        prev_fb: Optional[FB],
        fb_to_train: FB,
        first_it: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if first_it:
            if fb_to_train != "b":
                raise RuntimeError("IMF+DSBM initialization expects first direction 'b'.")
            z0 = x_pairs[:, 0]
            if self.cfg.first_coupling == "ref":
                z1 = z0 + self.cfg.sigma * self._rng.normal(size=z0.shape).astype(np.float32)
            else:
                z1 = x_pairs[:, 1].copy()
                perm = self._rng.permutation(len(z1))
                z1 = z1[perm]
            return z0, z1

        if prev_fb is None:
            raise RuntimeError("prev_fb is None while first_it=False.")

        if prev_fb == "f":
            zstart = x_pairs[:, 0]
            zend = self._sample_with_direction(zstart, "f")
            z0, z1 = zstart, zend
        else:
            zstart = x_pairs[:, 1]
            zend = self._sample_with_direction(zstart, "b")
            z0, z1 = zend, zstart

        return z0, z1

    def fit(self, train: pd.DataFrame | np.ndarray) -> "IMFDSBMDiscreteJointMLPSolver":
        x0 = self._as_array(train)
        x1 = self._sample_reference(len(x0), seed=self.cfg.seed + 999)
        x_pairs = np.stack([x0, x1], axis=1)

        prev_fb: Optional[FB] = None
        for it_idx, fb in enumerate(self.cfg.fb_sequence, start=1):
            z0, z1 = self._generate_coupling(
                x_pairs=x_pairs,
                prev_fb=prev_fb,
                fb_to_train=fb,
                first_it=(it_idx == 1),
            )
            self._train_direction(fb, z0, z1)
            prev_fb = fb

        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        if not self._fitted or self.field_b is None:
            raise RuntimeError("Call fit() before sample().")
        zstart = self._sample_reference(n=n, seed=seed)
        return self._sample_with_direction(zstart, "b")

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        arr = self.sample(n, seed)
        return pd.DataFrame(arr, columns=self.columns_)