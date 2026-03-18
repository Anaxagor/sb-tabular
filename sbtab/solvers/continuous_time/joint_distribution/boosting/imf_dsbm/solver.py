from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from sbtab.bridge.reference import GaussianReference
from sbtab.models.boosted.catboost_continuous_joint import (
    CatBoostContinuousField,
    CatBoostContinuousFieldConfig,
)


FB = Literal["f", "b"]


@dataclass
class IMFDSBMContinuousJointCatBoostConfig:
    fb_sequence: Sequence[FB] = ("b", "f", "b", "f", "b")

    num_steps: int = 1000      # sampling steps
    sigma: float = 0.1
    eps: float = 1e-3

    first_coupling: Literal["ref", "ind"] = "ref"
    n_noise_per_pair: int = 1

    noise: bool = True
    seed: int = 42

    field: CatBoostContinuousFieldConfig = field(default_factory=CatBoostContinuousFieldConfig)


class IMFDSBMContinuousJointCatBoostSolver:
    def __init__(self, dim: int, cfg: IMFDSBMContinuousJointCatBoostConfig):
        self.dim = int(dim)
        self.cfg = cfg

        self.columns_: Optional[list[str]] = None
        self.reference = GaussianReference(dim=self.dim)

        self.field_f: Optional[CatBoostContinuousField] = None
        self.field_b: Optional[CatBoostContinuousField] = None

        self._rng = np.random.default_rng(cfg.seed)
        self._x_data: Optional[np.ndarray] = None
        self._x_ref: Optional[np.ndarray] = None
        self._fitted = False

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

    def _build_continuous_dataset(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        fb: FB,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = z0.shape[0]
        reps = int(self.cfg.n_noise_per_pair)
        sigma = float(self.cfg.sigma)

        xt_list, t_list, x0_list, y_list = [], [], [], []

        for _ in range(reps):
            t = self._rng.uniform(low=self.cfg.eps, high=1.0 - self.cfg.eps, size=(n, 1)).astype(np.float32)
            noise = self._rng.normal(size=z0.shape).astype(np.float32)

            xt = (1.0 - t) * z0 + t * z1 + sigma * np.sqrt(t * (1.0 - t)) * noise
            delta = z1 - z0

            if fb == "f":
                target = delta - sigma * np.sqrt(t / (1.0 - t)) * noise
            else:
                target = -delta - sigma * np.sqrt((1.0 - t) / t) * noise

            xt_list.append(xt.astype(np.float32))
            t_list.append(t.astype(np.float32))
            x0_list.append(z0.astype(np.float32))
            y_list.append(target.astype(np.float32))

        return (
            np.concatenate(xt_list, axis=0),
            np.concatenate(t_list, axis=0),
            np.concatenate(x0_list, axis=0),
            np.concatenate(y_list, axis=0),
        )

    def _train_direction(self, fb: FB, z0: np.ndarray, z1: np.ndarray) -> None:
        field = CatBoostContinuousField(dim=self.dim, cfg=self.cfg.field)

        xt, t, x0_ctx, y = self._build_continuous_dataset(z0, z1, fb)
        X_feat = field._build_features(xt, t=t)
        field.fit(X_feat, y)

        if fb == "f":
            self.field_f = field
        else:
            self.field_b = field

    def _sample_with_direction(self, zstart: np.ndarray, direction: FB) -> np.ndarray:
        field = self.field_f if direction == "f" else self.field_b
        if field is None:
            raise RuntimeError(f"Direction '{direction}' has not been trained.")

        N = int(self.cfg.num_steps)
        dt = 1.0 / float(N)
        sigma = float(self.cfg.sigma)

        x = zstart.astype(np.float32).copy()
        x0 = x.copy()

        for i in range(N):
            tau = float(i) / float(N)
            if direction == "b":
                tau = 1.0 - tau

            t = np.full((x.shape[0], 1), tau, dtype=np.float32)
            drift = field.predict(x, t=t)

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

    def fit(self, train: pd.DataFrame | np.ndarray) -> "IMFDSBMContinuousJointCatBoostSolver":
        x0 = self._as_array(train)
        x1 = self._sample_reference(len(x0), seed=self.cfg.seed + 999)

        x_pairs = np.stack([x0, x1], axis=1)  # (N,2,D)

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

    def sample(self, n: int, seed: Optional[int] = None, steps: Optional[int] = None) -> np.ndarray:
        if not self._fitted or self.field_b is None:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive")

        zstart = self._sample_reference(n=n, seed=seed)
        return self._sample_with_direction(zstart, "b")

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        arr = self.sample(n, seed)
        return pd.DataFrame(arr, columns=self.columns_)