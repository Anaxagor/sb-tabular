from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from sbtab.bridge.reference import GaussianReference
from sbtab.models.boosted.catboost_discrete_scalar import (
    CatBoostScalarConfig,
    CatBoostTimeDiscretizedScalar,
)


FB = Literal["b", "f"]


@dataclass
class FeaturewiseDSBMBoostConfig:
    fb_sequence: Sequence[FB] = ("b", "f", "b", "f", "b")

    num_steps: int = 50
    sigma: float = 0.10
    eps: float = 1e-3

    first_coupling: Literal["ref", "ind"] = "ref"

    n_noise_per_pair: int = 1
    noise: bool = True

    feature_order: Optional[List[str]] = None
    context_cols_map: Optional[Dict[str, List[str]]] = None

    catboost: CatBoostScalarConfig = field(default_factory=CatBoostScalarConfig)

    seed: int = 0


class FeaturewiseDSBMBoostSolver:
    """
    Feature-wise CatBoost IMF+DSBM solver using the same GaussianReference abstraction
    as the continuous and joint variants.
    """

    def __init__(self, cfg: FeaturewiseDSBMBoostConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        self.columns_: Optional[List[str]] = None
        self.dim_: Optional[int] = None

        self.feature_order_: Optional[List[str]] = None
        self.feature_order_idx_: Optional[List[int]] = None

        self.context_idx_: Dict[int, List[int]] = {}

        self.fields_f_: Dict[int, CatBoostTimeDiscretizedScalar] = {}
        self.fields_b_: Dict[int, CatBoostTimeDiscretizedScalar] = {}

        self.t_grid_: Optional[np.ndarray] = None

        self._x_data: Optional[np.ndarray] = None
        self._x_ref: Optional[np.ndarray] = None

        self.reference: Optional[GaussianReference] = None
        self._fitted: bool = False

    def _sample_reference(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("Reference process is not initialized.")
        x = self.reference.sample(n=n, seed=seed)
        return x.detach().cpu().numpy().astype(np.float32)

    def _make_t_grid(self) -> np.ndarray:
        N = int(self.cfg.num_steps)
        if N <= 1:
            raise ValueError("num_steps must be > 1")
        t = (np.arange(N, dtype=np.float32) + 0.5) / float(N)
        return np.clip(t, float(self.cfg.eps), float(1.0 - self.cfg.eps))

    @staticmethod
    def _ensure_float_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _resolve_feature_order(self, cols: List[str]) -> List[str]:
        if self.cfg.feature_order is None:
            return cols
        order = list(self.cfg.feature_order)
        if set(order) != set(cols):
            missing = sorted(list(set(cols) - set(order)))
            extra = sorted(list(set(order) - set(cols)))
            raise ValueError(
                "feature_order must be a permutation of df.columns.\n"
                f"Missing in feature_order: {missing}\n"
                f"Extra in feature_order: {extra}"
            )
        return order

    def _resolve_context_for_feature(
        self,
        feat_name: str,
        feat_pos: int,
        feature_order: List[str],
    ) -> List[str]:
        if self.cfg.context_cols_map is not None and feat_name in self.cfg.context_cols_map:
            ctx = [c for c in self.cfg.context_cols_map[feat_name] if c != feat_name]
            return ctx
        return [c for c in feature_order[:feat_pos] if c != feat_name]

    def _validate_autoregressive_context(
        self,
        feature_order: List[str],
        contexts: Dict[str, List[str]],
    ) -> None:
        pos = {c: i for i, c in enumerate(feature_order)}
        for feat, ctx_list in contexts.items():
            for c in ctx_list:
                if c not in pos:
                    raise ValueError(f"Context feature '{c}' not in dataset columns.")
                if pos[c] >= pos[feat]:
                    raise ValueError(
                        f"Invalid context for '{feat}': '{c}' appears at position {pos[c]} "
                        f"but '{feat}' is at position {pos[feat]}.\n"
                        "For sequential generation, all context features must appear earlier in feature_order."
                    )

    def _build_feature_step_batch(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        ctx: np.ndarray,
        t: float,
        fb: FB,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = z0.shape[0]
        sigma = float(self.cfg.sigma)

        reps = int(self.cfg.n_noise_per_pair)
        if reps < 1:
            raise ValueError("n_noise_per_pair must be >= 1")

        X_list = []
        y_list = []

        for _ in range(reps):
            epsn = self._rng.normal(size=(n,)).astype(np.float32)

            xt = (1.0 - t) * z0 + t * z1
            xt = xt + sigma * np.sqrt(t * (1.0 - t)) * epsn

            delta = z1 - z0
            if fb == "f":
                target = delta - sigma * np.sqrt(t / (1.0 - t)) * epsn
            else:
                target = -delta - sigma * np.sqrt((1.0 - t) / t) * epsn

            xt_col = xt.reshape(n, 1)
            X_feat = np.concatenate([xt_col, ctx], axis=1) if ctx.size else xt_col

            X_list.append(X_feat.astype(np.float32))
            y_list.append(target.astype(np.float32))

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        return X_all, y_all

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

    def _train_direction_on_coupling(
        self,
        fb: FB,
        z0: np.ndarray,
        z1: np.ndarray,
    ) -> None:
        if self.t_grid_ is None or self.feature_order_ is None or self.columns_ is None:
            raise RuntimeError("Solver is not initialized for training.")

        col_to_idx = {c: i for i, c in enumerate(self.columns_)}
        new_fields: Dict[int, CatBoostTimeDiscretizedScalar] = {}

        for feat_name in self.feature_order_:
            j = col_to_idx[feat_name]
            ctx_idx = self.context_idx_[j]

            z0_j = z0[:, j].astype(np.float32)
            z1_j = z1[:, j].astype(np.float32)

            if fb == "b":
                ctx_source = z0[:, ctx_idx].astype(np.float32) if len(ctx_idx) else np.zeros((z0.shape[0], 0), dtype=np.float32)
            else:
                ctx_source = z1[:, ctx_idx].astype(np.float32) if len(ctx_idx) else np.zeros((z1.shape[0], 0), dtype=np.float32)

            field_j = CatBoostTimeDiscretizedScalar(
                t_grid=self.t_grid_,
                cfg=self.cfg.catboost,
            )

            for k, t in enumerate(self.t_grid_):
                X_feat, y = self._build_feature_step_batch(
                    z0=z0_j,
                    z1=z1_j,
                    ctx=ctx_source,
                    t=float(t),
                    fb=fb,
                )
                field_j.fit_step(k, X_feat, y)

            new_fields[j] = field_j

        if fb == "f":
            self.fields_f_ = new_fields
        else:
            self.fields_b_ = new_fields

    def _sample_with_direction(
        self,
        zstart: np.ndarray,
        direction: FB,
    ) -> np.ndarray:
        if self.columns_ is None or self.feature_order_idx_ is None or self.t_grid_ is None:
            raise RuntimeError("Call fit() before sample().")

        fields = self.fields_f_ if direction == "f" else self.fields_b_
        if len(fields) == 0:
            raise RuntimeError(f"Direction '{direction}' has not been trained yet.")

        n = zstart.shape[0]
        d = len(self.columns_)
        X_out = np.zeros((n, d), dtype=np.float32)

        N = len(self.t_grid_)
        dt = 1.0 / float(N)
        sigma = float(self.cfg.sigma)

        step_indices = range(N) if direction == "f" else range(N - 1, -1, -1)

        for j in self.feature_order_idx_:
            field_j = fields.get(j, None)
            if field_j is None:
                raise RuntimeError(f"No trained field for feature index {j} in direction '{direction}'.")

            x = zstart[:, j].astype(np.float32).copy()
            ctx_idx = self.context_idx_[j]
            ctx = X_out[:, ctx_idx] if len(ctx_idx) else np.zeros((n, 0), dtype=np.float32)

            for k in step_indices:
                x_col = x.reshape(-1, 1)
                X_feat = np.concatenate([x_col, ctx], axis=1) if ctx.size else x_col

                drift = field_j.predict_step(k, X_feat).astype(np.float32).reshape(-1)
                x = x + drift * dt

                if self.cfg.noise and sigma != 0.0:
                    x = x + sigma * np.sqrt(dt) * self._rng.normal(size=x.shape).astype(np.float32)

            X_out[:, j] = x

        return X_out

    def fit(self, train_df: pd.DataFrame) -> "FeaturewiseDSBMBoostSolver":
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("fit expects a pandas DataFrame.")
        df = self._ensure_float_df(train_df)

        self.columns_ = list(df.columns)
        self.dim_ = len(self.columns_)

        self.reference = GaussianReference(dim=self.dim_, device=torch.device("cpu"))

        feature_order = self._resolve_feature_order(self.columns_)
        self.feature_order_ = feature_order
        col_to_idx = {c: i for i, c in enumerate(self.columns_)}
        self.feature_order_idx_ = [col_to_idx[c] for c in feature_order]

        contexts_by_name: Dict[str, List[str]] = {}
        for pos, feat in enumerate(feature_order):
            contexts_by_name[feat] = self._resolve_context_for_feature(feat, pos, feature_order)
        self._validate_autoregressive_context(feature_order, contexts_by_name)

        self.context_idx_.clear()
        for feat, ctx_names in contexts_by_name.items():
            self.context_idx_[col_to_idx[feat]] = [col_to_idx[c] for c in ctx_names]

        self.t_grid_ = self._make_t_grid()

        X = df.to_numpy(dtype=np.float32, copy=True)
        self._x_data = X
        self._x_ref = self._sample_reference(len(X), seed=self.cfg.seed + 999)

        self.fields_f_.clear()
        self.fields_b_.clear()

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

    def sample(self, n: int, seed: Optional[int] = None, steps: Optional[int] = None) -> np.ndarray:
        if not self._fitted or len(self.fields_b_) == 0:
            raise RuntimeError("Call fit() before sample(); backward feature fields must be trained.")
        if n <= 0:
            raise ValueError("n must be positive")

        d = len(self.columns_)
        zstart = self._sample_reference(int(n), seed=seed)

        X_syn = self._sample_with_direction(zstart, "b")
        return X_syn

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        x = self.sample(n=n, seed=seed)
        return pd.DataFrame(x, columns=self.columns_)