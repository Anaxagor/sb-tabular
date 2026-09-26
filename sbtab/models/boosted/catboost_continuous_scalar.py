from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class CatBoostContinuousScalarConfig:
    """
    Continuous-time CatBoost scalar field for feature-wise solver.

    Learns
        f(x_t^j, context, t, [x0^j]) -> scalar drift
    """
    iterations: int = 2000
    depth: int = 8
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0

    task_type: Literal["CPU", "GPU"] = "CPU"
    thread_count: int = -1
    random_seed: int = 0
    verbose: bool = False
    allow_writing_files: bool = False
    # Features are always [x, parents, t]: the model is time-conditioned, so t
    # cannot be dropped. Any other value used to be accepted and silently ignored;
    # it is now rejected.
    feature_mode: str = "x_x0_t"

    # residual=True fits y - x and predicts x + model(.). Use it for MEAN-MATCHING
    # targets (IPF-DSB next-state means): those maps are identity plus an O(gamma)
    # correction, and trees cannot represent an identity map, so regressing y
    # directly leaves an error far above the correction. Keep it False for drift /
    # velocity targets (IMF-DSBM). The IPF solvers enforce True themselves.
    residual: bool = False



class CatBoostContinuousScalar:
    def __init__(self, cfg: CatBoostContinuousScalarConfig):
        if cfg.feature_mode != "x_x0_t":
            raise ValueError("CatBoostContinuousScalar always uses features [x, parents, t]; "
                             f"feature_mode={cfg.feature_mode!r} is not supported")
        self.cfg = cfg
        self.model = None
        self._checked = False

    def _check_deps(self) -> None:
        if self._checked:
            return
        try:
            import catboost  # noqa: F401
        except Exception as e:
            raise ImportError(
                "CatBoostContinuousScalar requires `catboost`.\n"
                "Install: pip install catboost"
            ) from e
        self._checked = True

    def _build_features(
        self,
        x: np.ndarray,
        ctx: np.ndarray,
        *,
        t: np.ndarray | float,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        ctx = np.asarray(ctx, dtype=np.float32)

        if np.isscalar(t):
            t_arr = np.full((x.shape[0], 1), float(t), dtype=np.float32)
        else:
            t_arr = np.asarray(t, dtype=np.float32)
            if t_arr.ndim == 1:
                t_arr = t_arr[:, None]
            if t_arr.shape[1] != 1:
                raise ValueError("t must have shape (n,) or (n,1)")

        parts = [x]
        if ctx.size:
            parts.append(ctx)
        parts.append(t_arr)
        return np.concatenate(parts, axis=1)

    def fit(
        self,
        x: np.ndarray,
        t: np.ndarray | float | None = None,
        y: np.ndarray | None = None,
        *,
        x0: Optional[np.ndarray] = None,
    ) -> None:
        self._check_deps()
        from catboost import CatBoostRegressor

        if y is None:
            X_feat = np.asarray(x, dtype=np.float32)
            y_arr = np.asarray(t, dtype=np.float32).reshape(-1)
        else:
            ctx = np.empty((len(x), 0), dtype=np.float32) if x0 is None else np.asarray(x0, dtype=np.float32)
            X_feat = self._build_features(x, ctx, t=t)
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        X_feat = np.asarray(X_feat, dtype=np.float32)
        if self.cfg.residual:
            y_arr = y_arr - X_feat[:, 0]

        boosting_type = "Plain" if self.cfg.task_type == "GPU" else "Ordered"

        model = CatBoostRegressor(
            iterations=self.cfg.iterations,
            depth=self.cfg.depth,
            learning_rate=self.cfg.learning_rate,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            loss_function="RMSE",
            task_type=self.cfg.task_type,
            boosting_type=boosting_type,
            thread_count=self.cfg.thread_count,
            random_seed=self.cfg.random_seed,
            verbose=self.cfg.verbose,
            allow_writing_files=self.cfg.allow_writing_files,
        )
        model.fit(np.asarray(X_feat, dtype=np.float32), y_arr)
        self.model = model

    def predict(
        self,
        x: np.ndarray,
        t: np.ndarray | float = 0.0,
        *,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        ctx_arr = np.empty((len(x), 0), dtype=np.float32) if x0 is None else np.asarray(x0, dtype=np.float32)
        X_feat = self._build_features(x, ctx_arr, t=t)
        pred = np.asarray(self.model.predict(X_feat), dtype=np.float32).reshape(-1)
        if self.cfg.residual:
            pred = pred + X_feat[:, 0]
        return pred.reshape(-1, 1)