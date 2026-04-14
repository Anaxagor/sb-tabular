from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class CatBoostDiscreteScalarConfig:
    """
    CatBoost regressor config for scalar drift/velocity prediction.
    """
    iterations: int = 2000
    depth: int = 8
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    loss_function: str = "RMSE"

    task_type: Literal["CPU", "GPU"] = "CPU"
    thread_count: int = -1
    random_seed: int = 0
    verbose: bool = False
    allow_writing_files: bool = False
    feature_mode: str = "x_x0"


class CatBoostDiscreteScalar:
    """
    Holds a list of CatBoostRegressor models {f_k} over discrete times {t_k}.
    Each f_k predicts a scalar drift/velocity.
    """

    def __init__(self, t_grid: np.ndarray, cfg: CatBoostDiscreteScalarConfig):
        self.t_grid = np.asarray(t_grid, dtype=np.float32)
        self.cfg = cfg
        self.models: list[object] = [None for _ in range(len(self.t_grid))]
        self._checked = False

    def _check_deps(self) -> None:
        if self._checked:
            return
        try:
            import catboost  # noqa: F401
        except Exception as e:
            raise ImportError(
                "CatBoostTimeDiscretizedScalar requires `catboost`.\n"
                "Install: pip install catboost"
            ) from e
        self._checked = True

    def _build_features(
        self,
        x: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
        t: np.ndarray | float | None = None,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
        parts = [x_arr]
        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=np.float32)
            if x0_arr.size:
                parts.append(x0_arr)
        if t is not None and "t" in self.cfg.feature_mode:
            if np.isscalar(t):
                t_arr = np.full((x_arr.shape[0], 1), float(t), dtype=np.float32)
            else:
                t_arr = np.asarray(t, dtype=np.float32)
                if t_arr.ndim == 1:
                    t_arr = t_arr[:, None]
            parts.append(t_arr)
        return np.concatenate(parts, axis=1)

    def fit_step(
        self,
        k: int,
        X_feat: np.ndarray,
        y: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit model at time index k.

        X_feat: (n, n_features)
        y     : (n,) or (n,1)
        """
        self._check_deps()
        from catboost import CatBoostRegressor

        if x0 is None:
            X_feat = np.asarray(X_feat, dtype=np.float32)
        else:
            X_feat = self._build_features(X_feat, x0=x0, t=float(self.t_grid[k]))
        y = np.asarray(y).reshape(-1).astype(np.float32)

        model = CatBoostRegressor(
            iterations=self.cfg.iterations,
            depth=self.cfg.depth,
            learning_rate=self.cfg.learning_rate,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            loss_function=self.cfg.loss_function,
            task_type=self.cfg.task_type,
            thread_count=self.cfg.thread_count,
            random_seed=self.cfg.random_seed,
            verbose=self.cfg.verbose,
            allow_writing_files=self.cfg.allow_writing_files,
        )
        model.fit(X_feat, y)
        self.models[k] = model

    def predict_step(
        self,
        k: int,
        X_feat: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict scalar drift/velocity at time index k.

        Returns: (n,)
        """
        model = self.models[k]
        if model is None:
            raise RuntimeError(f"Scalar model for step k={k} is not fitted.")

        if x0 is None:
            X_feat = np.asarray(X_feat, dtype=np.float32)
        else:
            X_feat = self._build_features(X_feat, x0=x0, t=float(self.t_grid[k]))
        yhat = model.predict(X_feat)
        return np.asarray(yhat, dtype=np.float32).reshape(-1, 1)


CatBoostScalarConfig = CatBoostDiscreteScalarConfig
CatBoostTimeDiscretizedScalar = CatBoostDiscreteScalar