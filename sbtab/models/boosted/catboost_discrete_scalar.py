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

    # residual=True fits y - x and predicts x + model(.). Use it for MEAN-MATCHING
    # targets (IPF-DSB next-state means): those maps are identity plus an O(gamma)
    # correction, and trees cannot represent an identity map, so regressing y
    # directly leaves an error far above the correction. Keep it False for drift /
    # velocity targets (IMF-DSBM). The IPF solvers enforce True themselves.
    residual: bool = False


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
        if self.cfg.residual:
            # feature column 0 is the feature's own state x
            y = y - X_feat[:, 0]
        boosting_type = "Plain" if self.cfg.task_type == "GPU" else "Ordered"

        model = CatBoostRegressor(
            iterations=self.cfg.iterations,
            depth=self.cfg.depth,
            learning_rate=self.cfg.learning_rate,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            loss_function=self.cfg.loss_function,
            task_type=self.cfg.task_type,
            boosting_type=boosting_type,
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
        Predict the scalar next-state mean at time index k.

        Returns: (n, 1)
        """
        model = self.models[k]
        if model is None:
            raise RuntimeError(f"Scalar model for step k={k} is not fitted.")

        if x0 is None:
            X_feat = np.asarray(X_feat, dtype=np.float32)
        else:
            X_feat = self._build_features(X_feat, x0=x0, t=float(self.t_grid[k]))
        yhat = np.asarray(model.predict(X_feat), dtype=np.float32).reshape(-1)
        if self.cfg.residual:
            yhat = yhat + X_feat[:, 0]
        return yhat.reshape(-1, 1)

    def is_fitted(self, k: int) -> bool:
        return self.models[k] is not None


CatBoostScalarConfig = CatBoostDiscreteScalarConfig
CatBoostTimeDiscretizedScalar = CatBoostDiscreteScalar