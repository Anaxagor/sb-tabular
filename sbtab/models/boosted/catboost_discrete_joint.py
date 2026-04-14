from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class CatBoostDiscreteJointConfig:
    """
    CatBoost field approximator on a discrete time grid.

    Each time step k has its own CatBoost model predicting
    a vector drift in R^dim.

    Uses native CatBoost multi-target regression with loss='MultiRMSE'.
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

    


class CatBoostDiscreteJoint:
    """
    Holds a list of CatBoost regressors {f_k} over discrete times {t_k}.

    Each model predicts drift vectors in R^dim using MultiRMSE.
    """

    def __init__(self, dim: int, t_grid: np.ndarray, cfg: CatBoostDiscreteJointConfig):

        self.dim = int(dim)
        self.t_grid = np.asarray(t_grid, dtype=np.float32)
        self.cfg = cfg

        self.models: list[object] = [None for _ in range(len(self.t_grid))]

        self._checked_deps = False

    def _check_deps(self):
        if self._checked_deps:
            return
        try:
            import catboost  # noqa
        except Exception as e:
            raise ImportError(
                "CatBoostTimeDiscretizedField requires catboost.\n"
                "Install with: pip install catboost"
            ) from e
        self._checked_deps = True

    def _build_features(
        self,
        x: np.ndarray,
        *,
        t: np.ndarray | float,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if np.isscalar(t):
            t_arr = np.full((x.shape[0], 1), float(t), dtype=np.float32)
        else:
            t_arr = np.asarray(t, dtype=np.float32)
            if t_arr.ndim == 1:
                t_arr = t_arr[:, None]
            if t_arr.shape[1] != 1:
                raise ValueError("t must have shape (n,) or (n,1)")
        return np.concatenate([x, t_arr], axis=1)

    def fit_step(
        self,
        k: int,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
    ):
        """
        Fit model at time index k.

        x:  (n, dim)       — raw state vector
        y:  (n, dim)       — regression target
        x0: (n, d_parent)  — optional parent context (for structural / AR modes)
        """

        self._check_deps()
        from catboost import CatBoostRegressor

        t = float(self.t_grid[k])
        del x0
        X_feat = self._build_features(x, t=t)

        boosting_type = "Plain" if self.cfg.task_type == "GPU" else "Ordered"

        model = CatBoostRegressor(
            iterations=self.cfg.iterations,
            depth=self.cfg.depth,
            learning_rate=self.cfg.learning_rate,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            loss_function="MultiRMSE",
            task_type=self.cfg.task_type,
            boosting_type=boosting_type,
            thread_count=self.cfg.thread_count,
            random_seed=self.cfg.random_seed,
            verbose=self.cfg.verbose,
            allow_writing_files=self.cfg.allow_writing_files,
        )

        model.fit(X_feat, y)

        self.models[k] = model
        

    # -------------------------------------------------------

    def predict_step(
        self,
        k: int,
        x: np.ndarray,
    ) -> np.ndarray:

        model = self.models[k]
        if model is None:
            raise RuntimeError(f"Model for time step {k} is not trained.")
        X_feat = self._build_features(x, t=float(self.t_grid[k]))
        pred = model.predict(X_feat)
        pred = np.asarray(pred, dtype=np.float32)
        return pred


CatBoostDiscreteFieldConfig = CatBoostDiscreteJointConfig
CatBoostTimeDiscretizedField = CatBoostDiscreteJoint