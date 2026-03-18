from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class CatBoostScalarConfig:
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


class CatBoostTimeDiscretizedScalar:
    """
    Holds a list of CatBoostRegressor models {f_k} over discrete times {t_k}.
    Each f_k predicts a scalar drift/velocity.
    """

    def __init__(self, t_grid: np.ndarray, cfg: CatBoostScalarConfig):
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

    def fit_step(self, k: int, X_feat: np.ndarray, y: np.ndarray) -> None:
        """
        Fit model at time index k.

        X_feat: (n, n_features)
        y     : (n,) or (n,1)
        """
        self._check_deps()
        from catboost import CatBoostRegressor

        X_feat = np.asarray(X_feat, dtype=np.float32)
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

    def predict_step(self, k: int, X_feat: np.ndarray) -> np.ndarray:
        """
        Predict scalar drift/velocity at time index k.

        Returns: (n,)
        """
        model = self.models[k]
        if model is None:
            raise RuntimeError(f"Scalar model for step k={k} is not fitted.")

        X_feat = np.asarray(X_feat, dtype=np.float32)
        yhat = model.predict(X_feat)
        return np.asarray(yhat, dtype=np.float32).reshape(-1)