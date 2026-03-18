from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class CatBoostDiscreteFieldConfig:
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

    


class CatBoostTimeDiscretizedField:
    """
    Holds a list of CatBoost regressors {f_k} over discrete times {t_k}.

    Each model predicts drift vectors in R^dim using MultiRMSE.
    """

    def __init__(self, dim: int, t_grid: np.ndarray, cfg: CatBoostDiscreteFieldConfig):

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

    def fit_step(
        self,
        k: int,
        X_feat: np.ndarray,
        y: np.ndarray,
    ):
        """
        Fit model at time index k.

        X_feat: (n, n_features)
        y: (n, dim)
        """

        self._check_deps()
        from catboost import CatBoostRegressor

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
        x = np.asarray(x, dtype=np.float32)
        pred = model.predict(x)
        pred = np.asarray(pred, dtype=np.float32)
        return pred