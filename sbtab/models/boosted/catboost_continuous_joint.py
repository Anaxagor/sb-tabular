from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class CatBoostContinuousFieldConfig:
    """
    Continuous-time CatBoost field for joint/vector prediction.

    A single CatBoost regressor learns
        f(x_t, t, [x0]) -> drift in R^dim
    and is used for all t in [0,1].
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



class CatBoostContinuousField:
    def __init__(self, dim: int, cfg: CatBoostContinuousFieldConfig):
        self.dim = int(dim)
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
                "CatBoostContinuousField requires `catboost`.\n"
                "Install: pip install catboost"
            ) from e
        self._checked = True

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

        parts = [x]
        parts.append(t_arr)
        return np.concatenate(parts, axis=1)

    def fit(self, X_feat: np.ndarray, y: np.ndarray) -> None:
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
        model.fit(np.asarray(X_feat, dtype=np.float32), np.asarray(y, dtype=np.float32))
        self.model = model

    def predict(
        self,
        x: np.ndarray,
        *,
        t: np.ndarray | float,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        X_feat = self._build_features(x, t=t)
        pred = self.model.predict(X_feat)
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred[:, None]
        return pred