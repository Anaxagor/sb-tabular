from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.boosted.catboost_discrete_joint import CatBoostDiscreteFieldConfig, CatBoostTimeDiscretizedField


@dataclass
class JointDiscreteBoostedConfig:
    num_steps: int = 20
    ipf_iters: int = 5
    alpha_ou: float = 1.0
    seed: int = 42
    catboost: CatBoostDiscreteFieldConfig = CatBoostDiscreteFieldConfig()

class JointDiscreteBoostedSolver:
    """
    [DT] + [Boosting] +[Joint]
    Implements IPF-DSB where each time step has its own CatBoost model predicting the full vector.
    """
    def __init__(self, dim: int, cfg: JointDiscreteBoostedConfig):
        self.dim = dim
        self.cfg = cfg
        self.timegrid = TimeGrid(num_steps=cfg.num_steps, T=1.0)
        self.gammas = self.timegrid.gammas().numpy()
        self.t_grid = self.timegrid.times().numpy()
        
        self.field_f = CatBoostTimeDiscretizedField(dim, self.t_grid, cfg.catboost)
        self.field_b = CatBoostTimeDiscretizedField(dim, self.t_grid, cfg.catboost)
        self._rng = np.random.default_rng(cfg.seed)
        self.columns_: Optional[list[str]] = None

    def fit(self, train_df: pd.DataFrame):
        self.columns_ = list(train_df.columns)
        X_train = train_df.to_numpy(dtype=np.float32)
        n = X_train.shape[0]
        
        print("Initial pretraining (OU approximation)...")
        for k in range(self.cfg.num_steps):
            target = X_train * (1.0 - self.cfg.alpha_ou * self.t_grid[k])
            self.field_f.fit_step(k, X_train, target)

        for it in range(self.cfg.ipf_iters):
            # 1. Train Backward field (B) on Forward paths
            print(f"IPF Iteration {it+1}/{self.cfg.ipf_iters} - Training B...")
            curr_x = X_train.copy()
            for k in range(self.cfg.num_steps):
                drift = self.field_f.predict_step(k, curr_x)
                noise = self._rng.normal(size=curr_x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                next_x = drift + noise
                
                # Residual target for B
                target = next_x + (drift - self.field_f.predict_step(k, next_x))
                self.field_b.fit_step(min(k+1, self.cfg.num_steps-1), next_x, target)
                curr_x = next_x
            
            # 2. Train Forward field (F) on Backward paths
            print(f"IPF Iteration {it+1}/{self.cfg.ipf_iters} - Training F...")
            curr_x = self._rng.normal(size=X_train.shape).astype(np.float32)
            for k in range(self.cfg.num_steps - 1, -1, -1):
                drift = self.field_b.predict_step(k, curr_x)
                noise = self._rng.normal(size=curr_x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                prev_x = drift + noise
                
                # Residual target for F
                target = prev_x + (drift - self.field_b.predict_step(k, prev_x))
                self.field_f.fit_step(max(k-1, 0), prev_x, target)
                curr_x = prev_x
                
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        x = rng.normal(size=(n, self.dim)).astype(np.float32)
        
        for k in range(self.cfg.num_steps - 1, -1, -1):
            drift = self.field_b.predict_step(k, x)
            noise = rng.normal(size=x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
            x = drift + noise
        return x

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame(self.sample(n, seed), columns=self.columns_)