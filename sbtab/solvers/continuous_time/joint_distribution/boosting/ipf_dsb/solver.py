from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.boosted.catboost_continuous_joint import CatBoostContinuousJointConfig, CatBoostContinuousJoint


@dataclass
class JointContinuousBoostedConfig:
    num_steps: int = 20
    ipf_iters: int = 5
    alpha_ou: float = 1.0
    seed: int = 42
    catboost: CatBoostContinuousJointConfig = field(default_factory=CatBoostContinuousJointConfig)

class JointContinuousBoostedSolver:
    """[CT] + [Boosting] + [Joint]
    Implements IPF-DSB using a single continuous-time CatBoost model for all features.
    """
    def __init__(self, dim: int, cfg: JointContinuousBoostedConfig):
        self.dim = dim
        self.cfg = cfg
        
        self.timegrid = TimeGrid(num_steps=cfg.num_steps)
        self.gammas = self.timegrid.gammas().numpy()
        self.times = self.timegrid.times().numpy()
        self._rng = np.random.default_rng(cfg.seed)

        self.F = CatBoostContinuousJoint(dim=dim, cfg=cfg.catboost)
        self.B = CatBoostContinuousJoint(dim=dim, cfg=cfg.catboost)
        
        self.columns_: Optional[list[str]] = None
        self._fitted = False

    def _sample_prior(self, n: int) -> np.ndarray:
        return self._rng.normal(size=(n, self.dim)).astype(np.float32)

    def pretrain_F_ou(self, X_data: np.ndarray):
        """Initializes the Forward field as an Ornstein-Uhlenbeck process."""
        n = len(X_data)
        x_prior = self._sample_prior(n)
        x = np.vstack([X_data, x_prior])
        
        k = self._rng.integers(0, len(self.times), size=x.shape[0])
        t_k = self.times[k]
        dt = self.gammas[k]
        
        target = x + dt[:, None] * (-self.cfg.alpha_ou * x)
        self.F.fit(x, t_k, target)

    def fit(self, train_df: pd.DataFrame):
        """Runs the main IPF training loop."""
        self.columns_ = list(train_df.columns)
        X_train = train_df.to_numpy(dtype=np.float32)
        n = len(X_train)

        print("Pretraining Forward field (OU)...")
        self.pretrain_F_ou(X_train)

        for i in range(self.cfg.ipf_iters):
            # 1. Train Backward (B) on Forward trajectories
            print(f"IPF Iteration {i+1}/{self.cfg.ipf_iters} - Training B...")
            xs, ts, ys = [], [],[]
            curr_x = X_train.copy()
            
            for k in range(len(self.times)):
                t_k = np.full((n,), self.times[k], dtype=np.float32)
                t_next = np.full((n,), self.times[min(k+1, len(self.times)-1)], dtype=np.float32)
                
                drift = self.F.predict(curr_x, t_k)
                noise = self._rng.normal(size=curr_x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                next_x = drift + noise
                
                # Residual matching target
                target = next_x + (drift - self.F.predict(next_x, t_next))
                xs.append(next_x); ts.append(t_next); ys.append(target)
                curr_x = next_x
                
            self.B.fit(np.vstack(xs), np.hstack(ts), np.vstack(ys))

            # 2. Train Forward (F) on Backward trajectories
            print(f"IPF Iteration {i+1}/{self.cfg.ipf_iters} - Training F...")
            xs, ts, ys = [], [],[]
            curr_x = self._sample_prior(n)
            
            for k in range(len(self.times) - 1, -1, -1):
                t_k = np.full((n,), self.times[k], dtype=np.float32)
                t_prev = np.full((n,), self.times[max(k-1, 0)], dtype=np.float32)
                
                drift = self.B.predict(curr_x, t_k)
                noise = self._rng.normal(size=curr_x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                prev_x = drift + noise
                
                target = prev_x + (drift - self.B.predict(prev_x, t_prev))
                xs.append(prev_x); ts.append(t_prev); ys.append(target)
                curr_x = prev_x
                
            self.F.fit(np.vstack(xs), np.hstack(ts), np.vstack(ys))

        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generates synthetic data from the backward process."""
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        x = rng.normal(size=(n, self.dim)).astype(np.float32)
        
        for k in range(len(self.times) - 1, -1, -1):
            t_k = np.full((n,), self.times[k], dtype=np.float32)
            drift = self.B.predict(x, t_k)
            noise = rng.normal(size=x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
            x = drift + noise
            
        return x

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame(self.sample(n, seed), columns=self.columns_)