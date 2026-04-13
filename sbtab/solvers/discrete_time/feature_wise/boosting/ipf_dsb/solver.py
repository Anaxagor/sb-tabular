from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pgmpy.estimators import HillClimbSearch
try:
    from pgmpy.estimators import BicScore
except ImportError:
    from pgmpy.estimators import BIC as BicScore
from sklearn.preprocessing import KBinsDiscretizer

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.boosted.catboost_discrete_scalar import CatBoostDiscreteScalarConfig, CatBoostDiscreteScalar

@dataclass
class StructuralDiscreteBoostedConfig:
    num_steps: int = 20
    ipf_iters: int = 3
    alpha_ou: float = 1.0
    n_bins: int = 5
    seed: int = 42
    # Mandatory for passing parents
    catboost: CatBoostDiscreteScalarConfig = field(default_factory=lambda: CatBoostDiscreteScalarConfig(feature_mode="x_x0"))

class StructuralDiscreteBoostedSolver:
    """
    [DT] + [Boosting] +[Feature-Wise: AR]
    Each feature has its own discrete time field (N models per feature), conditioned on DAG parents.
    """
    def __init__(self, cfg: StructuralDiscreteBoostedConfig):
        self.cfg = cfg
        self.timegrid = TimeGrid(num_steps=cfg.num_steps)
        self.gammas = self.timegrid.gammas().numpy()
        self.t_grid = self.timegrid.times().numpy()
        self._rng = np.random.default_rng(cfg.seed)
        
        self.fields: dict[str, CatBoostDiscreteScalar] = {}
        self.dag = None
        self.order =[]
        self.feature_cols =[]

    def _learn_structure(self, df: pd.DataFrame):
        print("Learning causal DAG structure...")
        disc = KBinsDiscretizer(n_bins=self.cfg.n_bins, encode='ordinal', strategy='quantile')
        df_b = pd.DataFrame(disc.fit_transform(df), columns=df.columns).astype(int)
        
        hc = HillClimbSearch(df_b)
        model = hc.estimate(scoring_method=BicScore(df_b))
        
        self.dag = nx.DiGraph(model.edges())
        self.dag.add_nodes_from(df.columns)
        
        while not nx.is_directed_acyclic_graph(self.dag):
            cycle = nx.find_cycle(self.dag)
            self.dag.remove_edge(cycle[-1][0], cycle[-1][1])
            
        self.order = list(nx.topological_sort(self.dag))

    def fit(self, df: pd.DataFrame):
        self.feature_cols = list(df.columns)
        self._learn_structure(df)
        n = len(df)
        
        for col in self.order:
            print(f"\nTraining Discrete Bridge for: {col}")
            parents = list(self.dag.predecessors(col))
            x_data = df[col].values.reshape(-1, 1).astype(np.float32)
            p_data_clean = df[parents].values.astype(np.float32) if parents else np.empty((n, 0), dtype=np.float32)

            field_f = CatBoostDiscreteScalar(self.t_grid, self.cfg.catboost)
            field_b = CatBoostDiscreteScalar(self.t_grid, self.cfg.catboost)

            # OU Pretrain
            for k in range(self.cfg.num_steps):
                dt = self.gammas[k]
                target_ou = x_data + dt * (-self.cfg.alpha_ou * x_data)
                field_f.fit_step(k, x_data, target_ou, x0=p_data_clean)

            for it in range(self.cfg.ipf_iters):
                p_data = p_data_clean + self._rng.normal(size=p_data_clean.shape).astype(np.float32) * 0.01 if parents else p_data_clean
                
                # Phase 1: Train B
                curr_x = x_data.copy()
                for k in range(self.cfg.num_steps):
                    drift = field_f.predict_step(k, curr_x, x0=p_data)
                    noise = self._rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                    next_x = drift + noise
                    target_b = next_x + (drift - field_f.predict_step(k, next_x, x0=p_data))
                    field_b.fit_step(min(k+1, self.cfg.num_steps-1), next_x, target_b, x0=p_data)
                    curr_x = next_x

                # Phase 2: Train F
                curr_x = self._rng.normal(size=(n, 1)).astype(np.float32)
                for k in range(self.cfg.num_steps - 1, -1, -1):
                    drift = field_b.predict_step(k, curr_x, x0=p_data)
                    noise = self._rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                    prev_x = drift + noise
                    target_f = prev_x + (drift - field_b.predict_step(k, prev_x, x0=p_data))
                    field_f.fit_step(max(k-1, 0), prev_x, target_f, x0=p_data)
                    curr_x = prev_x

            self.fields[col] = field_b
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        gen_df = pd.DataFrame(index=range(n))
        
        for col in self.order:
            parents = list(self.dag.predecessors(col))
            p_data = gen_df[parents].values.astype(np.float32) if parents else np.empty((n, 0), dtype=np.float32)
            x_i = rng.normal(size=(n, 1)).astype(np.float32)
            
            for k in range(self.cfg.num_steps - 1, -1, -1):
                drift = self.fields[col].predict_step(k, x_i, x0=p_data)
                noise = rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                x_i = drift + noise
                
            gen_df[col] = x_i.flatten()
            
        return gen_df[self.feature_cols]