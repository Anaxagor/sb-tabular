from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pgmpy.estimators import HillClimbSearch, BicScore
from sklearn.preprocessing import KBinsDiscretizer

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.boosted.catboost_continuous_scalar import CatBoostContinuousFieldConfig, CatBoostContinuousField


@dataclass
class StructuralContinuousBoostedConfig:
    num_steps: int = 30
    ipf_iters: int = 5
    alpha_ou: float = 1.0
    n_bins: int = 5
    seed: int = 42
    catboost: CatBoostContinuousFieldConfig = CatBoostContinuousFieldConfig(feature_mode="x_x0_t")

class StructuralContinuousBoostedSolver:
    """
    [CT] + [Boosting] + [Feature-Wise: AR]
    Structural Autoregressive DSB using continuous time fields conditioned on DAG parents.
    """
    def __init__(self, cfg: StructuralContinuousBoostedConfig):
        self.cfg = cfg
        self.timegrid = TimeGrid(num_steps=cfg.num_steps, T=1.0)
        self.gammas = self.timegrid.gammas().numpy()
        self.times = self.timegrid.times().numpy()
        self._rng = np.random.default_rng(cfg.seed)
        
        self.dag = None
        self.generation_order = []
        self.models = {} 
        self.feature_cols =[]

    def _learn_structure(self, df: pd.DataFrame):
        print("Learning causal DAG structure...")
        discretizer = KBinsDiscretizer(n_bins=self.cfg.n_bins, encode='ordinal', strategy='quantile')
        df_binned = pd.DataFrame(discretizer.fit_transform(df), columns=df.columns).astype(int)
        
        hc = HillClimbSearch(df_binned)
        best_model = hc.estimate(scoring_method=BicScore(df_binned))
        
        G = nx.DiGraph(best_model.edges())
        G.add_nodes_from(df.columns)
        
        # Cycle protection
        while not nx.is_directed_acyclic_graph(G):
            cycle = nx.find_cycle(G)
            G.remove_edge(cycle[-1][0], cycle[-1][1])
            
        self.dag = G
        self.generation_order = list(nx.topological_sort(G))
        print(f"Generation Order: {' -> '.join(self.generation_order)}")

    def _train_conditional_bridge(self, col: str, df: pd.DataFrame):
        parents = list(self.dag.predecessors(col))
        x_data = df[col].values.astype(np.float32).reshape(-1, 1)
        p_data_clean = df[parents].values.astype(np.float32) if parents else np.empty((len(df), 0), dtype=np.float32)
        n = len(x_data)

        # 1 dim since we predict column by column
        f_net = CatBoostContinuousField(dim=1, cfg=self.cfg.catboost)
        b_net = CatBoostContinuousField(dim=1, cfg=self.cfg.catboost)

        # 1. OU Pretrain
        k_rand = self._rng.integers(0, self.cfg.num_steps, size=n)
        t_k = self.times[k_rand]
        dt = self.gammas[k_rand]
        target_ou = x_data + dt[:, None] * (-self.cfg.alpha_ou * x_data)
        
        f_net.fit(x_data, t_k, target_ou, x0=p_data_clean)

        for it in range(self.cfg.ipf_iters):
            # Add micro-noise to parents to combat exposure bias
            p_data = p_data_clean + self._rng.normal(size=p_data_clean.shape).astype(np.float32) * 0.01 if parents else p_data_clean

            # --- Phase 1: Train B on Forward paths ---
            curr_x = x_data.copy()
            xs_train, ts_train, ys_target = [], [],[]

            for k in range(self.cfg.num_steps):
                t_val = np.full((n,), self.times[k], dtype=np.float32)
                t_val_next = np.full((n,), self.times[min(k+1, self.cfg.num_steps-1)], dtype=np.float32)
                
                mean_next = f_net.predict(curr_x, t_val, x0=p_data)
                noise = self._rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                next_x = mean_next + noise

                target_b = next_x + (mean_next - f_net.predict(next_x, t_val, x0=p_data))

                xs_train.append(next_x)
                ts_train.append(t_val_next)
                ys_target.append(target_b)
                curr_x = next_x

            b_net.fit(np.vstack(xs_train), np.hstack(ts_train), np.vstack(ys_target), x0=np.tile(p_data, (self.cfg.num_steps, 1)))

            # --- Phase 2: Train F on Backward paths ---
            curr_x = self._rng.normal(size=(n, 1)).astype(np.float32)
            xs_train, ts_train, ys_target = [], [],[]

            for k in range(self.cfg.num_steps - 1, -1, -1):
                t_val = np.full((n,), self.times[k], dtype=np.float32)
                t_val_prev = np.full((n,), self.times[max(k-1, 0)], dtype=np.float32)
                
                mean_prev = b_net.predict(curr_x, t_val, x0=p_data)
                noise = self._rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                prev_x = mean_prev + noise

                target_f = prev_x + (mean_prev - b_net.predict(prev_x, t_val, x0=p_data))

                xs_train.append(prev_x)
                ts_train.append(t_val_prev)
                ys_target.append(target_f)
                curr_x = prev_x

            f_net.fit(np.vstack(xs_train), np.hstack(ts_train), np.vstack(ys_target), x0=np.tile(p_data, (self.cfg.num_steps, 1)))
            
        print(f"  Column '{col}' | Bridge trained")
        self.models[col] = b_net

    def fit(self, df: pd.DataFrame):
        self.feature_cols = list(df.columns)
        self._learn_structure(df) 
        
        for col in self.generation_order:
            self._train_conditional_bridge(col, df)
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Sequential generation following the topological order."""
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        gen_df = pd.DataFrame(index=range(n))
        
        for col in self.generation_order:
            parents = list(self.dag.predecessors(col))
            p_data = gen_df[parents].values.astype(np.float32) if parents else np.empty((n, 0), dtype=np.float32)
            
            x_i = rng.normal(size=(n, 1)).astype(np.float32)
            for k in range(self.cfg.num_steps - 1, -1, -1):
                t_k = np.full((n,), self.times[k], dtype=np.float32)
                drift = self.models[col].predict(x_i, t_k, x0=p_data)
                noise = rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                x_i = drift + noise
                
            gen_df[col] = x_i.flatten()
            
        return gen_df[self.feature_cols]