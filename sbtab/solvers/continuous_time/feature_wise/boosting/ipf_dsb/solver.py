from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, replace
from typing import Optional

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.boosted.catboost_continuous_scalar import CatBoostContinuousScalarConfig, CatBoostContinuousScalar
from sbtab.solvers.structure import LearnedDAG, learn_dag, parent_matrix

CHECKPOINT_FORMAT = "sbtab.dsb_ct_structural_gbt/2"


@dataclass
class StructuralContinuousBoostedConfig:
    num_steps: int = 30
    ipf_iters: int = 5
    alpha_ou: float = 1.0
    n_bins: int = 5
    seed: int = 42
    catboost: CatBoostContinuousScalarConfig = field(default_factory=lambda: CatBoostContinuousScalarConfig(feature_mode="x_x0_t"))
    # std of the jitter added to parent values during training (exposure-bias heuristic)
    parent_noise: float = 0.01
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: str = "geom"

class StructuralContinuousBoostedSolver:
    """
    [CT] + [Boosting] + [Feature-Wise: AR]
    Structural Autoregressive DSB using continuous time fields conditioned on DAG parents.

    The DAG is learned from the rows passed to fit() and from nothing else. Each
    column j gets its own scalar bridge conditioned on its parents; parents are
    held fixed along a column's path. Generation is topological and a column's
    parents are read from the SAME generated row.

    Per-column contract: identical to JointContinuousBoostedSolver with dim = 1
    and features [x_j, parents, t] — edge k joins X_k and X_{k+1} with gamma_k and
    carries the time label times[k] for F and B, in training and in sampling;
    iteration 0 uses the analytic OU mean x (1 - alpha gamma_k); noise sqrt(2 gamma_k).
    """
    variant_id = "dsb_ct_structural_gbt"

    def __init__(self, cfg: StructuralContinuousBoostedConfig):
        self.cfg = cfg
        self.timegrid = TimeGrid(num_steps=cfg.num_steps, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max,
                                 schedule=cfg.schedule)
        self.gammas = self.timegrid.gammas().numpy()
        self.times = self.timegrid.times().numpy()
        if cfg.alpha_ou * float(self.gammas.max()) >= 1.0:
            raise ValueError("alpha_ou * max(gamma) must be < 1: the Euler step of the OU reference x (1 - alpha gamma) "
                             f"would flip sign (got {cfg.alpha_ou * float(self.gammas.max()):.3g})")
        self._rng = np.random.default_rng(cfg.seed)
        self._cb = replace(cfg.catboost, residual=True)  # mean-matching => residual fit

        self.structure: Optional[LearnedDAG] = None
        self.generation_order = []
        self.models = {}
        self.feature_cols =[]
        self.stage_log: list[dict] = []
        self._fitted = False

    @property
    def dag(self):
        return None if self.structure is None else self.structure.graph()

    def _learn_structure(self, df: pd.DataFrame):
        print("Learning causal DAG structure...")
        self.structure = learn_dag(df, n_bins=self.cfg.n_bins)
        self.generation_order = list(self.structure.order)
        print(f"Generation Order: {' -> '.join(str(c) for c in self.generation_order)}")

    def _reference_mean(self, k: int, x: np.ndarray) -> np.ndarray:
        return (x * (1.0 - self.cfg.alpha_ou * self.gammas[k])).astype(np.float32)

    def _train_conditional_bridge(self, col: str, df: pd.DataFrame):
        parents = self.structure.parents[col]
        x_data = df[col].values.astype(np.float32).reshape(-1, 1)
        n = len(x_data)
        p_data_clean = parent_matrix(df, parents, n)
        K = self.cfg.num_steps

        # 1 dim since we predict column by column
        f_net = CatBoostContinuousScalar(cfg=self._cb)
        b_net = CatBoostContinuousScalar(cfg=self._cb)

        for it in range(self.cfg.ipf_iters):
            use_reference = it == 0
            # Add micro-noise to parents to combat exposure bias
            p_data = p_data_clean
            if parents and self.cfg.parent_noise > 0:
                p_data = p_data_clean + self._rng.normal(size=p_data_clean.shape).astype(np.float32) * self.cfg.parent_noise

            def forward_mean(k, x, t_val):
                return self._reference_mean(k, x) if use_reference else f_net.predict(x, t_val, x0=p_data)

            # --- Phase 1: Train B on Forward paths ---
            curr_x = x_data.copy()
            xs_train, ts_train, ys_target = [], [],[]

            for k in range(K):
                t_val = np.full((n,), self.times[k], dtype=np.float32)

                mean_next = forward_mean(k, curr_x, t_val)
                noise = self._rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                next_x = mean_next + noise

                target_b = next_x + (mean_next - forward_mean(k, next_x, t_val))

                xs_train.append(next_x)
                ts_train.append(t_val)          # edge k is labelled times[k] everywhere
                ys_target.append(target_b)
                curr_x = next_x

            b_net.fit(np.vstack(xs_train), np.hstack(ts_train), np.vstack(ys_target), x0=np.tile(p_data, (K, 1)))

            # --- Phase 2: Train F on Backward paths ---
            curr_x = self._rng.normal(size=(n, 1)).astype(np.float32)
            xs_train, ts_train, ys_target = [], [],[]

            for k in range(K - 1, -1, -1):
                t_val = np.full((n,), self.times[k], dtype=np.float32)

                mean_prev = b_net.predict(curr_x, t_val, x0=p_data)
                noise = self._rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                prev_x = mean_prev + noise

                target_f = prev_x + (mean_prev - b_net.predict(prev_x, t_val, x0=p_data))

                xs_train.append(prev_x)
                ts_train.append(t_val)
                ys_target.append(target_f)
                curr_x = prev_x

            f_net.fit(np.vstack(xs_train), np.hstack(ts_train), np.vstack(ys_target), x0=np.tile(p_data, (K, 1)))
            self.stage_log.append(dict(column=col, iteration=it, parents=list(parents),
                                       simulated_with="reference" if use_reference else "F"))

        print(f"  Column '{col}' | Bridge trained")
        self.models[col] = b_net

    def fit(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("structural solvers need a DataFrame (column names define the graph)")
        if self.cfg.ipf_iters < 1:
            raise ValueError("ipf_iters must be >= 1")
        self.feature_cols = list(df.columns)
        self.models = {}
        self.stage_log = []
        self._learn_structure(df)

        for col in self.generation_order:
            self._train_conditional_bridge(col, df)
        self._fitted = True
        return self

    @property
    def n_updates(self) -> int:
        return 2 * len(self.stage_log)

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Sequential generation following the topological order."""
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        gen_df = pd.DataFrame(index=range(n))

        for col in self.generation_order:
            # parents come from the rows generated so far, i.e. the same row
            p_data = parent_matrix(gen_df, self.structure.parents[col], n)

            x_i = rng.normal(size=(n, 1)).astype(np.float32)
            for k in range(self.cfg.num_steps - 1, -1, -1):
                t_k = np.full((n,), self.times[k], dtype=np.float32)
                mean_prev = self.models[col].predict(x_i, t_k, x0=p_data)
                noise = rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                x_i = mean_prev + noise

            gen_df[col] = x_i.flatten()

        return gen_df[self.feature_cols]

    # ------------------------------------------------------------------ checkpoint
    def save_checkpoint(self, path) -> None:
        state = dict(format=CHECKPOINT_FORMAT, variant_id=self.variant_id, cfg=self.cfg,
                     feature_cols=self.feature_cols, structure=self.structure.state(),
                     gammas=self.gammas, times=self.times,
                     models={c: m.model for c, m in self.models.items()},
                     stage_log=self.stage_log, fitted=self._fitted)
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load_checkpoint(cls, path) -> "StructuralContinuousBoostedSolver":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format: {state.get('format')!r}")
        solver = cls(state["cfg"])
        if not np.allclose(solver.gammas, state["gammas"]):
            raise ValueError("checkpoint time grid does not match its configuration")
        solver.feature_cols = state["feature_cols"]
        # explicit ordered parent lists are restored verbatim, never re-derived
        solver.structure = LearnedDAG.from_state(state["structure"])
        solver.generation_order = list(solver.structure.order)
        for col, model in state["models"].items():
            net = CatBoostContinuousScalar(cfg=solver._cb)
            net.model = model
            solver.models[col] = net
        solver.stage_log = state["stage_log"]
        solver._fitted = state["fitted"]
        return solver
