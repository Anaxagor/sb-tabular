from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, replace
from typing import Optional

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.boosted.catboost_discrete_scalar import CatBoostDiscreteScalarConfig, CatBoostDiscreteScalar
from sbtab.solvers.structure import LearnedDAG, learn_dag, parent_matrix

CHECKPOINT_FORMAT = "sbtab.dsb_dt_structural_gbt/2"

@dataclass
class StructuralDiscreteBoostedConfig:
    num_steps: int = 20
    ipf_iters: int = 3
    alpha_ou: float = 1.0
    n_bins: int = 5
    seed: int = 42
    # Mandatory for passing parents
    catboost: CatBoostDiscreteScalarConfig = field(default_factory=lambda: CatBoostDiscreteScalarConfig(feature_mode="x_x0"))
    # std of the jitter added to parent values during training (exposure-bias heuristic)
    parent_noise: float = 0.01
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: str = "geom"

class StructuralDiscreteBoostedSolver:
    """
    [DT] + [Boosting] +[Feature-Wise: AR]
    Each feature has its own discrete time field (N models per feature), conditioned on DAG parents.

    The DAG is learned from the rows passed to fit() and from nothing else.
    Generation is topological; a column's parents are read from the SAME generated row.

    Per-column contract: identical to JointDiscreteBoostedSolver with dim = 1 and
    features [x_j, parents] — edge k joins X_k and X_{k+1} with step gamma_k;
    F[k]: X_k -> E[X_{k+1}], B[k]: X_{k+1} -> E[X_k]; every edge 0..K-1 is fitted
    and called under its own index in both directions; iteration 0 uses the
    analytic OU mean x (1 - alpha gamma_k); noise sqrt(2 gamma_k).
    """
    variant_id = "dsb_dt_structural_gbt"

    def __init__(self, cfg: StructuralDiscreteBoostedConfig):
        self.cfg = cfg
        self.timegrid = TimeGrid(num_steps=cfg.num_steps, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max,
                                 schedule=cfg.schedule)
        self.gammas = self.timegrid.gammas().numpy()
        self.t_grid = self.timegrid.times().numpy()
        if cfg.alpha_ou * float(self.gammas.max()) >= 1.0:
            raise ValueError("alpha_ou * max(gamma) must be < 1: the Euler step of the OU reference x (1 - alpha gamma) "
                             f"would flip sign (got {cfg.alpha_ou * float(self.gammas.max()):.3g})")
        self._rng = np.random.default_rng(cfg.seed)
        self._cb = replace(cfg.catboost, residual=True)  # mean-matching => residual fit

        self.fields: dict[str, CatBoostDiscreteScalar] = {}
        self.structure: Optional[LearnedDAG] = None
        self.order =[]
        self.feature_cols =[]
        self.stage_log: list[dict] = []
        self._fitted = False

    @property
    def dag(self):
        return None if self.structure is None else self.structure.graph()

    def _learn_structure(self, df: pd.DataFrame):
        print("Learning causal DAG structure...")
        self.structure = learn_dag(df, n_bins=self.cfg.n_bins)
        self.order = list(self.structure.order)

    def _reference_mean(self, k: int, x: np.ndarray) -> np.ndarray:
        return (x * (1.0 - self.cfg.alpha_ou * self.gammas[k])).astype(np.float32)

    def fit(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("structural solvers need a DataFrame (column names define the graph)")
        if self.cfg.ipf_iters < 1:
            raise ValueError("ipf_iters must be >= 1")
        self.feature_cols = list(df.columns)
        self.fields = {}
        self.stage_log = []
        self._learn_structure(df)
        n = len(df)
        K = self.cfg.num_steps

        for col in self.order:
            print(f"\nTraining Discrete Bridge for: {col}")
            parents = self.structure.parents[col]
            x_data = df[col].values.reshape(-1, 1).astype(np.float32)
            p_data_clean = parent_matrix(df, parents, n)

            field_f = CatBoostDiscreteScalar(self.t_grid, self._cb)
            field_b = CatBoostDiscreteScalar(self.t_grid, self._cb)

            for it in range(self.cfg.ipf_iters):
                use_reference = it == 0
                p_data = p_data_clean
                if parents and self.cfg.parent_noise > 0:
                    p_data = p_data_clean + self._rng.normal(size=p_data_clean.shape).astype(np.float32) * self.cfg.parent_noise

                def forward_mean(k, x):
                    return self._reference_mean(k, x) if use_reference else field_f.predict_step(k, x, x0=p_data)

                # Phase 1: Train B
                curr_x = x_data.copy()
                for k in range(K):
                    mean_next = forward_mean(k, curr_x)
                    noise = self._rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                    next_x = mean_next + noise
                    target_b = next_x + (mean_next - forward_mean(k, next_x))
                    field_b.fit_step(k, next_x, target_b, x0=p_data)
                    curr_x = next_x

                # Phase 2: Train F
                curr_x = self._rng.normal(size=(n, 1)).astype(np.float32)
                for k in range(K - 1, -1, -1):
                    mean_prev = field_b.predict_step(k, curr_x, x0=p_data)
                    noise = self._rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                    prev_x = mean_prev + noise
                    target_f = prev_x + (mean_prev - field_b.predict_step(k, prev_x, x0=p_data))
                    field_f.fit_step(k, prev_x, target_f, x0=p_data)
                    curr_x = prev_x
                self.stage_log.append(dict(column=col, iteration=it, parents=list(parents),
                                           simulated_with="reference" if use_reference else "F",
                                           edges_fitted=list(range(K))))

            missing = [k for k in range(K) if not field_b.is_fitted(k)]
            if missing:
                raise RuntimeError(f"backward edge models not fitted for column {col!r}: {missing}")
            self.fields[col] = field_b
        self._fitted = True
        return self

    @property
    def n_updates(self) -> int:
        return int(sum(2 * len(s["edges_fitted"]) for s in self.stage_log))

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        gen_df = pd.DataFrame(index=range(n))

        for col in self.order:
            # parents come from the rows generated so far, i.e. the same row
            p_data = parent_matrix(gen_df, self.structure.parents[col], n)
            x_i = rng.normal(size=(n, 1)).astype(np.float32)

            for k in range(self.cfg.num_steps - 1, -1, -1):
                mean_prev = self.fields[col].predict_step(k, x_i, x0=p_data)
                noise = rng.normal(size=(n, 1)).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                x_i = mean_prev + noise

            gen_df[col] = x_i.flatten()

        return gen_df[self.feature_cols]

    # ------------------------------------------------------------------ checkpoint
    def save_checkpoint(self, path) -> None:
        state = dict(format=CHECKPOINT_FORMAT, variant_id=self.variant_id, cfg=self.cfg,
                     feature_cols=self.feature_cols, structure=self.structure.state(),
                     gammas=self.gammas, t_grid=self.t_grid,
                     models={c: f.models for c, f in self.fields.items()},
                     stage_log=self.stage_log, fitted=self._fitted)
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load_checkpoint(cls, path) -> "StructuralDiscreteBoostedSolver":
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
        solver.order = list(solver.structure.order)
        for col, models in state["models"].items():
            f = CatBoostDiscreteScalar(solver.t_grid, solver._cb)
            f.models = models
            solver.fields[col] = f
        solver.stage_log = state["stage_log"]
        solver._fitted = state["fitted"]
        return solver
