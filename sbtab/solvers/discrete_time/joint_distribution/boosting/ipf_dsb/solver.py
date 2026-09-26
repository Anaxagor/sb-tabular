from __future__ import annotations

import pickle
from dataclasses import dataclass, field, replace
from typing import Optional
import numpy as np
import pandas as pd

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.boosted.catboost_discrete_joint import CatBoostDiscreteJointConfig, CatBoostDiscreteJoint

CHECKPOINT_FORMAT = "sbtab.dsb_dt_joint_gbt/2"


@dataclass
class JointDiscreteBoostedConfig:
    num_steps: int = 20
    ipf_iters: int = 5
    alpha_ou: float = 1.0
    seed: int = 42
    catboost: CatBoostDiscreteJointConfig = field(default_factory=CatBoostDiscreteJointConfig)
    # Time grid. The horizon T = sum(gamma) is a property of the declared OU
    # reference and is deliberately not forced to 1; training and sampling both
    # read it from the same TimeGrid.
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: str = "geom"

class JointDiscreteBoostedSolver:
    """
    [DT] + [Boosting] +[Joint]
    Implements IPF-DSB where each time step has its own CatBoost model predicting the full vector.

    Contract (De Bortoli et al., DSB, mean-matching form)
      orientation   data at k = 0, prior N(0, I) at k = K; generation runs B from K to 0
      reference     OU  dX = -alpha X dt + sqrt(2) dW, discretised on the grid:
                    X_{k+1} = X_k (1 - alpha gamma_k) + sqrt(2 gamma_k) Z
      edge k        joins X_k and X_{k+1} with step gamma_k, k = 0..K-1, in BOTH directions:
                      F[k] : X_k     -> E[X_{k+1}]      B[k] : X_{k+1} -> E[X_k]
      targets       B[k](X_{k+1}) <- X_{k+1} + F[k](X_k) - F[k](X_{k+1})   on forward paths
                    F[k](X_k)     <- X_k + B[k](X_{k+1}) - B[k](X_k)       on backward paths
                    (next-state MEANS, units of x; the sampler adds noise, no gamma factor)
      iteration 0   the forward map is the ANALYTIC reference mean, not a fitted model
      sampler       x <- B[k](x) + sqrt(2 gamma_k) Z,  k = K-1..0
    """
    variant_id = "dsb_dt_joint_gbt"

    def __init__(self, dim: int, cfg: JointDiscreteBoostedConfig):
        self.dim = dim
        self.cfg = cfg
        self.timegrid = TimeGrid(num_steps=cfg.num_steps, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max,
                                 schedule=cfg.schedule)
        self.gammas = self.timegrid.gammas().numpy()
        self.t_grid = self.timegrid.times().numpy()
        if cfg.alpha_ou * float(self.gammas.max()) >= 1.0:
            raise ValueError("alpha_ou * max(gamma) must be < 1: the Euler step of the OU reference x (1 - alpha gamma) "
                             f"would flip sign (got {cfg.alpha_ou * float(self.gammas.max()):.3g})")

        # mean-matching maps are identity + O(gamma): always fit the residual
        cb = replace(cfg.catboost, residual=True)
        self.field_f = CatBoostDiscreteJoint(dim, self.t_grid, cb)
        self.field_b = CatBoostDiscreteJoint(dim, self.t_grid, cb)
        self._rng = np.random.default_rng(cfg.seed)
        self.columns_: Optional[list[str]] = None
        self.stage_log: list[dict] = []
        self._fitted = False

    def _reference_mean(self, k: int, x: np.ndarray) -> np.ndarray:
        """One-step OU mean over the STEP interval gamma_k (not the elapsed time)."""
        return (x * (1.0 - self.cfg.alpha_ou * self.gammas[k])).astype(np.float32)

    def _forward_mean(self, k: int, x: np.ndarray, use_reference: bool) -> np.ndarray:
        return self._reference_mean(k, x) if use_reference else self.field_f.predict_step(k, x)

    def fit(self, train_df):
        if isinstance(train_df, pd.DataFrame):
            self.columns_ = list(train_df.columns)
            X_train = train_df.to_numpy(dtype=np.float32)
        else:
            X_train = np.asarray(train_df, dtype=np.float32)
            self.columns_ = None
        if X_train.ndim != 2 or X_train.shape[1] != self.dim:
            raise ValueError(f"expected training data of shape (N, {self.dim})")
        K = self.cfg.num_steps
        self.stage_log = []

        for it in range(self.cfg.ipf_iters):
            use_reference = it == 0
            # 1. Train Backward field (B) on Forward paths
            print(f"IPF Iteration {it+1}/{self.cfg.ipf_iters} - Training B...")
            curr_x = X_train.copy()
            for k in range(K):
                mean_next = self._forward_mean(k, curr_x, use_reference)
                noise = self._rng.normal(size=curr_x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                next_x = mean_next + noise

                target = next_x + (mean_next - self._forward_mean(k, next_x, use_reference))
                self.field_b.fit_step(k, next_x, target)
                curr_x = next_x
            self.stage_log.append(dict(iteration=it, trained="B", simulated_with="reference" if use_reference else "F",
                                       edges_fitted=list(range(K))))

            # 2. Train Forward field (F) on Backward paths
            print(f"IPF Iteration {it+1}/{self.cfg.ipf_iters} - Training F...")
            curr_x = self._rng.normal(size=X_train.shape).astype(np.float32)
            for k in range(K - 1, -1, -1):
                mean_prev = self.field_b.predict_step(k, curr_x)
                noise = self._rng.normal(size=curr_x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                prev_x = mean_prev + noise

                target = prev_x + (mean_prev - self.field_b.predict_step(k, prev_x))
                self.field_f.fit_step(k, prev_x, target)
                curr_x = prev_x
            self.stage_log.append(dict(iteration=it, trained="F", simulated_with="B",
                                       edges_fitted=list(range(K - 1, -1, -1))))

        self._assert_all_edges_fitted()
        self._fitted = True
        return self

    def _assert_all_edges_fitted(self) -> None:
        missing_b = [k for k in range(self.cfg.num_steps) if not self.field_b.is_fitted(k)]
        if missing_b:
            raise RuntimeError(f"backward edge models not fitted: {missing_b}")
        if self.cfg.ipf_iters > 0:
            missing_f = [k for k in range(self.cfg.num_steps) if not self.field_f.is_fitted(k)]
            if missing_f:
                raise RuntimeError(f"forward edge models not fitted: {missing_f}")

    @property
    def n_updates(self) -> int:
        """Number of per-edge model fits (the boosted analogue of optimizer updates)."""
        return int(sum(len(s["edges_fitted"]) for s in self.stage_log))

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        x = rng.normal(size=(n, self.dim)).astype(np.float32)

        for k in range(self.cfg.num_steps - 1, -1, -1):
            mean_prev = self.field_b.predict_step(k, x)
            noise = rng.normal(size=x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
            x = mean_prev + noise
        return x

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame(self.sample(n, seed), columns=self.columns_)

    # ------------------------------------------------------------------ checkpoint
    def save_checkpoint(self, path) -> None:
        state = dict(format=CHECKPOINT_FORMAT, variant_id=self.variant_id, cfg=self.cfg, dim=self.dim,
                     columns=self.columns_, gammas=self.gammas, t_grid=self.t_grid,
                     models_f=self.field_f.models, models_b=self.field_b.models,
                     stage_log=self.stage_log, fitted=self._fitted)
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load_checkpoint(cls, path) -> "JointDiscreteBoostedSolver":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format: {state.get('format')!r}")
        solver = cls(state["dim"], state["cfg"])
        if not np.allclose(solver.gammas, state["gammas"]):
            raise ValueError("checkpoint time grid does not match its configuration")
        solver.columns_ = state["columns"]
        solver.field_f.models = state["models_f"]
        solver.field_b.models = state["models_b"]
        solver.stage_log = state["stage_log"]
        solver._fitted = state["fitted"]
        return solver
