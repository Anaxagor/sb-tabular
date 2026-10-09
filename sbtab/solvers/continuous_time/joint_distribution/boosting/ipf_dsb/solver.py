from __future__ import annotations

import pickle
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd
from typing import Optional

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.boosted.catboost_continuous_joint import CatBoostContinuousJointConfig, CatBoostContinuousJoint

CHECKPOINT_FORMAT = "sbtab.dsb_ct_joint_gbt/2"


@dataclass
class JointContinuousBoostedConfig:
    num_steps: int = 20
    ipf_iters: int = 5
    alpha_ou: float = 1.0
    seed: int = 42
    catboost: CatBoostContinuousJointConfig = field(default_factory=CatBoostContinuousJointConfig)
    # Time grid; T = sum(gamma) belongs to the declared OU reference (not forced to 1).
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: str = "geom"
    horizon: Optional[float] = 2.0  # rescale steps; None keeps the raw gamma schedule

class JointContinuousBoostedSolver:
    """[CT] + [Boosting] + [Joint]
    Implements IPF-DSB using a single continuous-time CatBoost model for all features.

    "Continuous time" here means ONE time-conditioned regressor per direction with
    the edge's time label as a feature; it is trained and evaluated on the grid.

    Contract (De Bortoli et al., DSB, mean-matching form)
      orientation   data at k = 0, prior N(0, I) at k = K; generation runs B from K to 0
      reference     OU  dX = -alpha X dt + sqrt(2) dW:
                    X_{k+1} = X_k (1 - alpha gamma_k) + sqrt(2 gamma_k) Z
      edge k        joins X_k and X_{k+1} with step gamma_k. Its time label is
                    times[k] for F AND for B, in training AND in sampling.
      targets       B(X_{k+1}, t_k) <- X_{k+1} + F(X_k, t_k) - F(X_{k+1}, t_k)
                    F(X_k, t_k)     <- X_k + B(X_{k+1}, t_k) - B(X_k, t_k)
                    Both evaluations of the opposite map use the SAME edge k
                    (DSB Prop. 3); next-state means in units of x.
      iteration 0   the forward map is the ANALYTIC reference mean, not a fitted model
      sampler       x <- B(x, t_k) + sqrt(2 gamma_k) Z,  k = K-1..0
    """
    variant_id = "dsb_ct_joint_gbt"

    def __init__(self, dim: int, cfg: JointContinuousBoostedConfig):
        self.dim = dim
        self.cfg = cfg

        self.timegrid = TimeGrid(num_steps=cfg.num_steps, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max,
                                 schedule=cfg.schedule, horizon=cfg.horizon)
        self.gammas = self.timegrid.gammas().numpy()
        self.times = self.timegrid.times().numpy()
        if cfg.alpha_ou * float(self.gammas.max()) >= 1.0:
            raise ValueError("alpha_ou * max(gamma) must be < 1: the Euler step of the OU reference x (1 - alpha gamma) "
                             f"would flip sign (got {cfg.alpha_ou * float(self.gammas.max()):.3g})")
        self._rng = np.random.default_rng(cfg.seed)

        # mean-matching maps are identity + O(gamma): always fit the residual
        cb = replace(cfg.catboost, residual=True)
        self.F = CatBoostContinuousJoint(dim=dim, cfg=cb)
        self.B = CatBoostContinuousJoint(dim=dim, cfg=cb)

        self.columns_: Optional[list[str]] = None
        self.stage_log: list[dict] = []
        self._fitted = False

    def _sample_prior(self, n: int) -> np.ndarray:
        return self._rng.normal(size=(n, self.dim)).astype(np.float32)

    def _reference_mean(self, k: int, x: np.ndarray) -> np.ndarray:
        """One-step OU mean over the STEP interval gamma_k."""
        return (x * (1.0 - self.cfg.alpha_ou * self.gammas[k])).astype(np.float32)

    def _forward_mean(self, k: int, x: np.ndarray, t_k: np.ndarray, use_reference: bool) -> np.ndarray:
        return self._reference_mean(k, x) if use_reference else self.F.predict(x, t_k)

    def fit(self, train_df):
        """Runs the main IPF training loop."""
        if isinstance(train_df, pd.DataFrame):
            self.columns_ = list(train_df.columns)
            X_train = train_df.to_numpy(dtype=np.float32)
        else:
            X_train = np.asarray(train_df, dtype=np.float32)
            self.columns_ = None
        if X_train.ndim != 2 or X_train.shape[1] != self.dim:
            raise ValueError(f"expected training data of shape (N, {self.dim})")
        n = len(X_train)
        K = len(self.times)
        self.stage_log = []

        for i in range(self.cfg.ipf_iters):
            use_reference = i == 0
            # 1. Train Backward (B) on Forward trajectories
            print(f"IPF Iteration {i+1}/{self.cfg.ipf_iters} - Training B...")
            xs, ts, ys = [], [],[]
            curr_x = X_train.copy()

            for k in range(K):
                t_k = np.full((n,), self.times[k], dtype=np.float32)

                mean_next = self._forward_mean(k, curr_x, t_k, use_reference)
                noise = self._rng.normal(size=curr_x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                next_x = mean_next + noise

                # Mean-matching target: both F evaluations belong to edge k
                target = next_x + (mean_next - self._forward_mean(k, next_x, t_k, use_reference))
                xs.append(next_x); ts.append(t_k); ys.append(target)
                curr_x = next_x

            self.B.fit(np.vstack(xs), np.hstack(ts), np.vstack(ys))
            self.stage_log.append(dict(iteration=i, trained="B", simulated_with="reference" if use_reference else "F",
                                       edge_labels=[float(self.times[k]) for k in range(K)]))

            # 2. Train Forward (F) on Backward trajectories
            print(f"IPF Iteration {i+1}/{self.cfg.ipf_iters} - Training F...")
            xs, ts, ys = [], [],[]
            curr_x = self._sample_prior(n)

            for k in range(K - 1, -1, -1):
                t_k = np.full((n,), self.times[k], dtype=np.float32)

                mean_prev = self.B.predict(curr_x, t_k)
                noise = self._rng.normal(size=curr_x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
                prev_x = mean_prev + noise

                target = prev_x + (mean_prev - self.B.predict(prev_x, t_k))
                xs.append(prev_x); ts.append(t_k); ys.append(target)
                curr_x = prev_x

            self.F.fit(np.vstack(xs), np.hstack(ts), np.vstack(ys))
            self.stage_log.append(dict(iteration=i, trained="F", simulated_with="B",
                                       edge_labels=[float(self.times[k]) for k in range(K - 1, -1, -1)]))

        if self.B.model is None:
            raise RuntimeError("backward field was not fitted (ipf_iters must be >= 1)")
        self._fitted = True
        return self

    @property
    def n_updates(self) -> int:
        """Number of field fits (the boosted analogue of optimizer updates)."""
        return len(self.stage_log)

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generates synthetic data from the backward process."""
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        x = rng.normal(size=(n, self.dim)).astype(np.float32)

        for k in range(len(self.times) - 1, -1, -1):
            t_k = np.full((n,), self.times[k], dtype=np.float32)
            mean_prev = self.B.predict(x, t_k)
            noise = rng.normal(size=x.shape).astype(np.float32) * np.sqrt(2.0 * self.gammas[k])
            x = mean_prev + noise

        return x

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame(self.sample(n, seed), columns=self.columns_)

    # ------------------------------------------------------------------ checkpoint
    def save_checkpoint(self, path) -> None:
        state = dict(format=CHECKPOINT_FORMAT, variant_id=self.variant_id, cfg=self.cfg, dim=self.dim,
                     columns=self.columns_, gammas=self.gammas, times=self.times,
                     model_f=self.F.model, model_b=self.B.model, stage_log=self.stage_log, fitted=self._fitted)
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load_checkpoint(cls, path) -> "JointContinuousBoostedSolver":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format: {state.get('format')!r}")
        # Old pickled configs have no horizon field: preserve their raw gamma grid.
        cfg = replace(state["cfg"], horizon=vars(state["cfg"]).get("horizon"))
        solver = cls(state["dim"], cfg)
        if not np.allclose(solver.gammas, state["gammas"]):
            raise ValueError("checkpoint time grid does not match its configuration")
        solver.columns_ = state["columns"]
        solver.F.model = state["model_f"]
        solver.B.model = state["model_b"]
        solver.stage_log = state["stage_log"]
        solver._fitted = state["fitted"]
        return solver
