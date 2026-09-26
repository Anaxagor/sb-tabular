from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from sbtab.bridge.reference import GaussianReference
from sbtab.models.boosted.catboost_discrete_scalar import (
    CatBoostScalarConfig,
    CatBoostTimeDiscretizedScalar,
)
from sbtab.solvers.structure import LearnedDAG, learn_dag


FB = Literal["b", "f"]

CANONICAL_ID = "dsbm_dt_structural_gbt"
CHECKPOINT_FORMAT = "sbtab.dsbm_dt_structural_gbt.v1"

_STRUCTURES = ("autoregressive", "map", "learned")


@dataclass
class FeaturewiseDSBMBoostConfig:
    # IMF stage sequence: non-empty, only "f"/"b", strictly alternating (a stage is
    # trained on the coupling of the opposite direction). sample() needs a "b".
    fb_sequence: Sequence[FB] = ("b", "f", "b", "f", "b")

    num_steps: int = 50
    sigma: float = 0.10
    # Unused: the state-time grids need no clipping (both targets are finite on
    # them). Kept so existing configs keep loading.
    eps: float = 1e-3

    # "ind": independent coupling data (x) prior -- the DSBM-IMF initialiser.
    # "ref": z1 = z0 + sigma * N(0, I), the reference (IPF-like) coupling; its t=1
    #        marginal is data * N(0, sigma^2 I), not the N(0, I) prior.
    first_coupling: Literal["ind", "ref"] = "ind"

    n_noise_per_pair: int = 1
    # noise=False selects the NOISELESS HEURISTIC sampler for generation only.
    # Couplings simulated during fit() always use noise.
    noise: bool = True

    # Generation order. "autoregressive": optional (default: column order).
    # "map": optional (default: a topological order of the map). "learned": must
    # be None.
    feature_order: Optional[List[str]] = None
    # Explicit ordered parent lists; used ONLY with structure="map", where it must
    # have exactly one key per column.
    context_cols_map: Optional[Dict[str, List[str]]] = None

    # residual must stay False: DSBM targets are drifts, not next-state means.
    catboost: CatBoostScalarConfig = field(default_factory=CatBoostScalarConfig)

    seed: int = 0

    # Dependency structure between columns:
    #   "autoregressive": full chain -- every column is conditioned on ALL earlier
    #                     columns of feature_order. An exact factorisation.
    #   "map":            parents taken verbatim from context_cols_map.
    #   "learned":        sbtab.solvers.structure.learn_dag on the rows passed to
    #                     fit() only (hill-climb / BIC on quantile bins).
    structure: Literal["autoregressive", "map", "learned"] = "autoregressive"
    structure_n_bins: int = 5


class FeaturewiseDSBMBoostSolver:
    """
    Feature-wise CatBoost IMF+DSBM solver (registry id ``dsbm_dt_structural_gbt``):
    one scalar bridge per column, conditioned on the column's parents, one
    CatBoost model per column, step and direction.

    Structure. ``feature_order_`` is a topological generation order and
    ``parents_`` holds the explicit ORDERED parent list of every column; the
    parent order is the feature layout of the fitted models and is stored
    verbatim. Generation walks ``feature_order_`` and reads each column's parents
    from the SAME generated row.

    Time grids. Model k always drives the edge between k/N and (k+1)/N, and it is
    trained at the time of the state it is APPLIED to:
        forward  model k: t_f[k] = k/N       (state at the left end of the edge)
        backward model k: t_b[k] = (k+1)/N   (state at the right end of the edge)
    The forward target is finite at t=0 and the backward target at t=1, so no
    clipping is needed.

    Sampler: x <- x + drift_k(x, parents) dt + sigma sqrt(dt) eps. The drift is
    trained for the STOCHASTIC bridge and contains the score term. With
    ``cfg.noise=False`` the same drift is integrated without noise: that is a
    heuristic, NOT a probability-flow ODE, and it does not preserve the marginals.
    It is reported as ``variant_id == "dsbm_dt_structural_gbt_noiseless_heuristic"``.
    The flag only affects sample(); couplings simulated during fit() always use
    noise.
    """

    canonical_id = CANONICAL_ID

    def __init__(self, cfg: FeaturewiseDSBMBoostConfig):
        self.cfg = cfg
        self._validate_config(cfg)

        self.columns_: Optional[List[str]] = None
        self.dim_: Optional[int] = None

        # structure: topological order + explicit ordered parent lists
        self.feature_order_: Optional[List[str]] = None
        self.feature_order_idx_: Optional[List[int]] = None
        self.parents_: Dict[str, List[str]] = {}
        self.context_idx_: Dict[int, List[int]] = {}
        self.dag_: Optional[LearnedDAG] = None

        self.fields_f_: Dict[int, CatBoostTimeDiscretizedScalar] = {}
        self.fields_b_: Dict[int, CatBoostTimeDiscretizedScalar] = {}

        self.t_grid_f_, self.t_grid_b_ = self._make_t_grids(int(cfg.num_steps))

        self.reference: Optional[GaussianReference] = None
        self.stage_log: List[dict] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------ config
    @staticmethod
    def _validate_config(cfg: FeaturewiseDSBMBoostConfig) -> None:
        seq = tuple(cfg.fb_sequence)
        if len(seq) == 0:
            raise ValueError("fb_sequence must not be empty")
        for d in seq:
            if d not in ("f", "b"):
                raise ValueError(f"Unknown direction in fb_sequence: {d!r} (allowed: 'f', 'b')")
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                raise ValueError(
                    "fb_sequence must strictly alternate: each stage is trained on the coupling "
                    f"of the opposite direction, got {seq}"
                )
        if cfg.first_coupling not in ("ind", "ref"):
            raise ValueError(f"Unknown first_coupling={cfg.first_coupling!r} (allowed: 'ind', 'ref')")
        if int(cfg.num_steps) < 1:
            raise ValueError("num_steps must be >= 1")
        if int(cfg.n_noise_per_pair) < 1:
            raise ValueError("n_noise_per_pair must be >= 1")
        if not float(cfg.sigma) >= 0.0:
            raise ValueError("sigma must be >= 0")
        if getattr(cfg.catboost, "residual", False):
            raise ValueError("DSBM regresses drifts: the CatBoost field must use residual=False")

        if cfg.structure not in _STRUCTURES:
            raise ValueError(f"Unknown structure={cfg.structure!r} (allowed: {_STRUCTURES})")
        if cfg.structure == "map" and cfg.context_cols_map is None:
            raise ValueError("structure='map' requires context_cols_map")
        if cfg.structure != "map" and cfg.context_cols_map is not None:
            raise ValueError("context_cols_map is only used with structure='map'")
        if cfg.structure == "learned" and cfg.feature_order is not None:
            raise ValueError("structure='learned' derives the generation order; feature_order must be None")

    @property
    def variant_id(self) -> str:
        return self.canonical_id if self.cfg.noise else f"{self.canonical_id}_noiseless_heuristic"

    # ------------------------------------------------------------------ utilities
    @staticmethod
    def _make_t_grids(N: int) -> tuple[np.ndarray, np.ndarray]:
        """State-time grids: t_f[k] = k/N, t_b[k] = (k+1)/N, k = 0..N-1."""
        if N < 1:
            raise ValueError("num_steps must be >= 1")
        k = np.arange(N, dtype=np.float64)
        return k / N, (k + 1.0) / N

    def _t_grid(self, fb: FB) -> np.ndarray:
        return self.t_grid_f_ if fb == "f" else self.t_grid_b_

    @staticmethod
    def _make_generator(seed: Optional[int]) -> torch.Generator:
        """Seeded CPU generator; seed=None draws fresh entropy."""
        gen = torch.Generator(device="cpu")
        if seed is None:
            gen.seed()
        else:
            gen.manual_seed(int(seed))
        return gen

    @staticmethod
    def _randn(shape, generator: torch.Generator) -> np.ndarray:
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).numpy()

    def _sample_reference(self, n: int, generator: torch.Generator) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("Reference process is not initialized.")
        x = self.reference.sample(n=n, generator=generator)
        return x.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _ensure_float_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    # ------------------------------------------------------------------ structure
    def _resolve_feature_order(self, cols: List[str]) -> List[str]:
        if self.cfg.feature_order is None:
            return list(cols)
        order = list(self.cfg.feature_order)
        if len(order) != len(set(order)) or set(order) != set(cols):
            missing = sorted(set(cols) - set(order), key=str)
            extra = sorted(set(order) - set(cols), key=str)
            raise ValueError(
                "feature_order must be a permutation of df.columns.\n"
                f"Missing in feature_order: {missing}\n"
                f"Extra in feature_order: {extra}"
            )
        return order

    def _resolve_map_parents(self, cols: List[str]) -> Dict[str, List[str]]:
        cmap = self.cfg.context_cols_map or {}
        unknown = [k for k in cmap if k not in cols]
        if unknown:
            raise ValueError(f"context_cols_map has keys that are not dataset columns: {unknown}")
        missing = [c for c in cols if c not in cmap]
        if missing:
            raise ValueError(
                f"structure='map' needs a context_cols_map entry for every column; missing: {missing}. "
                "Use an empty list for a root column."
            )
        parents: Dict[str, List[str]] = {}
        for c in cols:
            ps = list(cmap[c])
            bad = [p for p in ps if p not in cols]
            if bad:
                raise ValueError(f"context_cols_map[{c!r}] names unknown columns: {bad}")
            if c in ps:
                raise ValueError(f"context_cols_map[{c!r}] lists the column as its own parent")
            if len(ps) != len(set(ps)):
                raise ValueError(f"context_cols_map[{c!r}] has duplicate parents")
            parents[c] = ps
        return parents

    @staticmethod
    def _topological_order(cols: List[str], parents: Dict[str, List[str]]) -> List[str]:
        """Kahn's algorithm; ties are broken by column position."""
        order: List[str] = []
        placed: set = set()
        remaining = list(cols)
        while remaining:
            ready = [c for c in remaining if all(p in placed for p in parents[c])]
            if not ready:
                raise ValueError(f"context_cols_map is cyclic among columns {remaining}")
            order.append(ready[0])
            placed.add(ready[0])
            remaining.remove(ready[0])
        return order

    def _resolve_structure(self, df: pd.DataFrame) -> tuple[List[str], Dict[str, List[str]], Optional[LearnedDAG]]:
        """(generation order, explicit ordered parent lists, learned DAG or None)."""
        cols = list(df.columns)
        structure = self.cfg.structure
        dag: Optional[LearnedDAG] = None

        if structure == "autoregressive":
            order = self._resolve_feature_order(cols)
            parents = {c: list(order[:i]) for i, c in enumerate(order)}
        elif structure == "map":
            parents = self._resolve_map_parents(cols)
            if self.cfg.feature_order is None:
                order = self._topological_order(cols, parents)
            else:
                order = self._resolve_feature_order(cols)
        elif structure == "learned":
            # learn_dag works on string labels; fit on exactly the rows given to fit()
            by_name = {str(c): c for c in cols}
            if len(by_name) != len(cols):
                raise ValueError("column names must be unique as strings for structure='learned'")
            named = df.copy()
            named.columns = list(by_name)
            dag = learn_dag(named, n_bins=int(self.cfg.structure_n_bins))
            order = [by_name[c] for c in dag.order]
            parents = {by_name[c]: [by_name[p] for p in ps] for c, ps in dag.parents.items()}
        else:
            raise ValueError(f"Unknown structure={structure!r}")

        self._validate_structure(order, parents)
        return order, parents, dag

    @staticmethod
    def _validate_structure(order: List[str], parents: Dict[str, List[str]]) -> None:
        if set(order) != set(parents) or len(order) != len(parents):
            raise ValueError("generation order and parent map cover different columns")
        seen: set = set()
        for col in order:
            late = [p for p in parents[col] if p not in seen]
            if late:
                raise ValueError(
                    f"Invalid structure: column {col!r} is generated before its parents {late}. "
                    "All parents must appear earlier in the generation order."
                )
            seen.add(col)

    def _setup(self, df: pd.DataFrame) -> None:
        """Columns, reference and structure for the (float) training frame ``df``."""
        self.columns_ = list(df.columns)
        self.dim_ = len(self.columns_)
        self.reference = GaussianReference(dim=self.dim_, device=torch.device("cpu"))

        order, parents, dag = self._resolve_structure(df)
        self._set_structure(order, parents, dag)

    def _set_structure(self, order: List[str], parents: Dict[str, List[str]], dag: Optional[LearnedDAG]) -> None:
        col_to_idx = {c: i for i, c in enumerate(self.columns_)}
        self.feature_order_ = list(order)
        self.feature_order_idx_ = [col_to_idx[c] for c in order]
        self.parents_ = {c: list(parents[c]) for c in order}
        self.context_idx_ = {col_to_idx[c]: [col_to_idx[p] for p in self.parents_[c]] for c in order}
        self.dag_ = dag

    # ------------------------------------------------------------------ DSBM pieces
    def _dsbm_train_tuple(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        t: float,
        fb: FB,
        *,
        generator: Optional[torch.Generator] = None,
        noise: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Scalar bridge-matching tuple at a fixed time t (z0, z1 of shape (n,)), the
        SAME noise in x_t and target:

          x_t = (1-t) z0 + t z1 + sigma sqrt(t(1-t)) noise
          fb == "f": target = (z1 - z0) - sigma sqrt(t/(1-t)) noise   = (z1 - x_t)/(1-t),  t < 1
          fb == "b": target = -(z1 - z0) - sigma sqrt((1-t)/t) noise  = (z0 - x_t)/t,      t > 0
        """
        t = float(t)
        if fb == "f" and not 0.0 <= t < 1.0:
            raise ValueError("the forward target needs t in [0, 1)")
        if fb == "b" and not 0.0 < t <= 1.0:
            raise ValueError("the backward target needs t in (0, 1]")
        sigma = float(self.cfg.sigma)

        if noise is None:
            noise = self._randn(z0.shape, generator)
        epsn = np.asarray(noise, dtype=np.float32)

        xt = (1.0 - t) * z0 + t * z1
        xt = xt + sigma * math.sqrt(t * (1.0 - t)) * epsn

        delta = z1 - z0
        if fb == "f":
            target = delta - sigma * math.sqrt(t / (1.0 - t)) * epsn
        else:
            target = -delta - sigma * math.sqrt((1.0 - t) / t) * epsn

        return xt.astype(np.float32), target.astype(np.float32)

    def _build_feature_step_batch(
        self,
        z0: np.ndarray,
        z1: np.ndarray,
        ctx: np.ndarray,
        t: float,
        fb: FB,
        generator: torch.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = z0.shape[0]
        X_list = []
        y_list = []
        for _ in range(int(self.cfg.n_noise_per_pair)):
            xt, target = self._dsbm_train_tuple(z0, z1, t, fb, generator=generator)
            xt_col = xt.reshape(n, 1)
            X_feat = np.concatenate([xt_col, ctx], axis=1) if ctx.shape[1] else xt_col
            X_list.append(X_feat.astype(np.float32))
            y_list.append(target)
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    def _generate_coupling(
        self,
        x_data: np.ndarray,
        x_prior: np.ndarray,
        prev_fb: Optional[FB],
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Coupling (z0, z1) for the next IMF stage and its source. Later stages
        simulate the latest OPPOSITE-direction model, always with noise, anchored at
        the real rows it starts from (data after "f", prior after "b").
        """
        gen = self._make_generator(seed)

        if prev_fb is None:
            z0 = x_data.copy()
            if self.cfg.first_coupling == "ind":
                perm = torch.randperm(len(x_prior), generator=gen).numpy()
                return z0, x_prior[perm].copy(), "independent"
            if self.cfg.first_coupling == "ref":
                return z0, z0 + float(self.cfg.sigma) * self._randn(z0.shape, gen), "reference"
            raise ValueError(f"Unknown first_coupling={self.cfg.first_coupling!r}")

        if prev_fb == "f":
            zstart = x_data.copy()
            zend = self._sample_with_direction(zstart, "f", generator=gen, noise=True)
            return zstart, zend, "forward_model"

        zstart = x_prior.copy()
        zend = self._sample_with_direction(zstart, "b", generator=gen, noise=True)
        return zend, zstart, "backward_model"

    def _train_direction(self, fb: FB, z0: np.ndarray, z1: np.ndarray, seed: int) -> int:
        """
        Fit fresh per-column, per-step models for ``fb`` on the coupling. The parent
        context comes from the endpoint the direction GENERATES (z0 for "b", z1 for
        "f"), which is what the sampler feeds. Returns the number of models fitted.
        """
        if self.feature_order_idx_ is None:
            raise RuntimeError("Solver is not initialized for training.")

        t_grid = self._t_grid(fb)
        gen = self._make_generator(seed)
        ctx_endpoint = z0 if fb == "b" else z1
        new_fields: Dict[int, CatBoostTimeDiscretizedScalar] = {}
        n_models = 0

        for j in self.feature_order_idx_:
            ctx_idx = self.context_idx_[j]
            z0_j = z0[:, j].astype(np.float32)
            z1_j = z1[:, j].astype(np.float32)
            ctx = ctx_endpoint[:, ctx_idx].astype(np.float32)   # (n, n_parents), explicit parent order

            field_j = CatBoostTimeDiscretizedScalar(t_grid=t_grid, cfg=self.cfg.catboost)
            for k, t in enumerate(t_grid):
                X_feat, y = self._build_feature_step_batch(z0_j, z1_j, ctx, float(t), fb, gen)
                field_j.fit_step(k, X_feat, y)
                n_models += 1
            new_fields[j] = field_j

        if fb == "f":
            self.fields_f_ = new_fields
        else:
            self.fields_b_ = new_fields
        return n_models

    def _sample_with_direction(
        self,
        zstart: np.ndarray,
        direction: FB,
        generator: Optional[torch.Generator] = None,
        noise: bool = True,
    ) -> np.ndarray:
        """
        Columns are generated in topological order; each column's parent context is
        read from the SAME generated row. Per column, Euler-Maruyama over the N
        edges with dt = 1/N: forward visits k = 0..N-1 with the state at k/N,
        backward visits k = N-1..0 with the state at (k+1)/N -- exactly the time
        model k was trained at.

        ``noise=False`` is the noiseless heuristic (see class docstring).
        """
        if self.columns_ is None or self.feature_order_idx_ is None:
            raise RuntimeError("Call fit() before sample().")

        fields = self.fields_f_ if direction == "f" else self.fields_b_
        if len(fields) == 0:
            raise RuntimeError(f"Direction '{direction}' has not been trained yet.")

        n = zstart.shape[0]
        X_out = np.zeros((n, len(self.columns_)), dtype=np.float32)

        N = int(self.cfg.num_steps)
        dt = 1.0 / float(N)
        sqrt_dt = math.sqrt(dt)
        sigma = float(self.cfg.sigma)
        if generator is None:
            generator = self._make_generator(None)

        step_indices = range(N) if direction == "f" else range(N - 1, -1, -1)

        for j in self.feature_order_idx_:
            field_j = fields.get(j, None)
            if field_j is None:
                raise RuntimeError(f"No trained field for feature index {j} in direction '{direction}'.")

            x = np.array(zstart[:, j], dtype=np.float32, copy=True)
            ctx = X_out[:, self.context_idx_[j]]   # parents are already generated (topological order)

            for k in step_indices:
                x_col = x.reshape(-1, 1)
                X_feat = np.concatenate([x_col, ctx], axis=1) if ctx.shape[1] else x_col

                drift = np.asarray(field_j.predict_step(k, X_feat), dtype=np.float32).reshape(-1)
                x = x + drift * dt

                if noise and sigma != 0.0:
                    x = x + sigma * sqrt_dt * self._randn(x.shape, generator)

            X_out[:, j] = x

        return X_out

    # ------------------------------------------------------------------ fit / sample
    def fit(self, train_df: pd.DataFrame) -> "FeaturewiseDSBMBoostSolver":
        """
        Fresh IMF fit. The structure (order, parents, learned DAG) is resolved from
        ``train_df`` only, and the training rows are not retained on the solver.
        """
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("fit expects a pandas DataFrame.")
        df = self._ensure_float_df(train_df)
        if len(df) == 0:
            raise ValueError("cannot fit on an empty training set")
        x_data = df.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(x_data).all():
            raise ValueError("fit expects finite numeric data (found NaN/inf after numeric coercion)")

        self._fitted = False
        self.fields_f_ = {}
        self.fields_b_ = {}
        self.stage_log = []
        self._setup(df)

        x_prior = self._sample_reference(len(x_data), self._make_generator(int(self.cfg.seed) + 999))

        prev_fb: Optional[FB] = None
        for idx, fb in enumerate(self.cfg.fb_sequence):
            coupling_seed = int(self.cfg.seed) + 10_000 + idx
            train_seed = int(self.cfg.seed) + 20_000 + idx

            z0, z1, source = self._generate_coupling(x_data, x_prior, prev_fb, coupling_seed)
            n_models = self._train_direction(fb, z0, z1, train_seed)

            self.stage_log.append({
                "stage": idx,
                "direction": fb,
                "coupling_source": source,
                # which endpoint of the coupling consists of real (non-simulated) rows
                "anchored_endpoint": {"independent": "both", "reference": "data",
                                      "forward_model": "data", "backward_model": "prior"}[source],
                "n_models_fitted": int(n_models),
                "n_train_rows": int(len(z0) * int(self.cfg.n_noise_per_pair)),
                "coupling_seed": int(coupling_seed),
                "train_seed": int(train_seed),
            })
            prev_fb = fb

        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Start from the Gaussian prior and run the latest backward chains. ONE
        generator drives the start sample and all path noise: the same seed
        reproduces the output exactly, seed=None uses fresh entropy.
        """
        if not self._fitted or len(self.fields_b_) == 0:
            raise RuntimeError("Call fit() before sample(); backward ('b') feature fields must be trained.")
        if int(n) <= 0:
            raise ValueError("n must be positive")

        gen = self._make_generator(seed)
        zstart = self._sample_reference(int(n), gen)
        return self._sample_with_direction(zstart, "b", generator=gen, noise=bool(self.cfg.noise))

    def sample_df(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        return pd.DataFrame(self.sample(n=n, seed=seed), columns=self.columns_)

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        """Inference-complete state; holds the fitted models, never training rows."""
        return {
            "format": CHECKPOINT_FORMAT,
            "canonical_id": self.canonical_id,
            "variant_id": self.variant_id,
            "config": asdict(self.cfg),
            "dim": self.dim_,
            "columns": None if self.columns_ is None else list(self.columns_),
            "orientation": {"x0": "data", "x1": "prior", "generation": "backward"},
            "sigma": float(self.cfg.sigma),
            "num_steps": int(self.cfg.num_steps),
            "t_grid_f": [float(t) for t in self.t_grid_f_],
            "t_grid_b": [float(t) for t in self.t_grid_b_],
            "structure": {
                "kind": self.cfg.structure,
                "order": None if self.feature_order_ is None else list(self.feature_order_),
                "parents": {c: list(ps) for c, ps in self.parents_.items()},
                "dag": None if self.dag_ is None else self.dag_.state(),
            },
            "models": {
                "f": {int(j): list(fld.models) for j, fld in self.fields_f_.items()},
                "b": {int(j): list(fld.models) for j, fld in self.fields_b_.items()},
            },
            "stage_log": self.stage_log,
            "fitted": bool(self._fitted),
        }

    def save_checkpoint(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.state_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_checkpoint(cls, path) -> "FeaturewiseDSBMBoostSolver":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
            fmt = state.get("format") if isinstance(state, dict) else type(state)
            raise ValueError(f"unsupported {CANONICAL_ID} checkpoint format: {fmt!r}")
        cfg_dict = dict(state["config"])
        cfg_dict["fb_sequence"] = tuple(cfg_dict["fb_sequence"])
        cfg_dict["catboost"] = CatBoostScalarConfig(**cfg_dict["catboost"])
        solver = cls(cfg=FeaturewiseDSBMBoostConfig(**cfg_dict))
        solver.t_grid_f_ = np.asarray(state["t_grid_f"], dtype=np.float64)
        solver.t_grid_b_ = np.asarray(state["t_grid_b"], dtype=np.float64)

        if state["columns"] is not None:
            solver.columns_ = list(state["columns"])
            solver.dim_ = int(state["dim"])
            solver.reference = GaussianReference(dim=solver.dim_, device=torch.device("cpu"))
            struct = state["structure"]
            solver._validate_structure(struct["order"], struct["parents"])
            dag = None if struct["dag"] is None else LearnedDAG.from_state(struct["dag"])
            solver._set_structure(struct["order"], struct["parents"], dag)

        for fb in ("f", "b"):
            fields: Dict[int, CatBoostTimeDiscretizedScalar] = {}
            for j, models in state["models"][fb].items():
                wrapper = CatBoostTimeDiscretizedScalar(t_grid=solver._t_grid(fb), cfg=solver.cfg.catboost)
                wrapper.models = list(models)
                fields[int(j)] = wrapper
            if fb == "f":
                solver.fields_f_ = fields
            else:
                solver.fields_b_ = fields
        solver.stage_log = state["stage_log"]
        solver._fitted = bool(state["fitted"])
        return solver
