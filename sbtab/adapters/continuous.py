"""
Adapters for continuous-state generators. They all model one real vector built by
ContinuousRepresentation (nominal columns one-hot, discrete columns standardised)
and decode back to the common schema (argmax / declared nearest-support decoding).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict

import numpy as np
import pandas as pd

from sbtab.adapters.base import ModelAdapter
from sbtab.adapters.representation import ContinuousRepresentation


class ContinuousSolverAdapter(ModelAdapter):
    native_regimes = ("continuous",)
    MODEL_FILE: ClassVar[str] = "model.bin"

    # ---- hooks for subclasses
    def _build(self, dim: int, columns):
        raise NotImplementedError

    def _fit_solver(self, encoded: pd.DataFrame) -> None:
        self.solver.fit(encoded)

    def _sample_solver(self, n: int, seed: int) -> np.ndarray:
        out = self.solver.sample(n, seed=seed)
        return out.to_numpy(dtype=np.float64) if isinstance(out, pd.DataFrame) else np.asarray(out, dtype=np.float64)

    def _solver_class(self):
        raise NotImplementedError

    def _describe_solver(self) -> dict:
        return {}

    # ---- ModelAdapter
    def _prepare(self, train: pd.DataFrame) -> None:
        self.rep = ContinuousRepresentation(self.schema).fit(train)
        self._encoded = self.rep.encode(train)

    def _build_model(self) -> None:
        self.solver = self._build(self.rep.dim, list(self._encoded.columns))

    def _fit_model(self) -> None:
        self._fit_solver(self._encoded)

    def _generate(self, n: int, seed: int) -> np.ndarray:
        return self._sample_solver(n, seed)

    def _decode(self, generated) -> pd.DataFrame:
        df, self.decoding_report_ = self.rep.decode(generated)
        return df

    @property
    def n_updates(self):
        return getattr(self.solver, "n_updates", None)

    def describe(self) -> dict:
        d = {"variant_id": getattr(self.solver, "variant_id", self.registry_id),
             "orientation": {"x0": "data", "x1": "prior N(0, I)", "generation": "backward"},
             "representation": self.rep.kind, "model_dim": self.rep.dim,
             "stages": getattr(self.solver, "stage_log", None)}
        d.update(self._describe_solver())
        return d

    def _state(self) -> dict:
        return {"representation": self.rep.state()}

    def _save_model(self, directory: Path) -> None:
        self.solver.save_checkpoint(directory / self.MODEL_FILE)

    def _load_model(self, directory: Path, state: dict) -> None:
        self.rep = ContinuousRepresentation.from_state(state["representation"], self.schema)
        self.solver = self._solver_class().load_checkpoint(directory / self.MODEL_FILE)


# --------------------------------------------------------------------------- boosted IPF-DSB
_CATBOOST = dict(cb_iterations=2000, cb_depth=8, cb_learning_rate=0.05, cb_l2_leaf_reg=3.0, cb_thread_count=4)
_IPF_GRID = dict(num_steps=20, ipf_iters=5, alpha_ou=1.0, gamma_min=1e-4, gamma_max=1e-2, schedule="geom")


class _BoostedIPFAdapter(ContinuousSolverAdapter):
    DEFAULTS: ClassVar[Dict[str, Any]] = {**_IPF_GRID, **_CATBOOST}

    def _catboost_kwargs(self) -> dict:
        c = self.config
        return dict(iterations=int(c["cb_iterations"]), depth=int(c["cb_depth"]),
                    learning_rate=float(c["cb_learning_rate"]), l2_leaf_reg=float(c["cb_l2_leaf_reg"]),
                    thread_count=int(c["cb_thread_count"]), random_seed=int(self.seed))

    def _grid_kwargs(self) -> dict:
        c = self.config
        if not float(c["gamma_min"]) <= float(c["gamma_max"]):
            raise ValueError("gamma_min must be <= gamma_max")
        return dict(num_steps=int(c["num_steps"]), ipf_iters=int(c["ipf_iters"]), alpha_ou=float(c["alpha_ou"]),
                    gamma_min=float(c["gamma_min"]), gamma_max=float(c["gamma_max"]), schedule=str(c["schedule"]),
                    seed=int(self.seed))

    def _describe_solver(self) -> dict:
        s = self.solver
        return {"reference": {"kind": "ornstein_uhlenbeck", "alpha": s.cfg.alpha_ou, "diffusion": "sqrt(2)",
                              "iteration_0": "analytic one-step mean x (1 - alpha gamma_k)"},
                "grid": {"schedule": s.cfg.schedule, "num_steps": s.cfg.num_steps, "horizon": float(s.timegrid.T),
                         "gamma_min": s.cfg.gamma_min, "gamma_max": s.cfg.gamma_max},
                "target": "next-state mean (units of x), mean-matching; residual tree fit"}


class DSBDiscreteJointGBTAdapter(_BoostedIPFAdapter):
    registry_id = "dsb_dt_joint_gbt"

    def _solver_class(self):
        from sbtab.solvers.discrete_time.joint_distribution.boosting.ipf_dsb.solver import JointDiscreteBoostedSolver
        return JointDiscreteBoostedSolver

    def _build(self, dim, columns):
        from sbtab.models.boosted.catboost_discrete_joint import CatBoostDiscreteJointConfig
        from sbtab.solvers.discrete_time.joint_distribution.boosting.ipf_dsb.solver import JointDiscreteBoostedConfig
        cfg = JointDiscreteBoostedConfig(catboost=CatBoostDiscreteJointConfig(**self._catboost_kwargs()), **self._grid_kwargs())
        return self._solver_class()(dim, cfg)


class DSBContinuousJointGBTAdapter(_BoostedIPFAdapter):
    registry_id = "dsb_ct_joint_gbt"

    def _solver_class(self):
        from sbtab.solvers.continuous_time.joint_distribution.boosting.ipf_dsb.solver import JointContinuousBoostedSolver
        return JointContinuousBoostedSolver

    def _build(self, dim, columns):
        from sbtab.models.boosted.catboost_continuous_joint import CatBoostContinuousJointConfig
        from sbtab.solvers.continuous_time.joint_distribution.boosting.ipf_dsb.solver import JointContinuousBoostedConfig
        cfg = JointContinuousBoostedConfig(catboost=CatBoostContinuousJointConfig(**self._catboost_kwargs()), **self._grid_kwargs())
        return self._solver_class()(dim, cfg)


class _StructuralIPFAdapter(_BoostedIPFAdapter):
    DEFAULTS = {**_BoostedIPFAdapter.DEFAULTS, "n_bins": 5, "parent_noise": 0.01}

    def _structural_kwargs(self) -> dict:
        return dict(n_bins=int(self.config["n_bins"]), parent_noise=float(self.config["parent_noise"]))

    def _describe_solver(self) -> dict:
        d = super()._describe_solver()
        st = self.solver.structure
        d["structure"] = {"learned_on": "current fit rows only", "fit_row_count": st.fit_row_count,
                          "order": st.order, "parents": st.parents}
        return d


class DSBDiscreteStructuralGBTAdapter(_StructuralIPFAdapter):
    registry_id = "dsb_dt_structural_gbt"

    def _solver_class(self):
        from sbtab.solvers.discrete_time.feature_wise.boosting.ipf_dsb.solver import StructuralDiscreteBoostedSolver
        return StructuralDiscreteBoostedSolver

    def _build(self, dim, columns):
        from sbtab.models.boosted.catboost_discrete_scalar import CatBoostDiscreteScalarConfig
        from sbtab.solvers.discrete_time.feature_wise.boosting.ipf_dsb.solver import StructuralDiscreteBoostedConfig
        cfg = StructuralDiscreteBoostedConfig(catboost=CatBoostDiscreteScalarConfig(**self._catboost_kwargs()),
                                              **self._grid_kwargs(), **self._structural_kwargs())
        return self._solver_class()(cfg)


class DSBContinuousStructuralGBTAdapter(_StructuralIPFAdapter):
    registry_id = "dsb_ct_structural_gbt"

    def _solver_class(self):
        from sbtab.solvers.continuous_time.feature_wise.boosting.ipf_dsb.solver import StructuralContinuousBoostedSolver
        return StructuralContinuousBoostedSolver

    def _build(self, dim, columns):
        from sbtab.models.boosted.catboost_continuous_scalar import CatBoostContinuousScalarConfig
        from sbtab.solvers.continuous_time.feature_wise.boosting.ipf_dsb.solver import StructuralContinuousBoostedConfig
        cfg = StructuralContinuousBoostedConfig(catboost=CatBoostContinuousScalarConfig(**self._catboost_kwargs()),
                                                **self._grid_kwargs(), **self._structural_kwargs())
        return self._solver_class()(cfg)
