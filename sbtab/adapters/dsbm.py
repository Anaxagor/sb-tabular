"""
Adapters for the IMF-DSBM solvers (unit-time Brownian reference, x0 = data,
x1 = N(0, I); generation runs the last backward stage).

Canonical entries always sample WITH dynamics noise: the drift is trained for the
stochastic bridge, and dropping the noise is not a marginal-preserving sampler.
``noise`` is therefore not an adapter option. The declared initial coupling is
the independent coupling data (x) prior ("ind"); "ref" selects the IPF-like variant.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

from sbtab.adapters.continuous import ContinuousSolverAdapter

_IMF = dict(n_stages=5, num_steps=32, sigma=0.1, first_coupling="ind")
_CATBOOST = dict(cb_iterations=1000, cb_depth=6, cb_learning_rate=0.05, cb_l2_leaf_reg=3.0, cb_thread_count=4)


def _fb_sequence(n_stages: int):
    n = int(n_stages)
    if n < 1:
        raise ValueError("n_stages must be >= 1")
    # alternate and END on the backward (generation) direction
    return tuple("b" if (n - 1 - i) % 2 == 0 else "f" for i in range(n))


class _DSBMAdapter(ContinuousSolverAdapter):
    DEFAULTS: ClassVar[Dict[str, Any]] = dict(_IMF)

    def _imf_kwargs(self) -> dict:
        c = self.config
        return dict(fb_sequence=_fb_sequence(c["n_stages"]), num_steps=int(c["num_steps"]), sigma=float(c["sigma"]),
                    first_coupling=str(c["first_coupling"]), noise=True, seed=int(self.seed))

    def _describe_solver(self) -> dict:
        s = self.solver
        return {"reference": {"kind": "brownian", "sigma": s.cfg.sigma, "horizon": 1.0},
                "grid": {"schedule": "uniform", "num_steps": s.cfg.num_steps, "horizon": 1.0},
                "initial_coupling": s.cfg.first_coupling,
                "target": "bridge-matching drift (units of x per unit time)",
                "generation": "last backward stage, stochastic Euler-Maruyama"}

    def _catboost_kwargs(self) -> dict:
        c = self.config
        return dict(iterations=int(c["cb_iterations"]), depth=int(c["cb_depth"]),
                    learning_rate=float(c["cb_learning_rate"]), l2_leaf_reg=float(c["cb_l2_leaf_reg"]),
                    thread_count=int(c["cb_thread_count"]), random_seed=int(self.seed))


class DSBMContinuousJointGBTAdapter(_DSBMAdapter):
    registry_id = "dsbm_ct_joint_gbt"
    DEFAULTS = {**_IMF, "num_steps": 100, "eps": 1e-3, "n_noise_per_pair": 4, **_CATBOOST}

    def _solver_class(self):
        from sbtab.solvers.continuous_time.joint_distribution.boosting.imf_dsbm.solver import (
            IMFDSBMContinuousJointCatBoostSolver)
        return IMFDSBMContinuousJointCatBoostSolver

    def _build(self, dim, columns):
        from sbtab.models.boosted.catboost_continuous_joint import CatBoostContinuousJointConfig
        from sbtab.solvers.continuous_time.joint_distribution.boosting.imf_dsbm.solver import (
            IMFDSBMContinuousJointCatBoostConfig)
        cfg = IMFDSBMContinuousJointCatBoostConfig(
            eps=float(self.config["eps"]), n_noise_per_pair=int(self.config["n_noise_per_pair"]),
            field=CatBoostContinuousJointConfig(**self._catboost_kwargs()), **self._imf_kwargs())
        return self._solver_class()(dim, cfg)


class DSBMDiscreteJointGBTAdapter(_DSBMAdapter):
    registry_id = "dsbm_dt_joint_gbt"
    DEFAULTS = {**_IMF, "n_noise_per_pair": 1, **_CATBOOST}

    def _solver_class(self):
        from sbtab.solvers.discrete_time.joint_distribution.boosting.imf_dsbm_boost.solver import IMFDSBMBoostSolver
        return IMFDSBMBoostSolver

    def _build(self, dim, columns):
        from sbtab.models.boosted.catboost_discrete_joint import CatBoostDiscreteJointConfig
        from sbtab.solvers.discrete_time.joint_distribution.boosting.imf_dsbm_boost.solver import IMFDSBMBoostConfig
        cfg = IMFDSBMBoostConfig(n_noise_per_pair=int(self.config["n_noise_per_pair"]),
                                 catboost=CatBoostDiscreteJointConfig(**self._catboost_kwargs()), **self._imf_kwargs())
        return self._solver_class()(dim, cfg)


class DSBMDiscreteJointMLPAdapter(_DSBMAdapter):
    registry_id = "dsbm_dt_joint_mlp"
    DEFAULTS = {**_IMF, "n_noise_per_pair": 1, "hidden_dim": 256, "n_layers": 4, "dropout": 0.0, "lr": 2e-4,
                "weight_decay": 0.0, "batch_size": 256, "n_epochs": 20, "grad_clip": 1.0, "device": "cpu"}

    def _solver_class(self):
        from sbtab.solvers.discrete_time.joint_distribution.mlp.imf_dsbm.solver import IMFDSBMDiscreteJointMLPSolver
        return IMFDSBMDiscreteJointMLPSolver

    def _build(self, dim, columns):
        from sbtab.models.neural.mlp_discrete_joint import StepMLPJointConfig
        from sbtab.solvers.discrete_time.joint_distribution.mlp.imf_dsbm.solver import IMFDSBMDiscreteJointMLPConfig
        c = self.config
        field = StepMLPJointConfig(hidden_dim=int(c["hidden_dim"]), n_layers=int(c["n_layers"]), dropout=float(c["dropout"]),
                                   lr=float(c["lr"]), weight_decay=float(c["weight_decay"]), batch_size=int(c["batch_size"]),
                                   n_epochs=int(c["n_epochs"]), grad_clip=None if c["grad_clip"] is None else float(c["grad_clip"]),
                                   device=str(c["device"]), feature_mode="x")
        cfg = IMFDSBMDiscreteJointMLPConfig(n_noise_per_pair=int(c["n_noise_per_pair"]), field=field, **self._imf_kwargs())
        return self._solver_class()(dim, cfg)


class DSBMDiscreteStructuralGBTAdapter(_DSBMAdapter):
    registry_id = "dsbm_dt_structural_gbt"
    DEFAULTS = {**_IMF, "num_steps": 50, "n_noise_per_pair": 1, "structure": "autoregressive", "structure_n_bins": 5,
                **_CATBOOST}

    def _solver_class(self):
        from sbtab.solvers.discrete_time.feature_wise.boosting.imf_dsbm_featurewise_boost.solver import (
            FeaturewiseDSBMBoostSolver)
        return FeaturewiseDSBMBoostSolver

    def _build(self, dim, columns):
        from sbtab.models.boosted.catboost_discrete_scalar import CatBoostDiscreteScalarConfig
        from sbtab.solvers.discrete_time.feature_wise.boosting.imf_dsbm_featurewise_boost.solver import (
            FeaturewiseDSBMBoostConfig)
        c = self.config
        if c["structure"] not in ("autoregressive", "learned"):
            raise ValueError("adapter structure must be 'autoregressive' or 'learned' (a parent map is dataset-specific)")
        cfg = FeaturewiseDSBMBoostConfig(n_noise_per_pair=int(c["n_noise_per_pair"]), structure=str(c["structure"]),
                                         structure_n_bins=int(c["structure_n_bins"]),
                                         catboost=CatBoostDiscreteScalarConfig(**self._catboost_kwargs()), **self._imf_kwargs())
        return self._solver_class()(cfg)

    def _describe_solver(self) -> dict:
        d = super()._describe_solver()
        s = self.solver
        d["structure"] = {"kind": s.cfg.structure, "learned_on": "current fit rows only" if s.cfg.structure == "learned" else None,
                          "order": list(getattr(s, "feature_order_", []) or []),
                          "parents": {k: list(v) for k, v in (getattr(s, "parents_", {}) or {}).items()}}
        return d
