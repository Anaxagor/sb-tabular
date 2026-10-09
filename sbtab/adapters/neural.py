"""Adapters for the neural continuous-state solvers: MLP IPF-DSB (CT / DT), MLP IMF-DSBM (CT) and LightSB."""
from __future__ import annotations

from typing import Any, ClassVar, Dict

from sbtab.adapters.continuous import ContinuousSolverAdapter
from sbtab.adapters.dsbm import _DSBMAdapter


# --------------------------------------------------------------------------- IPF-DSB (MLP)
class _DSBMLPAdapter(ContinuousSolverAdapter):
    """
    ``horizon`` rescales the gamma grid so that T = sum(gamma) equals it. The legacy
    grid (gamma in [1e-4, 1e-2]) gives T ~ 0.046 for 20 steps, far too short for an
    OU reference started at the data to approach the N(0, I) prior. Solvers and
    adapters therefore default to T = 2. T stays a property of the declared
    reference — training and sampling read it from the same grid.
    ``steps_per_phase`` is the declared training budget (exact optimiser updates per half-iteration).
    """
    DEFAULTS: ClassVar[Dict[str, Any]] = dict(
        ipf_iters=6, num_steps=20, gamma_min=1e-4, gamma_max=1e-2, schedule="geom", horizon=2.0, alpha_ou=1.0,
        sigma=2 ** 0.5, batch_size=512, cache_batches=200, steps_per_phase=2000, lr=2e-4, weight_decay=0.0,
        grad_clip=1.0, hidden_units=256, n_layers=4, dropout=0.0, device="cpu")
    EXTRA: ClassVar[Dict[str, Any]] = {}

    def _config_object(self):
        raise NotImplementedError

    def _build(self, dim, columns):
        c = {k: v for k, v in self.config.items()}
        cfg_cls = self._config_object()
        cfg = cfg_cls(noise=True, seed=int(self.seed), **c)     # canonical entries always sample with dynamics noise
        return self._solver_class()(dim, cfg)

    def _describe_solver(self) -> dict:
        d = self.solver.describe()
        return {"reference": d["reference"], "time_parameterization": d["time_parameterization"],
                "network_output": d["network_output"], "clock": d["clock"],
                "grid": {"schedule": self.solver.cfg.schedule, "num_steps": self.solver.cfg.num_steps,
                         "horizon": d["reference"].get("horizon_T")},
                "target": "mean-matching displacement (units of x); the sampler adds it without a second gamma factor"}


class DSBContinuousJointMLPAdapter(_DSBMLPAdapter):
    registry_id = "dsb_ct_joint_mlp"
    DEFAULTS = {**_DSBMLPAdapter.DEFAULTS, "time_features": 64, "time_scale": 1000.0}

    def _solver_class(self):
        from sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver import IPFDSBSolver
        return IPFDSBSolver

    def _config_object(self):
        from sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver import IPFDSBConfig
        return IPFDSBConfig


class DSBDiscreteJointMLPAdapter(_DSBMLPAdapter):
    """Genuinely per-step: one MLP per edge and direction, no time input (so no time_features / time_scale)."""
    registry_id = "dsb_dt_joint_mlp"
    DEFAULTS = {**_DSBMLPAdapter.DEFAULTS, "hidden_units": 128, "n_layers": 3}

    def _solver_class(self):
        from sbtab.solvers.discrete_time.joint_distribution.mlp.ipf_dsb.solver import IPFDSBSolver
        return IPFDSBSolver

    def _config_object(self):
        from sbtab.solvers.discrete_time.joint_distribution.mlp.ipf_dsb.solver import IPFDSBConfig
        return IPFDSBConfig


# --------------------------------------------------------------------------- IMF-DSBM (MLP, CT)
class DSBMContinuousJointMLPAdapter(_DSBMAdapter):
    registry_id = "dsbm_ct_joint_mlp"
    DEFAULTS = {"n_stages": 5, "num_steps": 100, "sigma": 0.1, "first_coupling": "ind", "eps": 1e-3, "inner_iters": 2000,
                "batch_size": 256, "lr": 1e-4, "weight_decay": 0.0, "grad_clip": 1.0, "hidden_dim": 256, "n_layers": 4,
                "dropout": 0.0, "time_emb_dim": 64, "device": "cpu"}

    def _solver_class(self):
        from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import IMFDSBMSolver
        return IMFDSBMSolver

    def _build(self, dim, columns):
        from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import IMFDSBMConfig
        c = self.config
        cfg = IMFDSBMConfig(eps=float(c["eps"]), inner_iters=int(c["inner_iters"]), batch_size=int(c["batch_size"]),
                            lr=float(c["lr"]), weight_decay=float(c["weight_decay"]),
                            grad_clip=None if c["grad_clip"] is None else float(c["grad_clip"]),
                            hidden_dim=int(c["hidden_dim"]), n_layers=int(c["n_layers"]), dropout=float(c["dropout"]),
                            time_emb_dim=int(c["time_emb_dim"]), device=str(c["device"]), **self._imf_kwargs())
        return self._solver_class()(dim, cfg)


# --------------------------------------------------------------------------- LightSB
class LightSBAdapter(ContinuousSolverAdapter):
    """
    LightSB (Korotin et al.), diagonal covariance, exact conditional sampler by
    default. NOT LightSB-M. Source x0 ~ N(0, I), target = data; ``max_iter`` is the
    declared training budget.
    """
    registry_id = "lightsb"
    DEFAULTS = dict(n_potentials=50, epsilon=0.1, S_diagonal_init=0.1, lr=1e-2, weight_decay=0.0, batch_size=256,
                    max_iter=10000, grad_clip=None, init_r_from_data=True, use_sde_sampling=False, n_euler_steps=100,
                    sampling_batch_size=4096, device="cpu")

    def _solver_class(self):
        from sbtab.solvers.light_sb.solver import LightSBSolver
        return LightSBSolver

    def _build(self, dim, columns):
        from sbtab.models.sb.light_sb import LightSBPotentialConfig
        from sbtab.solvers.light_sb.config import LightSBConfig
        c = self.config
        pot = LightSBPotentialConfig(n_potentials=int(c["n_potentials"]), epsilon=float(c["epsilon"]), is_diagonal=True,
                                     sampling_batch_size=int(c["sampling_batch_size"]),
                                     S_diagonal_init=float(c["S_diagonal_init"]))
        cfg = LightSBConfig(potential=pot, lr=float(c["lr"]), weight_decay=float(c["weight_decay"]),
                            batch_size=int(c["batch_size"]), max_iter=int(c["max_iter"]),
                            grad_clip=None if c["grad_clip"] is None else float(c["grad_clip"]),
                            init_r_from_data=bool(c["init_r_from_data"]), use_sde_sampling=bool(c["use_sde_sampling"]),
                            n_euler_steps=int(c["n_euler_steps"]), device=str(c["device"]), seed=int(self.seed),
                            verbose_every=10 ** 9)
        return self._solver_class()(dim, cfg)

    def _describe_solver(self) -> dict:
        c = self.config
        return {"algorithm": "LightSB (not LightSB-M)", "orientation": {"x0": "N(0, I) source", "x1": "data", "generation": "x0 -> pi(.|x0)"},
                "reference": {"kind": "brownian", "epsilon": c["epsilon"], "horizon": 1.0},
                "potential": {"n_potentials": c["n_potentials"], "covariance": "diagonal"},
                "sampler": "unit-horizon Euler-Maruyama SDE" if c["use_sde_sampling"] else "exact conditional mixture",
                "objective": "E_x0[log c(x0)] - E_x1[log v(x1)]"}
