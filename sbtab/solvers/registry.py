"""
solver_registry — stable IDs for every solver and baseline the benchmark can name.

An entry states what the implementation IS (validated against the code, not
against a label): algorithm family, state space, time parameterisation,
dependency structure, backend, native vs adapted regimes and checkpoint support.
``status``:
  supported     implemented, has an adapter, covered by tests
  heuristic     a declared non-canonical variant (kept under its own id)
  unavailable   not executable through the benchmark (absent implementation or
                missing adapter, or explicitly excluded). Never silently substituted.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

S = "sbtab/solvers/"


@dataclass(frozen=True)
class RegistryEntry:
    id: str
    status: str
    family: str
    implementation: Optional[str]
    adapter: Optional[str]                      # "module:Class"
    state_space: str = ""
    time_parameterization: str = ""
    dependency_structure: str = ""
    backend: str = ""
    native_regimes: Tuple[str, ...] = ()
    adapted_regimes: Tuple[str, ...] = ()
    checkpoint: str = "inference"
    requires: Tuple[str, ...] = ()              # optional third-party packages
    notes: str = ""

    @property
    def regimes(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(self.native_regimes + self.adapted_regimes))


_ADAPTED = ("discrete", "mixed")
_CONT = "R^d (continuous columns standardised; nominal one-hot, discrete standardised via the declared adapter)"

_ENTRIES = [
    # ---------------------------------------------------------------- IPF-DSB
    RegistryEntry("dsb_ct_joint_mlp", "supported", "IPF-DSB", S + "continuous_time/joint_distribution/mlp/ipf_dsb/",
                  "sbtab.adapters.neural:DSBContinuousJointMLPAdapter", _CONT, "time-conditioned networks on a gamma grid",
                  "joint", "torch MLP", ("continuous",), _ADAPTED),
    RegistryEntry("dsb_dt_joint_mlp", "supported", "IPF-DSB", S + "discrete_time/joint_distribution/mlp/ipf_dsb/",
                  "sbtab.adapters.neural:DSBDiscreteJointMLPAdapter", _CONT, "see adapter.describe(): per-step vs time-conditioned is reported by the solver",
                  "joint", "torch MLP", ("continuous",), _ADAPTED),
    RegistryEntry("dsb_ct_joint_gbt", "supported", "IPF-DSB", S + "continuous_time/joint_distribution/boosting/ipf_dsb/",
                  "sbtab.adapters.continuous:DSBContinuousJointGBTAdapter", _CONT,
                  "one time-conditioned regressor per direction, evaluated on the gamma grid", "joint",
                  "CatBoost MultiRMSE", ("continuous",), _ADAPTED),
    RegistryEntry("dsb_dt_joint_gbt", "supported", "IPF-DSB", S + "discrete_time/joint_distribution/boosting/ipf_dsb/",
                  "sbtab.adapters.continuous:DSBDiscreteJointGBTAdapter", _CONT, "one model per edge k and direction",
                  "joint", "CatBoost MultiRMSE", ("continuous",), _ADAPTED),
    RegistryEntry("dsb_ct_structural_gbt", "supported", "IPF-DSB", S + "continuous_time/feature_wise/boosting/ipf_dsb/",
                  "sbtab.adapters.continuous:DSBContinuousStructuralGBTAdapter", _CONT,
                  "one time-conditioned scalar regressor per column and direction",
                  "DAG learned on the fit rows; per-column bridge conditioned on parents", "CatBoost RMSE + pgmpy",
                  ("continuous",), _ADAPTED, requires=("pgmpy", "networkx")),
    RegistryEntry("dsb_dt_structural_gbt", "supported", "IPF-DSB", S + "discrete_time/feature_wise/boosting/ipf_dsb/",
                  "sbtab.adapters.continuous:DSBDiscreteStructuralGBTAdapter", _CONT,
                  "one scalar model per column, edge k and direction",
                  "DAG learned on the fit rows; per-column bridge conditioned on parents", "CatBoost RMSE + pgmpy",
                  ("continuous",), _ADAPTED, requires=("pgmpy", "networkx")),
    # ---------------------------------------------------------------- IMF-DSBM
    RegistryEntry("dsbm_ct_joint_mlp", "supported", "IMF-DSBM", S + "continuous_time/joint_distribution/mlp/imf_dsbm/",
                  "sbtab.adapters.neural:DSBMContinuousJointMLPAdapter", _CONT, "time-conditioned drift, t ~ U[eps, 1-eps], unit horizon",
                  "joint", "torch MLP", ("continuous",), _ADAPTED),
    RegistryEntry("dsbm_ct_joint_gbt", "supported", "IMF-DSBM", S + "continuous_time/joint_distribution/boosting/imf_dsbm/",
                  "sbtab.adapters.dsbm:DSBMContinuousJointGBTAdapter", _CONT, "one time-conditioned regressor per direction, unit horizon",
                  "joint", "CatBoost MultiRMSE", ("continuous",), _ADAPTED),
    RegistryEntry("dsbm_dt_joint_mlp", "supported", "IMF-DSBM", S + "discrete_time/joint_distribution/mlp/imf_dsbm/",
                  "sbtab.adapters.dsbm:DSBMDiscreteJointMLPAdapter", _CONT, "one MLP per edge and direction at the state time",
                  "joint", "torch MLP", ("continuous",), _ADAPTED),
    RegistryEntry("dsbm_dt_joint_gbt", "supported", "IMF-DSBM", S + "discrete_time/joint_distribution/boosting/imf_dsbm_boost/",
                  "sbtab.adapters.dsbm:DSBMDiscreteJointGBTAdapter", _CONT, "one model per edge and direction at the state time",
                  "joint", "CatBoost MultiRMSE", ("continuous",), _ADAPTED),
    RegistryEntry("dsbm_dt_structural_gbt", "supported", "IMF-DSBM",
                  S + "discrete_time/feature_wise/boosting/imf_dsbm_featurewise_boost/",
                  "sbtab.adapters.dsbm:DSBMDiscreteStructuralGBTAdapter", _CONT,
                  "one scalar model per column, edge and direction at the state time",
                  "autoregressive chain by default (exact factorisation); optional map / learned DAG",
                  "CatBoost RMSE", ("continuous",), _ADAPTED, requires=("pgmpy", "networkx"),
                  notes="The benchmark profile searches learned DAGs and requires the graph libraries before tuning. "
                        "The standalone autoregressive solver does not require pgmpy."),
    # ---------------------------------------------------------------- others
    RegistryEntry("lightsb", "supported", "LightSB", S + "light_sb/", "sbtab.adapters.neural:LightSBAdapter", _CONT,
                  "static potential; exact conditional sampler (optional unit-horizon SDE)", "joint",
                  "torch Gaussian-mixture potential", ("continuous",), _ADAPTED,
                  notes="Diagonal covariance. Full covariance needs the optional package geotorch. This is LightSB, NOT LightSB-M."),
    RegistryEntry("csbm", "supported", "CSBM / D-IMF", S + "csbm/", "sbtab.adapters.native:CSBMAdapter",
                  "finite states per column", "time-conditioned networks on a uniform unit-horizon grid",
                  "joint input, endpoint head factorised over columns", "torch MLP", ("discrete",), ()),
    RegistryEntry("csbm_annealed", "heuristic", "CSBM / D-IMF", S + "csbm/", "sbtab.adapters.native:AnnealedCSBMAdapter",
                  "finite states per column", "as csbm", "as csbm", "torch MLP", ("discrete",), (),
                  notes="Reference mixing rate is annealed between outer iterations; not the canonical fixed-reference IMF."),
    RegistryEntry("mixedsbm", "supported", "MixedSBM", S + "msbm/", "sbtab.adapters.native:MixedSBMAdapter",
                  "R^d x finite states", "time-conditioned networks on one uniform unit-horizon grid",
                  "joint input; drift head + factorised logit head", "torch MLP", ("continuous", "discrete", "mixed"), ()),
    # ---------------------------------------------------------------- baselines
    RegistryEntry("ctgan", "supported", "GAN baseline", "sbtab/baselines/ctgan/", "sbtab.adapters.baselines:CTGANAdapter",
                  "mixed (SDV metadata from the explicit schema)", "n/a", "joint", "sdv CTGANSynthesizer",
                  ("continuous", "discrete", "mixed"), (), requires=("sdv",)),
    RegistryEntry("tabddpm", "supported", "diffusion baseline", "sbtab/baselines/tabddpm/",
                  "sbtab.adapters.baselines:TabDDPMAdapter", "Gaussian + multinomial diffusion", "discrete diffusion timesteps",
                  "joint (row incl. target, unconditional)", "torch", ("continuous", "discrete", "mixed"), ()),
    RegistryEntry("ve_score_sde_simplified", "supported", "score-SDE baseline", "sbtab/baselines/stasy/",
                  "sbtab.adapters.baselines:VEScoreSDEAdapter", _CONT, "VE SDE, predictor-corrector sampler", "joint", "torch MLP",
                  ("continuous",), _ADAPTED,
                  notes="A simplified VE score-SDE model. It is NOT a faithful STaSy: no per-sample self-paced weights, "
                        "no fine-tuning stage, no VP/sub-VP option, no probability-flow ODE sampler, no ncsnpp-tabular network."),
    RegistryEntry("forestdiffusion", "supported", "tree flow/diffusion baseline",
                  "sbtab/baselines/forest_diffusion/", "sbtab.adapters.baselines:ForestDiffusionAdapter",
                  _CONT, "one XGBoost field per time level", "joint row incl. target, unconditional",
                  "XGBoost histogram trees", ("continuous",), _ADAPTED, requires=("xgboost",),
                  notes="Forest-Flow default; optional Forest-VP. Core imported from forest_diffusion/50635ca and audited. "
                        "Full one-hot and z-score wrapper; no continuous clipping."),
    RegistryEntry("tabbyflow", "supported", "flow baseline", "sbtab/baselines/tabbyflow/",
                  "sbtab.adapters.tabbyflow:TabbyFlowAdapter", "Gaussian numeric states + categorical endpoint heads",
                  "time-conditioned vector field", "joint row incl. target, unconditional", "torch MLP",
                  ("continuous", "discrete", "mixed"), (), requires=("torch", "sklearn"),
                  notes="TabVFM-MLP-style OT flow; train-fitted numeric quantiles and nominal one-hot encoding. "
                        "Ordered discrete values use numeric quantiles and training-support projection. "
                        "Gaussian source; OT endpoint retains residual noise 0.001."),
    # ---------------------------------------------------------------- named elsewhere, not implemented
    RegistryEntry("tabpfgen", "unavailable", "pretrained-prior baseline", None, None,
                  notes="Excluded from tuning and experiment pipelines. The historical standalone wrapper remains "
                        "in sbtab.baselines.tabpfn; benchmark runs require no TabPFN packages or pretrained weights."),
    RegistryEntry("stasy", "unavailable", "score-SDE baseline", None, None,
                  notes="No faithful STaSy implementation exists in this repository; see ve_score_sde_simplified. "
                        "Missing work: per-sample SPL weights with alpha0/beta0 thresholds, fine-tuning stage, VP/sub-VP SDEs, "
                        "probability-flow ODE sampler, ncsnpp-tabular architecture."),
    RegistryEntry("lightsb_m", "unavailable", "LightSB-M", None, None,
                  notes="Not implemented. A former class alias `LightSBM` pointed at the LightSB potential and "
                        "tuning_results/best_params/lightsbm_best_params.json holds LightSB-style parameters. "
                        "Missing work: the bridge-matching objective of Gushchin et al. with a differentiable drift."),
    RegistryEntry("tabsyn", "unavailable", "latent diffusion baseline", None, None,
                  notes="Not implemented anywhere in the tracked tree. Missing work: the whole wrapper (VAE + latent diffusion)."),
]

solver_registry: Dict[str, RegistryEntry] = {e.id: e for e in _ENTRIES}


class UnavailableModelError(LookupError):
    pass


def get_entry(model_id: str) -> RegistryEntry:
    try:
        return solver_registry[model_id]
    except KeyError as e:
        raise UnavailableModelError(f"unknown model id {model_id!r}; registered: {sorted(solver_registry)}") from e


def get_adapter_class(model_id: str):
    entry = get_entry(model_id)
    if entry.status == "unavailable" or entry.adapter is None:
        raise UnavailableModelError(f"{model_id!r} is registered as unavailable: {entry.notes}")
    module, cls = entry.adapter.split(":")
    return getattr(importlib.import_module(module), cls)


def missing_requirements(model_id: str) -> Tuple[str, ...]:
    out = []
    for pkg in get_entry(model_id).requires:
        try:
            importlib.import_module(pkg)
        except Exception:
            out.append(pkg)
    return tuple(out)
