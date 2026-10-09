"""Joint-row Forest-Flow / Forest-VP over a train-fitted continuous encoding.

The tree solver comes from branch forest_diffusion (50635ca). Numerical columns
are standardized, nominal columns use full one-hot blocks, discrete outputs use
nearest training support. All columns, including a target, are generated jointly;
this is the unconditional variant, not the paper's label-conditioned experiment.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

import pandas as pd

from sbtab.baselines.base import (BaselineFitInfo, BaselineGenerativeModel, ColumnRoles,
                                 FreshIdFactory, resolve_column_roles, validate_n)
from sbtab.baselines.encoding import MixedToContinuousCodec

CHECKPOINT_FORMAT = "sbtab.forestdiffusion/1"


@dataclass
class ForestDiffusionConfig:
    diffusion_type: str = "flow"
    n_t: int = 50
    n_estimators: int = 100
    max_depth: int = 7
    eta: float = 0.3
    duplicate_K: int = 100
    n_batch: int = 8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    subsample: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 8.0
    eps: float = 1e-3
    n_jobs: int = 1
    n_threads: int = 4
    sample_batch_size: int = 2048
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        if self.diffusion_type not in ("flow", "vp"):
            raise ValueError("diffusion_type must be flow or vp")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.n_t < 2 or any(getattr(self, k) < 1 for k in
                ("n_estimators", "duplicate_K", "n_batch", "n_threads", "sample_batch_size")):
            raise ValueError("n_t >= 2 and positive training/sampling budgets are required")
        if self.n_jobs == 0 or self.max_depth < 1 or not 0 < self.subsample <= 1 or not self.eta > 0:
            raise ValueError("invalid tree parameters")
        if not 0 < self.eps < 1 or not 0 < self.beta_min <= self.beta_max:
            raise ValueError("invalid VP schedule")


class ForestDiffusionWrapper(BaselineGenerativeModel):
    def __init__(self, cfg=None):
        self.cfg = cfg or ForestDiffusionConfig()
        super().__init__(seed=self.cfg.seed)
        self.model = None
        self.codec = None
        self.id_factory = None
        self.decoding_report_ = {}
        self.role_source_ = "explicit"

    @property
    def variant_id(self):
        return f"forest_{self.cfg.diffusion_type}_xgboost_joint_xy"

    def fit(self, data, *, continuous_cols=None, discrete_cols=None,
            categorical_cols=None, target_col=None, task=None, id_col=None):
        self.model = None
        self.n_updates_ = None
        if not isinstance(data, pd.DataFrame) or len(data) < 1:
            raise ValueError("fit expects a nonempty DataFrame with explicit column roles")
        self.roles_ = resolve_column_roles(data.columns, continuous_cols=continuous_cols,
            discrete_cols=discrete_cols, categorical_cols=categorical_cols,
            target_col=target_col, task=task, id_col=id_col)
        self.columns_ = list(data.columns)
        self.id_factory = FreshIdFactory.fit(data[id_col]) if id_col is not None else None
        self.codec = MixedToContinuousCodec(standardize_numeric=True).fit(data, self.roles_)
        matrix = self.codec.encode(data)
        if matrix.shape[1] == 0:
            raise ValueError("at least one modelled column is required")
        from sbtab.solvers.ForestDiffusion import ForestDiffusionModel
        c = self.cfg
        model = ForestDiffusionModel(matrix, n_t=c.n_t, diffusion_type=c.diffusion_type,
            n_estimators=c.n_estimators, max_depth=c.max_depth, eta=c.eta,
            duplicate_K=c.duplicate_K, n_batch=c.n_batch, n_jobs=c.n_jobs,
            reg_alpha=c.reg_alpha, reg_lambda=c.reg_lambda, subsample=c.subsample,
            eps=c.eps, beta_min=c.beta_min, beta_max=c.beta_max,
            gpu_hist=c.device == "cuda", seed=c.seed, nthread=c.n_threads)
        # Inference needs fitted trees, dimensions and codecs; retain no training
        # rows. The lower-level solver's separate imputation API is not exposed.
        model.X1 = None
        model.X_covs = None
        model.mask_y = {}
        self.model = model
        self.fit_info_ = BaselineFitInfo(len(data), len(data.columns), list(data.columns))
        self.n_updates_ = int(c.n_t * c.n_estimators)
        return self

    def sample(self, n, seed=None):
        if self.model is None:
            raise RuntimeError("fit() or load_checkpoint() first")
        n = validate_n(n)
        raw = self.model.generate(batch_size=n, seed=seed, max_batch_size=self.cfg.sample_batch_size)
        frame, report = self.codec.decode(raw)
        self.decoding_report_ = {"columns": report, "continuous_clipping": False}
        if self.id_factory is not None:
            frame[self.roles_.id_col] = self.id_factory.make(n)
        return frame[self.columns_]

    def save_checkpoint(self, path):
        if self.model is None:
            raise RuntimeError("nothing to save before fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"format": CHECKPOINT_FORMAT, "config": asdict(self.cfg), "model": self.model,
                 "codec": self.codec.state_dict(), "roles": self.roles_.to_dict(),
                 "columns": self.columns_, "fit_info": asdict(self.fit_info_),
                 "id_factory": None if self.id_factory is None else self.id_factory.state_dict(),
                 "n_updates": self.n_updates}
        with path.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return str(path)

    @classmethod
    def load_checkpoint(cls, path):
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported ForestDiffusion checkpoint format")
        obj = cls(ForestDiffusionConfig(**state["config"]))
        obj.model = state["model"]
        obj.codec = MixedToContinuousCodec.from_state(state["codec"])
        obj.roles_ = ColumnRoles.from_dict(state["roles"])
        obj.columns_ = state["columns"]
        obj.fit_info_ = BaselineFitInfo(**state["fit_info"])
        obj.id_factory = FreshIdFactory.from_state(state["id_factory"])
        obj.n_updates_ = state["n_updates"]
        return obj
