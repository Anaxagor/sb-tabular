"""
TabDDPM wrapper (vendored Gaussian + multinomial diffusion, MLP denoiser).

What is modelled
----------------
The COMPLETE row ``(X, y)`` is modelled jointly and unconditionally (``is_y_cond=False``):

* continuous and discrete-numeric columns (incl. a regression target) form the GAUSSIAN block;
* categorical columns (incl. a classification target, also when stored as integers) form the
  MULTINOMIAL block, one multinomial variable per column.

A classification target is therefore just one more multinomial column and the training label
prior is learned as its marginal; nothing is re-balanced.  The empirical training label
distribution is still recorded (``label_distribution_``) and stored in the checkpoint.

Column roles
------------
``fit(data, *, continuous_cols=, discrete_cols=, categorical_cols=, target_col=, task=, id_col=)``
- with explicit lists NO dtype/cardinality inference runs.  The legacy ``schema=`` /
``transforms=`` keywords are still accepted (explicit lists win); on that path the target role
comes from ``task`` when given and only otherwise from the deprecated dtype heuristic.

Internal representation (fitted on the FIT rows only, persisted, inverted on output)
-------------------------------------------------------------------------------------
* the whole Gaussian block is z-scored per column (zero variance -> scale 1).  Without this an
  unscaled regression target or count column wrecks the diffusion for every column;
* discrete numeric columns are decoded to the nearest value of their TRAINING support;
  ``decoding_report_`` holds the pre-decoding out-of-support / out-of-range rates per column;
* continuous columns are returned as generated - never clipped to the training range;
* original dtypes and the exact column order are restored; an ``id_col`` is never modelled and
  never resampled from the training ids - fresh ids disjoint from the training ids are emitted.

Training budget
---------------
ONE explicit budget: exactly one of ``cfg.steps`` (optimizer steps) or ``cfg.n_epochs``
(``n_epochs * ceil(n_rows / batch_size)`` steps; partial batches are kept) must be set.
It is resolved once at fit time into ``total_steps_`` (also ``n_updates``) and persisted.

EMA
---
Sampling uses the EMA denoiser in eval mode.  With ``ema_warmup=True`` (default) the decay at
update ``t`` is ``min(ema_decay, (1 + t) / (10 + t))`` so that a short, bounded run is not
dominated by the random initialisation (with a constant 0.999 the EMA after 300 updates is
still ~74% random init).  Both the EMA and the raw weights are persisted.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sbtab.numerics import check_gradients, require_finite

from sbtab.baselines.base import (
    ArrayLike,
    BaselineFitInfo,
    BaselineGenerativeModel,
    ColumnRoles,
    FreshIdFactory,
    empirical_distribution,
    explicit_roles_given,
    legacy_target_role,
    resolve_column_roles,
    seed_everything,
    to_python_scalar,
    validate_n,
)
from sbtab.baselines.encoding import (
    fit_standardizer,
    fit_vocabulary,
    nearest_support_decode,
    numeric_matrix,
    restore_dtype,
    standardize,
    unstandardize,
)

from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .modules import MLPDiffusion

_CHECKPOINT_FORMAT = "sbtab.baselines.tabddpm"
_CHECKPOINT_VERSION = 1


def validate_mlp_layers(d_layers: Sequence[int]) -> List[int]:
    """The vendored ``MLP.make_baseline`` needs equal middle layers; fail early and clearly."""
    try:
        layers = [int(d) for d in d_layers]
    except (TypeError, ValueError) as e:
        raise ValueError(f"d_layers must be a sequence of positive integers, got {d_layers!r}.") from e
    if len(layers) == 0:
        raise ValueError("d_layers must contain at least one layer width.")
    if any(d <= 0 for d in layers):
        raise ValueError(f"d_layers must be positive, got {layers}.")
    if len(layers) > 2 and len(set(layers[1:-1])) != 1:
        raise ValueError(
            "d_layers: all widths except the first and the last must be equal "
            f"(TabDDPM MLP.make_baseline constraint), got {layers}. "
            "Valid: [256], [128, 512], [128, 512, 512, 256]. Invalid: [128, 256, 512, 256]."
        )
    return layers


def ema_decay_at(update_index: int, ema_decay: float, warmup: bool) -> float:
    """EMA decay used for 0-based update ``update_index`` (bias-corrected warm-up)."""
    if not warmup:
        return float(ema_decay)
    t = float(update_index)
    return float(min(float(ema_decay), (1.0 + t) / (10.0 + t)))


@dataclass
class TabDDPMConfig:
    # ---- ONE explicit training budget: set exactly one of the two -------------------
    steps: Optional[int] = None      # optimizer steps
    n_epochs: Optional[int] = None   # epochs; steps = n_epochs * ceil(n_rows / batch_size)

    # original TabDDPM hyperparameters
    num_timesteps: int = 1000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4

    d_layers: List[int] = field(default_factory=lambda: [256, 512, 512, 256])
    dropout: float = 0.0

    gaussian_loss_type: str = "mse"   # "mse" | "kl" - passed through to the diffusion
    scheduler: str = "cosine"         # "cosine" | "linear"

    # EMA
    ema_decay: float = 0.999
    ema_warmup: bool = True           # decay_t = min(ema_decay, (1 + t) / (10 + t))

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        self._validate_budget()
        self.d_layers = validate_mlp_layers(self.d_layers)
        try:
            self.dropout = float(self.dropout)
        except (TypeError, ValueError) as e:
            raise ValueError(f"dropout must be a number, got {self.dropout!r}.") from e
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.gaussian_loss_type not in ("mse", "kl"):
            raise ValueError(f"gaussian_loss_type must be 'mse' or 'kl', got {self.gaussian_loss_type!r}.")
        if self.scheduler not in ("cosine", "linear"):
            raise ValueError(f"scheduler must be 'cosine' or 'linear', got {self.scheduler!r}.")
        if int(self.num_timesteps) < 2:
            raise ValueError("num_timesteps must be >= 2.")
        if self.scheduler == "linear" and int(self.num_timesteps) <= 20:
            # vendored linear schedule: beta_end = 0.02 * 1000 / T >= 1 for T <= 20 -> log(0)
            raise ValueError("scheduler='linear' needs num_timesteps > 20 (beta_end = 20 / num_timesteps must be < 1).")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be >= 1.")
        if not 0.0 <= float(self.ema_decay) < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}.")

    def _validate_budget(self) -> None:
        if self.steps is not None and self.n_epochs is not None:
            raise ValueError(
                f"Ambiguous training budget: both steps={self.steps} and n_epochs={self.n_epochs} "
                "are set. Set exactly one (neither silently wins)."
            )
        if self.steps is None and self.n_epochs is None:
            raise ValueError("No training budget: set exactly one of `steps` or `n_epochs`.")
        value = self.steps if self.steps is not None else self.n_epochs
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"The training budget must be a positive integer, got {value!r}.")

    def effective_budget(self, n_batches_per_epoch: int) -> int:
        """Total optimizer steps for a loader with ``n_batches_per_epoch`` batches (partial batch kept)."""
        self._validate_budget()
        if self.steps is not None:
            return int(self.steps)
        return int(self.n_epochs) * max(int(n_batches_per_epoch), 1)


class TabDDPMWrapper(BaselineGenerativeModel):
    """TabDDPM baseline; see the module docstring for the full contract."""

    variant_id = "tabddpm_mlp_joint_xy"

    def __init__(self, cfg: TabDDPMConfig):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self._fitted = False
        self.columns_: Optional[List[Any]] = None
        self._input_columns: Optional[List[Any]] = None
        self._dtypes: Dict[Any, str] = {}

        self.num_numerical_features: int = 0
        self.num_classes: np.ndarray = np.array([], dtype=np.int64)

        self._schema: Optional[Any] = None
        self.role_source_: Optional[str] = None

        self._id_col: Optional[Any] = None
        self._id_factory: Optional[FreshIdFactory] = None

        self._num_output_cols: List[Any] = []
        self._pending_discrete: List[Any] = []
        self._discrete_supports: Dict[Any, np.ndarray] = {}
        self._num_mean: np.ndarray = np.zeros(0)
        self._num_scale: np.ndarray = np.ones(0)
        self._cat_specs: List[Dict[str, Any]] = []

        self.total_steps_: Optional[int] = None
        self.n_batches_per_epoch_: Optional[int] = None
        self.label_distribution_: Optional[List[List[Any]]] = None
        self.decoding_report_: Dict[Any, Dict[str, Any]] = {}
        self.final_loss_: Optional[float] = None

        self.diffusion: Optional[GaussianMultinomialDiffusion] = None
        self.ema_model: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------
    # LEGACY helpers: discover categorical representation metadata in `transforms`
    # ------------------------------------------------------------------

    def _find_categorical_representation(self, obj: Any) -> Tuple[Optional[Any], Optional[str]]:
        visited = set()

        def infer_rep_name(x: Any) -> Optional[str]:
            rep_name = getattr(x, "representation_name", None)
            if isinstance(rep_name, str):
                return rep_name

            cls_name = x.__class__.__name__.lower()
            if "onehot" in cls_name:
                return "one_hot_representation"
            if "integercode" in cls_name or "integer_code" in cls_name:
                return "integer_code_representation"
            return None

        def is_rep_obj(x: Any) -> bool:
            return (
                hasattr(x, "categorical_cols_")
                and hasattr(x, "categories_")
                and hasattr(x, "fitted_")
            )

        def rec(x: Any) -> Tuple[Optional[Any], Optional[str]]:
            if x is None:
                return None, None
            xid = id(x)
            if xid in visited:
                return None, None
            visited.add(xid)

            if hasattr(x, "repr_") and getattr(x, "repr_", None) is not None:
                rep = getattr(x, "repr_")
                if is_rep_obj(rep):
                    return rep, infer_rep_name(x) or infer_rep_name(rep)

            if is_rep_obj(x):
                return x, infer_rep_name(x)

            for attr in ("transforms", "steps"):
                if hasattr(x, attr):
                    sub = getattr(x, attr)
                    if isinstance(sub, dict):
                        for v in sub.values():
                            obj2, name2 = rec(v)
                            if obj2 is not None:
                                return obj2, name2
                    else:
                        try:
                            for v in sub:
                                obj2, name2 = rec(v)
                                if obj2 is not None:
                                    return obj2, name2
                        except TypeError:
                            pass

            if isinstance(x, dict):
                for v in x.values():
                    obj2, name2 = rec(v)
                    if obj2 is not None:
                        return obj2, name2

            if isinstance(x, (list, tuple)):
                for v in x:
                    obj2, name2 = rec(v)
                    if obj2 is not None:
                        return obj2, name2

            return None, None

        return rec(obj)

    @staticmethod
    def _onehot_col_map(rep: Any) -> Dict[str, List[str]]:
        col_map: Dict[str, List[str]] = {}
        encoded_cols = list(getattr(rep, "encoded_cols_", []))
        categories = dict(getattr(rep, "categories_", {}))
        categorical_cols = list(getattr(rep, "categorical_cols_", []))

        cursor = 0
        for col in categorical_cols:
            cats = categories.get(col, [])
            width = len(cats)
            col_map[col] = encoded_cols[cursor: cursor + width]
            cursor += width
        return col_map

    @staticmethod
    def _intcode_col_map(rep: Any) -> Dict[str, List[str]]:
        categorical_cols = list(getattr(rep, "categorical_cols_", []))
        encoded_cols = list(getattr(rep, "encoded_cols_", categorical_cols))
        if encoded_cols and len(encoded_cols) == len(categorical_cols):
            return {src: [enc] for src, enc in zip(categorical_cols, encoded_cols)}
        return {col: [col] for col in categorical_cols}

    # ------------------------------------------------------------------
    # block layout
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_spec(df: pd.DataFrame, col: Any) -> Dict[str, Any]:
        vocab = fit_vocabulary(df[col], col)
        return {"name": col, "mode": "raw", "output_cols": [col], "num_classes": len(vocab), "categories": vocab}

    def _layout_from_roles(self, df: pd.DataFrame, roles: ColumnRoles) -> None:
        """Explicit path: no inference, no representation discovery."""
        self._num_output_cols = [c for c in df.columns if c in set(roles.numeric)]
        discrete = [c for c in self._num_output_cols if c in set(roles.discrete)]
        self._cat_specs = [self._raw_spec(df, c) for c in df.columns if c in set(roles.categorical)]
        self._pending_discrete = discrete

    def _layout_from_schema(self, df: pd.DataFrame, schema: Any, transforms: Any, task: Optional[str]) -> ColumnRoles:
        """Legacy path (``schema=`` / ``transforms=``)."""
        continuous = [c for c in getattr(schema, "continuous_cols", []) if c in df.columns]
        discrete = [c for c in getattr(schema, "discrete_cols", []) if c in df.columns]
        schema_categorical = list(getattr(schema, "categorical_cols", []))
        target_col = getattr(schema, "target_col", None)
        if target_col is not None and target_col not in df.columns:
            target_col = None

        inferred = False
        categorical_block = list(schema_categorical)
        if target_col is not None and target_col not in (*continuous, *discrete, *schema_categorical):
            role, inferred = legacy_target_role(df, target_col, task)
            if role == "categorical":
                categorical_block.append(target_col)
            elif role == "discrete":
                discrete.append(target_col)
            else:
                continuous.append(target_col)

        numeric = set(continuous) | set(discrete)
        self._num_output_cols = [c for c in df.columns if c in numeric]
        self._pending_discrete = [c for c in self._num_output_cols if c in set(discrete)]

        rep, rep_name = self._find_categorical_representation(transforms)
        onehot_map = self._onehot_col_map(rep) if rep is not None and rep_name == "one_hot_representation" else {}
        intcode_map = self._intcode_col_map(rep) if rep is not None and rep_name == "integer_code_representation" else {}
        categories_map = dict(getattr(rep, "categories_", {})) if rep is not None else {}

        specs: List[Dict[str, Any]] = []
        for col in categorical_block:
            cats = [to_python_scalar(v) for v in categories_map.get(col, [])]
            if col in intcode_map and all(c in df.columns for c in intcode_map[col]):
                specs.append({"name": col, "mode": "integer_code", "output_cols": list(intcode_map[col]),
                              "num_classes": len(cats), "categories": cats})
            elif col in onehot_map and all(c in df.columns for c in onehot_map[col]):
                specs.append({"name": col, "mode": "onehot", "output_cols": list(onehot_map[col]),
                              "num_classes": len(cats), "categories": cats})
            elif col in df.columns:
                specs.append(self._raw_spec(df, col))
            else:
                raise ValueError(
                    f"Categorical feature {col!r} is neither present as a raw column nor "
                    f"recoverable from fitted categorical transform metadata."
                )
        self._cat_specs = specs

        return ColumnRoles(
            continuous=tuple(c for c in df.columns if c in set(continuous)),
            discrete=tuple(c for c in df.columns if c in set(discrete)),
            categorical=tuple(categorical_block),
            target_col=target_col,
            task=task if task is not None else (
                None if target_col is None else ("classification" if target_col in categorical_block else "regression")
            ),
            id_col=getattr(schema, "id_col", None) if getattr(schema, "id_col", None) in df.columns else None,
            source="schema+inferred_target" if inferred else "schema",
        )

    # ------------------------------------------------------------------
    # internal TabDDPM matrix construction
    # ------------------------------------------------------------------

    def _categorical_codes(self, df: pd.DataFrame, spec: Dict[str, Any]) -> np.ndarray:
        if spec["mode"] == "raw":
            cat = pd.Categorical(df[spec["name"]], categories=spec["categories"])
            codes = np.asarray(cat.codes, dtype=np.int64)
            if (codes < 0).any():
                raise ValueError(f"Categorical column {spec['name']!r} contains unknown or missing values.")
            return codes
        if spec["mode"] == "onehot":
            block = df[spec["output_cols"]].to_numpy(dtype=np.float32, copy=True)
            return np.argmax(block, axis=1).astype(np.int64)
        if spec["mode"] == "integer_code":
            col = spec["output_cols"][0]
            codes = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.int64, copy=True)
            if (codes < 0).any() or (codes >= spec["num_classes"]).any():
                raise ValueError(f"Integer-coded categorical column {col!r} contains code(s) outside 0..S-1.")
            return codes
        raise RuntimeError(f"Unknown categorical mode: {spec['mode']!r}")

    def _build_training_matrix(self, df: pd.DataFrame) -> torch.Tensor:
        x_num = numeric_matrix(df, self._num_output_cols)
        self._num_mean, self._num_scale = fit_standardizer(x_num)          # FIT rows only
        self._discrete_supports = {
            c: np.unique(x_num[:, self._num_output_cols.index(c)]) for c in self._pending_discrete
        }
        z_num = standardize(x_num, self._num_mean, self._num_scale).astype(np.float32)
        self.num_numerical_features = z_num.shape[1]

        blocks = [z_num]
        for spec in self._cat_specs:
            if int(spec["num_classes"]) < 1:
                raise ValueError(f"Categorical column {spec['name']!r} has an empty vocabulary.")
            blocks.append(self._categorical_codes(df, spec).reshape(-1, 1).astype(np.float32))
        self.num_classes = np.asarray([int(s["num_classes"]) for s in self._cat_specs], dtype=np.int64)
        return torch.from_numpy(np.concatenate(blocks, axis=1)).to(self.device)

    # ------------------------------------------------------------------
    # model construction (shared by fit and load_checkpoint)
    # ------------------------------------------------------------------

    def _build_modules(self) -> None:
        num_classes = self.num_classes if len(self.num_classes) > 0 else np.array([0])
        d_in = self.num_numerical_features + int(num_classes.sum())
        if d_in == 0:
            raise ValueError("Nothing to model: no numeric and no categorical columns.")
        denoiser = MLPDiffusion(
            d_in=d_in,
            num_classes=0,
            is_y_cond=False,   # the row (X, y) is modelled jointly; y is an ordinary column
            rtdl_params={"d_layers": list(self.cfg.d_layers), "dropout": float(self.cfg.dropout)},
        ).to(self.device)

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=num_classes,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=denoiser,
            num_timesteps=int(self.cfg.num_timesteps),
            gaussian_loss_type=self.cfg.gaussian_loss_type,
            scheduler=self.cfg.scheduler,
            device=self.device,
        ).to(self.device)

        self.ema_model = copy.deepcopy(self.diffusion._denoise_fn).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _anneal_lr(optimizer: torch.optim.Optimizer, *, init_lr: float, step: int, total_steps: int) -> None:
        lr = init_lr * (1.0 - step / float(total_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    @torch.no_grad()
    def _update_ema(target_model: torch.nn.Module, source_model: torch.nn.Module, rate: float) -> None:
        for targ, src in zip(target_model.parameters(), source_model.parameters()):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1.0 - rate)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: ArrayLike,
        *,
        continuous_cols: Optional[Sequence[Any]] = None,
        discrete_cols: Optional[Sequence[Any]] = None,
        categorical_cols: Optional[Sequence[Any]] = None,
        target_col: Optional[Any] = None,
        task: Optional[str] = None,
        id_col: Optional[Any] = None,
        schema: Optional[Any] = None,
        transforms: Any = None,
        **kwargs: Any,
    ) -> "TabDDPMWrapper":
        """
        Train on exactly the rows of ``data`` (no internal split).

        Explicit roles (preferred; no inference runs):
            ``continuous_cols`` / ``discrete_cols`` -> Gaussian block (z-scored internally),
            ``categorical_cols`` -> multinomial block, ``target_col`` + ``task`` place the target
            when it is not listed, ``id_col`` is excluded and replaced by fresh ids on output.
        Legacy: ``schema=`` (object with ``continuous_cols/discrete_cols/categorical_cols/
            target_col/id_col``) and optional fitted ``transforms=`` (one-hot / integer-code
            representation discovery).  Explicit lists take precedence.
        """
        self._reject_unknown_kwargs(kwargs, "TabDDPMWrapper.fit()")
        self._fitted = False
        self.n_updates_ = 0
        # D5: seed BEFORE anything random (model init, timestep sampling, noise, batch order)
        seed_everything(self.cfg.seed)

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(np.asarray(data), columns=[f"f{i}" for i in range(np.asarray(data).shape[1])])
        if len(data) == 0:
            raise ValueError("Cannot fit on an empty frame.")

        if explicit_roles_given(continuous_cols, discrete_cols, categorical_cols):
            roles = resolve_column_roles(
                list(data.columns),
                continuous_cols=continuous_cols,
                discrete_cols=discrete_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
                task=task,
                id_col=id_col,
            )
            self._layout_from_roles(data, roles)
            self._schema = None
        elif schema is not None:
            if target_col is not None and target_col != getattr(schema, "target_col", None):
                raise ValueError("target_col disagrees with schema.target_col.")
            roles = self._layout_from_schema(data, schema, transforms, task)
            self._schema = schema
        else:
            raise ValueError(
                "Column roles are required: pass continuous_cols= / discrete_cols= / categorical_cols= "
                "(+ target_col=, task=) or, for backward compatibility, schema=."
            )
        self.roles_ = roles
        self.role_source_ = roles.source

        self.columns_ = list(data.columns)
        self._input_columns = list(data.columns)
        self._dtypes = {c: str(data[c].dtype) for c in data.columns}

        self._id_col = roles.id_col
        self._id_factory = FreshIdFactory.fit(data[self._id_col]) if self._id_col is not None else None

        self.label_distribution_ = None
        if roles.target_col is not None and roles.target_col in roles.categorical and roles.target_col in data.columns:
            self.label_distribution_ = empirical_distribution(data[roles.target_col].tolist())

        X = self._build_training_matrix(data)
        self._build_modules()
        assert self.diffusion is not None and self.ema_model is not None

        optimizer = torch.optim.AdamW(
            self.diffusion.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # D1: ONE budget, resolved here, exposed and persisted.
        n_rows = int(X.shape[0])
        batch_size = int(self.cfg.batch_size)
        self.n_batches_per_epoch_ = max((n_rows + batch_size - 1) // batch_size, 1)   # partial batch kept
        total_steps = self.cfg.effective_budget(self.n_batches_per_epoch_)
        self.total_steps_ = int(total_steps)

        # D5: seeded batch order (equivalent to DataLoader(shuffle=True, drop_last=False,
        # generator=g) but without per-row collation of an on-device tensor).
        order_gen = torch.Generator(device="cpu")
        order_gen.manual_seed(int(self.cfg.seed))

        def epoch_batches():
            perm = torch.randperm(n_rows, generator=order_gen).to(X.device)
            for start in range(0, n_rows, batch_size):
                yield X[perm[start: start + batch_size]]

        self.diffusion.train()
        batches = epoch_batches()
        n_done = 0
        last_loss = float("nan")
        for step in range(total_steps):
            try:
                x_batch = next(batches)
            except StopIteration:
                batches = epoch_batches()
                x_batch = next(batches)

            self._anneal_lr(optimizer, init_lr=self.cfg.lr, step=step, total_steps=total_steps)

            loss_multi, loss_gauss = self.diffusion.mixed_loss(x_batch, out_dict={"y": None})
            loss = loss_multi.to(self.device) + loss_gauss.to(self.device)
            context = dict(model=self.variant_id, stage="training", step=step)
            require_finite(loss, "loss", **context)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            check_gradients(self.diffusion._denoise_fn.parameters(), **context)
            optimizer.step()

            self._update_ema(
                self.ema_model,
                self.diffusion._denoise_fn,
                ema_decay_at(step, self.cfg.ema_decay, self.cfg.ema_warmup),
            )
            n_done += 1
            last_loss = loss.detach()

        self.final_loss_ = float(last_loss) if n_done else None
        self.n_updates_ = n_done
        self.diffusion.eval()

        self.fit_info_ = BaselineFitInfo(
            n_rows=int(data.shape[0]),
            n_cols=int(data.shape[1]),
            columns=list(self.columns_),
        )
        self._fitted = True
        return self

    # ------------------------------------------------------------------

    def _reconstruct_output_df(self, x_gen: np.ndarray) -> pd.DataFrame:
        n = x_gen.shape[0]
        n_num = self.num_numerical_features
        cols: Dict[Any, Any] = {}
        report: Dict[Any, Dict[str, Any]] = {}

        if n_num > 0:
            x_num = unstandardize(x_gen[:, :n_num], self._num_mean, self._num_scale)
            for j, col in enumerate(self._num_output_cols):
                values = x_num[:, j]
                if col in self._discrete_supports:
                    values, report[col] = nearest_support_decode(values, self._discrete_supports[col])
                # continuous columns: returned as generated, never clipped to the training range
                cols[col] = restore_dtype(values, self._dtypes[col])

        x_cat = x_gen[:, n_num:]
        for j, spec in enumerate(self._cat_specs):
            codes = np.rint(np.asarray(x_cat[:, j], dtype=np.float64)).astype(np.int64)
            if codes.size and (codes.min() < 0 or codes.max() >= spec["num_classes"]):
                raise RuntimeError(f"Multinomial sample outside 0..S-1 for column {spec['name']!r}.")

            if spec["mode"] == "raw":
                categories = np.empty(len(spec["categories"]), dtype=object)
                categories[:] = list(spec["categories"])
                out_col = spec["output_cols"][0]
                cols[out_col] = restore_dtype(categories[codes], self._dtypes[out_col])
            elif spec["mode"] == "onehot":
                oh = np.eye(spec["num_classes"], dtype=np.float64)[codes]
                for k, out_col in enumerate(spec["output_cols"]):
                    cols[out_col] = restore_dtype(oh[:, k], self._dtypes[out_col])
            elif spec["mode"] == "integer_code":
                out_col = spec["output_cols"][0]
                cols[out_col] = restore_dtype(codes, self._dtypes[out_col])
            else:
                raise RuntimeError(f"Unknown categorical mode: {spec['mode']!r}")

        if self._id_col is not None:
            # D10: NEVER resample real training ids into synthetic rows.
            factory = self._id_factory or FreshIdFactory()
            cols[self._id_col] = pd.Series(factory.make(n))

        missing = [c for c in (self._input_columns or []) if c not in cols]
        if missing:
            raise RuntimeError(
                f"Failed to reconstruct output columns: {missing}. "
                "Check categorical representation metadata handling."
            )
        self.decoding_report_ = report
        out = pd.DataFrame({c: pd.Series(cols[c]).reset_index(drop=True) for c in self._input_columns})
        return out[self._input_columns]

    @torch.no_grad()
    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        *,
        use_ema: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Exactly ``n`` rows, same columns / order / dtypes as the fit frame.  ``seed`` seeds the
        torch + numpy global generators (``None`` -> the global stream simply continues).
        Sampling uses the EMA denoiser in eval mode unless ``use_ema=False``.
        """
        self._reject_unknown_kwargs(kwargs, "TabDDPMWrapper.sample()")
        if not self._fitted or self.diffusion is None:
            raise RuntimeError("Call fit() before sample().")
        n = validate_n(n)

        if seed is not None:
            seed_everything(int(seed))

        self.diffusion.eval()
        y_dist = torch.ones(1, device=self.device)   # unused: is_y_cond=False

        denoiser_backup = self.diffusion._denoise_fn
        if use_ema and self.ema_model is not None:
            self.ema_model.eval()
            self.diffusion._denoise_fn = self.ema_model
        try:
            x_gen, _ = self.diffusion.sample_all(n, int(self.cfg.batch_size), y_dist)
        finally:
            self.diffusion._denoise_fn = denoiser_backup

        if isinstance(x_gen, torch.Tensor):
            x_gen = x_gen.detach().cpu().numpy()

        out = self._reconstruct_output_df(np.asarray(x_gen, dtype=np.float64))
        if len(out) != n:
            raise RuntimeError(f"TabDDPM produced {len(out)} rows, expected exactly {n}.")
        return out

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> str:
        """
        Single file written with ``torch.save``; contains only tensors and plain python, so it is
        loaded with ``weights_only=True``.  Holds the config, the RESOLVED budget, both the raw
        and the EMA denoiser weights, the fitted standardiser / supports / vocabularies, column
        order + dtypes, roles and the empirical training label distribution.
        """
        if not self._fitted or self.diffusion is None or self.ema_model is None:
            raise RuntimeError("Call fit() before save_checkpoint().")
        cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}  # noqa: E731
        state = {
            "format": _CHECKPOINT_FORMAT,
            "format_version": _CHECKPOINT_VERSION,
            "variant_id": self.variant_id,
            "cfg": asdict(self.cfg),
            "total_steps": int(self.total_steps_),
            "n_updates": int(self.n_updates_),
            "n_batches_per_epoch": int(self.n_batches_per_epoch_),
            "final_loss": self.final_loss_,
            "columns": list(self._input_columns),
            "dtypes": [[c, self._dtypes[c]] for c in self._input_columns],
            "roles": self.roles_.to_dict() if self.roles_ is not None else None,
            "num_cols": list(self._num_output_cols),
            "num_mean": torch.as_tensor(self._num_mean, dtype=torch.float64),
            "num_scale": torch.as_tensor(self._num_scale, dtype=torch.float64),
            "discrete_supports": [
                [c, torch.as_tensor(v, dtype=torch.float64)] for c, v in self._discrete_supports.items()
            ],
            "cat_specs": [
                {
                    "name": s["name"],
                    "mode": s["mode"],
                    "output_cols": list(s["output_cols"]),
                    "num_classes": int(s["num_classes"]),
                    "categories": [to_python_scalar(v) for v in s["categories"]],
                }
                for s in self._cat_specs
            ],
            "label_distribution": self.label_distribution_,
            "id_col": self._id_col,
            "id_factory": self._id_factory.state_dict() if self._id_factory is not None else None,
            "fit_info": asdict(self.fit_info_) if self.fit_info_ is not None else None,
            "denoiser_raw": cpu(self.diffusion._denoise_fn.state_dict()),
            "denoiser_ema": cpu(self.ema_model.state_dict()),
            "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__), "pandas": str(pd.__version__)},
        }
        torch.save(state, path)
        return str(path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None, **kwargs: Any) -> "TabDDPMWrapper":
        """Rebuild a fitted wrapper without refitting. ``device`` overrides the saved device."""
        cls._reject_unknown_kwargs(kwargs, "TabDDPMWrapper.load_checkpoint()")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"{path!r} is not a TabDDPM wrapper checkpoint.")
        if int(state.get("format_version", -1)) != _CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported TabDDPM checkpoint version: {state.get('format_version')!r}.")

        cfg_dict = dict(state["cfg"])
        if device is not None:
            cfg_dict["device"] = device
        elif str(cfg_dict.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("Checkpoint was trained on CUDA but CUDA is unavailable; loading on CPU.", UserWarning)
            cfg_dict["device"] = "cpu"
        obj = cls(TabDDPMConfig(**cfg_dict))

        obj._input_columns = list(state["columns"])
        obj.columns_ = list(state["columns"])
        obj._dtypes = {c: d for c, d in state["dtypes"]}
        obj.roles_ = ColumnRoles.from_dict(state["roles"]) if state["roles"] is not None else None
        obj.role_source_ = obj.roles_.source if obj.roles_ is not None else None
        obj._num_output_cols = list(state["num_cols"])
        obj._num_mean = state["num_mean"].numpy().astype(np.float64)
        obj._num_scale = state["num_scale"].numpy().astype(np.float64)
        obj._discrete_supports = {c: v.numpy().astype(np.float64) for c, v in state["discrete_supports"]}
        obj._cat_specs = [dict(s) for s in state["cat_specs"]]
        obj.num_numerical_features = len(obj._num_output_cols)
        obj.num_classes = np.asarray([int(s["num_classes"]) for s in obj._cat_specs], dtype=np.int64)
        obj.label_distribution_ = state["label_distribution"]
        obj._id_col = state["id_col"]
        obj._id_factory = FreshIdFactory.from_state(state["id_factory"])
        obj.total_steps_ = int(state["total_steps"])
        obj.n_updates_ = int(state["n_updates"])
        obj.n_batches_per_epoch_ = int(state["n_batches_per_epoch"])
        obj.final_loss_ = state.get("final_loss")
        if state.get("fit_info") is not None:
            obj.fit_info_ = BaselineFitInfo(**state["fit_info"])

        obj._build_modules()
        assert obj.diffusion is not None and obj.ema_model is not None
        obj.diffusion._denoise_fn.load_state_dict(state["denoiser_raw"])
        obj.ema_model.load_state_dict(state["denoiser_ema"])
        obj.diffusion.eval()
        obj.ema_model.eval()
        obj._fitted = True
        return obj
