"""
Simplified VE score-SDE baseline for tabular data.

THIS IS NOT STaSy.  The module lives in ``baselines/stasy`` for historical reasons, but what is
implemented is a plain variance-exploding (VE) score-based SDE with denoising score matching,
an MLP score network and a predictor-corrector sampler.  STaSy (Kim, Lee, Park - "STaSy:
Score-based Tabular data Synthesis", ICLR 2023) additionally needs the components listed under
``FAITHFULNESS["missing"]``; results of this model must therefore be reported under
``variant_id = "ve_score_sde_simplified"`` and never as "STaSy".

``STaSyGenerative`` / ``STaSyConfig`` / ``use_self_paced`` remain importable as DEPRECATED
aliases that emit a ``DeprecationWarning`` saying exactly that.

Column roles
------------
``fit(data, *, continuous_cols=, discrete_cols=, categorical_cols=, target_col=, task=, id_col=)``.
The model is continuous-only, so (all fitted on the FIT rows only, persisted, inverted on output)

* numeric columns (continuous + discrete) are z-scored (zero variance -> scale 1),
* categorical columns (incl. a classification target) are one-hot encoded over the train-fitted
  vocabulary and argmax-decoded on output,
* discrete numeric columns are decoded to the nearest value of their training support,
* ``decoding_report_`` records the pre-decoding statistics per decoded column,
* column order and dtypes are restored; continuous outputs are never clipped.

With no role information at all every column is treated as continuous (a declared default, not
an inference; non-numeric columns then raise).

Training budget
---------------
Exactly one of ``cfg.steps`` (optimizer steps) or ``cfg.n_epochs`` (``n_epochs *
ceil(n_rows / batch_size)`` steps, partial batches kept); resolved into ``total_steps_``.

Time curriculum (formerly mis-named ``use_self_paced``)
-------------------------------------------------------
``time_curriculum`` restricts the diffusion TIME range ``t <= t_max`` early in training.  It is a
curriculum over noise levels, NOT STaSy's per-sample self-paced weighting.  It is OFF by default:
the sampler starts at ``sigma_max`` (t = 1), and the old schedule reached ``t_max = 1`` only in
the last epoch, so the high-noise scores the sampler depends on first were barely trained
(synthetic std ~33 instead of 1).  When enabled the ramp now ends at ``curriculum_end_frac``
(default 0.5) of the budget, so at least half of the updates see the full noise range.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    validate_n,
)
from sbtab.baselines.encoding import MixedToContinuousCodec

VARIANT_ID = "ve_score_sde_simplified"

FAITHFULNESS: Dict[str, Any] = {
    "reference": "Kim, Lee, Park. STaSy: Score-based Tabular data Synthesis. ICLR 2023.",
    "variant_id": VARIANT_ID,
    "is_faithful_stasy": False,
    "present": [
        "VE SDE forward process with geometric sigma(t) = sigma_min * (sigma_max / sigma_min) ** t",
        "denoising score matching with sigma^2 weighting",
        "MLP score network conditioned on a sinusoidal embedding of log sigma",
        "reverse-diffusion predictor + Langevin corrector (PC) sampler, optional final denoising step",
        "one-hot encoding of categorical columns with argmax decoding (added by this wrapper)",
        "optional diffusion-TIME curriculum (`time_curriculum`) - not a STaSy component",
    ],
    "missing": [
        "self-paced learning: per-sample weights v_i with alpha0 / beta0 quantile thresholds on the per-sample DSM loss",
        "fine-tuning stage on top of the self-paced stage",
        "VP and sub-VP SDE options (only VE is implemented)",
        "probability-flow ODE sampler",
        "ncsnpp-tabular (ConcatSquash residual) score architecture",
        "STaSy preprocessing (min-max scaling of numeric columns to [0, 1]); z-scoring is used instead",
        "exact likelihood / log-probability evaluation",
    ],
}

_NOT_STASY = (
    "{old} is a DEPRECATED alias of {new}. The implementation is a simplified VE score-SDE "
    f"(variant_id={VARIANT_ID!r}); it is NOT a faithful implementation of STaSy (Kim et al., ICLR 2023) - "
    "see sbtab.baselines.stasy.FAITHFULNESS for the missing components."
)

_CHECKPOINT_FORMAT = "sbtab.baselines.ve_score_sde"
_CHECKPOINT_VERSION = 1


@dataclass
class VEScoreSDEConfig:
    hidden_dim: int = 256
    n_layers: int = 4
    time_emb_dim: int = 64
    dropout: float = 0.0

    sigma_min: float = 0.01
    sigma_max: float = 50.0

    # ---- ONE explicit training budget: set exactly one of n_epochs / steps -------------
    n_epochs: Optional[int] = None
    batch_size: int = 512
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0

    # diffusion-TIME curriculum (NOT STaSy self-paced learning); off by default
    time_curriculum: bool = False
    sp_start_ratio: float = 0.25

    n_sampling_steps: int = 1000
    n_corrector_steps: int = 1
    corrector_snr: float = 0.16
    eps: float = 1e-3
    denoise: bool = True

    device: str = "cpu"
    seed: int = 42

    steps: Optional[int] = None
    curriculum_end_frac: float = 0.5   # the t_max ramp reaches 1 after this fraction of the budget
    standardize_numeric: bool = True   # z-score the numeric block on the fit rows

    def __post_init__(self) -> None:
        self._validate_budget()
        if int(self.time_emb_dim) % 2 != 0 or int(self.time_emb_dim) <= 0:
            raise ValueError("time_emb_dim must be a positive even integer.")
        if not 0.0 < float(self.sigma_min) < float(self.sigma_max):
            raise ValueError("Need 0 < sigma_min < sigma_max.")
        if not 0.0 < float(self.eps) < 1.0:
            raise ValueError("eps must be in (0, 1).")
        if not 0.0 < float(self.sp_start_ratio) <= 1.0:
            raise ValueError("sp_start_ratio must be in (0, 1].")
        if not 0.0 < float(self.curriculum_end_frac) <= 1.0:
            raise ValueError("curriculum_end_frac must be in (0, 1].")
        if int(self.batch_size) < 1 or int(self.n_sampling_steps) < 1 or int(self.n_corrector_steps) < 0:
            raise ValueError("batch_size / n_sampling_steps must be >= 1 and n_corrector_steps >= 0.")

    def _validate_budget(self) -> None:
        if self.steps is not None and self.n_epochs is not None:
            raise ValueError(
                f"Ambiguous training budget: both steps={self.steps} and n_epochs={self.n_epochs} are set. "
                "Set exactly one."
            )
        if self.steps is None and self.n_epochs is None:
            raise ValueError("No training budget: set exactly one of `steps` or `n_epochs`.")
        value = self.steps if self.steps is not None else self.n_epochs
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"The training budget must be a positive integer, got {value!r}.")

    def effective_budget(self, n_batches_per_epoch: int) -> int:
        """Total optimizer steps (partial batches kept, so ``n_batches = ceil(n_rows / batch_size)``)."""
        self._validate_budget()
        if self.steps is not None:
            return int(self.steps)
        return int(self.n_epochs) * max(int(n_batches_per_epoch), 1)

    def curriculum_t_max(self, step: int, total_steps: int) -> float:
        """Upper end of the diffusion-time range used at 0-based optimizer step ``step``."""
        if not self.time_curriculum:
            return 1.0
        ramp_steps = max(float(self.curriculum_end_frac) * float(total_steps), 1.0)
        frac = min(float(step) / ramp_steps, 1.0)
        return float(min(self.sp_start_ratio + (1.0 - self.sp_start_ratio) * frac, 1.0))


class _SigmaEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(10_000.0)
                * torch.arange(dim // 2, dtype=torch.float32)
                / (dim // 2)
            ),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = torch.log(sigma.view(-1, 1))
        args = log_sigma * self.freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _ScoreNet(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        n_layers: int,
        time_emb_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = _SigmaEmbedding(time_emb_dim)
        in_dim = data_dim + time_emb_dim
        layers: list[nn.Module] = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        emb = self.emb(sigma)
        h = torch.cat([x, emb], dim=-1)
        return self.net(h)


class VEScoreSDEBaseline(BaselineGenerativeModel):
    """Simplified VE score-SDE baseline (NOT STaSy); see the module docstring."""

    variant_id = VARIANT_ID
    faithfulness = FAITHFULNESS

    def __init__(self, cfg: VEScoreSDEConfig) -> None:
        super().__init__(seed=cfg.seed)
        self.cfg = cfg
        self._score_net: Optional[_ScoreNet] = None
        self._dim: Optional[int] = None
        self._codec: Optional[MixedToContinuousCodec] = None
        self._input_columns: Optional[List[Any]] = None
        self._was_dataframe: bool = True
        self._id_col: Optional[Any] = None
        self._id_factory: Optional[FreshIdFactory] = None

        self.role_source_: Optional[str] = None
        self.total_steps_: Optional[int] = None
        self.n_batches_per_epoch_: Optional[int] = None
        self.label_distribution_: Optional[List[List[Any]]] = None
        self.decoding_report_: Dict[Any, Dict[str, Any]] = {}
        self.final_loss_: Optional[float] = None

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.cfg.sigma_min * (self.cfg.sigma_max / self.cfg.sigma_min) ** t

    def _build_net(self) -> _ScoreNet:
        cfg = self.cfg
        return _ScoreNet(
            data_dim=int(self._dim),
            hidden_dim=int(cfg.hidden_dim),
            n_layers=int(cfg.n_layers),
            time_emb_dim=int(cfg.time_emb_dim),
            dropout=float(cfg.dropout),
        ).to(torch.device(cfg.device))

    # ------------------------------------------------------------------

    def _resolve_roles(
        self,
        df: pd.DataFrame,
        continuous_cols, discrete_cols, categorical_cols, target_col, task, id_col, schema,
    ) -> ColumnRoles:
        columns = list(df.columns)
        if explicit_roles_given(continuous_cols, discrete_cols, categorical_cols):
            return resolve_column_roles(
                columns,
                continuous_cols=continuous_cols,
                discrete_cols=discrete_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
                task=task,
                id_col=id_col,
            )
        if schema is not None:
            continuous = [c for c in getattr(schema, "continuous_cols", []) if c in columns]
            discrete = [c for c in getattr(schema, "discrete_cols", []) if c in columns]
            categorical = [c for c in getattr(schema, "categorical_cols", []) if c in columns]
            s_target = getattr(schema, "target_col", None)
            s_id = getattr(schema, "id_col", None)
            inferred = False
            if s_target is not None and s_target in columns and s_target not in (*continuous, *discrete, *categorical):
                role, inferred = legacy_target_role(df, s_target, task)
                {"categorical": categorical, "discrete": discrete}.get(role, continuous).append(s_target)
            roles = resolve_column_roles(
                columns,
                continuous_cols=continuous,
                discrete_cols=discrete,
                categorical_cols=categorical,
                target_col=s_target if s_target in columns else None,
                id_col=s_id if s_id in columns else None,
            )
            return replace(roles, source="schema+inferred_target" if inferred else "schema")
        # Nothing declared: every column is continuous BY DECLARATION (nothing is inferred from
        # the data).  The only exception is a target whose role the caller did declare via `task`.
        if self._was_dataframe:
            warnings.warn(
                "No column roles were given to fit(): every column is treated as CONTINUOUS. Pass "
                "continuous_cols= / discrete_cols= / categorical_cols= for mixed-type data.",
                UserWarning,
                stacklevel=3,
            )
        placed_by_task = target_col is not None and task is not None
        modelled = [c for c in columns if c != id_col and not (placed_by_task and c == target_col)]
        roles = resolve_column_roles(
            columns, continuous_cols=modelled, target_col=target_col, task=task, id_col=id_col
        )
        return replace(roles, source="default_all_continuous")

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
    ) -> "VEScoreSDEBaseline":
        """
        Train on exactly the rows of ``data`` (no internal split).  ``transforms`` is accepted for
        signature compatibility and ignored: the frame is modelled in the representation given.
        """
        self._reject_unknown_kwargs(kwargs, f"{type(self).__name__}.fit()")
        cfg = self.cfg
        seed_everything(cfg.seed)   # model init, batch order, time / noise draws

        self._was_dataframe = isinstance(data, pd.DataFrame)
        if self._was_dataframe:
            df = data
            self.columns_ = list(df.columns)
        else:
            arr = np.asarray(data)
            df = pd.DataFrame(arr, columns=list(range(arr.shape[1])))
            self.columns_ = None
        if len(df) == 0:
            raise ValueError("Cannot fit on an empty frame.")
        self._input_columns = list(df.columns)

        roles = self._resolve_roles(df, continuous_cols, discrete_cols, categorical_cols, target_col, task, id_col, schema)
        self.roles_ = roles
        self.role_source_ = roles.source
        self._id_col = roles.id_col
        self._id_factory = FreshIdFactory.fit(df[self._id_col]) if self._id_col is not None else None

        self.label_distribution_ = None
        if roles.target_col is not None and roles.target_col in roles.categorical:
            self.label_distribution_ = empirical_distribution(df[roles.target_col].tolist())

        self._codec = MixedToContinuousCodec(standardize_numeric=cfg.standardize_numeric).fit(df, roles)
        X = self._codec.encode(df)

        n_rows, self._dim = int(X.shape[0]), int(X.shape[1])
        if self._dim == 0:
            raise ValueError("Nothing to model: no columns with a role.")
        self.fit_info_ = BaselineFitInfo(n_rows=int(df.shape[0]), n_cols=int(df.shape[1]), columns=list(df.columns))

        device = torch.device(cfg.device)
        score_net = self._build_net()
        opt = torch.optim.Adam(score_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        X_t = torch.from_numpy(X).to(device)
        batch_size = int(cfg.batch_size)
        self.n_batches_per_epoch_ = max((n_rows + batch_size - 1) // batch_size, 1)   # partial batch kept
        total_steps = cfg.effective_budget(self.n_batches_per_epoch_)
        self.total_steps_ = int(total_steps)

        order_gen = torch.Generator(device="cpu")
        order_gen.manual_seed(int(cfg.seed))

        def epoch_batches():
            perm = torch.randperm(n_rows, generator=order_gen).to(device)
            for start in range(0, n_rows, batch_size):
                yield X_t[perm[start: start + batch_size]]

        score_net.train()
        batches = epoch_batches()
        n_done = 0
        last_loss = None
        for step in range(total_steps):
            try:
                x_batch = next(batches)
            except StopIteration:
                batches = epoch_batches()
                x_batch = next(batches)

            t_max = cfg.curriculum_t_max(step, total_steps)
            B = x_batch.shape[0]
            t = torch.rand(B, device=device) * (t_max - cfg.eps) + cfg.eps
            sigma = self._sigma(t).unsqueeze(1)

            noise = torch.randn_like(x_batch)
            x_noisy = x_batch + sigma * noise

            score_pred = score_net(x_noisy, sigma)
            loss = (sigma * score_pred + noise).pow(2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(score_net.parameters(), cfg.grad_clip)
            opt.step()
            n_done += 1
            last_loss = loss.detach()

        score_net.eval()
        self._score_net = score_net
        self.n_updates_ = n_done
        self.final_loss_ = float(last_loss) if last_loss is not None else None
        return self

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_encoded(self, n: int, seed: Optional[int]) -> np.ndarray:
        cfg = self.cfg
        device = torch.device(cfg.device)
        gen = torch.Generator(device=device)
        if seed is None:
            # continue the global stream: draw a sub-seed from it
            gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        else:
            gen.manual_seed(int(seed))

        def randn(shape) -> torch.Tensor:
            return torch.randn(shape, generator=gen, device=device)

        net = self._score_net
        net.eval()
        N = int(cfg.n_sampling_steps)
        ts = torch.linspace(1.0, cfg.eps, N + 1, device=device)
        x = randn((n, self._dim)) * cfg.sigma_max

        for i in range(N):
            sigma_cur = self._sigma(ts[i])
            diff = sigma_cur ** 2 - self._sigma(ts[i + 1]) ** 2
            sigma_cur_vec = sigma_cur.view(1, 1).expand(n, 1)

            for _ in range(int(cfg.n_corrector_steps)):
                score = net(x, sigma_cur_vec)
                noise = randn(x.shape)
                grad_norm = score.norm(dim=-1).mean().clamp(min=1e-8)
                noise_norm = noise.norm(dim=-1).mean().clamp(min=1e-8)
                alpha = 2.0 * (cfg.corrector_snr * noise_norm / grad_norm) ** 2
                x = x + alpha * score + (2.0 * alpha).sqrt() * noise

            score = net(x, sigma_cur_vec)
            x = x + diff * score + diff.sqrt() * randn(x.shape)

        if cfg.denoise:
            sigma_last = self._sigma(ts[-1]).view(1, 1).expand(n, 1)
            x = x + (sigma_last ** 2) * net(x, sigma_last)

        return x.detach().cpu().numpy()

    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> ArrayLike:
        """
        Exactly ``n`` rows in the fit frame's columns / order / dtypes.  All sampling noise comes
        from a private generator seeded with ``seed`` (``None`` -> a sub-seed drawn from the global
        torch stream), so ``sample(n, seed)`` is reproducible and independent of global RNG state.
        """
        self._reject_unknown_kwargs(kwargs, f"{type(self).__name__}.sample()")
        if self._score_net is None or self._dim is None or self._codec is None:
            raise RuntimeError("Call fit() before sample().")
        n = validate_n(n)

        z = self._sample_encoded(n, seed)
        out, self.decoding_report_ = self._codec.decode(z)
        if self._id_col is not None:
            out[self._id_col] = (self._id_factory or FreshIdFactory()).make(n)
        out = out[self._input_columns]
        if len(out) != n:
            raise RuntimeError(f"Produced {len(out)} rows, expected exactly {n}.")
        if not self._was_dataframe:
            return out.to_numpy(dtype=np.float32)
        return out

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> str:
        """Single ``torch.save`` file (tensors + plain python; loaded with ``weights_only=True``)."""
        if self._score_net is None or self._codec is None:
            raise RuntimeError("Call fit() before save_checkpoint().")
        state = {
            "format": _CHECKPOINT_FORMAT,
            "format_version": _CHECKPOINT_VERSION,
            "variant_id": self.variant_id,
            "is_faithful_stasy": False,
            "cfg": asdict(self.cfg),
            "total_steps": int(self.total_steps_),
            "n_updates": int(self.n_updates_),
            "n_batches_per_epoch": int(self.n_batches_per_epoch_),
            "final_loss": self.final_loss_,
            "dim": int(self._dim),
            "input_columns": list(self._input_columns),
            "was_dataframe": bool(self._was_dataframe),
            "roles": self.roles_.to_dict() if self.roles_ is not None else None,
            "codec": self._codec.state_dict(),
            "label_distribution": self.label_distribution_,
            "id_col": self._id_col,
            "id_factory": self._id_factory.state_dict() if self._id_factory is not None else None,
            "fit_info": asdict(self.fit_info_) if self.fit_info_ is not None else None,
            "score_net": {k: v.detach().cpu() for k, v in self._score_net.state_dict().items()},
            "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__), "pandas": str(pd.__version__)},
        }
        torch.save(state, path)
        return str(path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None, **kwargs: Any) -> "VEScoreSDEBaseline":
        cls._reject_unknown_kwargs(kwargs, f"{cls.__name__}.load_checkpoint()")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"{path!r} is not a VE score-SDE checkpoint.")
        if int(state.get("format_version", -1)) != _CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {state.get('format_version')!r}.")

        known = {f.name for f in fields(VEScoreSDEConfig)}
        cfg_dict = {k: v for k, v in state["cfg"].items() if k in known}
        if device is not None:
            cfg_dict["device"] = device
        elif str(cfg_dict.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("Checkpoint was trained on CUDA but CUDA is unavailable; loading on CPU.", UserWarning)
            cfg_dict["device"] = "cpu"
        # always rebuild as the honest class, even if a deprecated alias saved the checkpoint
        obj = VEScoreSDEBaseline(VEScoreSDEConfig(**cfg_dict))

        obj._dim = int(state["dim"])
        obj._input_columns = list(state["input_columns"])
        obj._was_dataframe = bool(state["was_dataframe"])
        obj.columns_ = list(state["input_columns"]) if obj._was_dataframe else None
        obj.roles_ = ColumnRoles.from_dict(state["roles"]) if state["roles"] is not None else None
        obj.role_source_ = obj.roles_.source if obj.roles_ is not None else None
        obj._codec = MixedToContinuousCodec.from_state(state["codec"])
        obj.label_distribution_ = state["label_distribution"]
        obj._id_col = state["id_col"]
        obj._id_factory = FreshIdFactory.from_state(state["id_factory"])
        obj.total_steps_ = int(state["total_steps"])
        obj.n_updates_ = int(state["n_updates"])
        obj.n_batches_per_epoch_ = int(state["n_batches_per_epoch"])
        obj.final_loss_ = state.get("final_loss")
        if state.get("fit_info") is not None:
            obj.fit_info_ = BaselineFitInfo(**state["fit_info"])

        net = obj._build_net()
        net.load_state_dict(state["score_net"])
        net.eval()
        obj._score_net = net
        return obj


# ----------------------------------------------------------------------
# DEPRECATED aliases - kept importable, but they say what they are
# ----------------------------------------------------------------------


class STaSyConfig(VEScoreSDEConfig):
    """
    DEPRECATED alias of :class:`VEScoreSDEConfig` (NOT a faithful STaSy configuration).

    ``use_self_paced`` maps onto ``time_curriculum`` (a diffusion-time curriculum, not STaSy's
    per-sample self-paced learning) and now defaults to ``False``.  For backward compatibility
    only, a missing budget falls back to the old default ``n_epochs=100``.
    """

    def __init__(self, *args: Any, use_self_paced: Optional[bool] = None, **kwargs: Any) -> None:
        warnings.warn(_NOT_STASY.format(old="STaSyConfig", new="VEScoreSDEConfig"), DeprecationWarning, stacklevel=2)
        if use_self_paced is not None:
            if "time_curriculum" in kwargs:
                raise TypeError("Pass either `time_curriculum` or the deprecated `use_self_paced`, not both.")
            warnings.warn(
                "`use_self_paced` is DEPRECATED and misnamed: it enables a diffusion-TIME curriculum "
                "(`time_curriculum`), NOT STaSy's per-sample self-paced learning.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["time_curriculum"] = bool(use_self_paced)
        n_positional_budget = len(args) > 6   # n_epochs is the 7th positional field
        if not n_positional_budget and kwargs.get("n_epochs") is None and kwargs.get("steps") is None:
            kwargs["n_epochs"] = 100
        super().__init__(*args, **kwargs)

    @property
    def use_self_paced(self) -> bool:
        return bool(self.time_curriculum)


class STaSyGenerative(VEScoreSDEBaseline):
    """DEPRECATED alias of :class:`VEScoreSDEBaseline` - a simplified VE score-SDE, NOT STaSy."""

    def __init__(self, cfg: VEScoreSDEConfig) -> None:
        warnings.warn(_NOT_STASY.format(old="STaSyGenerative", new="VEScoreSDEBaseline"), DeprecationWarning, stacklevel=2)
        super().__init__(cfg)
