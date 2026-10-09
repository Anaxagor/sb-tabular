"""TabbyFlow (TabVFM-MLP) baseline used in the final experiments.

Adapted from the Apache-2.0 tabular-flow-matching reference implementation:
https://github.com/rulnasution/tabular-flow-matching, baselines/tabvvfm.

The default OT path has a standard Gaussian source and residual endpoint noise
of 0.001. VP uses the usual small-residual Gaussian source approximation. VE
uses N(0, 4I) as an approximation to data + N(0, 4I), not an exact Gaussian
source; use OT or cosine when an exact Gaussian source is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sbtab.data.schema import TabularSchema
from sbtab.baselines.base import FreshIdFactory, resolve_column_roles, to_python_scalar, validate_n
from sbtab.baselines.encoding import fit_vocabulary, nearest_support_decode, numeric_matrix, restore_dtype
from sbtab.numerics import check_gradients, require_finite


TABBYFLOW_OFFICIAL_REPOSITORY = "https://github.com/rulnasution/tabular-flow-matching"
TABBYFLOW_OFFICIAL_VARIANT = "baselines/tabvvfm (MLP TabVFM/TabbyFlow)"
TABBYFLOW_CHECKPOINT_FORMAT = "sbtab.tabbyflow/1"


@dataclass
class TabbyFlowConfig:
    max_train_steps: int = 2000
    batch_size: int = 1024
    n_frequencies: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    cond_vel: str = "ot"
    ode_solver: str = "euler"
    ode_steps: int = 100
    scheduler_factor: float = 0.9
    scheduler_patience_epochs: int = 20
    early_stopping_patience_epochs: int = 200
    sample_batch_size: int = 512
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("max_train_steps", "batch_size", "n_frequencies", "ode_steps", "sample_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if str(self.cond_vel).lower() not in {"ot", "vp", "ve", "cos"}:
            raise ValueError(f"Unsupported TabbyFlow conditional path: {self.cond_vel!r}")
        if str(self.ode_solver).lower() not in {"euler", "midpoint", "rk4"}:
            raise ValueError(f"Unknown ODE method: {self.ode_solver!r}")
        for name in ("lr", "weight_decay", "scheduler_factor"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.lr <= 0 or self.weight_decay < 0 or not 0 < self.scheduler_factor < 1:
            raise ValueError("lr must be positive, weight_decay nonnegative, scheduler_factor in (0, 1)")
        for name, minimum in (("scheduler_patience_epochs", 0), ("early_stopping_patience_epochs", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("TabbyFlow supports cpu or cuda devices")
        # Keep checkpoint config independent of NumPy scalar pickle globals.
        for name in ("max_train_steps", "batch_size", "n_frequencies", "ode_steps", "sample_batch_size",
                     "scheduler_patience_epochs", "early_stopping_patience_epochs", "seed"):
            setattr(self, name, int(getattr(self, name)))
        for name in ("lr", "weight_decay", "scheduler_factor"):
            setattr(self, name, float(getattr(self, name)))
        for name in ("cond_vel", "ode_solver", "device"):
            setattr(self, name, str(getattr(self, name)))


def _checkpoint_category(value: Any, column: str) -> Any:
    """The standalone categorical API accepts scalar labels portable to weights-only checkpoints."""
    value = to_python_scalar(value)
    if value is not None and type(value) not in {bool, int, float, str}:
        raise TypeError(f"TabbyFlow categorical column {column!r} has unsupported label type "
                        f"{type(value).__name__}; use string, bool, integer or float labels")
    return value


class TabbyFlowNet(nn.Module):
    """Official tabvvfm Net: fixed 3-layer MLP with sinusoidal time embedding."""

    def __init__(self, in_dim: int, n_frequencies: int) -> None:
        super().__init__()
        dim_t = 2 * int(n_frequencies)
        ins = [dim_t, dim_t * 2, dim_t * 2]
        outs = [dim_t * 2, dim_t * 2, dim_t]
        self.n_frequencies = int(n_frequencies)
        self.proj = nn.Linear(int(in_dim), dim_t)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_d, out_d), nn.SiLU())
            for in_d, out_d in zip(ins, outs)
        ])
        self.top = nn.Linear(dim_t, int(in_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

    def time_encoder(self, t: torch.Tensor) -> torch.Tensor:
        freq = (
            2
            * torch.arange(self.n_frequencies, device=t.device, dtype=t.dtype)
            * torch.pi
        )
        phase = freq * t[..., None]
        return torch.cat((phase.cos(), phase.sin()), dim=-1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(self.time_encoder(t))
        h = self.proj(x) + emb
        for layer in self.layers:
            h = layer(h)
        return self.top(h)


class TabbyFlowConditionalPath:
    """Official OT/VP/VE/cos interpolation families with analytic derivatives."""

    def __init__(self, name: str) -> None:
        name = str(name).lower()
        if name not in {"ot", "vp", "ve", "cos"}:
            raise ValueError(f"Unsupported TabbyFlow conditional path: {name!r}")
        self.name = name
        self.eps = 1e-5

    def coefficients(
        self,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = t.reshape(-1, 1)

        if self.name == "ot":
            t_use = t.clamp(0.0, 1.0)
            alpha = t_use
            beta = 1.0 - 0.999 * t_use
            d_alpha = torch.ones_like(t_use)
            d_beta = torch.full_like(t_use, -0.999)

        elif self.name == "vp":
            t_use = t.clamp(0.0, 1.0 - self.eps)
            beta_min, beta_max = 0.1, 20.0
            s = 1.0 - t_use
            beta_schedule = beta_min + s * (beta_max - beta_min)
            integrated_beta = beta_min * s + 0.5 * s.square() * (beta_max - beta_min)
            alpha = torch.exp(-0.5 * integrated_beta)
            beta = torch.sqrt(torch.clamp(1.0 - alpha.square(), min=1e-10))
            d_alpha = 0.5 * beta_schedule * alpha
            d_beta = -(alpha * d_alpha) / beta

        elif self.name == "ve":
            t_use = t.clamp(0.0, 1.0)
            sigma_min, sigma_max = 0.01, 2.0
            log_ratio = math.log(sigma_max / sigma_min)
            alpha = torch.ones_like(t_use)
            beta = sigma_min * torch.exp((1.0 - t_use) * log_ratio)
            d_alpha = torch.zeros_like(t_use)
            d_beta = -log_ratio * beta

        else:  # cosine interpolation from the official repository
            t_use = t.clamp(0.0, 1.0 - self.eps)
            half_pi = 0.5 * math.pi
            alpha = torch.sin(half_pi * t_use)
            beta = torch.cos(half_pi * t_use)
            d_alpha = half_pi * torch.cos(half_pi * t_use)
            d_beta = -half_pi * torch.sin(half_pi * t_use)

        beta_safe = beta.clamp_min(1e-8)
        p2 = d_beta / beta_safe
        atx = d_alpha - p2 * alpha
        atx = torch.nan_to_num(atx, nan=1.0, posinf=1e6, neginf=1e-6)
        atx = atx.clamp(1e-6, 1e6)
        p2 = torch.nan_to_num(p2, nan=0.0, posinf=1e6, neginf=-1e6)
        return alpha, beta, atx, p2


class TabbyFlowMatchingLoss(nn.Module):
    """Variational flow-matching loss from baselines/tabvvfm/flow_matching.py."""

    def __init__(self, d_cont: int, cat_sizes: List[int], path: TabbyFlowConditionalPath) -> None:
        super().__init__()
        self.d_cont = int(d_cont)
        self.cat_sizes = [int(v) for v in cat_sizes]
        self.path = path
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, net: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        t = torch.rand((x_1.shape[0], 1), device=x_1.device) * (1.0 - 1e-5)
        x_0 = torch.randn_like(x_1)
        alpha, beta, atx, _ = self.path.coefficients(t)
        x_t = alpha * x_1 + beta * x_0
        theta = net(t[:, 0], x_t)

        loss = torch.zeros((), device=x_1.device, dtype=x_1.dtype)
        if self.d_cont > 0:
            std = torch.rsqrt(2.0 * atx).expand(-1, self.d_cont)
            dist = Normal(theta[:, : self.d_cont], std)
            loss = loss + torch.mean(-dist.log_prob(x_1[:, : self.d_cont]))

        start = self.d_cont
        for width in self.cat_sizes:
            end = start + width
            loss = loss + self.cross_entropy(
                theta[:, start:end],
                torch.argmax(x_1[:, start:end], dim=-1),
            )
            start = end
        return loss


class TabbyFlowODEField(nn.Module):
    """Conditional vector field used by the repository's CondVF decoder."""

    def __init__(
        self,
        net: nn.Module,
        path: TabbyFlowConditionalPath,
        d_cont: int,
        cat_sizes: List[int],
    ) -> None:
        super().__init__()
        self.net = net
        self.path = path
        self.d_cont = int(d_cont)
        self.cat_sizes = [int(v) for v in cat_sizes]

    def forward(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = torch.full(
            (x.shape[0],),
            float(t_scalar),
            device=x.device,
            dtype=x.dtype,
        )
        theta = self.net(t, x)
        parts: List[torch.Tensor] = []
        if self.d_cont > 0:
            parts.append(theta[:, : self.d_cont])
        start = self.d_cont
        for width in self.cat_sizes:
            end = start + width
            parts.append(F.softmax(theta[:, start:end], dim=-1))
            start = end
        x_1_hat = torch.cat(parts, dim=1)
        _, _, atx, p2 = self.path.coefficients(t)
        return atx * x_1_hat + p2 * x


def integrate_tabbyflow_fixed_step(
    field: TabbyFlowODEField,
    x: torch.Tensor,
    *,
    n_steps: int,
    method: str,
) -> torch.Tensor:
    """Fixed-step Euler/midpoint/RK4, matching the methods exposed by CondVF."""
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    method = str(method).lower()
    if method not in {"euler", "midpoint", "rk4"}:
        raise ValueError(f"Unknown ODE method: {method!r}")

    dt = 1.0 / float(n_steps)
    for step in range(n_steps):
        t = torch.tensor(step * dt, device=x.device, dtype=x.dtype)
        if method == "euler":
            x = x + dt * field(t, x)
        elif method == "midpoint":
            k1 = field(t, x)
            t_mid = t + 0.5 * dt
            k2 = field(t_mid, x + 0.5 * dt * k1)
            x = x + dt * k2
        else:
            k1 = field(t, x)
            k2 = field(t + 0.5 * dt, x + 0.5 * dt * k1)
            k3 = field(t + 0.5 * dt, x + 0.5 * dt * k2)
            k4 = field(t + dt, x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


class TabbyFlowSynthesizer:
    """Joint MLP TabVFM/TabbyFlow with inference-complete, train-fitted checkpoints."""

    variant_id = "tabbyflow_tabvfm_mlp_joint_xy"

    def __init__(self, cfg: TabbyFlowConfig) -> None:
        self.cfg = cfg
        requested = str(cfg.device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for TabbyFlow but is unavailable")
        self.device = torch.device(requested)
        self.net: Optional[TabbyFlowNet] = None
        self.path: Optional[TabbyFlowConditionalPath] = None
        self.quantile: Optional[QuantileTransformer] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.columns_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.cat_sizes_: List[int] = []
        self.cat_decode_maps_: Dict[str, Dict[str, Any]] = {}
        self.raw_dtypes_: Dict[str, Any] = {}
        self.discrete_supports_: Dict[str, np.ndarray] = {}
        self.decoding_report_: Dict[str, Dict[str, Any]] = {}
        self._id_col: Optional[str] = None
        self._id_factory: Optional[FreshIdFactory] = None
        self.d_cont_: int = 0
        self.d_total_: int = 0
        self.actual_train_steps_: int = 0
        self.best_train_loss_: float = float("inf")
        self._fitted = False

    def _column_groups(
        self,
        data: pd.DataFrame,
        schema: TabularSchema,
        task_type: str,
    ) -> Tuple[List[str], List[str]]:
        task = str(task_type).lower()
        if task in {"binclass", "multiclass"}:
            task = "classification"
        roles = resolve_column_roles(
            list(data.columns),
            continuous_cols=schema.continuous_cols,
            discrete_cols=schema.discrete_cols,
            categorical_cols=schema.categorical_cols,
            target_col=schema.target_col,
            task=task,
            id_col=schema.id_col,
        )
        self._id_col = roles.id_col
        self._id_factory = FreshIdFactory.fit(data[self._id_col]) if self._id_col is not None else None
        self.discrete_supports_ = {
            col: np.unique(pd.to_numeric(data[col], errors="raise").to_numpy(dtype=np.float64))
            for col in roles.discrete
        }
        # An undeclared feature must not silently become an independent bootstrap
        # marginal. resolve_column_roles rejects it; only identifiers are excluded.
        return list(roles.numeric), list(roles.categorical)

    def _fit_preprocessor(
        self,
        data: pd.DataFrame,
        schema: TabularSchema,
        task_type: str,
    ) -> np.ndarray:
        self.quantile = None
        self.encoder = None
        self.cat_decode_maps_ = {}
        self.columns_ = list(data.columns)
        self.raw_dtypes_ = {col: data[col].dtype for col in data.columns}
        (
            self.numeric_cols_,
            self.categorical_cols_,
        ) = self._column_groups(data, schema, task_type)

        # Validate every value that will be persisted, including unused levels of
        # pandas categorical dtypes, before fitting any transform or network.
        for col in self.categorical_cols_:
            for value in pd.unique(data[col]):
                _checkpoint_category(value, col)
            if isinstance(data[col].dtype, pd.CategoricalDtype):
                for value in data[col].dtype.categories:
                    _checkpoint_category(value, col)

        if self.numeric_cols_:
            x_num = numeric_matrix(data, self.numeric_cols_)
            self.quantile = QuantileTransformer(
                n_quantiles=max(1, min(1000, len(data))),
                output_distribution="uniform",
                random_state=int(self.cfg.seed),
            )
            x_cont = self.quantile.fit_transform(x_num).astype(np.float32)
        else:
            x_cont = np.empty((len(data), 0), dtype=np.float32)

        if self.categorical_cols_:
            cat_frame = pd.DataFrame(index=data.index)
            for col in self.categorical_cols_:
                vocabulary = fit_vocabulary(data[col], col)
                cat_frame[col] = pd.Categorical(data[col], categories=vocabulary).codes
                self.cat_decode_maps_[col] = {str(i): value for i, value in enumerate(vocabulary)}
            try:
                self.encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )
            except TypeError:
                self.encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                )
            x_cat = self.encoder.fit_transform(cat_frame).astype(np.float32)
            self.cat_sizes_ = [len(values) for values in self.encoder.categories_]
        else:
            x_cat = np.empty((len(data), 0), dtype=np.float32)
            self.cat_sizes_ = []

        self.d_cont_ = int(x_cont.shape[1])
        matrix = np.concatenate([x_cont, x_cat], axis=1).astype(np.float32)
        self.d_total_ = int(matrix.shape[1])
        if self.d_total_ <= 0:
            raise ValueError("TabbyFlow found no modelled columns")
        return matrix

    def fit(
        self,
        data: pd.DataFrame,
        *,
        schema: TabularSchema,
        task_type: str,
    ) -> "TabbyFlowSynthesizer":
        # Any failed refit invalidates the old fitted model, including validation failures.
        self._fitted = False
        self.net = None
        self.actual_train_steps_ = 0
        devices = list(range(torch.cuda.device_count())) if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            return self._fit(data, schema=schema, task_type=task_type)

    def _fit(self, data: pd.DataFrame, *, schema: TabularSchema, task_type: str) -> "TabbyFlowSynthesizer":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("TabbyFlowSynthesizer.fit expects a pandas DataFrame")
        if len(data) < 2:
            raise ValueError("TabbyFlow requires at least two rows")

        self._fitted = False
        self.best_train_loss_ = float("inf")
        self.actual_train_steps_ = 0
        if str(self.cfg.cond_vel).lower() == "ve":
            warnings.warn(
                "TabbyFlow VE starts from N(0, 4I), an approximation to the path's "
                "data + N(0, 4I) source. OT and cosine have an exact Gaussian source.",
                UserWarning,
                stacklevel=2,
            )

        seed = int(self.cfg.seed)
        torch.random.default_generator.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        matrix = self._fit_preprocessor(data.reset_index(drop=True), schema, task_type)
        tensor = torch.from_numpy(matrix)
        loader_generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=min(int(self.cfg.batch_size), len(tensor)),
            shuffle=True,
            drop_last=False,
            generator=loader_generator,
        )

        self.net = TabbyFlowNet(self.d_total_, int(self.cfg.n_frequencies)).to(self.device)
        self.path = TabbyFlowConditionalPath(self.cfg.cond_vel)
        loss_fn = TabbyFlowMatchingLoss(self.d_cont_, self.cat_sizes_, self.path)
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.cfg.scheduler_factor),
            patience=int(self.cfg.scheduler_patience_epochs),
        )

        max_steps = int(self.cfg.max_train_steps)
        max_epochs = max(1, math.ceil(max_steps / max(1, len(loader))))
        best_state = None
        stale_epochs = 0
        steps = 0
        self.net.train()

        for _epoch in range(max_epochs):
            epoch_sum = 0.0
            epoch_rows = 0
            for (x_batch,) in loader:
                if steps >= max_steps:
                    break
                x_batch = x_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.net, x_batch)
                context = dict(model=self.variant_id, stage="training", step=steps)
                require_finite(loss, "loss", **context)
                loss.backward()
                check_gradients(self.net.parameters(), **context)
                optimizer.step()
                epoch_sum += float(loss.detach().cpu()) * len(x_batch)
                epoch_rows += len(x_batch)
                steps += 1

            if epoch_rows == 0:
                break
            epoch_loss = epoch_sum / epoch_rows
            scheduler.step(epoch_loss)
            if epoch_loss < self.best_train_loss_:
                self.best_train_loss_ = float(epoch_loss)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.net.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if (
                stale_epochs >= int(self.cfg.early_stopping_patience_epochs)
                and steps >= min(500, max_steps)
            ):
                break
            if steps >= max_steps:
                break

        if best_state is None:
            raise RuntimeError("TabbyFlow did not produce a valid checkpoint")
        self.net.load_state_dict(best_state)
        self.net.to(self.device).eval()
        self.actual_train_steps_ = int(steps)
        self._fitted = True
        return self

    def _decode(self, latent: np.ndarray, *, seed: int) -> pd.DataFrame:
        if latent.ndim != 2 or latent.shape[1] != self.d_total_ or not np.isfinite(latent).all():
            raise FloatingPointError("TabbyFlow produced invalid or non-finite latent samples")
        n = len(latent)
        out = pd.DataFrame(index=np.arange(n))
        self.decoding_report_ = {}

        if self.numeric_cols_:
            if self.quantile is None:
                raise RuntimeError("Missing fitted QuantileTransformer")
            numeric = self.quantile.inverse_transform(latent[:, : self.d_cont_])
            if not np.isfinite(numeric).all():
                raise FloatingPointError("TabbyFlow inverse quantile transform produced non-finite samples")
            for idx, col in enumerate(self.numeric_cols_):
                values = numeric[:, idx]
                if col in self.discrete_supports_:
                    values, self.decoding_report_[col] = nearest_support_decode(values, self.discrete_supports_[col])
                out[col] = restore_dtype(values, str(self.raw_dtypes_[col]))

        if self.categorical_cols_:
            start = self.d_cont_
            for col, width in zip(self.categorical_cols_, self.cat_sizes_):
                end = start + width
                idx = np.argmax(latent[:, start:end], axis=1)
                # Full one-hot blocks are ordered by integer vocabulary position.
                # Inference needs only that mapping, not a fitted sklearn encoder.
                mapping = self.cat_decode_maps_[col]
                out[col] = [mapping[str(int(value))] for value in idx]
                try:
                    out[col] = out[col].astype(self.raw_dtypes_[col])
                except Exception:
                    pass
                start = end

        if self._id_col is not None:
            out[self._id_col] = self._id_factory.make(n)

        return out.reindex(columns=self.columns_)

    def sample(self, n: int, *, seed: int) -> pd.DataFrame:
        if not self._fitted or self.net is None or self.path is None:
            raise RuntimeError("Call fit() before sample()")
        n = validate_n(n)
        if int(self.cfg.sample_batch_size) < 1:
            raise ValueError("sample_batch_size must be positive")

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        field = TabbyFlowODEField(
            self.net,
            self.path,
            self.d_cont_,
            self.cat_sizes_,
        ).to(self.device)
        chunks = []
        remaining = int(n)
        with torch.inference_mode():
            _, source_std, _, _ = self.path.coefficients(torch.zeros(1, device=self.device))
            while remaining > 0:
                size = min(int(self.cfg.sample_batch_size), remaining)
                x_0 = source_std * torch.randn(size, self.d_total_, device=self.device, generator=generator)
                x_1 = integrate_tabbyflow_fixed_step(
                    field,
                    x_0,
                    n_steps=int(self.cfg.ode_steps),
                    method=str(self.cfg.ode_solver),
                )
                chunks.append(x_1.detach().cpu().numpy())
                remaining -= size
        latent = np.concatenate(chunks, axis=0)[: int(n)]
        return self._decode(latent, seed=int(seed))

    @property
    def n_updates(self) -> int:
        return self.actual_train_steps_

    def save_checkpoint(self, path) -> str:
        """Save inference state as tensors and plain values; no training rows or optimizer."""
        if not self._fitted or self.net is None:
            raise RuntimeError("Call fit() before save_checkpoint()")
        dtypes = {}
        for col, dtype in self.raw_dtypes_.items():
            dtypes[col] = {"name": str(dtype)}
            if isinstance(dtype, pd.CategoricalDtype):
                dtypes[col].update(categories=[_checkpoint_category(v, col) for v in dtype.categories], ordered=dtype.ordered)
        quantile = None if self.quantile is None else {
            "quantiles": torch.from_numpy(self.quantile.quantiles_.copy()),
            "references": torch.from_numpy(self.quantile.references_.copy()),
            "n_quantiles": int(self.quantile.n_quantiles_),
            "n_features": int(self.quantile.n_features_in_),
        }
        state = {"format": TABBYFLOW_CHECKPOINT_FORMAT, "variant_id": self.variant_id,
                 "config": asdict(TabbyFlowConfig(**asdict(self.cfg))),
                 "columns": self.columns_, "numeric_columns": self.numeric_cols_,
                 "categorical_columns": self.categorical_cols_, "categorical_sizes": self.cat_sizes_,
                 "categorical_maps": {c: {k: _checkpoint_category(v, c) for k, v in m.items()}
                                      for c, m in self.cat_decode_maps_.items()},
                 "dtypes": dtypes, "quantile": quantile,
                 "discrete_supports": {c: torch.from_numpy(s.copy()) for c, s in self.discrete_supports_.items()},
                 "id_column": self._id_col,
                 "id_factory": None if self._id_factory is None else self._id_factory.state_dict(),
                 "d_cont": self.d_cont_, "d_total": self.d_total_, "n_updates": self.actual_train_steps_,
                 "best_train_loss": self.best_train_loss_,
                 "network": {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        return str(path)

    @classmethod
    def load_checkpoint(cls, path, *, device: Optional[str] = None) -> "TabbyFlowSynthesizer":
        """Restore inference directly; quantiles, vocabularies and supports are never refitted."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != TABBYFLOW_CHECKPOINT_FORMAT or state.get("variant_id") != cls.variant_id:
            raise ValueError("unsupported TabbyFlow checkpoint")
        cfg = dict(state["config"])
        if device is not None:
            cfg["device"] = str(device)
        elif str(cfg["device"]).startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA checkpoint is being loaded on CPU for inference", UserWarning)
            cfg["device"] = "cpu"
        obj = cls(TabbyFlowConfig(**cfg))
        obj.columns_ = list(state["columns"])
        obj.numeric_cols_ = list(state["numeric_columns"])
        obj.categorical_cols_ = list(state["categorical_columns"])
        obj.cat_sizes_ = list(state["categorical_sizes"])
        obj.cat_decode_maps_ = state["categorical_maps"]
        obj.raw_dtypes_ = {c: pd.CategoricalDtype(s["categories"], ordered=s["ordered"])
                           if "categories" in s else pd.api.types.pandas_dtype(s["name"])
                           for c, s in state["dtypes"].items()}
        obj.discrete_supports_ = {c: s.numpy().copy() for c, s in state["discrete_supports"].items()}
        obj._id_col = state["id_column"]
        obj._id_factory = FreshIdFactory.from_state(state["id_factory"])
        obj.d_cont_, obj.d_total_ = int(state["d_cont"]), int(state["d_total"])
        obj.actual_train_steps_ = int(state["n_updates"])
        obj.best_train_loss_ = float(state["best_train_loss"])
        if state["quantile"] is not None:
            q = state["quantile"]
            obj.quantile = QuantileTransformer(n_quantiles=q["n_quantiles"], output_distribution="uniform",
                                               random_state=int(obj.cfg.seed))
            obj.quantile.quantiles_ = q["quantiles"].numpy().copy()
            obj.quantile.references_ = q["references"].numpy().copy()
            obj.quantile.n_quantiles_ = int(q["n_quantiles"])
            obj.quantile.n_features_in_ = int(q["n_features"])
        with torch.random.fork_rng(devices=[]):
            obj.net = TabbyFlowNet(obj.d_total_, int(obj.cfg.n_frequencies))
        obj.net.load_state_dict(state["network"])
        obj.net.to(obj.device).eval()
        obj.path = TabbyFlowConditionalPath(obj.cfg.cond_vel)
        obj._fitted = True
        return obj
