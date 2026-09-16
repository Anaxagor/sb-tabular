"""TabbyFlow (TabVFM-MLP) baseline used in the final experiments.

Adapted from the Apache-2.0 tabular-flow-matching reference implementation:
https://github.com/rulnasution/tabular-flow-matching, baselines/tabvvfm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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


TABBYFLOW_OFFICIAL_REPOSITORY = "https://github.com/rulnasution/tabular-flow-matching"
TABBYFLOW_OFFICIAL_VARIANT = "baselines/tabvvfm (MLP TabVFM/TabbyFlow)"


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
    """Notebook wrapper around the official MLP TabVFM/TabbyFlow implementation."""

    def __init__(self, cfg: TabbyFlowConfig) -> None:
        self.cfg = cfg
        requested = str(cfg.device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.net: Optional[TabbyFlowNet] = None
        self.path: Optional[TabbyFlowConditionalPath] = None
        self.quantile: Optional[QuantileTransformer] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.columns_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.extra_cols_: List[str] = []
        self.cat_sizes_: List[int] = []
        self.cat_decode_maps_: Dict[str, Dict[str, Any]] = {}
        self.raw_dtypes_: Dict[str, Any] = {}
        self.extra_values_: Dict[str, np.ndarray] = {}
        self.d_cont_: int = 0
        self.d_total_: int = 0
        self.actual_train_steps_: int = 0
        self.best_train_loss_: float = float("inf")
        self._fitted = False

    @staticmethod
    def _unique_existing(cols: List[str], frame: pd.DataFrame) -> List[str]:
        seen = set()
        out = []
        for col in cols:
            if col in frame.columns and col not in seen:
                out.append(col)
                seen.add(col)
        return out

    def _column_groups(
        self,
        data: pd.DataFrame,
        schema: TabularSchema,
        task_type: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        numeric = [*schema.continuous_cols, *schema.discrete_cols]
        categorical = list(schema.categorical_cols)
        target = schema.target_col
        if target is not None and target in data.columns:
            if str(task_type).lower() == "regression":
                numeric.append(target)
            else:
                categorical.append(target)
        numeric = self._unique_existing(numeric, data)
        categorical = self._unique_existing(categorical, data)
        categorical = [col for col in categorical if col not in set(numeric)]
        extras = [
            col for col in data.columns
            if col not in set(numeric) and col not in set(categorical)
        ]
        return numeric, categorical, extras

    def _fit_preprocessor(
        self,
        data: pd.DataFrame,
        schema: TabularSchema,
        task_type: str,
    ) -> np.ndarray:
        self.columns_ = list(data.columns)
        self.raw_dtypes_ = {col: data[col].dtype for col in data.columns}
        (
            self.numeric_cols_,
            self.categorical_cols_,
            self.extra_cols_,
        ) = self._column_groups(data, schema, task_type)

        if self.numeric_cols_:
            x_num_frame = data[self.numeric_cols_].apply(pd.to_numeric, errors="raise")
            self.quantile = QuantileTransformer(
                n_quantiles=max(1, min(1000, len(data))),
                output_distribution="uniform",
                random_state=int(self.cfg.seed),
            )
            x_cont = self.quantile.fit_transform(x_num_frame).astype(np.float32)
        else:
            x_cont = np.empty((len(data), 0), dtype=np.float32)

        if self.categorical_cols_:
            cat_frame = data[self.categorical_cols_].astype(str)
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
            for col in self.categorical_cols_:
                mapping: Dict[str, Any] = {}
                for value in data[col].drop_duplicates().tolist():
                    mapping.setdefault(str(value), value)
                self.cat_decode_maps_[col] = mapping
        else:
            x_cat = np.empty((len(data), 0), dtype=np.float32)
            self.cat_sizes_ = []

        self.extra_values_ = {
            col: data[col].to_numpy(copy=True)
            for col in self.extra_cols_
        }
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
        if not isinstance(data, pd.DataFrame):
            raise TypeError("TabbyFlowSynthesizer.fit expects a pandas DataFrame")
        if len(data) < 2:
            raise ValueError("TabbyFlow requires at least two rows")

        seed = int(self.cfg.seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
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
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite TabbyFlow training loss")
                loss.backward()
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
        n = len(latent)
        out = pd.DataFrame(index=np.arange(n))

        if self.numeric_cols_:
            if self.quantile is None:
                raise RuntimeError("Missing fitted QuantileTransformer")
            numeric = self.quantile.inverse_transform(latent[:, : self.d_cont_])
            for idx, col in enumerate(self.numeric_cols_):
                out[col] = numeric[:, idx]

        if self.categorical_cols_:
            if self.encoder is None:
                raise RuntimeError("Missing fitted OneHotEncoder")
            hard_blocks = []
            start = self.d_cont_
            for width in self.cat_sizes_:
                end = start + width
                idx = np.argmax(latent[:, start:end], axis=1)
                hard_blocks.append(np.eye(width, dtype=np.float32)[idx])
                start = end
            hard = np.concatenate(hard_blocks, axis=1)
            decoded = self.encoder.inverse_transform(hard)
            for idx, col in enumerate(self.categorical_cols_):
                mapping = self.cat_decode_maps_[col]
                fallback = next(iter(mapping.values()))
                out[col] = [mapping.get(str(value), fallback) for value in decoded[:, idx]]
                try:
                    out[col] = out[col].astype(self.raw_dtypes_[col])
                except Exception:
                    pass

        rng = np.random.default_rng(int(seed))
        for col in self.extra_cols_:
            values = self.extra_values_[col]
            out[col] = values[rng.integers(0, len(values), size=n)]

        return out.reindex(columns=self.columns_)

    def sample(self, n: int, *, seed: int) -> pd.DataFrame:
        if not self._fitted or self.net is None or self.path is None:
            raise RuntimeError("Call fit() before sample()")
        if int(n) <= 0:
            raise ValueError("n must be positive")

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        field = TabbyFlowODEField(
            self.net,
            self.path,
            self.d_cont_,
            self.cat_sizes_,
        ).to(self.device)
        chunks = []
        remaining = int(n)
        with torch.inference_mode():
            while remaining > 0:
                size = min(int(self.cfg.sample_batch_size), remaining)
                x_0 = torch.randn(size, self.d_total_, device=self.device)
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
