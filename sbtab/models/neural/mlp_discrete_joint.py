from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class StepMLPJointConfig:
    hidden_dim: int = 256
    n_layers: int = 4
    dropout: float = 0.0

    lr: float = 2e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    n_epochs: int = 1
    grad_clip: Optional[float] = 1.0
    device: str = "cpu"

    feature_mode: Literal["x", "x_x0", "x_t", "x_x0_t"] = "x"


class _PlainMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        d0 = in_dim
        for i in range(n_layers):
            di = d0 if i == 0 else hidden_dim
            layers.append(nn.Linear(di, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPTimeDiscretizedField:
    """
    One plain MLP per discrete time step, predicting a vector drift in R^dim.
    """

    def __init__(self, dim: int, t_grid: np.ndarray, cfg: StepMLPJointConfig):
        self.dim = int(dim)
        self.t_grid = np.asarray(t_grid, dtype=np.float32)
        self.cfg = cfg
        self.models: list[Optional[nn.Module]] = [None for _ in range(len(self.t_grid))]

    def _build_features(
        self,
        x: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
        t: Optional[float] = None,
    ) -> np.ndarray:
        parts = [np.asarray(x, dtype=np.float32)]
        if self.cfg.feature_mode in ("x_x0", "x_x0_t"):
            if x0 is None:
                raise ValueError("feature_mode requires x0")
            parts.append(np.asarray(x0, dtype=np.float32))
        if self.cfg.feature_mode in ("x_t", "x_x0_t"):
            if t is None:
                raise ValueError("feature_mode requires t")
            parts.append(np.full((x.shape[0], 1), float(t), dtype=np.float32))
        return np.concatenate(parts, axis=1)

    def fit_step(self, k: int, X_feat: np.ndarray, y: np.ndarray) -> None:
        device = torch.device(self.cfg.device)
        X_t = torch.from_numpy(np.asarray(X_feat, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))

        model = _PlainMLP(
            in_dim=X_t.shape[1],
            out_dim=self.dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        model.train()
        for _ in range(self.cfg.n_epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.cfg.grad_clip))
                opt.step()

        model.eval()
        self.models[k] = model

    @torch.no_grad()
    def predict_step(
        self,
        k: int,
        x: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        model = self.models[k]
        if model is None:
            raise RuntimeError(f"Model for time step {k} is not trained.")

        t = float(self.t_grid[k])
        X_feat = self._build_features(x, x0=x0, t=t)

        device = next(model.parameters()).device
        X_t = torch.from_numpy(X_feat).to(device)
        pred = model(X_t).cpu().numpy().astype(np.float32)
        return pred