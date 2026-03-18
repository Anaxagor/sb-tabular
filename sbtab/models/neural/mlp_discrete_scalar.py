from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class StepMLPScalarConfig:
    hidden_dim: int = 256
    n_layers: int = 4
    dropout: float = 0.0

    lr: float = 2e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    n_epochs: int = 1
    grad_clip: Optional[float] = 1.0
    device: str = "cpu"


class _PlainScalarMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        d0 = in_dim
        for i in range(n_layers):
            di = d0 if i == 0 else hidden_dim
            layers.append(nn.Linear(di, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPTimeDiscretizedScalar:
    """
    One plain scalar-output MLP per discrete time step.
    """

    def __init__(self, t_grid: np.ndarray, cfg: StepMLPScalarConfig):
        self.t_grid = np.asarray(t_grid, dtype=np.float32)
        self.cfg = cfg
        self.models: list[Optional[nn.Module]] = [None for _ in range(len(self.t_grid))]

    def fit_step(self, k: int, X_feat: np.ndarray, y: np.ndarray) -> None:
        device = torch.device(self.cfg.device)
        X_t = torch.from_numpy(np.asarray(X_feat, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1))

        model = _PlainScalarMLP(
            in_dim=X_t.shape[1],
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
    def predict_step(self, k: int, X_feat: np.ndarray) -> np.ndarray:
        model = self.models[k]
        if model is None:
            raise RuntimeError(f"Scalar model for time step {k} is not trained.")

        device = next(model.parameters()).device
        X_t = torch.from_numpy(np.asarray(X_feat, dtype=np.float32)).to(device)
        pred = model(X_t).cpu().numpy().astype(np.float32)
        return pred.reshape(-1)