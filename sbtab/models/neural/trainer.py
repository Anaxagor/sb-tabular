
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sbtab.bridge.losses import RegressionLoss
from sbtab.numerics import check_gradients, require_finite


@dataclass
class NeuralTrainerConfig:
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 1
    grad_clip: Optional[float] = 1.0
    device: str = "cpu"  # "cuda" | "cpu"


class NeuralTrainer:
    """
    Minimal trainer for time-conditioned field models.
    """
    def __init__(self, cfg: NeuralTrainerConfig, loss: Optional[RegressionLoss] = None):
        self.cfg = cfg
        self.loss_fn = loss or RegressionLoss(kind="mse", reduction="mean")

    def fit(
        self,
        model: nn.Module,
        loader: DataLoader,
        predict_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        device = torch.device(self.cfg.device)
        model.to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        model.train()
        for _epoch in range(self.cfg.max_epochs):
            for step, batch in enumerate(loader):
                x, t, target = batch  # shapes: (B,D), (B,1), (B,D)
                x = x.to(device)
                t = t.to(device)
                target = target.to(device)

                pred = predict_fn(model, x, t)
                loss = self.loss_fn(pred, target)
                context = dict(model=type(model).__name__, stage="training", epoch=_epoch, step=step)
                require_finite(loss, "loss", **context)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                check_gradients(model.parameters(), max_norm=self.cfg.grad_clip, **context)
                opt.step()
