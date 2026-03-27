
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from sbtab.bridge.reference import CategoricalReference


@dataclass(frozen=True)
class RegressionLoss:
    """
    Basic regression loss for field/drift/mean-map training.

    In DSB/IPF caches typically you regress:
      - target = (x_prev - x_next)  OR  (x_prev) depending on parametrization
    We keep it generic: predict -> target.
    """
    kind: str = "mse"  # "mse" | "huber"
    huber_delta: float = 1.0
    reduction: str = "mean"  # "mean" | "sum"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.kind == "mse":
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.kind == "huber":
            loss = F.huber_loss(pred, target, reduction="none", delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss kind: {self.kind}")

        loss = loss.mean(dim=1)

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unknown reduction: {self.reduction}")

class CSBMLoss:
    def __init__(self, reference: CategoricalReference, lmbda: float = 0.001):
        self.lmbda = lmbda
        self.reference = reference

    def forward_loss(self, pred_logits_x1, x_1_true, x_t, n, K):
        model_transition = self.reference.model_induced_next_step(pred_logits_x1, x_t, n, K)

        target_transition = self.reference.bridge_next_given_prev(x_t, x_1_true, n, K)

        kl_input = torch.log(model_transition.view(-1, self.reference.S_max) + 1e-12)
        kl_target = target_transition.view(-1, self.reference.S_max)

        kl_term = F.kl_div(kl_input, kl_target, reduction="batchmean")

        ce_input = pred_logits_x1.view(-1, self.reference.S_max)
        ce_target = x_1_true.view(-1)
        simple_term = F.cross_entropy(ce_input, ce_target)

        return kl_term + self.lmbda * simple_term

    def backward_loss(self, pred_logits_x0, x_0_true, x_t, n):
        model_transition = self.reference.model_induced_prev_step(pred_logits_x0, x_t, n)
        target_transition = self.reference.bridge_prev_given_next(x_0_true, x_t, n)

        kl_input = torch.log(model_transition.view(-1, self.reference.S_max) + 1e-12)
        kl_target = target_transition.view(-1, self.reference.S_max)

        kl_term = F.kl_div(kl_input, kl_target, reduction="batchmean")

        ce_input = pred_logits_x0.view(-1, self.reference.S_max)
        ce_target = x_0_true.view(-1)
        simple_term = F.cross_entropy(ce_input, ce_target)

        return kl_term + self.lmbda * simple_term