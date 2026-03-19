
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


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

class EfficientCSBMLoss:
    def __init__(self, lmbda: float = 0.001):
        self.lmbda = lmbda

    def forward_loss(self, pred_logits_x1, x_1_true, x_t_minus_1, n, N, ref):
        B, D, S_max = pred_logits_x1.shape

        l_simple = F.cross_entropy(pred_logits_x1.view(-1, S_max), x_1_true.view(-1))

        log_q_theta = F.log_softmax(pred_logits_x1, dim=-1)
        target_dist = F.one_hot(x_1_true, num_classes=S_max).float()

        kl = F.kl_div(log_q_theta, target_dist, reduction="batchmean")

        return kl + self.lmbda * l_simple

    def backward_loss(self, pred_logits_x0, x_0_true, x_t_n, n, N, ref):
        B, D, S_max = pred_logits_x0.shape

        l_simple = F.cross_entropy(pred_logits_x0.view(-1, S_max), x_0_true.view(-1))

        log_q_eta = F.log_softmax(pred_logits_x0, dim=-1)
        target_dist = F.one_hot(x_0_true, num_classes=S_max).float()

        kl = F.kl_div(log_q_eta, target_dist, reduction="batchmean")

        return kl + self.lmbda * l_simple

# class EfficientCSBMLoss:
#     def __init__(self, lmbda: float = 0.5):
#         self.lmbda = lmbda
#
#     def forward_loss(self, pred_logits_x1, x_1_true, x_t_prev, n, N, ref):
#         B, D, S_max = pred_logits_x1.shape
#         p_theta_x1 = F.softmax(pred_logits_x1, dim=-1)
#
#         with torch.no_grad():
#             q_ref_true = ref.bridge_at_time(x_t_prev, x_1_true, n + 1, N)

#         q_ref_all = ref.bridge_at_time(
#             x_t_prev.unsqueeze(-1).expand(-1, -1, S_max).reshape(-1, D),
#             all_possible_x1.expand(B, D, -1).reshape(-1, D),
#             n.repeat_interleave(S_max),
#             N
#         ).view(B, D, S_max, S_max)
#
#         q_ref_pred = torch.einsum('bds,bdsk->bdk', p_theta_x1, q_ref_all)
#
#         kl = F.kl_div(q_ref_pred.log() + 1e-12, q_ref_true, reduction="batchmean")
#
#         l_simple = F.cross_entropy(pred_logits_x1.view(-1, S_max), x_1_true.view(-1))
#
#         return kl + self.lmbda * l_simple
#
#     def backward_loss(self, pred_logits_x0, x_0_true, x_t, n, N, ref):
#         B, D, S_max = pred_logits_x0.shape
#         p_eta_x0 = F.softmax(pred_logits_x0, dim=-1)
#
#         with torch.no_grad():
#             q_ref_true = ref.bridge_at_time(x_0_true, x_t, n - 1, N)
#
#         all_possible_x0 = torch.arange(S_max, device=x_t.device).view(1, 1, S_max)
#         q_ref_all = ref.bridge_at_time(
#             all_possible_x0.expand(B, D, -1).reshape(-1, D),
#             x_t.unsqueeze(-1).expand(-1, -1, S_max).reshape(-1, D),
#             n.repeat_interleave(S_max),
#             N
#         ).view(B, D, S_max, S_max)
#
#         q_ref_pred = torch.einsum('bds,bdsk->bdk', p_eta_x0, q_ref_all)
#
#         kl = F.kl_div(q_ref_pred.log() + 1e-12, q_ref_true, reduction="batchmean")
#         l_simple = F.cross_entropy(pred_logits_x0.view(-1, S_max), x_0_true.view(-1))
#
#         return kl + self.lmbda * l_simple