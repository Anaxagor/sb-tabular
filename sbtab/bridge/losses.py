
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sbtab.bridge.reference import CategoricalReference


@dataclass(frozen=True)
class RegressionLoss:
    """
    Basic regression loss for field/drift/mean-map training.

    In DSB/IPF caches typically you regress:
      - target = (x_prev - x_next)  OR  (x_prev) depending on parametrization
    We keep it generic: predict -> target.

    Normalisation: the per-row loss is the MEAN over coordinates (not the squared
    norm), then reduced over the batch. It is therefore a per-feature average,
    the same unit as the per-column categorical loss below.
    """
    kind: str = "mse"  # "mse" | "huber"
    huber_delta: float = 1.0
    reduction: str = "mean"  # "mean" | "sum"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pred.ndim != 2 or pred.shape[1] == 0:
            raise ValueError("RegressionLoss needs at least one coordinate; branch on absent blocks before calling")
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
    """
    Categorical bridge-matching loss: KL(bridge step || model-induced step) plus
    ``lmbda`` times the endpoint cross-entropy.

    Normalisation: each (row, column) pair yields one scalar; the loss is their
    mean over the batch AND over columns, exactly once. It is a per-column
    average, so duplicating an identical column leaves the value unchanged.

    Padded categories are masked to -inf before the softmax and carry zero mass.
    """

    def __init__(self, reference: CategoricalReference, lmbda: float = 0.001):
        self.lmbda = lmbda
        self.reference = reference

    @staticmethod
    def _kl(target: torch.Tensor, model: torch.Tensor) -> torch.Tensor:
        """sum_s target log(target / model) per (row, column), with 0 log 0 = 0."""
        pos = target > 0
        one = torch.ones_like(target)
        t_safe = torch.where(pos, target, one)
        m_safe = torch.where(pos, model, one)
        return torch.where(pos, target * (torch.log(t_safe) - torch.log(m_safe)), torch.zeros_like(target)).sum(dim=-1)

    def _ce(self, logits: torch.Tensor, true_idx: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(self.reference.masked_logits(logits).double(), dim=-1)
        return -logp.gather(-1, true_idx.long().unsqueeze(-1)).squeeze(-1)

    def per_column(self, model_transition, target_transition, logits, true_idx) -> torch.Tensor:
        """(B, D) matrix of per-row, per-column losses."""
        return self._kl(target_transition, model_transition) + self.lmbda * self._ce(logits, true_idx)

    def forward_loss(self, pred_logits_x1, x_1_true, x_t, n, K=None):
        model_transition = self.reference.model_induced_next_step(pred_logits_x1, x_t, n)
        target_transition = self.reference.bridge_next_given_prev(x_t, x_1_true, n)
        return self.per_column(model_transition, target_transition, pred_logits_x1, x_1_true).mean()

    def backward_loss(self, pred_logits_x0, x_0_true, x_t, n):
        model_transition = self.reference.model_induced_prev_step(pred_logits_x0, x_t, n)
        target_transition = self.reference.bridge_prev_given_next(x_0_true, x_t, n)
        return self.per_column(model_transition, target_transition, pred_logits_x0, x_0_true).mean()


class MixedSBMLoss(nn.Module):
    """
    lambda_num * L_num + lambda_cat * L_cat.

    L_num is the mean squared error per numerical coordinate and L_cat the mean
    categorical loss per column, so both terms are per-feature averages and the
    two weights are directly comparable. L_cat is normalised once, inside
    CSBMLoss; there is no further division by the column count.

    An absent block (no numerical or no categorical columns) is skipped before
    its loss is computed. It contributes nothing, rather than NaN times zero.
    """

    def __init__(
            self,
            reference: Optional[CategoricalReference],
            lambda_num: float = 0.5,
            lambda_cat: float = 0.5,
            ce_lambda: float = 0.001
    ):
        super().__init__()
        self.num_loss_fn = RegressionLoss(kind="mse", reduction="mean")
        self.cat_loss_fn = CSBMLoss(reference=reference, lmbda=ce_lambda) if reference is not None else None

        self.lambda_num = lambda_num
        self.lambda_cat = lambda_cat

    def forward(
            self,
            pred_num: Optional[torch.Tensor],
            target_num: Optional[torch.Tensor],
            pred_logits_cat: Optional[torch.Tensor],
            true_cat: Optional[torch.Tensor],
            x_t_cat: Optional[torch.Tensor],
            n: torch.Tensor,
            K: int = None,
            direction: str = "forward"
    ) -> torch.Tensor:
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")

        has_num = pred_num is not None and pred_num.ndim == 2 and pred_num.shape[1] > 0
        has_cat = (
            self.cat_loss_fn is not None
            and pred_logits_cat is not None
            and pred_logits_cat.ndim == 3
            and pred_logits_cat.shape[1] > 0
        )
        if not has_num and not has_cat:
            raise ValueError("MixedSBMLoss needs at least one numerical or categorical column")

        total = None
        if has_num:
            total = self.lambda_num * self.num_loss_fn(pred_num, target_num).double()
        if has_cat:
            if direction == "forward":
                l_cat = self.cat_loss_fn.forward_loss(pred_logits_cat, true_cat, x_t_cat, n)
            else:
                l_cat = self.cat_loss_fn.backward_loss(pred_logits_cat, true_cat, x_t_cat, n)
            total = self.lambda_cat * l_cat if total is None else total + self.lambda_cat * l_cat
        return total
