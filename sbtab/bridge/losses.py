
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

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
    def __init__(self, reference: CategoricalReference, lmbda: float = 0.001) -> None:
        """
        Initialization of Categorical Schrödinger Bridge Matching (CSBM) loss.

        Args:
            reference (CategoricalReference): Categorical reference process.
            lmbda (float, optional): Weight parameter for the auxiliary cross-entropy loss term. Defaults to 0.001.
        """
        self.lmbda = lmbda
        self.reference = reference

    def forward_loss(
        self,
        pred_logits_x1: torch.Tensor,
        x_1_true: torch.Tensor,
        x_t: torch.Tensor,
        n: torch.Tensor,
        K: int
    ) -> torch.Tensor:
        """
        Computes the forward direction loss (towards t=1) for the categorical data.

        Args:
            pred_logits_x1 (torch.Tensor): Model's predicted unnormalized logits for the target
                                           terminal state x_1 (Shape: [B, D, S_max] or flattened).
            x_1_true (torch.Tensor): Ground-truth target categorical indices at the end of the bridge (x_1).
            x_t (torch.Tensor): Current intermediate categorical indices sampled at step n.
            n (torch.Tensor): Current integer time steps for each sample in the batch.
            K (int): Total number of discretization steps in the time grid.

        Returns:
            torch.Tensor: Scalar loss combining KL-divergence of transition probabilities and auxiliary CE.
        """
        model_transition = self.reference.model_induced_next_step(pred_logits_x1, x_t, n, K)

        target_transition = self.reference.bridge_next_given_prev(x_t, x_1_true, n, K)

        kl_input = torch.log(model_transition.view(-1, self.reference.S_max) + 1e-12)
        kl_target = target_transition.view(-1, self.reference.S_max)

        kl_term = F.kl_div(kl_input, kl_target, reduction="batchmean")

        ce_input = pred_logits_x1.view(-1, self.reference.S_max)
        ce_target = x_1_true.view(-1)
        simple_term = F.cross_entropy(ce_input, ce_target)

        return kl_term + self.lmbda * simple_term

    def backward_loss(
        self,
        pred_logits_x0: torch.Tensor,
        x_0_true: torch.Tensor,
        x_t: torch.Tensor,
        n: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the backward direction loss (towards t=0) for the categorical components.

        Args:
            pred_logits_x0 (torch.Tensor): Model's predicted unnormalized logits for the starting
                                           source state x_0 (Shape: [B, D, S_max] or flattened).
            x_0_true (torch.Tensor): Ground-truth target categorical indices at the start of the bridge (x_0).
            x_t (torch.Tensor): Current intermediate categorical indices sampled at step n.
            n (torch.Tensor): Current integer time steps for each sample in the batch.

        Returns:
            torch.Tensor: Scalar loss combining backward KL-divergence and auxiliary CE.
        """
        model_transition = self.reference.model_induced_prev_step(pred_logits_x0, x_t, n)
        target_transition = self.reference.bridge_prev_given_next(x_0_true, x_t, n)

        kl_input = torch.log(model_transition.view(-1, self.reference.S_max) + 1e-12)
        kl_target = target_transition.view(-1, self.reference.S_max)

        kl_term = F.kl_div(kl_input, kl_target, reduction="batchmean")

        ce_input = pred_logits_x0.view(-1, self.reference.S_max)
        ce_target = x_0_true.view(-1)
        simple_term = F.cross_entropy(ce_input, ce_target)

        return kl_term + self.lmbda * simple_term

class MixedSBMLoss(nn.Module):
    def __init__(
            self,
            reference: Optional[CategoricalReference],
            lambda_num: float = 0.5,
            lambda_cat: float = 0.5,
            ce_lambda: float = 0.001
    ) -> None:
        """
        Initialization of MSBM loss constructed as a weighted sum of continuous
        regression loss and categorical (CSBM) loss functions.

        Args:
            reference (Optional[CategoricalReference]): Categorical reference process helper.
                                                       Pass None if the dataset has no categorical features.
            lambda_num (float, optional): Scaling weight for the continuous feature loss. Defaults to 0.5.
            lambda_cat (float, optional): Scaling weight for the categorical feature loss. Defaults to 0.5.
            ce_lambda (float, optional): Scaling weight for the auxiliary cross-entropy inside CSBM. Defaults to 0.001.

        Returns:
            None
        """
        super().__init__()
        self.num_loss_fn = RegressionLoss(kind="mse", reduction="mean")
        self.cat_loss_fn = CSBMLoss(reference=reference, lmbda=ce_lambda) if reference is not None else None

        self.lambda_num = lambda_num
        self.lambda_cat = lambda_cat

    def forward(
        self,
        pred_num: torch.Tensor,
        true_num: torch.Tensor,
        pred_logits_cat: torch.Tensor,
        true_cat: torch.Tensor,
        x_t_cat: torch.Tensor,
        n: torch.Tensor,
        K: int | None = None,
        direction: Literal['f', 'b'] = 'f'
    ) -> torch.Tensor:
        """
        Forward pass that aggregates both continuous regression matching loss and categorical bridge matching loss.

        Args:
            pred_num (torch.Tensor): Target vectors (e.g. vector field, velocity or drift) predicted
                                     by the model for continuous features (Shape: [B, cont_dim]).
            true_num (torch.Tensor): Ground-truth target vectors calculated from the continuous bridge formula (Shape: [B, cont_dim]).
            pred_logits_cat (torch.Tensor): Unnormalized logit predictions from the categorical head (Shape: [B, D, S_max]).
            true_cat (torch.Tensor): Ground-truth target categorical indices for the current direction (z_1 for 'f', z_0 for 'b').
            x_t_cat (torch.Tensor): Intermediate categorical indices sampled at the current step n.
            n (torch.Tensor): Current integer time steps for each sample in the batch.
            K (Optional[int], optional): Total number of steps in the time grid discretization. Required if direction is 'f'.
            direction (Literal['f', 'b'], optional): Current training direction ('f' for forward, 'b' for backward). Defaults to 'f'.

        Returns:
            loss (torch.Tensor): Total combined loss scalar on the same device as the input tensors.
        """
        loss = torch.tensor(0.0, device=true_num.device)
        if pred_num is not None and true_num.shape[-1] > 0:
            l_num = self.num_loss_fn(pred_num, true_num)
            loss += self.lambda_num * l_num

        if pred_logits_cat is not None and true_cat.shape[-1] > 0 and self.cat_loss_fn is not None:
            if direction == 'f':
                l_cat = self.cat_loss_fn.forward_loss(pred_logits_cat, true_cat, x_t_cat, n, K)
            elif direction == 'b':
                l_cat = self.cat_loss_fn.backward_loss(pred_logits_cat, true_cat, x_t_cat, n)
            else:
                raise ValueError("direction must be 'f' or 'b'")
            loss += self.lambda_cat * l_cat

        return loss
