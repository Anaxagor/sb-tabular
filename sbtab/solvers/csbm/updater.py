import torch
from dataclasses import dataclass

from sbtab.bridge.losses import EfficientCSBMLoss
from sbtab.bridge.reference import CategoricalReference

@dataclass
class CSBMUpdater:
    forward_model: torch.nn.Module
    backward_model: torch.nn.Module
    forward_opt: torch.optim.Optimizer
    backward_opt: torch.optim.Optimizer
    ref_process: "CategoricalReference"
    loss_fn: "EfficientCSBMLoss"

    def train_forward_step(self, x_t_prev, x_1_true, n, N) -> float:
        self.forward_model.train()
        self.forward_opt.zero_grad()
        pred_logits = self.forward_model(x_t_prev, n.float() / N)
        loss = self.loss_fn.forward_loss(pred_logits, x_1_true, x_t_prev, n, N, self.ref_process)
        loss.backward()
        self.forward_opt.step()
        return loss.item()

    def train_backward_step(self, x_t, x_0_true, n, N) -> float:
        self.backward_model.train()
        self.backward_opt.zero_grad()
        pred_logits = self.backward_model(x_t, n.float() / N)
        loss = self.loss_fn.backward_loss(pred_logits, x_0_true, x_t, n, N, self.ref_process)
        loss.backward()
        self.backward_opt.step()
        return loss.item()