import torch
from dataclasses import dataclass

from sbtab.bridge.losses import CSBMLoss
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid

@dataclass
class CSBMUpdater:
    """
    Both networks are fed the grid time of the CURRENT state, t[n] — the same
    clock DiscretePathSampler uses. forward: n in [0, N-1] predicts x_N;
    backward: n in [1, N] predicts x_0.
    """
    forward_model: torch.nn.Module
    backward_model: torch.nn.Module
    forward_opt: torch.optim.Optimizer
    backward_opt: torch.optim.Optimizer
    ref_process: "CategoricalReference"
    loss_fn: "CSBMLoss"
    timegrid: TimeGrid
    n_forward_updates: int = 0
    n_backward_updates: int = 0

    def _time(self, n: torch.Tensor) -> torch.Tensor:
        return self.timegrid.grid().to(n.device)[n]

    def train_forward_step(self, x_t, x_1_true, n, K=None) -> float:
        self.forward_model.train()
        self.forward_opt.zero_grad()

        pred_logits = self.forward_model(x_t, self._time(n))

        loss = self.loss_fn.forward_loss(pred_logits, x_1_true, x_t, n)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CSBM forward loss")

        loss.backward()
        self.forward_opt.step()
        self.n_forward_updates += 1
        return loss.item()

    def train_backward_step(self, x_t, x_0_true, n, K=None) -> float:
        self.backward_model.train()
        self.backward_opt.zero_grad()

        pred_logits = self.backward_model(x_t, self._time(n))

        loss = self.loss_fn.backward_loss(pred_logits, x_0_true, x_t, n)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CSBM backward loss")

        loss.backward()
        self.backward_opt.step()
        self.n_backward_updates += 1
        return loss.item()
