
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GaussianReference:
    """
    Simple Gaussian reference distribution/process endpoints.

    In DSB/IPF practice for tabular:
      - terminal distribution at t=T is often standard Gaussian
      - initial distribution at t=0 is data distribution

    This class provides sampling for the "noise/prior" endpoint.
    """
    dim: int
    mean: float = 0.0
    std: float = 1.0
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def sample(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        if n <= 0:
            raise ValueError("n must be positive")
        g = torch.Generator(device=str(self.device) if self.device is not None else "cpu")
        if seed is not None:
            g.manual_seed(int(seed))
        dev = self.device or torch.device("cpu")
        x = torch.randn((n, self.dim), generator=g, device=dev, dtype=self.dtype)
        return x * self.std + self.mean

@dataclass
class CategoricalReference:
    cardinalities: list[int]
    is_ordered: torch.Tensor
    alpha: float = 0.05
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.S = torch.tensor(self.cardinalities, device=self.device)
        self.is_ordered = torch.tensor(self.is_ordered, device=self.device, dtype=torch.bool)
        self.S_max = int(self.S.max().item())
        self.D = len(self.cardinalities)

    def get_q_k_probs(self, x_start: torch.Tensor, k: torch.Tensor | int) -> torch.Tensor:
        curr_batch_size = x_start.shape[0]

        if isinstance(k, int):
            k_val = torch.full((curr_batch_size, 1, 1), k, device=self.device, dtype=self.dtype)
        else:
            k_val = k.reshape(-1)[:curr_batch_size].view(curr_batch_size, 1, 1).to(self.dtype)

        S_tensor = self.S.view(1, self.D, 1)
        b = self.alpha * S_tensor / (S_tensor - 1 + 1e-6)
        alpha_bar_k = (1 - b) ** k_val

        p_stay = alpha_bar_k + (1 - alpha_bar_k) / S_tensor
        p_jump = (1 - alpha_bar_k) / S_tensor

        probs_uniform = p_jump.expand(curr_batch_size, self.D, self.S_max).clone()
        target_indices = x_start.unsqueeze(-1)
        probs_uniform.scatter_(-1, target_indices, p_stay.expand(curr_batch_size, self.D, 1))

        all_cats = torch.arange(self.S_max, device=self.device).view(1, 1, self.S_max)
        delta = S_tensor - 1
        curr_alpha = self.alpha * (k_val ** 0.5)
        dist_sq = (all_cats - x_start.unsqueeze(-1)) ** 2
        variance_k = (curr_alpha ** 2 * k_val) * (delta ** 2) + 1e-12
        logits = -4 * dist_sq / variance_k
        probs_gaussian = torch.softmax(logits, dim=-1)

        ordered_mask = self.is_ordered.view(1, self.D, 1)
        final_probs = torch.where(ordered_mask, probs_gaussian, probs_uniform)

        valid_cats_mask = all_cats < S_tensor
        final_probs = torch.where(valid_cats_mask, final_probs, torch.zeros_like(final_probs))

        return final_probs / (final_probs.sum(dim=-1, keepdim=True) + 1e-12)

    def sample(self, x_prev: torch.Tensor, x_target: torch.Tensor, t: int, total_steps: int) -> torch.Tensor:
        steps_left = total_steps - t
        w_step = self.get_q_k_probs(x_prev, k=1)

        if steps_left > 1:
            w_to_end = self.get_q_k_probs(x_target, k=steps_left - 1)
        else:
            w_to_end = torch.zeros_like(w_step)
            w_to_end.scatter_(-1, x_target.unsqueeze(-1), 1.0)

        probs = w_step * w_to_end

        probs[:, :, 0] += 1e-12
        probs /= probs.sum(dim=-1, keepdim=True)

        flat_probs = probs.reshape(-1, self.S_max)
        return torch.multinomial(flat_probs, num_samples=1).view(x_prev.shape)

    def bridge_at_time(self, x_start: torch.Tensor, x_target: torch.Tensor, t: torch.Tensor,
                       total_steps: int) -> torch.Tensor:
        curr_batch_size = x_start.shape[0]

        t_batch = t.reshape(-1)[:curr_batch_size]
        steps_left = total_steps - t_batch

        q_start_to_t = self.get_q_k_probs(x_start, k=t_batch)
        q_end_to_t = self.get_q_k_probs(x_target, k=steps_left)

        q_full_probs = self.get_q_k_probs(x_start, k=total_steps)
        q_full = q_full_probs.gather(-1, x_target.unsqueeze(-1))

        probs = (q_start_to_t * q_end_to_t) / (q_full + 1e-12)

        valid_mask = torch.arange(self.S_max, device=self.device).view(1, 1, self.S_max) < self.S.view(1, self.D, 1)
        probs = torch.where(valid_mask, probs + 1e-15, torch.zeros_like(probs))

        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)