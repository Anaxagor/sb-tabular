
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
    total_number_of_q_powers: int
    alpha: float = 0.05
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.S = torch.tensor(self.cardinalities, device=self.device)
        self.is_ordered = self.is_ordered.clone().detach().to(device=self.device, dtype=torch.bool)
        self.S_max = int(self.S.max().item())
        self.D = len(self.cardinalities)
        self._powers = []

        for d in range(self.D):
            S_d = int(self.S[d].item())
            is_ord = bool(self.is_ordered[d].item())

            feat_powers = []
            for k in range(self.total_number_of_q_powers + 1):
                if is_ord:
                    matrix = self._build_gaussian_k_matrix(S_d, k)
                else:
                    matrix = self._build_uniform_k_matrix(S_d, k)
                feat_powers.append(matrix)

            self._powers.append(torch.stack(feat_powers))

    def _build_uniform_k_matrix(self, S_d: int, k: int = 1) -> torch.Tensor:
        """
        Private method to calculate uniform reference transition matrix of the power k.
        By default, k = 1 so it's the simple transition matrix.
        """
        if k == 0: return torch.eye(S_d, device=self.device)

        b = self.alpha * S_d / (S_d - 1 + 1e-6)
        alpha_bar_k = (1 - b) ** k

        p_stay = alpha_bar_k + (1 - alpha_bar_k) / S_d
        p_jump = (1 - alpha_bar_k) / S_d

        transition_matrix = torch.full((S_d, S_d), p_jump, device=self.device)
        transition_matrix.fill_diagonal_(p_stay)

        return transition_matrix

    def _build_gaussian_k_matrix(self, S_d: int, k: int = 1) -> torch.Tensor:
        """
        Private method to calculate gaussian reference transition matrix of the power k.
        By default, k = 1 so it's the simple transition matrix.
        """
        if k == 0: return torch.eye(S_d, device=self.device)

        idx = torch.arange(S_d, device=self.device)
        i = idx.view(S_d, 1)
        j = idx.view(1, S_d)

        delta = S_d - 1

        dist_sq = (i - j) ** 2

        if k < 30:
            variance_1 = (self.alpha ** 2) * (delta ** 2) + 1e-12
            logits_1 = -4 * dist_sq / variance_1
            Q = torch.softmax(logits_1, dim=-1)

            return torch.matrix_power(Q, k)
        else:
            variance_k = (self.alpha ** 2 * k) * (delta ** 2) + 1e-12
            logits_k = -4 * dist_sq / variance_k

            return torch.softmax(logits_k, dim=-1)


    def _bridge_at_time_1d(self, x_start_idx, x_target_idx, t, total_steps, d):
        batch_size = x_start_idx.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device)

        t = t.long()
        t_rest = (total_steps - t).clamp(0, total_steps)

        Q_t = self._powers[d][t]
        Q_rest = self._powers[d][t_rest]
        Q_all = self._powers[d][total_steps]

        row_start = Q_t[batch_indices, x_start_idx]
        col_end = Q_rest[batch_indices, :, x_target_idx]

        norm = Q_all[x_start_idx, x_target_idx].unsqueeze(-1)

        return (row_start * col_end) / (norm + 1e-12)

    def bridge_at_time(self, x_start: torch.Tensor, x_target: torch.Tensor, t: torch.Tensor,
                       total_steps: int) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x_start.shape[0])

        probs_all_dims = []
        for d in range(self.D):
            probs_d = self._bridge_at_time_1d(x_start[:, d], x_target[:, d], t, total_steps, d)

            if probs_d.shape[-1] < self.S_max:
                padding = torch.zeros((probs_d.shape[0], self.S_max - probs_d.shape[-1]),
                                      device=self.device, dtype=self.dtype)
                probs_d = torch.cat([probs_d, padding], dim=-1)

            probs_all_dims.append(probs_d)

        return torch.stack(probs_all_dims, dim=1)

    def bridge_next_given_prev(self, x_t: torch.Tensor, x_target: torch.Tensor, n: torch.Tensor, K: int):
        batch_size = x_t.shape[0]

        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), n, device=self.device, dtype=torch.long)
        elif n.dim() == 0:
            n = n.expand(batch_size).long()

        batch_indices = torch.arange(batch_size, device=self.device)

        if n.dim() == 0:
            n = n.expand(batch_size)
        n = n.long()

        all_dims_probs = []
        for d in range(self.D):
            S_d = int(self.S[d].item())
            Q_1 = self._powers[d][1]

            Q_rest = self._powers[d][(K - n - 1).clamp(0, K)]
            Q_total = self._powers[d][(K - n).clamp(0, K)]

            w_step = Q_1[x_t[:, d]]

            w_to_end = Q_rest[batch_indices, :, x_target[:, d]]

            norm = Q_total[batch_indices, x_t[:, d], x_target[:, d]].unsqueeze(-1)

            probs_d = (w_step * w_to_end) / (norm + 1e-12)

            if probs_d.shape[-1] < self.S_max:
                padding = torch.zeros((batch_size, self.S_max - S_d), device=self.device)
                probs_d = torch.cat([probs_d, padding], dim=-1)
            all_dims_probs.append(probs_d)

        return torch.stack(all_dims_probs, dim=1)

    def bridge_prev_given_next(self, x_start: torch.Tensor, x_t: torch.Tensor, n: torch.Tensor):
        batch_size = x_t.shape[0]

        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), n, device=self.device, dtype=torch.long)
        elif n.dim() == 0:
            n = n.expand(batch_size).long()

        batch_indices = torch.arange(batch_size, device=self.device)

        if n.dim() == 0:
            n = n.expand(batch_size)
        n = n.long()

        all_dims_probs = []
        for d in range(self.D):
            S_d = int(self.S[d].item())
            Q_1 = self._powers[d][1]
            Q_from_start = self._powers[d][(n - 1).clamp(0, self.total_number_of_q_powers)]
            Q_total = self._powers[d][n.clamp(0, self.total_number_of_q_powers)]

            w_from_start = Q_from_start[batch_indices, x_start[:, d], :]

            w_step_back = Q_1[:, x_t[:, d]].T

            norm = Q_total[batch_indices, x_start[:, d], x_t[:, d]].unsqueeze(-1)

            probs_d = (w_from_start * w_step_back) / (norm + 1e-12)

            if probs_d.shape[-1] < self.S_max:
                padding = torch.zeros((batch_size, self.S_max - S_d), device=self.device)
                probs_d = torch.cat([probs_d, padding], dim=-1)
            all_dims_probs.append(probs_d)

        return torch.stack(all_dims_probs, dim=1)

    def model_induced_next_step(self, model_logits, x_t, n, K):
        batch_size = x_t.shape[0]

        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), n, device=self.device, dtype=torch.long)
        elif n.dim() == 0:
            n = n.expand(batch_size).long()

        p_model_xK = torch.softmax(model_logits, dim=-1)
        batch_indices = torch.arange(batch_size, device=self.device)

        if n.dim() == 0:
            n = n.expand(batch_size)
        n = n.long()

        induced_probs = []
        for d in range(self.D):
            S_d = int(self.S[d].item())
            Q_1 = self._powers[d][1]
            Q_rest = self._powers[d][(K - n - 1).clamp(0, K)]
            Q_total = self._powers[d][(K - n).clamp(0, K)]

            norm_den = Q_total[batch_indices, x_t[:, d], :S_d]

            p_model_d = p_model_xK[:, d, :S_d]
            term_to_sum = p_model_d / (norm_den + 1e-12)

            summed_targets = torch.einsum('bj,bsj->bs', term_to_sum, Q_rest)

            w_step = Q_1[x_t[:, d]]
            probs_d = w_step * summed_targets

            probs_d = probs_d / (probs_d.sum(dim=-1, keepdim=True) + 1e-12)

            if S_d < self.S_max:
                padding = torch.zeros((batch_size, self.S_max - S_d), device=self.device)
                probs_d = torch.cat([probs_d, padding], dim=-1)
            induced_probs.append(probs_d)

        return torch.stack(induced_probs, dim=1)

    def model_induced_prev_step(self, model_logits, x_t, n):
        p_model_x0 = torch.softmax(model_logits, dim=-1)
        batch_size = x_t.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device)

        if not isinstance(n, torch.Tensor):
            n = torch.tensor(n, device=self.device)
        if n.dim() == 0:
            n = n.expand(batch_size)
        n = n.long()

        induced_probs = []
        for d in range(self.D):
            S_d = int(self.S[d].item())
            Q_1 = self._powers[d][1]
            Q_from_start_to_prev = self._powers[d][(n - 1).clamp(0, self.total_number_of_q_powers)]
            Q_from_start_to_curr = self._powers[d][n.clamp(0, self.total_number_of_q_powers)]

            w_step_back = Q_1[:, x_t[:, d]].T
            p_model_d = p_model_x0[:, d, :S_d]

            norm_den = Q_from_start_to_curr[batch_indices, :, x_t[:, d]]

            term_to_sum = p_model_d / (norm_den + 1e-12)
            summed_starts = torch.einsum('bi, bij -> bj', term_to_sum, Q_from_start_to_prev)

            probs_d = w_step_back * summed_starts
            probs_d = probs_d / (probs_d.sum(dim=-1, keepdim=True) + 1e-12)

            if S_d < self.S_max:
                padding = torch.zeros((batch_size, self.S_max - S_d), device=self.device)
                probs_d = torch.cat([probs_d, padding], dim=-1)
            induced_probs.append(probs_d)

        return torch.stack(induced_probs, dim=1)

    def update_alpha(self, new_alpha: float):
        self.alpha = new_alpha
        self._powers = []
        for d in range(self.D):
            S_d = int(self.S[d].item())
            is_ord = bool(self.is_ordered[d].item())
            feat_powers = []
            for k in range(self.total_number_of_q_powers + 1):
                if is_ord:
                    matrix = self._build_gaussian_k_matrix(S_d, k)
                else:
                    matrix = self._build_uniform_k_matrix(S_d, k)
                feat_powers.append(matrix)
            self._powers.append(torch.stack(feat_powers))

    def sample_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        batch_size, dims, s_max = probs.shape

        arange = torch.arange(s_max, device=self.device).view(1, 1, s_max)
        mask = arange < self.S.view(1, dims, 1)

        masked_probs = (probs + 1e-12) * mask

        flat_probs = masked_probs.reshape(-1, s_max)
        samples = torch.multinomial(flat_probs, num_samples=1)
        return samples.view(batch_size, dims)

    def sample_x_t(self, x_start: torch.Tensor, x_target: torch.Tensor, t: torch.Tensor, total_steps: int) -> torch.Tensor:
        probs = self.bridge_at_time(x_start, x_target, t, total_steps)
        return self.sample_from_probs(probs)

    def sample_step(self, x_t: torch.Tensor, x_target: torch.Tensor, n: torch.Tensor, total_steps: int) -> torch.Tensor:
        probs = self.bridge_next_given_prev(x_t, x_target, n, total_steps)
        return self.sample_from_probs(probs)