from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

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
    """
    Manages transition matrices and Markov bridge probabilities for categorical features.

    Supports both ordered (Gaussian-like transitions) and unordered (uniform transitions)
    categorical variables across a discretized time grid. Precomputes multi-step
    transition probabilities to optimize bridge sampling and loss evaluations.

    Attributes:
        cardinalities (list[int]): Number of classes/categories for each feature.
        is_ordered (torch.Tensor): Boolean tensor mask of shape [D], where True
            indicates that the corresponding categorical feature is ordered.
        total_number_of_q_powers (int): Number of discretization steps in the time grid (K).
        alpha (float): Noise/diffusion rate parameter governing transition speeds.
        device (torch.device): Compute device for tensor operations.
        dtype (torch.dtype): Floating point precision for probability computations.
    """
    cardinalities: list[int]
    is_ordered: torch.Tensor
    total_number_of_q_powers: int
    alpha: float = 0.05
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        """Initializes categorical dimensions, validation masks, and caches transition powers."""
        self.S = torch.tensor(self.cardinalities, device=self.device)
        self.is_ordered = self.is_ordered.clone().detach().to(device=self.device, dtype=torch.bool)
        self.S_max = int(self.S.max().item())
        self.D = len(self.cardinalities)
        self._powers = torch.zeros((self.D, self.total_number_of_q_powers + 1, self.S_max, self.S_max), device=self.device)

        arange = torch.arange(self.S_max, device=self.device).view(1, 1, self.S_max)
        self.valid_mask = arange < self.S.view(1, self.D, 1)

        for d in range(self.D):
            S_d = int(self.S[d].item())
            is_ord = bool(self.is_ordered[d].item())
            for k in range(self.total_number_of_q_powers + 1):
                if is_ord:
                    matrix = self._build_gaussian_k_matrix(S_d, k)
                else:
                    matrix = self._build_uniform_k_matrix(S_d, k)
                self._powers[d, k, :S_d, :S_d] = matrix

    def _build_uniform_k_matrix(self, S_d: int, k: int = 1) -> torch.Tensor:
        """
        Computes the k-step transition matrix for an unordered categorical feature.

        Args:
            S_d (int): Cardinality of the specific feature.
            k (int): Number of steps (matrix power). Defaults to 1.

        Returns:
            torch.Tensor: Transition probability matrix of shape [S_d, S_d].
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
        Computes the k-step transition matrix for an ordered categorical feature.

        Uses local random walk transitions embedded via softmax distance metrics.

        Args:
            S_d (int): Cardinality of the specific feature.
            k (int): Number of steps (matrix power). Defaults to 1.

        Returns:
            torch.Tensor: Transition probability matrix of shape [S_d, S_d].
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

    def bridge_at_time(
        self,
        x_start: torch.Tensor,
        x_target: torch.Tensor,
        t: torch.Tensor,
        total_steps: int
    ) -> torch.Tensor:
        """
        Computes the bridge probability distribution P(x_t | x_0=x_start, x_K=x_target).

        Args:
            x_start (torch.Tensor): Starting state class indices (Shape: [B, D]).
            x_target (torch.Tensor): Target endpoint class indices (Shape: [B, D]).
            t (torch.Tensor): Discrete time steps for the current batch (Shape: [B] or scalar).
            total_steps (int): Total number of discrete steps in the setup grid (K).

        Returns:
            torch.Tensor: Conditional probability distributions of shape [B, D, S_max].
        """
        x_start = torch.as_tensor(x_start, device=self.device, dtype=torch.long)
        x_target = torch.as_tensor(x_target, device=self.device, dtype=torch.long)

        batch_size = x_start.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device).view(batch_size, 1)
        dim_indices = torch.arange(self.D, device=self.device).view(1, self.D)

        if t.dim() == 0 or t.numel() == 1:
            t = t.expand(batch_size)
        t = t.view(-1).long()
        t_rest = (total_steps - t).clamp(0, total_steps)

        Q_t = self._powers[:, t].permute(1, 0, 2, 3)
        Q_rest = self._powers[:, t_rest].permute(1, 0, 2, 3)
        Q_all = self._powers[:, total_steps]

        row_start = Q_t[batch_indices, dim_indices, x_start, :]
        col_end = Q_rest[batch_indices, dim_indices, :, x_target]

        norm = Q_all[dim_indices, x_start, x_target]
        probs = (row_start * col_end) / (norm.unsqueeze(-1) + 1e-12)

        return probs

    def bridge_next_given_prev(
        self,
        x_t: torch.Tensor,
        x_target: torch.Tensor,
        n: Union[torch.Tensor, int],
        K: int
    ) -> torch.Tensor:
        """
        Calculates the conditional transition distribution P(x_{t+1} | x_t, x_K=x_target).

        Used during forward IMF training updates.

        Args:
            x_t (torch.Tensor): Categorical feature class indices at step n (Shape: [B, D]).
            x_target (torch.Tensor): Endpoint target indices (z_1) (Shape: [B, D]).
            n (Union[torch.Tensor, int]): Discrete time steps for the batch.
            K (int): Total number of time grid steps.

        Returns:
            torch.Tensor: Next-step transition distribution vectors of shape [B, D, S_max].
        """
        x_t = torch.as_tensor(x_t, device=self.device, dtype=torch.long)
        x_target = torch.as_tensor(x_target, device=self.device, dtype=torch.long)

        batch_size = x_t.shape[0]

        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), n, device=self.device, dtype=torch.long)
        elif n.dim() == 0:
            n = n.expand(batch_size).long()
        n = n.view(-1).long()

        t_rest = (K - n - 1).clamp(0, K)
        t_total = (K - n).clamp(0, K)

        b_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1)  # [B, 1]
        d_idx = torch.arange(self.D, device=self.device).view(1, self.D)  # [1, D]

        Q_1 = self._powers[:, 1]  # [D, S_max, S_max]
        Q_rest = self._powers[:, t_rest].permute(1, 0, 2, 3)  # [B, D, S_max, S_max]
        Q_total = self._powers[:, t_total].permute(1, 0, 2, 3)  # [B, D, S_max, S_max]

        w_step = Q_1[d_idx, x_t, :]  # [B, D, S_max]
        w_to_end = Q_rest[b_idx, d_idx, :, x_target]  # [B, D, S_max]

        norm = Q_total[b_idx, d_idx, x_t, x_target]  # [B, D]

        return (w_step * w_to_end) / (norm.unsqueeze(-1) + 1e-12)  # [B, D, S_max]

    def bridge_prev_given_next(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        n: Union[torch.Tensor, int]
    ) -> torch.Tensor:
        """
        Calculates the posterior transition distribution P(x_{t-1} | x_0=x_start, x_t).

        Used during backward IMF training updates.

        Args:
            x_start (torch.Tensor): Starting source indices (z_0) (Shape: [B, D]).
            x_t (torch.Tensor): Categorical feature class indices at step n (Shape: [B, D]).
            n (Union[torch.Tensor, int]): Discrete time steps for the batch.

        Returns:
            torch.Tensor: Backward step transition distribution vectors of shape [B, D, S_max].
        """
        x_start = torch.as_tensor(x_start, device=self.device, dtype=torch.long)
        x_t = torch.as_tensor(x_t, device=self.device, dtype=torch.long)

        batch_size = x_t.shape[0]

        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), n, device=self.device, dtype=torch.long)
        elif n.dim() == 0:
            n = n.expand(batch_size).long()
        n = n.view(-1).long()

        t_prev = (n - 1).clamp(0, self.total_number_of_q_powers)
        t_curr = n.clamp(0, self.total_number_of_q_powers)

        b_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1)  # [B, 1]
        d_idx = torch.arange(self.D, device=self.device).view(1, self.D)  # [1, D]

        Q_1 = self._powers[:, 1]  # [D, S_max, S_max]
        Q_from_start = self._powers[:, t_prev].permute(1, 0, 2, 3)  # [B, D, S_max, S_max]
        Q_total = self._powers[:, t_curr].permute(1, 0, 2, 3)  # [B, D, S_max, S_max]

        w_from_start = Q_from_start[b_idx, d_idx, x_start, :]  # [B, D, S_max]

        w_step_back = Q_1[d_idx, :, x_t]  # [B, D, S_max]

        norm = Q_total[b_idx, d_idx, x_start, x_t]  # [B, D]

        return (w_from_start * w_step_back) / (norm.unsqueeze(-1) + 1e-12)

    def model_induced_next_step(
        self,
        model_logits: torch.Tensor,
        x_t: torch.Tensor,
        n: Union[torch.Tensor, int],
        K: int
    ) -> torch.Tensor:
        """
        Predicts forward step probabilities P(x_{t+1} | x_t) marginalized over endpoint estimates.

        Integrates the network's endpoint logits output to perform forward path reconstruction.

        Args:
            model_logits (torch.Tensor): Unnormalized predicted logits for the target state x_K (Shape: [B, D, S_max]).
            x_t (torch.Tensor): Class indices at the current step (Shape: [B, D]).
            n (Union[torch.Tensor, int]): Current discrete time step index.
            K (int): Total number of time grid discretization intervals.

        Returns:
            torch.Tensor: Aggregated transition probabilities of shape [B, D, S_max].
        """
        x_t = torch.as_tensor(x_t, device=self.device, dtype=torch.long)
        batch_size = x_t.shape[0]

        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), n, device=self.device, dtype=torch.long)
        elif n.dim() == 0:
            n = n.expand(batch_size).long()
        n = n.view(-1).long()

        t_rest = (K - n - 1).clamp(0, K)
        t_total = (K - n).clamp(0, K)

        b_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1)
        d_idx = torch.arange(self.D, device=self.device).view(1, self.D)

        masked_logits = model_logits.masked_fill(~self.valid_mask, -1e9)
        p_model = torch.softmax(masked_logits, dim=-1)  # [B, D, S_max]

        Q_1 = self._powers[:, 1]  # [D, S_max, S_max]
        Q_rest = self._powers[:, t_rest].permute(1, 0, 2, 3)  # [B, D, S_max, S_max]
        Q_total = self._powers[:, t_total].permute(1, 0, 2, 3)  # [B, D, S_max, S_max]

        norm_den = Q_total[b_idx, d_idx, x_t, :]  # [B, D, S_max]

        term_to_sum = p_model / (norm_den + 1e-12)  # [B, D, S_max]

        # b (batch), d (dim), i (next_state), j (target_state).
        # term_to_sum(b, d, j) * Q_rest(b, d, i, j) -> (b, d, i)
        summed_targets = torch.einsum('bdj, bdij -> bdi', term_to_sum, Q_rest)  # [B, D, S_max]

        w_step = Q_1[d_idx, x_t, :]  # [B, D, S_max]

        probs = w_step * summed_targets  # [B, D, S_max]
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

        return probs

    def model_induced_prev_step(
        self,
        model_logits: torch.Tensor,
        x_t: torch.Tensor,
        n: Union[torch.Tensor, int]
    ) -> torch.Tensor:
        """
        Predicts backward step probabilities P(x_{t-1} | x_t) marginalized over source estimates.

        Integrates the network's source logits output to perform reverse trajectory sampling.

        Args:
            model_logits (torch.Tensor): Unnormalized predicted logits for the source state x_0 (Shape: [B, D, S_max]).
            x_t (torch.Tensor): Class indices at the current step (Shape: [B, D]).
            n (Union[torch.Tensor, int]): Current discrete time step index.

        Returns:
            torch.Tensor: Aggregated reverse transition probabilities of shape [B, D, S_max].
        """
        x_t = torch.as_tensor(x_t, device=self.device, dtype=torch.long)

        batch_size = x_t.shape[0]

        if not isinstance(n, torch.Tensor):
            n = torch.tensor(n, device=self.device)
        if n.dim() == 0:
            n = n.expand(batch_size)
        n = n.view(-1).long()

        t_prev = (n - 1).clamp(0, self.total_number_of_q_powers)
        t_curr = n.clamp(0, self.total_number_of_q_powers)

        b_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1)
        d_idx = torch.arange(self.D, device=self.device).view(1, self.D)

        masked_logits = model_logits.masked_fill(~self.valid_mask, -1e9)
        p_model = torch.softmax(masked_logits, dim=-1)  # [B, D, S_max]

        Q_1 = self._powers[:, 1]  # [D, S_max, S_max]
        Q_to_prev = self._powers[:, t_prev].permute(1, 0, 2, 3)  # [B, D, S_max, S_max]
        Q_to_curr = self._powers[:, t_curr].permute(1, 0, 2, 3)  # [B, D, S_max, S_max]

        norm_den = Q_to_curr[b_idx, d_idx, :, x_t]  # [B, D, S_max]

        term_to_sum = p_model / (norm_den + 1e-12)  # [B, D, S_max]

        # b (batch), d (dim), i (start_state), j (prev_state x_{t-1}).
        # term_to_sum(b, d, i) * Q_to_prev(b, d, i, j) -> (b, d, j)
        summed_starts = torch.einsum('bdi, bdij -> bdj', term_to_sum, Q_to_prev)  # [B, D, S_max]

        w_step_back = Q_1[d_idx, :, x_t]  # [B, D, S_max]

        probs = w_step_back * summed_starts  # [B, D, S_max]
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

        return probs

    def update_alpha(self, new_alpha: float) -> None:
        """
        Updates the reference model's alpha rate parameter and updates transition matrix caches.

        Args:
            new_alpha (float): New alpha parameter value.
        """
        self.alpha = new_alpha

        self._powers.zero_()

        for d in range(self.D):
            S_d = int(self.S[d].item())
            is_ord = bool(self.is_ordered[d].item())
            for k in range(self.total_number_of_q_powers + 1):
                if is_ord:
                    matrix = self._build_gaussian_k_matrix(S_d, k)
                else:
                    matrix = self._build_uniform_k_matrix(S_d, k)

                self._powers[d, k, :S_d, :S_d] = matrix

    def sample_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Draws discrete feature class realizations from a given probability distribution.

        Args:
            probs (torch.Tensor): Transition probabilities tensor (Shape: [B, D, S_max]).

        Raises:
            ValueError: If probabilities contain NaN/Inf or if a dimension sum maps to an invalid distribution.

        Returns:
            torch.Tensor: Randomly drawn categorical index realizations of shape [B, D] (dtype=long).
        """
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            raise ValueError(
                "Categorical reference received NaNs or Infs in probabilities. "
                "The neural network diverged due to bad hyperparameters."
            )

        batch_size, dims, s_max = probs.shape

        arange = torch.arange(s_max, device=self.device).view(1, 1, s_max)
        mask = arange < self.S.view(1, dims, 1)

        masked_probs = (probs + 1e-12) * mask

        flat_probs = masked_probs.reshape(-1, s_max)

        if (flat_probs.sum(dim=-1) <= 0).any():
            raise ValueError("Some rows in flat_probs have zero or negative sum after masking.")

        samples = torch.multinomial(flat_probs, num_samples=1)
        return samples.view(batch_size, dims)

    def sample_x_t(
        self,
        x_start: torch.Tensor,
        x_target: torch.Tensor,
        t: torch.Tensor,
        total_steps: int
    ) -> torch.Tensor:
        """
        Samples an explicit intermediate state representation from a conditional path bridge.

        Args:
            x_start (torch.Tensor): Source class assignments indices (Shape: [B, D]).
            x_target (torch.Tensor): Endpoint target class assignments indices (Shape: [B, D]).
            t (torch.Tensor): Time step index slice array.
            total_steps (int): Comprehensive time-grid discretization size (K).

        Returns:
            torch.Tensor: Categorical features tensor sampled at step t (Shape: [B, D]).
        """
        probs = self.bridge_at_time(x_start, x_target, t, total_steps)
        return self.sample_from_probs(probs)

    def sample_step(
        self,
        x_t: torch.Tensor,
        x_target: torch.Tensor,
        n: torch.Tensor,
        total_steps: int
    ) -> torch.Tensor:
        """
        Draws a single-step discrete feature transition using endpoint-conditioned lookaheads.

        Args:
            x_t (torch.Tensor): Current state class assignments indices (Shape: [B, D]).
            x_target (torch.Tensor): Endpoint goal class indices (Shape: [B, D]).
            n (torch.Tensor): Grid location timeline step tracker.
            total_steps (int): Total number of time grid steps.

        Returns:
            torch.Tensor: Sampled adjacent next-step class assignments indices (Shape: [B, D]).
        """
        probs = self.bridge_next_given_prev(x_t, x_target, n, total_steps)
        return self.sample_from_probs(probs)
