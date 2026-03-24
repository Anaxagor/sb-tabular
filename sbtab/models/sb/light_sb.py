from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


@dataclass(frozen=True)
class LightSBPotentialConfig:
    """
    Parameterization config for the LightSB adjusted Schrödinger potential.

    The model follows the reference implementation and paper:
      v_theta(x) = sum_k alpha_k * N(x | r_k, epsilon * S_k)

    Notes:
      - `is_diagonal=True` is the recommended practical setting from the paper.
      - `sampling_batch_size` only affects conditional sampling in `forward()`.
    """

    n_potentials: int = 50
    epsilon: float = 0.1
    is_diagonal: bool = True
    sampling_batch_size: int = 256
    S_diagonal_init: float = 0.1


class LightSBPotential(nn.Module):
    """
    Proper LightSB model.

    Implements the same core logic as the official LightSB reference model:
      - conditional plan sampling π_theta(x1 | x0) via `forward`
      - exact drift g_theta(x, t) via `get_drift`
      - log unnormalized potential log v_theta(x) via `get_log_potential`
      - log normalizer log C_theta(x0) via `get_log_C`

    The diagonal case is dependency-free. The full-covariance case uses geotorch
    to maintain orthogonal rotation matrices, matching the reference code.
    """

    def __init__(self, dim: int, cfg: LightSBPotentialConfig):
        super().__init__()
        self.dim = int(dim)
        self.cfg = cfg
        self.is_diagonal = bool(cfg.is_diagonal)
        self.n_potentials = int(cfg.n_potentials)
        self.sampling_batch_size = int(cfg.sampling_batch_size)

        if self.n_potentials <= 0:
            raise ValueError("n_potentials must be positive")
        if cfg.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if cfg.S_diagonal_init <= 0:
            raise ValueError("S_diagonal_init must be positive")

        if not self.is_diagonal:
            try:
                import geotorch  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "LightSBPotential with is_diagonal=False requires `geotorch`. "
                    "Install it with: pip install geotorch"
                ) from e

        self.register_buffer("epsilon", torch.tensor(float(cfg.epsilon), dtype=torch.float32))

        # Reference LightSB parameterization: store epsilon * log(alpha) and divide by
        # epsilon in `get_log_alpha()`. This stabilizes optimization for small epsilon.
        init_log_alpha = self.epsilon * torch.log(
            torch.ones(self.n_potentials, dtype=torch.float32) / self.n_potentials
        )
        self.log_alpha_raw = nn.Parameter(init_log_alpha)

        # Gaussian centers.
        self.r = nn.Parameter(torch.randn(self.n_potentials, self.dim, dtype=torch.float32))

        # Positive diagonal entries (or eigenvalues in the non-diagonal case).
        self.S_log_diagonal_matrix = nn.Parameter(
            torch.log(
                float(cfg.S_diagonal_init)
                * torch.ones(self.n_potentials, self.dim, dtype=torch.float32)
            )
        )

        # Orthogonal rotations for the non-diagonal case.
        self.S_rotation_matrix = nn.Parameter(
            torch.randn(self.n_potentials, self.dim, self.dim, dtype=torch.float32)
        )
        if not self.is_diagonal:
            import geotorch

            geotorch.orthogonal(self, "S_rotation_matrix")

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------

    def init_r_by_samples(self, samples: torch.Tensor) -> None:
        """
        Initialize centers r_k from data samples.

        Expects exactly n_potentials samples, matching the reference implementation.
        """
        if samples.ndim != 2 or samples.shape[1] != self.dim:
            raise ValueError(
                f"samples must have shape ({self.n_potentials}, {self.dim}) or at least (?, {self.dim})"
            )
        if samples.shape[0] != self.n_potentials:
            raise ValueError(
                f"init_r_by_samples expects exactly {self.n_potentials} samples, got {samples.shape[0]}"
            )
        self.r.data = torch.clone(samples.to(self.r.device, dtype=self.r.dtype))

    def get_S(self) -> torch.Tensor:
        """
        Return covariance factors S_k.

        Shape:
          - diagonal case: (K, D), interpreted as diagonal entries
          - full case:     (K, D, D)
        """
        if self.is_diagonal:
            S = torch.exp(self.S_log_diagonal_matrix)
        else:
            diag = torch.exp(self.S_log_diagonal_matrix)[:, None, :]  # (K,1,D)
            R = self.S_rotation_matrix
            S = (R * diag) @ torch.permute(R, (0, 2, 1))
        return S

    def get_r(self) -> torch.Tensor:
        return self.r

    def get_log_alpha(self) -> torch.Tensor:
        return (1.0 / self.epsilon) * self.log_alpha_raw

    # ------------------------------------------------------------------
    # Conditional plan π_theta(x1 | x0)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Sample x1 ~ π_theta(. | x0).

        This matches Proposition 3.2 / the official implementation. The returned
        samples are *conditional plan* samples, not SDE trajectories.
        """
        if x0.ndim != 2 or x0.shape[1] != self.dim:
            raise ValueError(f"x0 must have shape (B, {self.dim}), got {tuple(x0.shape)}")

        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.get_log_alpha()

        samples = []
        batch_size = x0.shape[0]
        chunk = max(1, int(self.sampling_batch_size))
        n_iter = batch_size // chunk if batch_size % chunk == 0 else batch_size // chunk + 1

        for i in range(n_iter):
            sub_x = x0[chunk * i : chunk * (i + 1)]
            if sub_x.shape[0] == 0:
                continue

            if self.is_diagonal:
                x_S_x = (sub_x[:, None, :] * S[None, :, :] * sub_x[:, None, :]).sum(dim=-1)
                x_r = (sub_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + S[None, :, :] * sub_x[:, None, :]

                logits = (x_S_x + 2.0 * x_r) / (2.0 * epsilon) + log_alpha[None, :]
                mix = Categorical(logits=logits)
                comp = Independent(
                    Normal(loc=r_x, scale=torch.sqrt(epsilon * S)[None, :, :]), 1
                )
                gmm = MixtureSameFamily(mix, comp)
            else:
                x_S_x = (
                    sub_x[:, None, None, :] @ (S[None, :, :, :] @ sub_x[:, None, :, None])
                )[:, :, 0, 0]
                x_r = (sub_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + (S[None, :, :, :] @ sub_x[:, None, :, None])[:, :, :, 0]

                logits = (x_S_x + 2.0 * x_r) / (2.0 * epsilon) + log_alpha[None, :]
                mix = Categorical(logits=logits)
                comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon * S)
                gmm = MixtureSameFamily(mix, comp)

            samples.append(gmm.sample())

        return torch.cat(samples, dim=0)

    # ------------------------------------------------------------------
    # Exact drift of the associated bridge process
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def get_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the exact LightSB drift g_theta(x, t).

        This matches Proposition 3.3 / the official implementation.

        Args:
            x: (B, D)
            t: (B,) or (B,1), values in [0, 1)
        Returns:
            drift: (B, D)
        """
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (B, {self.dim}), got {tuple(x.shape)}")

        x = torch.clone(x)
        x.requires_grad_(True)

        t = t.reshape(-1).to(x.device, dtype=x.dtype)
        if t.shape[0] != x.shape[0]:
            raise ValueError("t must have the same batch size as x")
        t = torch.clamp(t, min=0.0, max=1.0 - 1e-7)

        epsilon = self.epsilon
        r = self.get_r()

        # The reference implementation uses the diagonal entries even in the full case,
        # because log|S_k| only depends on the diagonal spectrum under orthogonal rotation.
        S_diagonal = torch.exp(self.S_log_diagonal_matrix)  # (K, D)
        A_diagonal = (
            (t / (epsilon * (1.0 - t)))[:, None, None]
            + 1.0 / (epsilon * S_diagonal)[None, :, :]
        )  # (B, K, D)

        S_log_det = torch.sum(self.S_log_diagonal_matrix, dim=-1)    # (K,)
        A_log_det = torch.sum(torch.log(A_diagonal), dim=-1)         # (B, K)
        log_alpha = self.get_log_alpha()                             # (K,)

        if self.is_diagonal:
            S = S_diagonal
            A = A_diagonal
            S_inv = 1.0 / S
            A_inv = 1.0 / A

            c = ((1.0 / (epsilon * (1.0 - t)))[:, None] * x)[:, None, :] + (
                r / (epsilon * S_diagonal)
            )[None, :, :]

            exp_arg = (
                log_alpha[None, :]
                - 0.5 * S_log_det[None, :]
                - 0.5 * A_log_det
                - 0.5 * ((r * S_inv * r) / epsilon).sum(dim=-1)[None, :]
                + 0.5 * (c * A_inv * c).sum(dim=-1)
            )
        else:
            R = self.S_rotation_matrix
            S = (R * S_diagonal[:, None, :]) @ torch.permute(R, (0, 2, 1))
            A = (
                R[None, :, :, :] * A_diagonal[:, :, None, :]
            ) @ torch.permute(R, (0, 2, 1))[None, :, :, :]

            S_inv = (R * (1.0 / S_diagonal[:, None, :])) @ torch.permute(R, (0, 2, 1))
            A_inv = (
                R[None, :, :, :] * (1.0 / A_diagonal[:, :, None, :])
            ) @ torch.permute(R, (0, 2, 1))[None, :, :, :]

            c = ((1.0 / (epsilon * (1.0 - t)))[:, None] * x)[:, None, :] + (
                S_inv @ (r[:, :, None])
            )[None, :, :, 0] / epsilon

            c_A_inv_c = (c[:, :, None, :] @ A_inv @ c[:, :, :, None])[:, :, 0, 0]
            r_S_inv_r = (r[:, None, :] @ S_inv @ r[:, :, None])[None, :, 0, 0]

            exp_arg = (
                log_alpha[None, :]
                - 0.5 * S_log_det[None, :]
                - 0.5 * A_log_det
                - 0.5 * r_S_inv_r / epsilon
                + 0.5 * c_A_inv_c
            )

        lse = torch.logsumexp(exp_arg, dim=-1)
        grad = torch.autograd.grad(
            lse,
            x,
            grad_outputs=torch.ones_like(lse, device=lse.device),
            create_graph=False,
        )[0]

        drift = -x / (1.0 - t[:, None]) + epsilon * grad
        return drift.detach()

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_euler_maruyama(
        self,
        x0: torch.Tensor,
        n_steps: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Simulate the associated bridge process with Euler-Maruyama.

        Returns a trajectory tensor of shape (B, n_steps + 1, D).
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        x = x0
        t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        dt = 1.0 / float(n_steps)
        trajectory = [x.clone()]

        for _ in range(n_steps):
            drift = self.get_drift(x, t)
            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            x = x + drift * dt + math.sqrt(dt) * torch.sqrt(self.epsilon) * noise
            t = t + dt
            trajectory.append(x.clone())

        return torch.stack(trajectory, dim=1)

    @torch.no_grad()
    def sample_at_time_moment(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from the Brownian bridge interpolation used by the model.
        """
        t = t.to(x0.device, dtype=x0.dtype)
        y = self(x0)
        return t * y + (1.0 - t) * x0 + torch.sqrt(t * (1.0 - t) * self.epsilon) * torch.randn_like(x0)

    # ------------------------------------------------------------------
    # Quantities for the LightSB objective
    # ------------------------------------------------------------------

    def get_log_potential(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log v_theta(x), i.e. the log of the unnormalized Gaussian mixture.
        """
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (B, {self.dim}), got {tuple(x.shape)}")

        S = self.get_S()
        r = self.get_r()
        log_alpha = self.get_log_alpha()

        if self.is_diagonal:
            mix = Categorical(logits=log_alpha)
            comp = Independent(Normal(loc=r, scale=torch.sqrt(self.epsilon * S)), 1)
            gmm = MixtureSameFamily(mix, comp)
        else:
            mix = Categorical(logits=log_alpha)
            comp = MultivariateNormal(loc=r, covariance_matrix=self.epsilon * S)
            gmm = MixtureSameFamily(mix, comp)

        # MixtureSameFamily uses normalized mixture weights internally, so we add back
        # logsumexp(log_alpha) to recover the *unnormalized* potential v_theta.
        return gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)

    def get_log_C(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log C_theta(x0) = log ∫ exp(<x0, x1>/eps) v_theta(x1) dx1
        in closed form, as in the official implementation.
        """
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (B, {self.dim}), got {tuple(x.shape)}")

        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.get_log_alpha()

        if self.is_diagonal:
            x_S_x = (x[:, None, :] * S[None, :, :] * x[:, None, :]).sum(dim=-1)
            x_r = (x[:, None, :] * r[None, :, :]).sum(dim=-1)
        else:
            x_S_x = (
                x[:, None, None, :] @ (S[None, :, :, :] @ x[:, None, :, None])
            )[:, :, 0, 0]
            x_r = (x[:, None, :] * r[None, :, :]).sum(dim=-1)

        exp_argument = (x_S_x + 2.0 * x_r) / (2.0 * epsilon) + log_alpha[None, :]
        return torch.logsumexp(exp_argument, dim=-1)

    def set_epsilon(self, new_epsilon: float) -> None:
        with torch.no_grad():
            self.epsilon.fill_(float(new_epsilon))


# Backward-compatible alias; if existing code expects LightSBM, it can import this.
LightSBM = LightSBPotential
