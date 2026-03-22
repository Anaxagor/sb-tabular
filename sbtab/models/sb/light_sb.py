from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class LightSBM(nn.Module):

    def __init__(
        self,
        dim: int = 2,
        n_potentials: int = 5,
        epsilon: float = 1.0,
        is_diagonal: bool = True,
        sampling_batch_size: int = 1,
        S_diagonal_init: float = 0.1,
    ):
        super().__init__()

        if not is_diagonal:
            try:
                import geotorch  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "is_diagonal=False requires the `geotorch` package. "
                    "Install it via: pip install geotorch"
                ) from e

        self.is_diagonal = is_diagonal
        self.dim = int(dim)
        self.n_potentials = int(n_potentials)
        self.register_buffer("epsilon", torch.tensor(float(epsilon)))
        self.sampling_batch_size = int(sampling_batch_size)

        self.log_alpha = nn.Parameter(
            torch.log(torch.ones(n_potentials) / n_potentials)
        )
        self.r = nn.Parameter(torch.randn(n_potentials, dim))
        self.S_log_diagonal_matrix = nn.Parameter(
            torch.log(S_diagonal_init * torch.ones(n_potentials, self.dim))
        )
        self.S_rotation_matrix = nn.Parameter(
            torch.randn(n_potentials, self.dim, self.dim)
        )

        if not is_diagonal:
            import geotorch
            geotorch.orthogonal(self, "S_rotation_matrix")

    def get_S(self) -> torch.Tensor:
        if self.is_diagonal:
            return torch.exp(self.S_log_diagonal_matrix)
        else:
            S_diag = torch.exp(self.S_log_diagonal_matrix)
            R = self.S_rotation_matrix
            return (R * S_diag[:, None, :]) @ R.permute(0, 2, 1)

    def get_r(self) -> torch.Tensor:
        return self.r

    def init_r_by_samples(self, samples: torch.Tensor) -> None:
        assert samples.shape[0] == self.r.shape[0]
        self.r.data = samples.clone().to(self.r.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.log_alpha

        samples = []
        batch_size = x.shape[0]
        sbs = self.sampling_batch_size

        n_iter = (
            batch_size // sbs
            if batch_size % sbs == 0
            else (batch_size // sbs) + 1
        )

        for i in range(n_iter):
            sub_x = x[sbs * i : sbs * (i + 1)]

            if self.is_diagonal:
                x_S_x = (sub_x[:, None, :] * S[None, :, :] * sub_x[:, None, :]).sum(dim=-1)
                x_r = (sub_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + S[None, :] * sub_x[:, None, :]
            else:
                x_S_x = (
                    sub_x[:, None, None, :] @ (S[None, :, :, :] @ sub_x[:, None, :, None])
                )[:, :, 0, 0]
                x_r = (sub_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + (S[None, :, :, :] @ sub_x[:, None, :, None])[:, :, :, 0]

            exp_argument = (x_S_x + 2 * x_r) / (2 * epsilon) + log_alpha[None, :]

            if self.is_diagonal:
                mix = Categorical(logits=exp_argument)
                comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon * S)[None, :, :]), 1)
            else:
                mix = Categorical(logits=exp_argument)
                comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon * S)

            gmm = MixtureSameFamily(mix, comp)
            samples.append(gmm.sample())

        return torch.cat(samples, dim=0)

    @torch.enable_grad()
    def get_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)

        epsilon = self.epsilon
        r = self.get_r()

        S_diagonal = torch.exp(self.S_log_diagonal_matrix)
        A_diagonal = (
            (t / (epsilon * (1 - t)))[:, None, None]
            + 1 / (epsilon * S_diagonal)[None, :, :]
        )

        S_log_det = torch.sum(self.S_log_diagonal_matrix, dim=-1)
        A_log_det = torch.sum(torch.log(A_diagonal), dim=-1)

        log_alpha = self.log_alpha

        if self.is_diagonal:
            S = S_diagonal
            A = A_diagonal
            S_inv = 1 / S
            A_inv = 1 / A

            c = (
                ((1 / (epsilon * (1 - t)))[:, None] * x)[:, None, :]
                + (r / (epsilon * S_diagonal))[None, :, :]
            )

            exp_arg = (
                log_alpha[None, :]
                - 0.5 * S_log_det[None, :]
                - 0.5 * A_log_det
                - 0.5 * ((r * S_inv * r) / epsilon).sum(dim=-1)[None, :]
                + 0.5 * (c * A_inv * c).sum(dim=-1)
            )

        else:
            R = self.S_rotation_matrix
            S = (R * S_diagonal[:, None, :]) @ R.permute(0, 2, 1)
            A = (R[None, :, :, :] * A_diagonal[:, :, None, :]) @ R.permute(0, 2, 1)[None, :, :, :]

            S_inv = (R * (1 / S_diagonal[:, None, :])) @ R.permute(0, 2, 1)
            A_inv = (R[None, :, :, :] * (1 / A_diagonal[:, :, None, :])) @ R.permute(0, 2, 1)[None, :, :, :]

            c = (
                ((1 / (epsilon * (1 - t)))[:, None] * x)[:, None, :]
                + (S_inv @ r[:, :, None])[None, :, :, 0] / epsilon
            )

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
            grad_outputs=torch.ones_like(lse),
            create_graph=True,
        )[0]

        drift = -x / (1 - t[:, None]) + epsilon * grad
        return drift

    def sample_at_time_moment(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.to(x.device)
        y = self(x)
        return t * y + (1 - t) * x + torch.sqrt(t * (1 - t) * self.epsilon) * torch.randn_like(x)

    def get_log_potential(self, x: torch.Tensor) -> torch.Tensor:
        S = self.get_S()
        r = self.get_r()

        if self.is_diagonal:
            mix = Categorical(logits=self.log_alpha)
            comp = Independent(Normal(loc=r, scale=torch.sqrt(self.epsilon * S)), 1)
        else:
            mix = Categorical(logits=self.log_alpha)
            comp = MultivariateNormal(loc=r, covariance_matrix=self.epsilon * S)

        gmm = MixtureSameFamily(mix, comp)
        return gmm.log_prob(x) + torch.logsumexp(self.log_alpha, dim=-1)

    def get_log_C(self, x: torch.Tensor) -> torch.Tensor:
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.log_alpha

        if self.is_diagonal:
            x_S_x = (x[:, None, :] * S[None, :, :] * x[:, None, :]).sum(dim=-1)
            x_r = (x[:, None, :] * r[None, :, :]).sum(dim=-1)
        else:
            x_S_x = (x[:, None, None, :] @ (S[None, :, :, :] @ x[:, None, :, None]))[:, :, 0, 0]
            x_r = (x[:, None, :] * r[None, :, :]).sum(dim=-1)

        exp_argument = (x_S_x + 2 * x_r) / (2 * epsilon) + log_alpha[None, :]
        return torch.logsumexp(exp_argument, dim=-1)

    def get_alpha(self, x: torch.Tensor) -> torch.Tensor:
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.log_alpha

        if self.is_diagonal:
            x_S_x = (x[:, None, :] * S[None, :, :] * x[:, None, :]).sum(dim=-1)
            x_r = (x[:, None, :] * r[None, :, :]).sum(dim=-1)
        else:
            x_S_x = (x[:, None, None, :] @ (S[None, :, :, :] @ x[:, None, :, None]))[:, :, 0, 0]
            x_r = (x[:, None, :] * r[None, :, :]).sum(dim=-1)

        exp_argument = (x_S_x + 2 * x_r) / (2 * epsilon) + log_alpha[None, :]
        return torch.softmax(exp_argument, dim=-1)

    @torch.no_grad()
    def sample_euler_maruyama(
        self,
        x: torch.Tensor,
        n_steps: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        epsilon = self.epsilon
        t = torch.zeros(x.shape[0], device=x.device)
        dt = 1.0 / n_steps
        trajectory = [x]

        for _ in range(n_steps):
            drift = self.get_drift(x, t)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = x + drift * dt + math.sqrt(dt) * torch.sqrt(epsilon) * noise
            t = t + dt
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)

    def set_epsilon(self, new_epsilon: float) -> None:
        self.epsilon = torch.tensor(float(new_epsilon), device=self.epsilon.device)
