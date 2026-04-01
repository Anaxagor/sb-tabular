from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sbtab.baselines.base import ArrayLike, BaselineFitInfo, BaselineGenerativeModel


@dataclass
class STaSyConfig:
    hidden_dim: int = 256
    n_layers: int = 4
    time_emb_dim: int = 64
    dropout: float = 0.0

    sigma_min: float = 0.01
    sigma_max: float = 50.0

    n_epochs: int = 100
    batch_size: int = 512
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0

    use_self_paced: bool = True
    sp_start_ratio: float = 0.25

    n_sampling_steps: int = 1000
    n_corrector_steps: int = 1
    corrector_snr: float = 0.16
    eps: float = 1e-3
    denoise: bool = True

    device: str = "cpu"
    seed: int = 42


class _SigmaEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(10_000.0)
                * torch.arange(dim // 2, dtype=torch.float32)
                / (dim // 2)
            ),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = torch.log(sigma.view(-1, 1))
        args = log_sigma * self.freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _ScoreNet(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        n_layers: int,
        time_emb_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = _SigmaEmbedding(time_emb_dim)
        in_dim = data_dim + time_emb_dim
        layers: list[nn.Module] = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        emb = self.emb(sigma)
        h = torch.cat([x, emb], dim=-1)
        return self.net(h)


class STaSyGenerative(BaselineGenerativeModel):
    def __init__(self, cfg: STaSyConfig) -> None:
        super().__init__(seed=cfg.seed)
        self.cfg = cfg
        self._score_net: Optional[_ScoreNet] = None
        self._dim: Optional[int] = None

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.cfg.sigma_min * (self.cfg.sigma_max / self.cfg.sigma_min) ** t

    def fit(self, data: ArrayLike, **kwargs: Any) -> "STaSyGenerative":
        if isinstance(data, pd.DataFrame):
            self.columns_ = list(data.columns)
            X = data.to_numpy(dtype=np.float32, copy=True)
        else:
            X = np.asarray(data, dtype=np.float32)

        n_rows, n_cols = X.shape
        self._dim = n_cols
        self.fit_info_ = BaselineFitInfo(
            n_rows=int(n_rows),
            n_cols=int(n_cols),
            columns=self.columns_ or [],
        )

        cfg = self.cfg
        device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        score_net = _ScoreNet(
            data_dim=self._dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            time_emb_dim=cfg.time_emb_dim,
            dropout=cfg.dropout,
        ).to(device)

        opt = torch.optim.Adam(
            score_net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        sp_ratio = cfg.sp_start_ratio if cfg.use_self_paced else 1.0
        sp_step = (1.0 - sp_ratio) / max(cfg.n_epochs - 1, 1)

        score_net.train()
        for epoch in range(cfg.n_epochs):
            t_max = min(sp_ratio + sp_step * epoch, 1.0)

            for (x_batch,) in loader:
                x_batch = x_batch.to(device)
                B = x_batch.shape[0]

                t = torch.rand(B, device=device) * (t_max - cfg.eps) + cfg.eps
                sigma = self._sigma(t).unsqueeze(1)

                noise = torch.randn_like(x_batch)
                x_noisy = x_batch + sigma * noise

                score_pred = score_net(x_noisy, sigma)
                loss = (sigma * score_pred + noise).pow(2).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(score_net.parameters(), cfg.grad_clip)
                opt.step()

        score_net.eval()
        self._score_net = score_net
        return self

    @torch.no_grad()
    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> ArrayLike:
        if self._score_net is None or self._dim is None:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        cfg = self.cfg
        device = torch.device(cfg.device)

        if seed is not None:
            torch.manual_seed(seed)

        N = cfg.n_sampling_steps
        ts = torch.linspace(1.0, cfg.eps, N + 1, device=device)

        x = torch.randn(n, self._dim, device=device) * cfg.sigma_max

        for i in range(N):
            t_cur = ts[i]
            t_next = ts[i + 1]

            sigma_cur = self._sigma(t_cur)
            sigma_sq_cur = sigma_cur ** 2
            sigma_sq_next = self._sigma(t_next) ** 2
            diff = sigma_sq_cur - sigma_sq_next

            sigma_cur_vec = sigma_cur.view(1, 1).expand(n, 1)

            for _ in range(cfg.n_corrector_steps):
                score = self._score_net(x, sigma_cur_vec)
                noise = torch.randn_like(x)

                grad_norm = score.norm(dim=-1).mean().clamp(min=1e-8)
                noise_norm = noise.norm(dim=-1).mean().clamp(min=1e-8)
                alpha = 2.0 * (cfg.corrector_snr * noise_norm / grad_norm) ** 2

                x = x + alpha * score + (2.0 * alpha).sqrt() * noise

            score = self._score_net(x, sigma_cur_vec)
            noise = torch.randn_like(x)
            x = x + diff * score + diff.sqrt() * noise

        if cfg.denoise:
            sigma_last = self._sigma(ts[-1]).view(1, 1).expand(n, 1)
            score = self._score_net(x, sigma_last)
            x = x + (sigma_last ** 2) * score

        return self._format_output(x.cpu().numpy())
