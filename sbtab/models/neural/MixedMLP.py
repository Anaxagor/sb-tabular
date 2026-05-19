from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from sbtab.models.neural.time_embedding import SinusoidalTimeEmbedding, SinusoidalTimeEmbeddingConfig

class MixedSbmMlp(nn.Module):
    def __init__(self, continuous_dim: int, cardinalities: list, cat_emb_dim: int, hidden_dim: int, time_dim: int,
                 n_layers: int, dropout: float = 0.0):
        super(MixedSbmMlp, self).__init__()

        self.D_num = continuous_dim
        self.cardinalities = [int(c) for c in cardinalities]
        self.D_cat = len(self.cardinalities)
        self.S_max = int(max(self.cardinalities)) if self.D_cat > 0 else 0

        self.time_emb = SinusoidalTimeEmbedding(SinusoidalTimeEmbeddingConfig(dim=time_dim))
        self.cat_embs = nn.ModuleList([nn.Embedding(c, cat_emb_dim) for c in self.cardinalities])

        layers = []
        in_dim = self.D_num + self.D_cat * cat_emb_dim + self.time_emb.dim

        for i in range(n_layers):
            di = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(di, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)

        self.head_cont = nn.Linear(hidden_dim, self.D_num)

        if self.D_cat > 0:
            self.head_cat = nn.Linear(hidden_dim, self.D_cat * self.S_max)

            mask = torch.full((self.D_cat, self.S_max), -1e9)
            for i, c in enumerate(self.cardinalities):
                mask[i, :c] = 0.0
            self.register_buffer("logit_mask", mask)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor, t: torch.Tensor) -> tuple[Any, Tensor | Any] | tuple[
        Any, None]:
        if x_cont.ndim != 2:
            raise ValueError("x_cont must have shape (B, D_num)")
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("t must have shape (B, 1)")

        B = x_cont.shape[0]

        te = self.time_emb(t)

        cat_embeddings = []
        for i in range(self.D_cat):
            idx = torch.clamp(x_cat[:, i].long(), 0, self.cardinalities[i] - 1)
            cat_embeddings.append(self.cat_embs[i](idx))

        if self.D_cat > 0:
            h_cat = torch.cat(cat_embeddings, dim=-1)
            h_input = torch.cat([x_cont, h_cat, te], dim=1)
        else:
            h_input = torch.cat([x_cont, te], dim=1)

        h = self.trunk(h_input)

        out_cont = self.head_cont(h)

        if self.D_cat > 0:
            logits = self.head_cat(h)

            logits = logits.view(B, self.D_cat, self.S_max)
            logits = logits + self.logit_mask.unsqueeze(0)
            return out_cont, logits

        return out_cont, None
