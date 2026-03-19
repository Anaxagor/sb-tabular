import torch
from torch import nn

from sbtab.models.neural.time_embedding import SinusoidalTimeEmbedding, SinusoidalTimeEmbeddingConfig


class CSBMTableMLP(nn.Module):
    def __init__(self, cardinalities, emb_dim=16, hidden_dim=256, time_dim=64):
        super().__init__()
        self.cardinalities = [int(c) for c in cardinalities]
        self.D = len(self.cardinalities)
        self.S_max = int(max(self.cardinalities))
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in self.cardinalities])
        self.time_emb = SinusoidalTimeEmbedding(SinusoidalTimeEmbeddingConfig(dim=time_dim))
        self.net = nn.Sequential(
            nn.Linear(self.D * emb_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.D * self.S_max)
        )

    def forward(self, x, t):
        B = x.shape[0]
        embeddings = [self.embs[i](torch.clamp(x[:, i], 0, self.cardinalities[i] - 1)) for i in range(self.D)]
        h = torch.cat(embeddings, dim=-1)
        te = self.time_emb(t.view(-1, 1))
        logits = self.net(torch.cat([h, te], dim=-1))
        return logits.view(B, self.D, self.S_max)