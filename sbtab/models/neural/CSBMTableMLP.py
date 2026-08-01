from typing import List

import torch
from torch import nn

from sbtab.models.neural.time_embedding import SinusoidalTimeEmbedding, SinusoidalTimeEmbeddingConfig


class CSBMTableMLP(nn.Module):
    def __init__(
            self,
            cardinalities: List[int],
            n_layers: int = 2,
            emb_dim: int = 16,
            hidden_dim: int = 256,
            time_dim: int = 64,
            dropout: float = 0.0
    ) -> None:
        """
        Init of the MLP backbone module which is used for the CSBM solver.

        Args:
            cardinalities (List[int]): List of cardinalities (number of possible values) of each category.
            n_layers (int, optional): Number of layers in the MLP. Defaults to 2.
            emb_dim (int, optional): Dimension of the categorical embeddings. Defaults to 16.
            hidden_dim (int, optional): Dimension of the hidden layers in the MLP. Defaults to 256.
            time_dim (int, optional): Dimension of the time embeddings. Defaults to 64.
            dropout (float, optional): Dropout rate. Defaults to 0.0.

        Returns:
            None
        """
        super().__init__()
        self.cardinalities = cardinalities
        self.D = len(self.cardinalities)
        self.S_max = max(self.cardinalities)
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in self.cardinalities])
        self.time_emb = SinusoidalTimeEmbedding(SinusoidalTimeEmbeddingConfig(dim=time_dim))

        layers = []
        in_dim = self.D * emb_dim + time_dim

        for i in range(n_layers):
            di = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(di, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, self.D * self.S_max)

        mask = torch.full((self.D, self.S_max), -1e9)
        for i, c in enumerate(self.cardinalities):
            mask[i, :c] = 0.0
        self.register_buffer("logit_mask", mask)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of the CSBM MLP backbone.

        Args:
            x (torch.Tensor): Input categorical features tensor of shape (B, D).
            t (torch.Tensor): Normalized time steps of shape (B,) or (B, 1).
                Must contain float values scaled to the [0, 1] range. Typically
                computed as (n / K), where 'n' is the discrete time step index
                and 'K' is the total number of steps in the time grid.

        Returns:
            torch.Tensor: Predicted categorical logits of shape (B, D, S_max).
        """
        B = x.shape[0]
        embeddings = [self.embs[i](torch.clamp(x[:, i].long(), 0, self.cardinalities[i] - 1)) for i in range(self.D)]
        h = torch.cat(embeddings, dim=-1)
        te = self.time_emb(t.view(-1, 1))

        logits = self.output_layer(self.net(torch.cat([h, te], dim=-1)))
        logits = logits.view(B, self.D, self.S_max)
        logits += self.logit_mask.unsqueeze(0)

        return logits
