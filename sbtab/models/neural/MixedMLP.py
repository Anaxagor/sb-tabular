from typing import List

import torch
import torch.nn as nn

from sbtab.models.neural.time_embedding import SinusoidalTimeEmbedding, SinusoidalTimeEmbeddingConfig

class MixedSbmMlp(nn.Module):
    def __init__(
        self,
        continuous_dim: int,
        cardinalities: List[int],
        cat_emb_dim: int,
        hidden_dim: int,
        time_dim: int,
        n_layers: int,
        dropout: float = 0.0
    ) -> None:
        """
        Init of the MLP backbone module which is used for the MSBM solver.

        Args:
            continuous_dim (int): Dimension of the continuous part of the data.
            cardinalities (List[int]): List of cardinalities (number of possible values) of each category.
            cat_emb_dim (int): Dimension of the embeddings of categorical part of the data.
            hidden_dim (int): Dimension of the hidden layers in the MLP.
            time_dim (int): Dimension of the time embeddings.
            n_layers (int): Number of layers in the MLP.
            dropout (float): Dropout rate for the MLP. Defaults to 0.0.

        Raises:
            TypeError: If any element of cardinalities is not an integer.

        Returns:
            None
        """

        super().__init__()
        if not all(isinstance(c, int) for c in cardinalities):
            raise TypeError("All elements of cardinalities list should be integers")

        self.num_of_num_features = continuous_dim
        self.cardinalities = cardinalities
        self.num_of_cat_features = len(self.cardinalities)
        self.S_max = max(self.cardinalities) if self.num_of_cat_features > 0 else 0

        self.time_emb = SinusoidalTimeEmbedding(SinusoidalTimeEmbeddingConfig(dim=time_dim))
        total_categories = sum(self.cardinalities)
        self.cat_emb = nn.Embedding(total_categories, cat_emb_dim)

        offsets = []
        curr_offset = 0
        for c in self.cardinalities:
            offsets.append(curr_offset)
            curr_offset += c
        self.register_buffer("cat_offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("cat_bounds", torch.tensor(self.cardinalities, dtype=torch.long) - 1)


        layers = []
        in_dim = self.num_of_num_features + self.num_of_cat_features * cat_emb_dim + self.time_emb.dim

        for i in range(n_layers):
            di = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(di, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)

        if self.num_of_num_features > 0:
            self.head_cont = nn.Linear(hidden_dim, self.num_of_num_features)

        if self.num_of_cat_features > 0:
            self.head_cat = nn.Linear(hidden_dim, self.num_of_cat_features * self.S_max)

            mask = torch.full((self.num_of_cat_features, self.S_max), -1e9)
            for i, c in enumerate(self.cardinalities):
                mask[i, :c] = 0.0
            self.register_buffer("logit_mask", mask)

    def forward(
        self,
        x_cont: torch.Tensor | None,
        x_cat: torch.Tensor | None,
        t: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Forward propagation of the MLP.

        Args:
            x_cont (torch.Tensor): Continuous part of the data.
            x_cat (torch.Tensor): Categorical part of the data.
            t (torch.Tensor): Time embeddings.

        Raises:
            ValueError: If there's no data to use or incoming function parameters have wrong shape.

        Returns:
            out_cont (torch.Tensor): continuous head (logits).
            logits (torch.Tensor): categorical head (logits).
        """
        if (x_cont is None or x_cont.nelement() == 0) and (x_cat is None or x_cat.nelement() == 0):
            raise ValueError("There's no data to use.")
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("t must have shape (batch, 1)")

        batch = x_cont.shape[0] if x_cont is not None and x_cont.nelement() != 0 else x_cat.shape[0]
        inputs = []

        if self.num_of_num_features > 0:
            if x_cont is None or x_cont.nelement() == 0:
                raise ValueError("Model expects continuous features, but received None or empty tensor.")
            if x_cont.ndim != 2:
                raise ValueError("x_cont must have shape (batch, num_of_num_features)")
            inputs.append(x_cont)

        if self.num_of_cat_features > 0:
            if x_cat is None or x_cat.nelement() == 0:
                raise ValueError("Model expects categorical features, but received None or empty tensor.")
            if x_cat.ndim != 2:
                raise ValueError("x_cat must have shape (batch, num_of_cat_features)")

            x_cat_long = x_cat.long()
            idx = torch.clamp(x_cat_long, min=torch.zeros_like(self.cat_bounds), max=self.cat_bounds)
            idx += self.cat_offsets

            h_cat = self.cat_emb(idx)
            h_cat = h_cat.view(batch, -1)
            inputs.append(h_cat)

        inputs.append(self.time_emb(t))

        h_input = torch.cat(inputs, dim=1)
        h = self.trunk(h_input)

        out_cont = self.head_cont(h) if self.num_of_num_features > 0 else None
        logits = None

        if self.num_of_cat_features > 0:
            logits = self.head_cat(h)
            logits = logits.view(batch, self.num_of_cat_features, self.S_max)
            logits = logits + self.logit_mask.unsqueeze(0)

        return out_cont, logits

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
