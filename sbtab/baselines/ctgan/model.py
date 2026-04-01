"""
from sbtab.baselines.ctgan.model import CTGANConfig, CTGANWrapper
cfg = CTGANConfig(**ctgan_params)
model = CTGANWrapper(cfg)

model.fit(D_train_norm)
X_syn_norm = model.sample(len(D_train_norm))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN

from sbtab.baselines.base import BaselineGenerativeModel, BaselineFitInfo, ArrayLike

@dataclass
class CTGANConfig:
    """
    Configuration for the CTGAN baseline.
    """
    embedding_dim: int = 128
    generator_dim: Tuple[int, int] = (512, 512)
    discriminator_dim: Tuple[int, int] = (512, 512)
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    batch_size: int = 500
    epochs: int = 300
    pac: int = 10
    cuda: bool = True
    seed: int = 42
    verbose: bool = False
    # Adding the ability to explicitly specify discrete speakers
    discrete_columns: Optional[List[str]] = None

class CTGANWrapper(BaselineGenerativeModel):
    """
    General-purpose CTGAN wrapper. 
    Handles both continuous and discrete data automatically.
    """

    def __init__(self, cfg: CTGANConfig):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.model = CTGAN(
            embedding_dim=cfg.embedding_dim,
            generator_dim=cfg.generator_dim,
            discriminator_dim=cfg.discriminator_dim,
            generator_lr=cfg.generator_lr,
            discriminator_lr=cfg.discriminator_lr,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            pac=cfg.pac,
            verbose=cfg.verbose,
            cuda=cfg.cuda if torch.cuda.is_available() else False
        )
        self._fitted = False
        self.columns_: Optional[list[str]] = None

    def fit(self, data: ArrayLike, **kwargs: Any) -> "CTGANWrapper":
        """
        Fits CTGAN on any tabular data.
        Automatically detects discrete columns if not provided in config.
        """
        # 1. Conversion to pandas.DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            self.columns_ = list(df.columns)
        else:
            self.columns_ = [f"f{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=self.columns_)

        # 2. Definition of discrete columns (Logic "in general")
        # First, we check whether they are passed to the fit method directly
        discrete_cols = kwargs.get("discrete_columns", self.cfg.discrete_columns)

        if discrete_cols is None:
            # If no list is specified, we find all non-numeric columns automatically
            # This makes the model suitable for any data "out of the box"
            discrete_cols = [
                    col for col in df.columns 
                    if not pd.api.types.is_numeric_dtype(df[col]) 
                    or pd.api.types.is_bool_dtype(df[col])
                ]

        #3. Saving training metadata
        self.fit_info_ = BaselineFitInfo(
            n_rows=int(df.shape[0]),
            n_cols=int(df.shape[1]),
            columns=self.columns_,
        )

        #4. Training
        self.model.fit(df, discrete_columns=discrete_cols)
        
        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Generates n synthetic samples in the original data format.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        
        if n <= 0:
            raise ValueError("n must be positive")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        syn_df = self.model.sample(n)
        
        # We return the columns in the same order as they were at the entrance
        return syn_df[self.columns_]