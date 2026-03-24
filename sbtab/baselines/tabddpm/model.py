"""
This file is adapted from the official TabDDPM repository: https://github.com/yandex-research/tab-ddpm
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sbtab.baselines.base import BaselineGenerativeModel, BaselineFitInfo, ArrayLike
from sbtab.data.schema import TabularSchema

# import modules TabDDPM
from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .modules import MLPDiffusion
from .utils import ohe_to_categories

@dataclass
class TabDDPMConfig:
    """
    Configuration for the official TabDDPM implementation.
    """
    num_timesteps: int = 1000
    n_epochs: int = 1000
    batch_size: int = 4096
    lr: float = 0.001
    weight_decay: float = 1e-4
    
    # MLP params (default from original paper)
    d_layers: List[int] = field(default_factory=lambda: [256, 512, 512, 256])
    dropout: float = 0.0
    
    # Diffusion params
    gaussian_loss_type: str = 'mse' # mse or kl
    scheduler: str = 'cosine'      # cosine or linear
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

class TabDDPMWrapper(BaselineGenerativeModel):
    """
    Wrapper for Yandex TabDDPM. 
    Handles mixed data (numerical + categorical) using Multinomial diffusion.
    """

    def __init__(self, cfg: TabDDPMConfig):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        self._fitted = False
        self.columns_: Optional[List[str]] = None
        self.num_numerical_features = 0
        self.num_classes = np.array([], dtype=np.int64)
        self.ordered_cols: List[str] = []

    def _preprocess_data(self, df: pd.DataFrame, schema: TabularSchema) -> torch.Tensor:
        """
        Build a tensor for TabDDPM from ``df`` using ``TabularSchema`` for column roles:
        ``schema.continuous_cols`` for numerical block; optional ``categorical_cols`` on
        the schema object (if present) for categorical columns, else none.
        """
        self.columns_ = list(df.columns)

        num_cols = [c for c in schema.continuous_cols if c in df.columns]
        cat_cols_raw = getattr(schema, "categorical_cols", None)
        if cat_cols_raw is None:
            cat_cols: List[str] = []
        else:
            cat_cols = [c for c in list(cat_cols_raw) if c in df.columns]
        
        # Original TabDDPM expects numerical features first, then categorical
        self.num_numerical_features = len(num_cols)
        self.ordered_cols = num_cols + cat_cols
        
        X_num = df[num_cols].values.astype(np.float32) if num_cols else np.empty((len(df), 0), dtype=np.float32)
        X_cat: List[np.ndarray] = []
        num_classes_list: List[int] = []

        for col in cat_cols:
            codes = df[col].astype("category").cat.codes.values
            n_classes = int(df[col].nunique())
            X_cat.append(codes)
            num_classes_list.append(n_classes)

        if X_cat:
            X_cat_arr = np.stack(X_cat, axis=1).astype(np.float32)
            X = np.concatenate([X_num, X_cat_arr], axis=1)
        else:
            X = X_num

        self.num_classes = np.array(num_classes_list, dtype=np.int64)
        return torch.from_numpy(X).to(self.device)

    def fit(self, data: ArrayLike, **kwargs: Any) -> "TabDDPMWrapper":
        schema = kwargs.get("schema")
        if schema is None:
            raise ValueError("TabularSchema must be provided in kwargs (e.g., model.fit(data, schema=schema))")

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])

        X = self._preprocess_data(data, schema)
        
        # 1. Define internal model (denoising function)
        model = MLPDiffusion(
            d_in=X.shape[1] + (sum(self.num_classes) - len(self.num_classes) if len(self.num_classes) > 0 else 0),
            num_classes=0, # Conditional class-y is disabled for simplicity here
            is_y_cond=False,
            rtdl_params={
                'd_layers': self.cfg.d_layers,
                'dropout': self.cfg.dropout,
                'activation': 'ReLU'
            }
        ).to(self.device)
        
        # 2. Define Diffusion wrapper
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=model,
            num_timesteps=self.cfg.num_timesteps,
            scheduler=self.cfg.scheduler,
            device=self.device
        ).to(self.device)
        
        # 3. Training Loop
        optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), 
            lr=self.cfg.lr, 
            weight_decay=self.cfg.weight_decay
        )
        
        loader = DataLoader(TensorDataset(X), batch_size=self.cfg.batch_size, shuffle=True)
        
        self.diffusion.train()
        for epoch in range(self.cfg.n_epochs):
            for batch in loader:
                x_batch = batch[0]
                loss_multi, loss_gauss = self.diffusion.mixed_loss(x_batch, out_dict={'y': None})
                loss = loss_multi + loss_gauss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.fit_info_ = BaselineFitInfo(
            n_rows=int(data.shape[0]),
            n_cols=int(data.shape[1]),
            columns=self.columns_,
        )
        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.diffusion.eval()
        # y_dist is uniform since we trained without conditions
        y_dist = torch.ones(1).to(self.device) 
        
        # Generate in chunks if n is very large
        x_gen, _ = self.diffusion.sample_all(n, self.cfg.batch_size, y_dist)
        
        # Convert back to DataFrame
        res_df = pd.DataFrame(x_gen.numpy(), columns=self.ordered_cols)
        # Restore original column order
        return res_df[self.columns_]