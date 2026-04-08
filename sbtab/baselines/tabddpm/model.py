
"""
Updated TabDDPM wrapper for the repository mixed-type data logic.

Key assumptions:
  - `fit()` receives an already-preprocessed DataFrame (e.g. from TabularDataModule.get_fold/get_holdout).
  - Continuous and discrete columns are treated as the numerical block for TabDDPM.
  - Categorical columns may already be represented via a categorical transform (e.g. one-hot).
  - The wrapper uses `schema` plus optional `transforms` metadata to reconstruct the
    numerical / categorical split expected by the official TabDDPM implementation.

This file is adapted from the official TabDDPM repository:
https://github.com/yandex-research/tab-ddpm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_object_dtype, is_string_dtype
from torch.utils.data import DataLoader, TensorDataset

from sbtab.baselines.base import ArrayLike, BaselineFitInfo, BaselineGenerativeModel
from sbtab.data.schema import TabularSchema, classify_feature_type

from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .modules import MLPDiffusion


@dataclass
class TabDDPMConfig:
    """
    Configuration for the official TabDDPM implementation.

    Updated assumptions:
      - the input DataFrame is already preprocessed by the repository pipeline
      - "numerical" = continuous_cols + discrete_cols (+ numeric target if target is present)
      - "categorical" = categorical_cols (+ categorical target if target is present)
    """
    num_timesteps: int = 1000
    n_epochs: int = 1000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4

    d_layers: List[int] = field(default_factory=lambda: [256, 512, 512, 256])
    dropout: float = 0.0

    gaussian_loss_type: str = "mse"
    scheduler: str = "cosine"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class TabDDPMWrapper(BaselineGenerativeModel):
    """
    Wrapper for TabDDPM that follows the repository mixed-type data handling logic.

    Important behavior:
      - uses schema.continuous_cols + schema.discrete_cols as the numerical block
      - uses schema.categorical_cols as the categorical block
      - if categorical features have already been transformed (e.g. one-hot), the wrapper
        reconstructs integer category codes internally using the fitted transform metadata
      - sampling returns data in the SAME preprocessed column layout as the input DataFrame
    """

    def __init__(self, cfg: TabDDPMConfig):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self._fitted = False
        self.columns_: Optional[List[str]] = None
        self._input_columns: Optional[List[str]] = None

        self.num_numerical_features: int = 0
        self.num_classes: np.ndarray = np.array([], dtype=np.int64)

        self._ordered_internal_cols: List[str] = []   # internal order: numerical + categorical variables (raw variable names)
        self._schema: Optional[TabularSchema] = None

        # metadata for rebuilding output layout
        self._id_col: Optional[str] = None
        self._id_values: Optional[pd.Series] = None

        self._num_output_cols: List[str] = []
        self._cat_specs: List[Dict[str, Any]] = []

        self.diffusion: Optional[GaussianMultinomialDiffusion] = None

    # ------------------------------------------------------------------
    # helpers for discovering categorical representation metadata
    # ------------------------------------------------------------------

    def _find_onehot_representation(self, obj: Any) -> Optional[Any]:
        """
        Recursively search for a fitted OneHotRepresentation-like object inside transforms.

        We intentionally use duck-typing because the pipeline class is not hard-coded here.
        """
        visited = set()

        def _rec(x: Any) -> Optional[Any]:
            if x is None:
                return None
            xid = id(x)
            if xid in visited:
                return None
            visited.add(xid)

            # direct representation object
            if hasattr(x, "categorical_cols_") and hasattr(x, "encoded_cols_") and hasattr(x, "categories_"):
                return x

            # transform that wraps a representation
            if hasattr(x, "repr_") and getattr(x, "repr_", None) is not None:
                rep = getattr(x, "repr_")
                if hasattr(rep, "categorical_cols_") and hasattr(rep, "encoded_cols_") and hasattr(rep, "categories_"):
                    return rep

            # common pipeline-like patterns
            if hasattr(x, "transforms"):
                sub = getattr(x, "transforms")
                if isinstance(sub, dict):
                    for v in sub.values():
                        found = _rec(v)
                        if found is not None:
                            return found
                else:
                    try:
                        for v in sub:
                            found = _rec(v)
                            if found is not None:
                                return found
                    except TypeError:
                        pass

            if hasattr(x, "steps"):
                sub = getattr(x, "steps")
                if isinstance(sub, dict):
                    for v in sub.values():
                        found = _rec(v)
                        if found is not None:
                            return found
                else:
                    try:
                        for v in sub:
                            found = _rec(v)
                            if found is not None:
                                return found
                    except TypeError:
                        pass

            if isinstance(x, dict):
                for v in x.values():
                    found = _rec(v)
                    if found is not None:
                        return found

            if isinstance(x, (list, tuple)):
                for v in x:
                    found = _rec(v)
                    if found is not None:
                        return found

            return None

        return _rec(obj)

    @staticmethod
    def _representation_col_map(rep: Any) -> Dict[str, List[str]]:
        """
        Reconstruct mapping:
            original categorical feature -> encoded column names
        from a fitted OneHotRepresentation object.
        """
        col_map: Dict[str, List[str]] = {}
        encoded_cols = list(getattr(rep, "encoded_cols_", []))
        categories = dict(getattr(rep, "categories_", {}))
        categorical_cols = list(getattr(rep, "categorical_cols_", []))

        cursor = 0
        for col in categorical_cols:
            cats = categories.get(col, [])
            width = len(cats)
            col_map[col] = encoded_cols[cursor: cursor + width]
            cursor += width
        return col_map

    # ------------------------------------------------------------------
    # target helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_categorical_series(s: pd.Series) -> bool:
        return bool(
            is_object_dtype(s.dtype)
            or is_string_dtype(s.dtype)
            or is_bool_dtype(s.dtype)
            or is_categorical_dtype(s.dtype)
        )

    def _numeric_block_cols(self, df: pd.DataFrame, schema: TabularSchema) -> List[str]:
        """
        Numerical block = continuous + discrete (+ numeric target if present in df).
        """
        cols = [c for c in [*schema.continuous_cols, *schema.discrete_cols] if c in df.columns]

        if schema.target_col is not None and schema.target_col in df.columns:
            target_type = classify_feature_type(df[schema.target_col])
            if target_type in ("continuous", "discrete"):
                cols.append(schema.target_col)

        # preserve input DataFrame order
        ordered = [c for c in df.columns if c in set(cols)]
        return ordered

    def _categorical_block_specs(
        self,
        df: pd.DataFrame,
        schema: TabularSchema,
        transforms: Any,
    ) -> List[Dict[str, Any]]:
        """
        Determine the categorical variables to model and how they are represented
        in the already-preprocessed DataFrame.

        Each spec contains:
          - name: logical variable name
          - mode: "raw" or "onehot"
          - output_cols: columns present in the input/output DataFrame
          - num_classes: number of categories
          - categories: training categories for reconstruction
        """
        specs: List[Dict[str, Any]] = []

        rep = self._find_onehot_representation(transforms)
        rep_col_map = self._representation_col_map(rep) if rep is not None else {}

        # regular feature categorical columns
        for col in schema.categorical_cols:
            if col in df.columns:
                # raw categorical column still present
                cat = pd.Categorical(df[col])
                specs.append(
                    {
                        "name": col,
                        "mode": "raw",
                        "output_cols": [col],
                        "num_classes": len(cat.categories),
                        "categories": list(cat.categories),
                    }
                )
            elif col in rep_col_map and all(ec in df.columns for ec in rep_col_map[col]):
                specs.append(
                    {
                        "name": col,
                        "mode": "onehot",
                        "output_cols": list(rep_col_map[col]),
                        "num_classes": len(rep_col_map[col]),
                        "categories": list(getattr(rep, "categories_", {}).get(col, [])),
                    }
                )
            else:
                raise ValueError(
                    f"Categorical feature {col!r} is neither present as a raw column nor "
                    f"recoverable from fitted categorical transform metadata."
                )

        # optional target column if categorical and present
        if schema.target_col is not None and schema.target_col in df.columns:
            target_type = classify_feature_type(df[schema.target_col])
            if target_type == "categorical":
                col = schema.target_col
                cat = pd.Categorical(df[col])
                specs.append(
                    {
                        "name": col,
                        "mode": "raw",
                        "output_cols": [col],
                        "num_classes": len(cat.categories),
                        "categories": list(cat.categories),
                    }
                )

        return specs

    # ------------------------------------------------------------------
    # internal TabDDPM matrix construction
    # ------------------------------------------------------------------

    def _preprocess_data(
        self,
        df: pd.DataFrame,
        schema: TabularSchema,
        transforms: Any = None,
    ) -> torch.Tensor:
        """
        Build the internal matrix expected by TabDDPM:
            [numerical block | categorical-code block]

        where:
          - numerical block = continuous + discrete (+ numeric target if present)
          - categorical-code block = integer-coded categorical variables reconstructed from
            either raw categorical columns or already-transformed one-hot blocks
        """
        self.columns_ = list(df.columns)
        self._input_columns = list(df.columns)
        self._schema = schema

        self._id_col = schema.id_col if schema.id_col in df.columns else None
        if self._id_col is not None:
            self._id_values = df[self._id_col].reset_index(drop=True).copy()

        # 1. Numerical block
        num_cols = self._numeric_block_cols(df, schema)
        self._num_output_cols = num_cols

        X_num = (
            df[num_cols].to_numpy(dtype=np.float32, copy=True)
            if num_cols
            else np.empty((len(df), 0), dtype=np.float32)
        )
        self.num_numerical_features = X_num.shape[1]

        # 2. Categorical block
        self._cat_specs = self._categorical_block_specs(df, schema, transforms)

        X_cat_list: List[np.ndarray] = []
        num_classes_list: List[int] = []
        ordered_cat_names: List[str] = []

        for spec in self._cat_specs:
            ordered_cat_names.append(spec["name"])
            num_classes_list.append(int(spec["num_classes"]))

            if spec["mode"] == "raw":
                cat = pd.Categorical(df[spec["name"]], categories=spec["categories"])
                codes = cat.codes.astype(np.int64, copy=False)
                if (codes < 0).any():
                    raise ValueError(
                        f"Categorical column {spec['name']!r} contains unknown or missing values "
                        "after preprocessing."
                    )
                X_cat_list.append(codes.reshape(-1, 1).astype(np.float32))

            elif spec["mode"] == "onehot":
                block = df[spec["output_cols"]].to_numpy(dtype=np.float32, copy=True)
                codes = np.argmax(block, axis=1).astype(np.int64)
                X_cat_list.append(codes.reshape(-1, 1).astype(np.float32))

            else:
                raise RuntimeError(f"Unknown categorical spec mode: {spec['mode']!r}")

        if X_cat_list:
            X_cat = np.concatenate(X_cat_list, axis=1)
            X = np.concatenate([X_num, X_cat], axis=1)
        else:
            X = X_num

        self.num_classes = np.asarray(num_classes_list, dtype=np.int64)
        self._ordered_internal_cols = num_cols + ordered_cat_names

        return torch.from_numpy(X).to(self.device)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, data: ArrayLike, **kwargs: Any) -> "TabDDPMWrapper":
        schema = kwargs.get("schema")
        transforms = kwargs.get("transforms", None)

        if schema is None:
            raise ValueError(
                "TabularSchema must be provided in kwargs "
                "(e.g., model.fit(data, schema=schema, transforms=pipe))."
            )

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])

        X = self._preprocess_data(data, schema, transforms)
        if len(self.num_classes) == 0:
            self.num_classes = np.array([0])

        # Internal denoising model
        d_in = X.shape[1] + (int(self.num_classes.sum()) if len(self.num_classes) > 0 else 0)
        model = MLPDiffusion(
            d_in=d_in,
            num_classes=0,
            is_y_cond=False,
            rtdl_params={
                "d_layers": self.cfg.d_layers,
                "dropout": self.cfg.dropout,
            },
        ).to(self.device)
        # Diffusion wrapper
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=model,
            num_timesteps=self.cfg.num_timesteps,
            scheduler=self.cfg.scheduler,
            device=self.device,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.diffusion.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        loader = DataLoader(TensorDataset(X), batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        self.diffusion.train()
        for _ in range(self.cfg.n_epochs):
            for (x_batch,) in loader:
                loss_multi, loss_gauss = self.diffusion.mixed_loss(x_batch, out_dict={"y": None})
                loss = loss_multi + loss_gauss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.fit_info_ = BaselineFitInfo(
            n_rows=int(data.shape[0]),
            n_cols=int(data.shape[1]),
            columns=self.columns_ or [],
        )
        self._fitted = True
        return self

    def _reconstruct_output_df(self, x_gen: np.ndarray) -> pd.DataFrame:
        """
        Convert internal TabDDPM output:
            [numerical block | categorical-code block]
        back into the SAME preprocessed column layout as the training DataFrame.
        """
        n = x_gen.shape[0]
        out = pd.DataFrame(index=np.arange(n))

        # Split internal output
        X_num = x_gen[:, : self.num_numerical_features] if self.num_numerical_features > 0 else np.empty((n, 0), dtype=np.float32)
        X_cat = x_gen[:, self.num_numerical_features:] if len(self.num_classes) > 0 else np.empty((n, 0), dtype=np.float32)

        # numerical block
        for j, col in enumerate(self._num_output_cols):
            out[col] = X_num[:, j]

        # categorical block
        for j, spec in enumerate(self._cat_specs):
            codes = np.asarray(X_cat[:, j]).reshape(-1)
            codes = np.clip(np.rint(codes).astype(np.int64), 0, spec["num_classes"] - 1)

            if spec["mode"] == "raw":
                categories = np.asarray(spec["categories"], dtype=object)
                out[spec["output_cols"][0]] = categories[codes]
            elif spec["mode"] == "onehot":
                oh = np.eye(spec["num_classes"], dtype=np.float32)[codes]
                for k, col in enumerate(spec["output_cols"]):
                    out[col] = oh[:, k]
            else:
                raise RuntimeError(f"Unknown categorical spec mode: {spec['mode']!r}")

        # preserve id column if it existed in input
        if self._id_col is not None and self._id_col not in out.columns:
            if self._id_values is None:
                out[self._id_col] = np.arange(n)
            else:
                out[self._id_col] = self._id_values.sample(n=n, replace=True, random_state=self.seed).reset_index(drop=True)

        # return columns exactly in the same layout/order as the input DataFrame
        if self._input_columns is None:
            return out
        for col in self._input_columns:
            if col not in out.columns:
                raise RuntimeError(
                    f"Failed to reconstruct output column {col!r}. "
                    "Check categorical transform metadata handling."
                )
        return out[self._input_columns]

    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> pd.DataFrame:
        if not self._fitted or self.diffusion is None:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive.")

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        self.diffusion.eval()

        # Unconditional generation
        y_dist = torch.ones(1, device=self.device)

        x_gen, _ = self.diffusion.sample_all(n, self.cfg.batch_size, y_dist)
        if isinstance(x_gen, torch.Tensor):
            x_gen = x_gen.detach().cpu().numpy()

        return self._reconstruct_output_df(np.asarray(x_gen, dtype=np.float32))
