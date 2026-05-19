
"""
Refined TabDDPM wrapper that only addresses comments 1, 2, and 3 from the training-logic review:

1. train for a fixed number of optimizer STEPS (not only epochs)
2. apply linear learning-rate annealing across training steps
3. maintain an EMA copy of the denoiser, and optionally sample with EMA

All other wrapper behavior is intentionally left unchanged:
  - mixed-type schema handling
  - support for raw / one-hot / integer-coded categoricals
  - reconstruction of the same representation on sampling
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sbtab.baselines.base import ArrayLike, BaselineFitInfo, BaselineGenerativeModel
from sbtab.data.schema import TabularSchema, classify_feature_type

from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .modules import MLPDiffusion


@dataclass
class TabDDPMConfig:
    # --- comment 1: fixed number of training steps ---
    steps: Optional[int] = 10000

    # optional backward-compatibility fallback; if `steps` is None, use old epoch logic
    n_epochs: Optional[int] = None

    # original TabDDPM hyperparameters
    num_timesteps: int = 1000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4

    d_layers: List[int] = field(default_factory=lambda: [256, 512, 512, 256])
    dropout: float = 0.0

    gaussian_loss_type: str = "mse"
    scheduler: str = "cosine"

    # --- comment 3: EMA ---
    ema_decay: float = 0.999

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class TabDDPMWrapper(BaselineGenerativeModel):
    """
    TabDDPM wrapper with:
      - fixed-step training
      - linear LR annealing
      - EMA denoiser

    Other data-handling logic is unchanged from the mixed-type wrapper.
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

        self._schema: Optional[TabularSchema] = None

        self._id_col: Optional[str] = None
        self._id_values: Optional[pd.Series] = None

        self._num_output_cols: List[str] = []
        self._cat_specs: List[Dict[str, Any]] = []

        self.diffusion: Optional[GaussianMultinomialDiffusion] = None
        self.ema_model: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------
    # helpers for discovering categorical representation metadata
    # ------------------------------------------------------------------

    def _find_categorical_representation(self, obj: Any) -> Tuple[Optional[Any], Optional[str]]:
        visited = set()

        def infer_rep_name(x: Any) -> Optional[str]:
            rep_name = getattr(x, "representation_name", None)
            if isinstance(rep_name, str):
                return rep_name

            cls_name = x.__class__.__name__.lower()
            if "onehot" in cls_name:
                return "one_hot_representation"
            if "integercode" in cls_name or "integer_code" in cls_name:
                return "integer_code_representation"
            return None

        def is_rep_obj(x: Any) -> bool:
            return (
                hasattr(x, "categorical_cols_")
                and hasattr(x, "categories_")
                and hasattr(x, "fitted_")
            )

        def rec(x: Any) -> Tuple[Optional[Any], Optional[str]]:
            if x is None:
                return None, None
            xid = id(x)
            if xid in visited:
                return None, None
            visited.add(xid)

            if hasattr(x, "repr_") and getattr(x, "repr_", None) is not None:
                rep = getattr(x, "repr_")
                if is_rep_obj(rep):
                    return rep, infer_rep_name(x) or infer_rep_name(rep)

            if is_rep_obj(x):
                return x, infer_rep_name(x)

            for attr in ("transforms", "steps"):
                if hasattr(x, attr):
                    sub = getattr(x, attr)
                    if isinstance(sub, dict):
                        for v in sub.values():
                            obj2, name2 = rec(v)
                            if obj2 is not None:
                                return obj2, name2
                    else:
                        try:
                            for v in sub:
                                obj2, name2 = rec(v)
                                if obj2 is not None:
                                    return obj2, name2
                        except TypeError:
                            pass

            if isinstance(x, dict):
                for v in x.values():
                    obj2, name2 = rec(v)
                    if obj2 is not None:
                        return obj2, name2

            if isinstance(x, (list, tuple)):
                for v in x:
                    obj2, name2 = rec(v)
                    if obj2 is not None:
                        return obj2, name2

            return None, None

        return rec(obj)

    @staticmethod
    def _onehot_col_map(rep: Any) -> Dict[str, List[str]]:
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

    @staticmethod
    def _intcode_col_map(rep: Any) -> Dict[str, List[str]]:
        categorical_cols = list(getattr(rep, "categorical_cols_", []))
        encoded_cols = list(getattr(rep, "encoded_cols_", categorical_cols))
        if encoded_cols and len(encoded_cols) == len(categorical_cols):
            return {src: [enc] for src, enc in zip(categorical_cols, encoded_cols)}
        return {col: [col] for col in categorical_cols}

    # ------------------------------------------------------------------
    # block resolution
    # ------------------------------------------------------------------

    def _numeric_block_cols(self, df: pd.DataFrame, schema: TabularSchema) -> List[str]:
        cols = [c for c in [*schema.continuous_cols, *schema.discrete_cols] if c in df.columns]

        if schema.target_col is not None and schema.target_col in df.columns:
            target_type = classify_feature_type(df[schema.target_col])
            if target_type in ("continuous", "discrete"):
                cols.append(schema.target_col)

        ordered = [c for c in df.columns if c in set(cols)]
        return ordered

    def _categorical_block_specs(
        self,
        df: pd.DataFrame,
        schema: TabularSchema,
        transforms: Any,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []

        rep, rep_name = self._find_categorical_representation(transforms)
        onehot_map = self._onehot_col_map(rep) if rep is not None and rep_name == "one_hot_representation" else {}
        intcode_map = self._intcode_col_map(rep) if rep is not None and rep_name == "integer_code_representation" else {}
        categories_map = dict(getattr(rep, "categories_", {})) if rep is not None else {}

        def _append_spec(col: str, mode: str, output_cols: List[str], categories: List[Any]) -> None:
            specs.append(
                {
                    "name": col,
                    "mode": mode,
                    "output_cols": output_cols,
                    "num_classes": len(categories),
                    "categories": list(categories),
                }
            )

        for col in schema.categorical_cols:
            if col in intcode_map and all(c in df.columns for c in intcode_map[col]):
                cats = list(categories_map.get(col, []))
                _append_spec(col, "integer_code", list(intcode_map[col]), cats)
                continue

            if col in onehot_map and all(c in df.columns for c in onehot_map[col]):
                cats = list(categories_map.get(col, []))
                _append_spec(col, "onehot", list(onehot_map[col]), cats)
                continue

            if col in df.columns:
                cat = pd.Categorical(df[col])
                _append_spec(col, "raw", [col], list(cat.categories))
                continue

            raise ValueError(
                f"Categorical feature {col!r} is neither present as a raw column nor "
                f"recoverable from fitted categorical transform metadata."
            )

        if schema.target_col is not None and schema.target_col in df.columns:
            target_type = classify_feature_type(df[schema.target_col])
            if target_type == "categorical":
                col = schema.target_col

                if col in intcode_map and all(c in df.columns for c in intcode_map[col]):
                    cats = list(categories_map.get(col, []))
                    _append_spec(col, "integer_code", list(intcode_map[col]), cats)
                elif col in onehot_map and all(c in df.columns for c in onehot_map[col]):
                    cats = list(categories_map.get(col, []))
                    _append_spec(col, "onehot", list(onehot_map[col]), cats)
                else:
                    cat = pd.Categorical(df[col])
                    _append_spec(col, "raw", [col], list(cat.categories))

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
        self.columns_ = list(df.columns)
        self._input_columns = list(df.columns)
        self._schema = schema

        self._id_col = schema.id_col if schema.id_col in df.columns else None
        if self._id_col is not None:
            self._id_values = df[self._id_col].reset_index(drop=True).copy()

        num_cols = self._numeric_block_cols(df, schema)
        self._num_output_cols = num_cols

        X_num = (
            df[num_cols].to_numpy(dtype=np.float32, copy=True)
            if num_cols
            else np.empty((len(df), 0), dtype=np.float32)
        )
        self.num_numerical_features = X_num.shape[1]

        self._cat_specs = self._categorical_block_specs(df, schema, transforms)

        X_cat_list: List[np.ndarray] = []
        num_classes_list: List[int] = []

        for spec in self._cat_specs:
            num_classes_list.append(int(spec["num_classes"]))

            if spec["mode"] == "raw":
                cat = pd.Categorical(df[spec["name"]], categories=spec["categories"])
                codes = cat.codes.astype(np.int64, copy=False)
                if (codes < 0).any():
                    raise ValueError(
                        f"Categorical column {spec['name']!r} contains unknown or missing values."
                    )
                X_cat_list.append(codes.reshape(-1, 1).astype(np.float32))

            elif spec["mode"] == "onehot":
                block = df[spec["output_cols"]].to_numpy(dtype=np.float32, copy=True)
                codes = np.argmax(block, axis=1).astype(np.int64)
                X_cat_list.append(codes.reshape(-1, 1).astype(np.float32))

            elif spec["mode"] == "integer_code":
                col = spec["output_cols"][0]
                codes = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.int64, copy=True)
                if (codes < 0).any():
                    raise ValueError(
                        f"Integer-coded categorical column {col!r} contains unknown code(s) < 0."
                    )
                X_cat_list.append(codes.reshape(-1, 1).astype(np.float32))

            else:
                raise RuntimeError(f"Unknown categorical mode: {spec['mode']!r}")

        if X_cat_list:
            X_cat = np.concatenate(X_cat_list, axis=1)
            X = np.concatenate([X_num, X_cat], axis=1)
        else:
            X = X_num

        self.num_classes = np.asarray(num_classes_list, dtype=np.int64)
        return torch.from_numpy(X).to(self.device)

    # ------------------------------------------------------------------
    # comment 2: linear LR annealing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _anneal_lr(optimizer: torch.optim.Optimizer, *, init_lr: float, step: int, total_steps: int) -> None:
        frac_done = step / float(total_steps)
        lr = init_lr * (1.0 - frac_done)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # ------------------------------------------------------------------
    # comment 3: EMA helper
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _update_ema(target_model: torch.nn.Module, source_model: torch.nn.Module, rate: float) -> None:
        for targ, src in zip(target_model.parameters(), source_model.parameters()):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1.0 - rate)

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

        d_in = self.num_numerical_features + int(self.num_classes.sum())
        model = MLPDiffusion(
            d_in=d_in,
            num_classes=0,
            is_y_cond=False,
            rtdl_params={
                "d_layers": self.cfg.d_layers,
                "dropout": self.cfg.dropout,
            },
        ).to(self.device)

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

        # --- comment 3: initialize EMA model from the denoiser ---
        self.ema_model = copy.deepcopy(self.diffusion._denoise_fn).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        loader = DataLoader(TensorDataset(X), batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        # --- comment 1: use fixed number of optimizer steps when cfg.steps is provided ---
        if self.cfg.steps is not None:
            total_steps = int(self.cfg.steps)
        else:
            if self.cfg.n_epochs is None:
                raise ValueError("Either cfg.steps or cfg.n_epochs must be provided.")
            total_steps = int(self.cfg.n_epochs) * max(len(loader), 1)

        loader_iter = iter(loader)
        self.diffusion.train()

        for step in range(total_steps):
            try:
                (x_batch,) = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                (x_batch,) = next(loader_iter)

            # --- comment 2: linear LR annealing ---
            self._anneal_lr(optimizer, init_lr=self.cfg.lr, step=step, total_steps=total_steps)

            loss_multi, loss_gauss = self.diffusion.mixed_loss(x_batch, out_dict={"y": None})
            loss = loss_multi + loss_gauss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # --- comment 3: EMA update after optimizer step ---
            self._update_ema(self.ema_model, self.diffusion._denoise_fn, self.cfg.ema_decay)

        self.fit_info_ = BaselineFitInfo(
            n_rows=int(data.shape[0]),
            n_cols=int(data.shape[1]),
            columns=self.columns_ or [],
        )
        self._fitted = True
        return self

    def _reconstruct_output_df(self, x_gen: np.ndarray) -> pd.DataFrame:
        n = x_gen.shape[0]
        out = pd.DataFrame(index=np.arange(n))

        X_num = (
            x_gen[:, : self.num_numerical_features]
            if self.num_numerical_features > 0
            else np.empty((n, 0), dtype=np.float32)
        )
        X_cat = (
            x_gen[:, self.num_numerical_features:]
            if len(self.num_classes) > 0
            else np.empty((n, 0), dtype=np.float32)
        )

        for j, col in enumerate(self._num_output_cols):
            out[col] = X_num[:, j]

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

            elif spec["mode"] == "integer_code":
                out[spec["output_cols"][0]] = codes.astype(np.int64)

            else:
                raise RuntimeError(f"Unknown categorical mode: {spec['mode']!r}")

        if self._id_col is not None and self._id_col not in out.columns:
            if self._id_values is None:
                out[self._id_col] = np.arange(n)
            else:
                out[self._id_col] = self._id_values.sample(n=n, replace=True, random_state=self.seed).reset_index(drop=True)

        if self._input_columns is None:
            return out

        missing = [c for c in self._input_columns if c not in out.columns]
        if missing:
            raise RuntimeError(
                f"Failed to reconstruct output columns: {missing}. "
                "Check categorical representation metadata handling."
            )

        return out[self._input_columns]

    @torch.no_grad()
    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        *,
        use_ema: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self._fitted or self.diffusion is None:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive.")

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        self.diffusion.eval()
        y_dist = torch.ones(1, device=self.device)

        # --- comment 3: sample with EMA model by temporarily swapping denoiser ---
        denoiser_backup = self.diffusion._denoise_fn
        if use_ema and self.ema_model is not None:
            self.diffusion._denoise_fn = self.ema_model

        try:
            x_gen, _ = self.diffusion.sample_all(n, self.cfg.batch_size, y_dist)
        finally:
            self.diffusion._denoise_fn = denoiser_backup

        if isinstance(x_gen, torch.Tensor):
            x_gen = x_gen.detach().cpu().numpy()

        return self._reconstruct_output_df(np.asarray(x_gen, dtype=np.float32))
