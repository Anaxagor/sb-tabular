
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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sbtab.baselines.base import ArrayLike, BaselineFitInfo, BaselineGenerativeModel
from sbtab.data.schema import TabularSchema, classify_feature_type

from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .native import TabDDPMConfig, TabDDPMSolver


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
        self._solver: Optional[TabDDPMSolver] = None

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
        cardinalities = self.num_classes.tolist()
        train_num = X[:, : self.num_numerical_features].to(dtype=torch.float32)
        train_state = X[:, self.num_numerical_features :].to(dtype=torch.int64)
        solver = TabDDPMSolver(
            num_numerical_features=self.num_numerical_features,
            cardinalities=cardinalities,
            cfg=self.cfg,
        )
        solver.fit(train_num, train_state)

        self._solver = solver
        self.diffusion = solver.diffusion
        self.ema_model = solver.ema_model
        if len(self.num_classes) == 0:
            self.num_classes = np.array([0], dtype=np.int64)

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
        if not self._fitted or self._solver is None:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive.")

        generated_num, generated_state = self._solver.sample(
            n_samples=n,
            seed=seed,
            use_ema=use_ema,
        )
        generated = torch.cat(
            (generated_num, generated_state.to(dtype=torch.float32)),
            dim=1,
        )
        return self._reconstruct_output_df(
            generated.detach().cpu().numpy().astype(np.float32, copy=False)
        )
