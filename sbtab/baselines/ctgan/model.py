
"""
Updated CTGAN wrapper for the repository mixed-type data handling logic.

Key design choices:
  - Uses SDV's CTGANSynthesizer (not the standalone `ctgan.CTGAN`)
  - Accepts already-preprocessed DataFrames from the repository pipeline
  - Requires `schema` and optionally the fitted split-specific `transforms`
  - Reconstructs a schema-level "raw-like" table via `transforms.inverse_transform(...)`
    before fitting CTGAN
  - Models:
      * numerical = continuous + discrete (+ numeric target)
      * categorical = categorical (+ categorical target)
  - Excludes `id_col` from modeling and re-attaches sampled IDs on output
  - Returns synthetic data in the SAME representation as the input to `fit`
    (processed if transforms were provided, otherwise raw)

This keeps the wrapper compatible with the repository data flow while using
the SDV library's end-to-end CTGAN synthesizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sbtab.baselines.base import ArrayLike, BaselineFitInfo, BaselineGenerativeModel
from sbtab.data.schema import TabularSchema, classify_feature_type


@dataclass
class CTGANConfig:
    """
    Configuration for the SDV CTGAN synthesizer.
    """
    embedding_dim: int = 128
    generator_dim: Tuple[int, ...] = (256, 256)
    discriminator_dim: Tuple[int, ...] = (256, 256)

    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6
    discriminator_steps: int = 1

    batch_size: int = 500
    epochs: int = 300
    pac: int = 10
    log_frequency: bool = True

    enforce_rounding: bool = True
    enforce_min_max_values: bool = True
    locales: Tuple[str, ...] = ("en_US",)

    enable_gpu: bool = True
    seed: int = 42
    verbose: bool = False


class CTGANWrapper(BaselineGenerativeModel):
    """
    SDV CTGAN wrapper compatible with the repository mixed-type data flow.

    Expected usage in the new module logic:
        model.fit(train_proc_df, schema=schema, transforms=fold.transforms)
        synth_proc_df = model.sample(n)

    Internally:
      1. inverse-transform the processed dataframe back to a schema-level raw-like table
      2. build SDV metadata using repository schema:
           numerical = continuous + discrete (+ numeric target)
           categorical = categorical (+ categorical target)
      3. train SDV CTGANSynthesizer on that raw-like table
      4. sample raw-like synthetic data
      5. postprocess:
           - snap discrete numeric columns back to observed support
           - reattach id column
           - transform synthetic raw-like table back to the same processed representation
    """

    def __init__(self, cfg: CTGANConfig):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg

        self._fitted = False
        self.columns_: Optional[List[str]] = None          # columns of the DataFrame passed to fit()
        self._schema: Optional[TabularSchema] = None
        self._fitted_transforms: Optional[Any] = None

        self._model = None
        self._metadata = None

        self._train_repr: Optional[str] = None             # "raw" or "processed"
        self._raw_model_cols: List[str] = []               # columns CTGAN actually models
        self._raw_return_cols: List[str] = []              # schema-level raw columns including target / id if present

        self._id_col: Optional[str] = None
        self._id_values: Optional[pd.Series] = None

        self._categorical_model_cols: List[str] = []
        self._numerical_model_cols: List[str] = []
        self._discrete_supports: Dict[str, np.ndarray] = {}
        self._raw_dtypes: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_schema_and_training_repr(
        self,
        data: pd.DataFrame,
        schema: Optional[TabularSchema],
        transforms: Any,
    ) -> Tuple[TabularSchema, pd.DataFrame]:
        if schema is None:
            raise ValueError(
                "CTGANWrapper requires `schema` in fit(..., schema=schema). "
                "This is needed to follow the repository mixed-type logic."
            )

        self._schema = schema
        self.columns_ = list(data.columns)
        self._fitted_transforms = transforms

        # If fitted transforms are provided, the input is the processed split representation.
        # Reconstruct a raw-like table for SDV CTGAN.
        if transforms is not None:
            raw_like = transforms.inverse_transform(data)
            self._train_repr = "processed"
        else:
            raw_like = data.copy()
            self._train_repr = "raw"

        # Validate that the raw-like table matches schema-level columns
        schema.validate(raw_like)
        return schema, raw_like

    def _resolve_model_column_groups(self, df_raw: pd.DataFrame, schema: TabularSchema) -> None:
        """
        Determine which schema-level raw columns are modeled as numerical vs categorical.

        Numerical block = continuous + discrete (+ numeric target)
        Categorical block = categorical (+ categorical target)

        `id_col` is excluded from modeling.
        """
        self._id_col = schema.id_col if schema.id_col in df_raw.columns else None
        if self._id_col is not None:
            self._id_values = df_raw[self._id_col].reset_index(drop=True).copy()

        self._raw_return_cols = list(df_raw.columns)

        numerical_cols = [c for c in [*schema.continuous_cols, *schema.discrete_cols] if c in df_raw.columns]
        categorical_cols = [c for c in schema.categorical_cols if c in df_raw.columns]

        if schema.target_col is not None and schema.target_col in df_raw.columns:
            ttype = classify_feature_type(df_raw[schema.target_col])
            if ttype == "categorical":
                categorical_cols.append(schema.target_col)
            else:
                numerical_cols.append(schema.target_col)

        # preserve raw dataframe column order and exclude id_col from the modeled subset
        model_cols_ordered = [c for c in df_raw.columns if c != self._id_col]
        self._numerical_model_cols = [c for c in model_cols_ordered if c in set(numerical_cols)]
        self._categorical_model_cols = [c for c in model_cols_ordered if c in set(categorical_cols)]
        self._raw_model_cols = [c for c in model_cols_ordered if c in (set(self._numerical_model_cols) | set(self._categorical_model_cols))]

        # store training dtypes and observed support for discrete numeric cols
        self._raw_dtypes = {c: str(df_raw[c].dtype) for c in self._raw_model_cols}
        self._discrete_supports = {}
        for c in schema.discrete_cols:
            if c in self._raw_model_cols:
                vals = pd.to_numeric(df_raw[c], errors="raise").to_numpy(dtype=np.float32, copy=True)
                uniq = np.sort(pd.unique(vals)).astype(np.float32)
                self._discrete_supports[c] = uniq

        # handle target if it is discrete
        if schema.target_col is not None and schema.target_col in self._raw_model_cols:
            ttype = classify_feature_type(df_raw[schema.target_col])
            if ttype == "discrete":
                vals = pd.to_numeric(df_raw[schema.target_col], errors="raise").to_numpy(dtype=np.float32, copy=True)
                uniq = np.sort(pd.unique(vals)).astype(np.float32)
                self._discrete_supports[schema.target_col] = uniq

    def _build_metadata(self, df_model: pd.DataFrame):
        """
        Build SDV metadata from the schema-level raw-like modeling table.
        """
        try:
            from sdv.metadata import Metadata
            metadata = Metadata.detect_from_dataframe(
                data=df_model,
                table_name="table",
                infer_sdtypes=False,
                infer_keys=None,
            )
        except TypeError:
            # Compatibility fallback for older/newer SDV signatures
            from sdv.metadata import Metadata
            metadata = Metadata.detect_from_dataframe(data=df_model, table_name="table")

        for col in self._raw_model_cols:
            sdtype = "categorical" if col in self._categorical_model_cols else "numerical"
            metadata.update_column(column_name=col, sdtype=sdtype)

        metadata.validate()
        return metadata

    @staticmethod
    def _nearest_values(values: np.ndarray, support: np.ndarray) -> np.ndarray:
        idx = np.argmin(np.abs(values[:, None] - support[None, :]), axis=1)
        return support[idx]

    def _postprocess_raw_sample(self, raw_synth: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess raw synthetic rows before optionally re-applying repository transforms.
        """
        out = raw_synth.copy()

        # Snap discrete numeric columns back to observed support
        for col, support in self._discrete_supports.items():
            if col not in out.columns:
                continue
            vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float32, copy=True)
            snapped = self._nearest_values(vals, support)

            # cast back to integer if the original looked integer-like
            orig_dtype = self._raw_dtypes.get(col, "")
            if "int" in orig_dtype.lower():
                out[col] = np.rint(snapped).astype(np.int64)
            else:
                out[col] = snapped

        # Reattach id column if it was excluded from modeling
        if self._id_col is not None and self._id_col not in out.columns:
            if self._id_values is None:
                out[self._id_col] = np.arange(len(out))
            else:
                out[self._id_col] = self._id_values.sample(
                    n=len(out),
                    replace=True,
                    random_state=self.seed,
                ).reset_index(drop=True)

        # Restore raw schema-level column order
        missing = [c for c in self._raw_return_cols if c not in out.columns]
        if missing:
            raise RuntimeError(
                f"CTGAN synthetic raw table is missing expected columns after postprocessing: {missing}"
            )
        return out[self._raw_return_cols]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, data: ArrayLike, **kwargs: Any) -> "CTGANWrapper":
        schema: Optional[TabularSchema] = kwargs.get("schema")
        transforms = kwargs.get("transforms", None)

        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "CTGANWrapper expects a pandas DataFrame so it can follow the repository's "
                "mixed-type data handling logic."
            )

        schema, raw_like = self._resolve_schema_and_training_repr(data, schema, transforms)
        self._resolve_model_column_groups(raw_like, schema)

        df_model = raw_like[self._raw_model_cols].copy()

        metadata = self._build_metadata(df_model)

        try:
            from sdv.single_table import CTGANSynthesizer
        except Exception as e:
            raise ImportError(
                "CTGANWrapper requires the SDV library with CTGANSynthesizer. "
                "Install it with: pip install sdv"
            ) from e

        synthesizer = CTGANSynthesizer(
            metadata,
            enforce_rounding=self.cfg.enforce_rounding,
            enforce_min_max_values=self.cfg.enforce_min_max_values,
            epochs=int(self.cfg.epochs),
            verbose=bool(self.cfg.verbose),
            embedding_dim=int(self.cfg.embedding_dim),
            generator_dim=tuple(int(v) for v in self.cfg.generator_dim),
            discriminator_dim=tuple(int(v) for v in self.cfg.discriminator_dim),
            generator_lr=float(self.cfg.generator_lr),
            generator_decay=float(self.cfg.generator_decay),
            discriminator_lr=float(self.cfg.discriminator_lr),
            discriminator_decay=float(self.cfg.discriminator_decay),
            discriminator_steps=int(self.cfg.discriminator_steps),
            batch_size=int(self.cfg.batch_size),
            log_frequency=bool(self.cfg.log_frequency),
            pac=int(self.cfg.pac),
            enable_gpu=bool(self.cfg.enable_gpu),
            locales=list(self.cfg.locales),
        )

        synthesizer.fit(df_model)

        self._metadata = metadata
        self._model = synthesizer
        self.fit_info_ = BaselineFitInfo(
            n_rows=int(data.shape[0]),
            n_cols=int(data.shape[1]),
            columns=list(self.columns_ or []),
        )
        self._fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> pd.DataFrame:
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive.")

        if seed is not None:
            np.random.seed(int(seed))

        raw_synth = self._model.sample(num_rows=int(n))
        raw_synth = self._postprocess_raw_sample(raw_synth)

        # Return in the same representation as fit input
        if self._train_repr == "processed":
            if self._fitted_transforms is None:
                raise RuntimeError("Processed training representation requires fitted transforms.")
            proc = self._fitted_transforms.transform(raw_synth)

            # Ensure exact column order as the DataFrame passed into fit()
            if self.columns_ is not None:
                missing = [c for c in self.columns_ if c not in proc.columns]
                if missing:
                    raise RuntimeError(
                        f"Processed synthetic data is missing expected columns: {missing}"
                    )
                proc = proc[self.columns_]
            return proc

        # raw representation
        if self.columns_ is not None:
            missing = [c for c in self.columns_ if c not in raw_synth.columns]
            if missing:
                raise RuntimeError(f"Raw synthetic data is missing expected columns: {missing}")
            raw_synth = raw_synth[self.columns_]
        return raw_synth
