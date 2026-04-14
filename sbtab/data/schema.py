
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import pandas as pd


@dataclass(frozen=True)
class TabularSchema:
    """
    Schema for tabular features: continuous columns and optional categorical columns.

    Columns listed in ``categorical_cols`` must be a subset of ``feature_cols``.
    All other feature columns are treated as continuous (numeric).
    """
    feature_cols: List[str]
    target_col: Optional[str] = None
    id_col: Optional[str] = None  # optional row identifier column
    categorical_cols: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        feats = set(self.feature_cols)
        for c in self.categorical_cols:
            if c not in feats:
                raise ValueError(
                    f"categorical_cols contains '{c}' which is not in feature_cols"
                )

    @property
    def continuous_cols(self) -> List[str]:
        cat = set(self.categorical_cols)
        return [c for c in self.feature_cols if c not in cat]

    @property
    def all_cols(self) -> List[str]:
        cols: List[str] = []
        if self.id_col is not None:
            cols.append(self.id_col)
        cols.extend(self.feature_cols)
        if self.target_col is not None:
            cols.append(self.target_col)
        return cols

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        for c in self.continuous_cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                raise TypeError(
                    f"Continuous feature '{c}' must be numeric; "
                    "add it to categorical_cols if it is categorical."
                )

    @classmethod
    def infer_from_dataframe(
        cls,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        id_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        drop_non_numeric: bool = False,
        categorical_cols: Optional[Sequence[str]] = None,
        infer_categorical: bool = False,
    ) -> "TabularSchema":
        """
        Infer schema from a DataFrame.

        If feature_cols is None:
          - uses all columns except target_col and id_col
          - optionally drops non-numeric columns if drop_non_numeric=True

        categorical_cols:
          Explicit list of feature columns to treat as categorical.

        infer_categorical:
          If True and categorical_cols is None, columns with object/categorical/bool
          dtype are marked categorical (heuristic).
        """
        cols = list(df.columns)

        if id_col is not None and id_col not in cols:
            raise ValueError(f"id_col='{id_col}' not found in df.columns")
        if target_col is not None and target_col not in cols:
            raise ValueError(f"target_col='{target_col}' not found in df.columns")

        if feature_cols is None:
            exclude = set([c for c in [id_col, target_col] if c is not None])
            feats = [c for c in cols if c not in exclude]
        else:
            feats = list(feature_cols)

        if drop_non_numeric:
            feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]

        inferred_cat: Tuple[str, ...] = ()
        if categorical_cols is not None:
            inferred_cat = tuple(categorical_cols)
        elif infer_categorical:
            inferred_cat = tuple(
                c
                for c in feats
                if pd.api.types.is_object_dtype(df[c])
                or pd.api.types.is_categorical_dtype(df[c])
                or pd.api.types.is_bool_dtype(df[c])
            )

        schema = cls(
            feature_cols=feats,
            target_col=target_col,
            id_col=id_col,
            categorical_cols=inferred_cat,
        )
        schema.validate(df)
        return schema
