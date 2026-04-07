from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api.types import is_categorical_dtype

from .base import TransformState
from sbtab.data.schema import TabularSchema


@dataclass
class DropMissingRows:
    name: str = "drop_missing_rows"
    subset_cols: Optional[List[str]] = None
    include_target: bool = True

    resolved_subset_: Optional[List[str]] = None
    kept_index_: Optional[pd.Index] = None
    dropped_index_: Optional[pd.Index] = None

    def requires_fit(self) -> bool:
        return False

    def is_invertible(self) -> bool:
        return False

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "DropMissingRows":
        if self.subset_cols is not None:
            self.resolved_subset_ = list(self.subset_cols)
        else:
            cols = list(schema.feature_cols)
            if self.include_target and schema.target_col is not None:
                cols.append(schema.target_col)
            self.resolved_subset_ = cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        subset = self.resolved_subset_ if self.resolved_subset_ is not None else self.subset_cols
        if subset is None:
            mask_keep = ~df.isna().any(axis=1)
        else:
            mask_keep = ~df[subset].isna().any(axis=1)
        self.kept_index_ = df.index[mask_keep]
        self.dropped_index_ = df.index[~mask_keep]
        return df.loc[self.kept_index_].copy()

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_state(self) -> TransformState:
        return TransformState(
            name=self.name,
            params={
                "subset_cols": self.subset_cols,
                "include_target": self.include_target,
            },
        )

    @classmethod
    def from_state(cls, state: TransformState) -> "DropMissingRows":
        return cls(
            subset_cols=state.params.get("subset_cols"),
            include_target=bool(state.params.get("include_target", True)),
        )


@dataclass
class TypeAwareImputer:
    name: str = "type_aware_imputer"
    continuous_strategy: str = "median"
    discrete_strategy: str = "most_frequent"
    categorical_strategy: str = "most_frequent"
    numeric_fill_value: float = 0.0
    categorical_fill_value: str = "__missing__"

    fitted_: bool = False
    fill_values_: Dict[str, Any] = field(default_factory=dict)
    feature_cols_: List[str] = field(default_factory=list)

    def requires_fit(self) -> bool:
        return True

    def is_invertible(self) -> bool:
        return False

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "TypeAwareImputer":
        fill_values: Dict[str, Any] = {}

        for col in schema.continuous_cols:
            fill_values[col] = self._fit_numeric(df[col], strategy=self.continuous_strategy)

        for col in schema.discrete_cols:
            value = self._fit_numeric(df[col], strategy=self.discrete_strategy)
            if pd.notna(value):
                try:
                    value = int(round(float(value)))
                except (TypeError, ValueError):
                    pass
            fill_values[col] = value

        for col in schema.categorical_cols:
            fill_values[col] = self._fit_categorical(df[col], strategy=self.categorical_strategy)

        self.fill_values_ = fill_values
        self.feature_cols_ = list(schema.feature_cols)
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TypeAwareImputer must be fitted before transform().")

        out = df.copy()
        for col, fill_value in self.fill_values_.items():
            if col not in out.columns:
                raise ValueError(f"Column {col!r} is missing from DataFrame during imputation.")
            series = out[col]
            if is_categorical_dtype(series.dtype) and pd.notna(fill_value):
                if fill_value not in series.cat.categories:
                    series = series.cat.add_categories([fill_value])
            out[col] = series.fillna(fill_value)
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_state(self) -> TransformState:
        return TransformState(
            name=self.name,
            params={
                "continuous_strategy": self.continuous_strategy,
                "discrete_strategy": self.discrete_strategy,
                "categorical_strategy": self.categorical_strategy,
                "numeric_fill_value": self.numeric_fill_value,
                "categorical_fill_value": self.categorical_fill_value,
                "fitted": self.fitted_,
                "fill_values": self.fill_values_,
                "feature_cols": self.feature_cols_,
            },
        )

    @classmethod
    def from_state(cls, state: TransformState) -> "TypeAwareImputer":
        obj = cls(
            continuous_strategy=str(state.params.get("continuous_strategy", "median")),
            discrete_strategy=str(state.params.get("discrete_strategy", "most_frequent")),
            categorical_strategy=str(state.params.get("categorical_strategy", "most_frequent")),
            numeric_fill_value=float(state.params.get("numeric_fill_value", 0.0)),
            categorical_fill_value=str(state.params.get("categorical_fill_value", "__missing__")),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.fill_values_ = dict(state.params.get("fill_values", {}))
        obj.feature_cols_ = list(state.params.get("feature_cols", []))
        return obj

    def _fit_numeric(self, series: pd.Series, *, strategy: str) -> Any:
        non_null = series.dropna()
        if non_null.empty:
            return self.numeric_fill_value

        if strategy == "median":
            return float(non_null.median())
        if strategy == "mean":
            return float(non_null.mean())
        if strategy == "most_frequent":
            mode = non_null.mode(dropna=True)
            return mode.iloc[0] if not mode.empty else self.numeric_fill_value
        if strategy == "constant":
            return self.numeric_fill_value

        raise ValueError(
            f"Unsupported numeric imputation strategy {strategy!r}. "
            "Use one of: 'median', 'mean', 'most_frequent', 'constant'."
        )

    def _fit_categorical(self, series: pd.Series, *, strategy: str) -> Any:
        non_null = series.dropna()
        if strategy == "constant":
            return self.categorical_fill_value

        if strategy == "most_frequent":
            if non_null.empty:
                return self.categorical_fill_value
            mode = non_null.mode(dropna=True)
            if mode.empty:
                return self.categorical_fill_value
            return mode.iloc[0]

        raise ValueError(
            f"Unsupported categorical imputation strategy {strategy!r}. "
            "Use one of: 'most_frequent', 'constant'."
        )
