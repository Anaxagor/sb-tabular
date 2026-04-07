from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

DEFAULT_DISCRETE_MAX_UNIQUE = 20
_INTEGER_TOL = 1e-12


def _is_supported_categorical_dtype(series: pd.Series) -> bool:
    return bool(
        is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or is_categorical_dtype(series.dtype)
    )


def _dropna_numeric(series: pd.Series) -> np.ndarray:
    non_null = series.dropna()
    if non_null.empty:
        return np.asarray([], dtype=np.float64)
    return pd.to_numeric(non_null, errors="raise").to_numpy(dtype=np.float64)


def _is_integer_like_numeric(series: pd.Series, *, tol: float = _INTEGER_TOL) -> bool:
    if is_bool_dtype(series.dtype):
        return False
    if is_integer_dtype(series.dtype):
        return True
    if not is_numeric_dtype(series.dtype) or is_complex_dtype(series.dtype):
        return False

    values = _dropna_numeric(series)
    if values.size == 0:
        return bool(is_integer_dtype(series.dtype))

    rounded = np.round(values)
    return bool(np.all(np.isclose(values, rounded, atol=tol, rtol=0.0)))


def classify_feature_type(
    series: pd.Series,
    *,
    discrete_max_unique: int = DEFAULT_DISCRETE_MAX_UNIQUE,
    integer_tol: float = _INTEGER_TOL,
) -> str:
    """
    Classify a feature using the requested logic.

    Rules
    -----
    categorical:
        object, string, bool, category
    discrete:
        integer dtype with fewer than `discrete_max_unique` unique non-null values,
        or float/numeric columns whose values are all integer-like and also have fewer
        than `discrete_max_unique` unique non-null values
    continuous:
        all other real numeric columns, including true floats and integer-like numeric
        columns at or above the uniqueness threshold

    Notes
    -----
    The user's original rule leaves the case `nunique == discrete_max_unique` undefined.
    This implementation resolves it deterministically by treating those columns as
    continuous (`nunique < threshold` -> discrete, otherwise continuous).
    """
    if discrete_max_unique <= 0:
        raise ValueError("discrete_max_unique must be a positive integer.")

    if _is_supported_categorical_dtype(series):
        return "categorical"

    if is_complex_dtype(series.dtype):
        raise TypeError(
            f"Complex-valued columns are not supported for schema inference: {series.name!r}."
        )

    if not is_numeric_dtype(series.dtype):
        raise TypeError(
            f"Cannot classify column {series.name!r} with dtype {series.dtype!r}. "
            "Cast it to a supported dtype first."
        )

    non_null = series.dropna()
    if non_null.empty:
        # Fall back to dtype when the column is entirely missing.
        if _is_integer_like_numeric(series, tol=integer_tol):
            return "discrete"
        return "continuous"

    nunique = int(non_null.nunique(dropna=True))
    if _is_integer_like_numeric(series, tol=integer_tol):
        return "discrete" if nunique < discrete_max_unique else "continuous"

    return "continuous"


@dataclass(frozen=True)
class TabularSchema:
    continuous_cols: List[str]
    discrete_cols: List[str]
    categorical_cols: List[str]
    target_col: Optional[str] = None
    id_col: Optional[str] = None

    @property
    def feature_cols(self) -> List[str]:
        return [*self.continuous_cols, *self.discrete_cols, *self.categorical_cols]

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

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
    def has_continuous(self) -> bool:
        return bool(self.continuous_cols)

    @property
    def has_discrete(self) -> bool:
        return bool(self.discrete_cols)

    @property
    def has_categorical(self) -> bool:
        return bool(self.categorical_cols)

    def validate(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.all_cols if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        groups: Dict[str, List[str]] = {
            "continuous_cols": list(self.continuous_cols),
            "discrete_cols": list(self.discrete_cols),
            "categorical_cols": list(self.categorical_cols),
        }
        for group_name, cols in groups.items():
            duplicates = sorted({col for col in cols if cols.count(col) > 1})
            if duplicates:
                raise ValueError(f"Duplicate columns in {group_name}: {duplicates}")

        feature_cols = self.feature_cols
        overlaps = sorted({col for col in feature_cols if feature_cols.count(col) > 1})
        if overlaps:
            raise ValueError(
                "A column is assigned to multiple feature groups "
                f"(continuous/discrete/categorical): {overlaps}"
            )

        if self.id_col is not None and self.id_col in feature_cols:
            raise ValueError("id_col cannot also appear in the feature columns.")
        if self.target_col is not None and self.target_col in feature_cols:
            raise ValueError("target_col cannot also appear in the feature columns.")
        if self.id_col is not None and self.target_col is not None and self.id_col == self.target_col:
            raise ValueError("id_col and target_col must be different columns.")

        bad_continuous: List[str] = []
        for col in self.continuous_cols:
            series = df[col]
            if _is_supported_categorical_dtype(series) or is_complex_dtype(series.dtype):
                bad_continuous.append(col)
                continue
            if not is_numeric_dtype(series.dtype):
                bad_continuous.append(col)
        if bad_continuous:
            raise TypeError(
                "Continuous columns must be real numeric columns. "
                f"Bad continuous columns: {bad_continuous}"
            )

        bad_discrete: List[str] = []
        for col in self.discrete_cols:
            series = df[col]
            if not is_numeric_dtype(series.dtype) or is_complex_dtype(series.dtype):
                bad_discrete.append(col)
                continue
            if not _is_integer_like_numeric(series):
                bad_discrete.append(col)
        if bad_discrete:
            raise TypeError(
                "Discrete columns must be integer-like numeric columns "
                "(integer dtype or numeric values with x mod 1 == 0). "
                f"Bad discrete columns: {bad_discrete}"
            )

        bad_categorical: List[str] = []
        for col in self.categorical_cols:
            if not _is_supported_categorical_dtype(df[col]):
                bad_categorical.append(col)
        if bad_categorical:
            raise TypeError(
                "Categorical columns must be object, string, bool, or category dtype. "
                f"Bad categorical columns: {bad_categorical}"
            )

        if self.id_col is not None:
            sid = df[self.id_col]
            if sid.dropna().empty:
                raise ValueError(f"id_col {self.id_col!r} is entirely missing.")

    @classmethod
    def infer_from_dataframe(
        cls,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        id_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        continuous_cols: Optional[Sequence[str]] = None,
        discrete_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        discrete_max_unique: int = DEFAULT_DISCRETE_MAX_UNIQUE,
    ) -> "TabularSchema":
        cols = list(df.columns)

        if id_col is not None and id_col not in cols:
            raise ValueError(f"id_col={id_col!r} not found in the DataFrame.")
        if target_col is not None and target_col not in cols:
            raise ValueError(f"target_col={target_col!r} not found in the DataFrame.")

        if feature_cols is None:
            exclude = {col for col in (id_col, target_col) if col is not None}
            inferred_feature_cols = [col for col in cols if col not in exclude]
        else:
            inferred_feature_cols = list(feature_cols)
            missing_feature_cols = [col for col in inferred_feature_cols if col not in cols]
            if missing_feature_cols:
                raise ValueError(
                    "feature_cols contains columns that are not present in the DataFrame: "
                    f"{missing_feature_cols}"
                )

        feature_set = set(inferred_feature_cols)
        explicit_cont = set(continuous_cols or [])
        explicit_disc = set(discrete_cols or [])
        explicit_cat = set(categorical_cols or [])

        for label, explicit_group in (
            ("continuous_cols", explicit_cont),
            ("discrete_cols", explicit_disc),
            ("categorical_cols", explicit_cat),
        ):
            extras = sorted(explicit_group - feature_set)
            if extras:
                raise ValueError(f"{label} contains columns outside feature_cols: {extras}")

        overlap = (explicit_cont & explicit_disc) | (explicit_cont & explicit_cat) | (explicit_disc & explicit_cat)
        if overlap:
            raise ValueError(
                "A column cannot be explicitly assigned to more than one feature group: "
                f"{sorted(overlap)}"
            )

        continuous: List[str] = [col for col in inferred_feature_cols if col in explicit_cont]
        discrete: List[str] = [col for col in inferred_feature_cols if col in explicit_disc]
        categorical: List[str] = [col for col in inferred_feature_cols if col in explicit_cat]
        assigned = set(continuous) | set(discrete) | set(categorical)

        for col in inferred_feature_cols:
            if col in assigned:
                continue
            feature_type = classify_feature_type(df[col], discrete_max_unique=discrete_max_unique)
            if feature_type == "continuous":
                continuous.append(col)
            elif feature_type == "discrete":
                discrete.append(col)
            elif feature_type == "categorical":
                categorical.append(col)
            else:
                raise RuntimeError(f"Unexpected inferred feature type {feature_type!r} for column {col!r}.")

        schema = cls(
            continuous_cols=continuous,
            discrete_cols=discrete,
            categorical_cols=categorical,
            target_col=target_col,
            id_col=id_col,
        )
        schema.validate(df)
        return schema
