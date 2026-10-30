"""
Train-fitted encoders / decoders shared by the baseline wrappers.

Everything here is fitted on the rows passed to ``fit`` ONLY and is plain numpy/pandas, so it
can be unit-tested without any generator and persisted as plain python + arrays.

* ``fit_standardizer`` / ``standardize`` / ``unstandardize`` - per-column z-scoring with a
  deterministic zero-variance rule (scale := 1).
* ``nearest_support_decode`` - decode a declared-DISCRETE numeric column to the nearest value
  of its training support and report how far the raw generator output was from it.
  It is never applied to continuous columns (continuous outputs are never clipped/snapped).
* ``fit_vocabulary`` / ``encode_with_vocabulary`` - train-fitted categorical vocabulary.
* ``MixedToContinuousCodec`` - mixed-type frame <-> all-continuous matrix (z-scored numeric
  block + one-hot categorical blocks, argmax decoding) for continuous-only generators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sbtab.baselines.base import ColumnRoles, to_python_scalar

# A column whose std is below this fraction of max(1, |mean|) is treated as constant.
_ZERO_VARIANCE_REL_TOL = 1e-12


# ----------------------------------------------------------------------
# numeric block
# ----------------------------------------------------------------------


def numeric_matrix(df: pd.DataFrame, cols: Sequence[Any]) -> np.ndarray:
    """``df[cols]`` as a finite float64 matrix; raises a clear error on NaN / inf / non-numeric."""
    cols = list(cols)
    if not cols:
        return np.empty((len(df), 0), dtype=np.float64)
    try:
        x = df[cols].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Numeric (continuous/discrete) columns must be numeric: {cols}. {e}") from e
    bad = [c for j, c in enumerate(cols) if not np.all(np.isfinite(x[:, j]))]
    if bad:
        raise ValueError(
            f"Numeric columns contain NaN/inf: {bad}. Impute or drop missing rows before fit(); "
            "the baselines do not model missingness."
        )
    return x


def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-column mean / std (population std, float64) of the FIT rows.
    Zero-variance columns get scale 1.0 (deterministic; they encode to exactly 0).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] == 0:
        return np.zeros(0, dtype=np.float64), np.ones(0, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    constant = std <= _ZERO_VARIANCE_REL_TOL * np.maximum(1.0, np.abs(mean))
    scale = np.where(constant, 1.0, std)
    return mean, scale


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean) / scale


def unstandardize(z: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) * scale + mean


def nearest_support_decode(values: np.ndarray, support: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decode ``values`` to the nearest element of ``support`` (the sorted unique TRAINING
    values of a declared-discrete column).  O(n log s) via searchsorted; ties go to the
    lower support value.

    The report describes the generator output BEFORE decoding:
      * ``out_of_support_rate`` - fraction of raw values that are not (numerically) a support value
      * ``out_of_range_rate``   - fraction of raw values outside ``[min(support), max(support)]``
      * ``mean_abs_shift`` / ``max_abs_shift`` - size of the decoding correction, original units
    Non-finite raw values cannot be decoded and raise.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    support = np.unique(np.asarray(support, dtype=np.float64))
    if support.size == 0:
        raise ValueError("support must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot decode non-finite generated values to a discrete support.")

    hi = np.clip(np.searchsorted(support, values, side="left"), 0, support.size - 1)
    lo = np.clip(hi - 1, 0, support.size - 1)
    pick_lo = np.abs(values - support[lo]) <= np.abs(values - support[hi])
    decoded = np.where(pick_lo, support[lo], support[hi])

    shift = np.abs(values - decoded)
    span = float(support[-1] - support[0])
    tol = 1e-6 * max(1.0, span)
    n = int(values.size)
    report = {
        "kind": "nearest_support",
        "n": n,
        "support_size": int(support.size),
        "out_of_support_rate": float(np.mean(shift > tol)) if n else 0.0,
        "out_of_range_rate": float(np.mean((values < support[0] - tol) | (values > support[-1] + tol))) if n else 0.0,
        "mean_abs_shift": float(shift.mean()) if n else 0.0,
        "max_abs_shift": float(shift.max()) if n else 0.0,
    }
    return decoded, report


# ----------------------------------------------------------------------
# categorical block
# ----------------------------------------------------------------------


def fit_vocabulary(series: pd.Series, col: Any = None) -> List[Any]:
    """Sorted unique TRAINING values as python scalars. Missing values are rejected."""
    if series.isna().any():
        raise ValueError(
            f"Categorical column {col!r} contains missing values; encode missingness as its own "
            "category before fit()."
        )
    uniq = [to_python_scalar(v) for v in pd.unique(series)]
    try:
        return sorted(uniq)
    except TypeError:
        return sorted(uniq, key=lambda v: (type(v).__name__, str(v)))


def encode_with_vocabulary(series: pd.Series, vocab: Sequence[Any], col: Any = None) -> np.ndarray:
    """Values -> codes ``0..S-1``; values outside the vocabulary raise."""
    index = {v: i for i, v in enumerate(vocab)}
    raw = [to_python_scalar(v) for v in series.tolist()]
    try:
        return np.fromiter((index[v] for v in raw), dtype=np.int64, count=len(raw))
    except KeyError as e:
        raise ValueError(f"Categorical column {col!r} contains a value outside its vocabulary: {e}") from e


def decode_with_vocabulary(codes: np.ndarray, vocab: Sequence[Any]) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.float64).reshape(-1)
    if not np.isfinite(codes).all() or not np.equal(codes, np.floor(codes)).all():
        raise ValueError("Generated categorical codes must be finite integers.")
    if codes.size and (codes.min() < 0 or codes.max() >= len(vocab)):
        raise ValueError("Generated categorical code outside the training vocabulary.")
    codes = codes.astype(np.int64)
    arr = np.empty(len(vocab), dtype=object)
    arr[:] = list(vocab)
    return arr[codes]


def argmax_decode(block: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Argmax-decode a relaxed one-hot block.  ``mean_top1_margin`` (top-1 minus top-2 score)
    says how decisive the raw generator output was; ``ambiguous_rate`` is the fraction of
    rows whose margin is below 0.5 (half the distance between two one-hot vertices).
    """
    block = np.asarray(block, dtype=np.float64)
    if not np.all(np.isfinite(block)):
        raise ValueError("Cannot argmax-decode non-finite generated values.")
    codes = block.argmax(axis=1).astype(np.int64)
    if block.shape[1] > 1:
        part = np.sort(block, axis=1)
        margin = part[:, -1] - part[:, -2]
    else:
        margin = np.full(block.shape[0], np.inf)
    n = int(block.shape[0])
    finite = margin[np.isfinite(margin)]
    report = {
        "kind": "argmax",
        "n": n,
        "n_classes": int(block.shape[1]),
        "mean_top1_margin": float(finite.mean()) if finite.size else float("inf"),
        "ambiguous_rate": float(np.mean(margin < 0.5)) if n else 0.0,
    }
    return codes, report


def restore_dtype(values: np.ndarray, dtype: str) -> pd.Series:
    """Cast a decoded column back to the dtype it had in the frame passed to ``fit``."""
    s = pd.Series(values)
    if dtype == "object":
        return s.astype(object)
    try:
        if pd.api.types.is_integer_dtype(dtype) and pd.api.types.is_float_dtype(s.dtype):
            s = s.round()
        return s.astype(dtype)
    except (TypeError, ValueError):
        return s


# ----------------------------------------------------------------------
# mixed frame <-> continuous matrix
# ----------------------------------------------------------------------


class MixedToContinuousCodec:
    """
    Mixed-type frame <-> all-continuous float matrix, fitted on the FIT rows only.

    Layout of the encoded matrix: ``[numeric block | one-hot(cat_1) | ... | one-hot(cat_m)]``
    where the numeric block is ``continuous + discrete`` columns in frame order, z-scored when
    ``standardize_numeric`` (zero variance -> scale 1).

    ``decode`` inverts the z-scoring, decodes declared-discrete columns to the nearest training
    support value, argmax-decodes the one-hot blocks, and restores the original column order
    and dtypes.  Continuous columns are returned as generated (no clipping).
    """

    def __init__(self, standardize_numeric: bool = True):
        self.standardize_numeric = bool(standardize_numeric)
        self.columns: List[Any] = []
        self.dtypes: Dict[Any, str] = {}
        self.numeric_cols: List[Any] = []
        self.discrete_cols: List[Any] = []
        self.categorical_cols: List[Any] = []
        self.mean: np.ndarray = np.zeros(0)
        self.scale: np.ndarray = np.ones(0)
        self.supports: Dict[Any, np.ndarray] = {}
        self.vocab: Dict[Any, List[Any]] = {}
        self.fitted = False

    # -- fit / encode --------------------------------------------------

    def fit(self, df: pd.DataFrame, roles: ColumnRoles, columns: Optional[Sequence[Any]] = None) -> "MixedToContinuousCodec":
        cols = list(columns) if columns is not None else [c for c in df.columns if c != roles.id_col]
        self.columns = cols
        self.dtypes = {c: str(df[c].dtype) for c in cols}
        numeric = set(roles.numeric)
        self.numeric_cols = [c for c in cols if c in numeric]
        self.discrete_cols = [c for c in cols if c in set(roles.discrete)]
        self.categorical_cols = [c for c in cols if c in set(roles.categorical)]
        missing_role = [c for c in cols if c not in numeric and c not in set(roles.categorical)]
        if missing_role:
            raise ValueError(f"Columns without a role: {missing_role}")

        x = numeric_matrix(df, self.numeric_cols)
        if self.standardize_numeric:
            self.mean, self.scale = fit_standardizer(x)
        else:
            self.mean = np.zeros(x.shape[1], dtype=np.float64)
            self.scale = np.ones(x.shape[1], dtype=np.float64)
        self.supports = {
            c: np.unique(x[:, self.numeric_cols.index(c)]) for c in self.discrete_cols
        }
        self.vocab = {c: fit_vocabulary(df[c], c) for c in self.categorical_cols}
        self.fitted = True
        return self

    @property
    def dim(self) -> int:
        return len(self.numeric_cols) + sum(len(v) for v in self.vocab.values())

    @property
    def encoded_names(self) -> List[str]:
        names = [str(c) for c in self.numeric_cols]
        for c in self.categorical_cols:
            names.extend(f"{c}=={v!r}" for v in self.vocab[c])
        return names

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Codec is not fitted.")
        blocks = [standardize(numeric_matrix(df, self.numeric_cols), self.mean, self.scale)]
        for c in self.categorical_cols:
            codes = encode_with_vocabulary(df[c], self.vocab[c], c)
            blocks.append(np.eye(len(self.vocab[c]), dtype=np.float64)[codes])
        return np.concatenate(blocks, axis=1).astype(np.float32)

    # -- decode --------------------------------------------------------

    def decode(self, z: np.ndarray) -> Tuple[pd.DataFrame, Dict[Any, Dict[str, Any]]]:
        if not self.fitted:
            raise RuntimeError("Codec is not fitted.")
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError(f"Expected an (n, {self.dim}) matrix, got {z.shape}.")
        if not np.all(np.isfinite(z)):
            raise ValueError("Generator produced non-finite values.")

        n_num = len(self.numeric_cols)
        x_num = unstandardize(z[:, :n_num], self.mean, self.scale)
        report: Dict[Any, Dict[str, Any]] = {}
        decoded: Dict[Any, pd.Series] = {}

        for j, c in enumerate(self.numeric_cols):
            col = x_num[:, j]
            if c in self.supports:
                col, report[c] = nearest_support_decode(col, self.supports[c])
            decoded[c] = restore_dtype(col, self.dtypes[c])

        cursor = n_num
        for c in self.categorical_cols:
            width = len(self.vocab[c])
            codes, report[c] = argmax_decode(z[:, cursor: cursor + width])
            decoded[c] = restore_dtype(decode_with_vocabulary(codes, self.vocab[c]), self.dtypes[c])
            cursor += width

        out = pd.DataFrame({c: decoded[c] for c in self.columns}, columns=self.columns)
        return out, report

    # -- persistence ---------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "standardize_numeric": self.standardize_numeric,
            "columns": list(self.columns),
            "dtypes": [[c, self.dtypes[c]] for c in self.columns],
            "numeric_cols": list(self.numeric_cols),
            "discrete_cols": list(self.discrete_cols),
            "categorical_cols": list(self.categorical_cols),
            "mean": [float(v) for v in self.mean],
            "scale": [float(v) for v in self.scale],
            "supports": [[c, [float(v) for v in self.supports[c]]] for c in self.discrete_cols],
            "vocab": [[c, list(self.vocab[c])] for c in self.categorical_cols],
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "MixedToContinuousCodec":
        obj = cls(standardize_numeric=bool(state["standardize_numeric"]))
        obj.columns = list(state["columns"])
        obj.dtypes = {c: d for c, d in state["dtypes"]}
        obj.numeric_cols = list(state["numeric_cols"])
        obj.discrete_cols = list(state["discrete_cols"])
        obj.categorical_cols = list(state["categorical_cols"])
        obj.mean = np.asarray(state["mean"], dtype=np.float64)
        obj.scale = np.asarray(state["scale"], dtype=np.float64)
        obj.supports = {c: np.asarray(v, dtype=np.float64) for c, v in state["supports"]}
        obj.vocab = {c: list(v) for c, v in state["vocab"]}
        obj.fitted = True
        return obj
