"""
Shared numerical helpers. Every metric module (tuning objective, final marginal
metrics, conditional metrics) calls THESE functions, so a formula exists once.

Conventions
  * natural logarithms everywhere;
  * float64 everywhere;
  * finite-support columns are aligned BY LABEL VALUE (hash/equality of the label),
    never by array position or by a per-table integer recoding.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance


# --------------------------------------------------------------------------- JSON
def json_safe(obj: Any) -> Any:
    """Recursively convert to plain JSON types; NaN / +-inf become ``None``."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {_json_key(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return {str(c): json_safe(obj[c].tolist()) for c in obj.columns}
    if isinstance(obj, (pd.Series, pd.Index)):
        return json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return json_safe(obj.to_dict())
    return str(obj)


def _json_key(k: Any) -> str:
    if isinstance(k, str):
        return k
    if isinstance(k, (np.generic,)):
        k = k.item()
    return str(k)


# --------------------------------------------------------------------------- labels / numerics
def to_native(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def label_array(values: Any) -> np.ndarray:
    """Label values of a column as a 1-D numpy array (Categorical dtype -> its labels)."""
    if isinstance(values, (pd.Series, pd.Index)):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    if arr.dtype.kind in "US":          # fixed-width numpy strings: keep them python strings
        arr = arr.astype(object)
    return arr.reshape(-1)


def sorted_labels(values: Any) -> List[Any]:
    """Unique labels, sorted; mixed incomparable types fall back to (type name, str) order."""
    labs = [to_native(v) for v in pd.unique(label_array(values))]
    if any(_is_null(v) for v in labs):
        raise ValueError("null label in a finite-support training column")
    try:
        return sorted(labs)
    except TypeError:
        return sorted(labs, key=lambda v: (type(v).__name__, str(v)))


def _is_null(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def numeric_array(values: Any) -> np.ndarray:
    """float64 view of a numeric column; anything unparseable becomes NaN (never dropped)."""
    s = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values).reshape(-1))
    if s.dtype.kind in "fiub":
        return s.to_numpy(dtype=np.float64, na_value=np.nan)
    # object / string storage: a number stored as text is NOT a number of this column
    out = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan).copy()
    is_text = s.map(lambda v: isinstance(v, (str, bytes))).to_numpy(dtype=bool)
    out[is_text] = np.nan
    return out


def null_mask(values: Any) -> np.ndarray:
    return np.asarray(pd.isna(pd.Series(label_array(values))), dtype=bool)


def joint_codes(a: Any, b: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integer codes of two label arrays over their COMMON label universe.

    Returns (codes_a, codes_b, labels). Codes are assigned by label value through a
    single hash-based factorisation of the concatenated arrays, so the same label
    receives the same code in both tables whatever its position or storage dtype
    (1 and 1.0 are one label; 1 and "1" are two).
    """
    a = label_array(a)
    b = label_array(b)
    if a.dtype != b.dtype and not (a.dtype.kind in "fiub" and b.dtype.kind in "fiub"):
        a = a.astype(object)
        b = b.astype(object)
    both = np.concatenate([a, b])
    codes, labels = pd.factorize(both, use_na_sentinel=False)
    codes = np.asarray(codes, dtype=np.int64)
    return codes[: len(a)], codes[len(a):], np.asarray(labels)


def aligned_label_counts(a: Any, b: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts of two samples over the union of their labels -> (labels, counts_a, counts_b)."""
    ca, cb, labels = joint_codes(a, b)
    k = len(labels)
    return labels, np.bincount(ca, minlength=k).astype(np.float64), np.bincount(cb, minlength=k).astype(np.float64)


def support_codes(values: Any, support: Sequence[Any]) -> np.ndarray:
    """Position of each value in the saved training support; -1 = unexpected value."""
    arr = label_array(values)
    index = pd.Index(list(support))
    if index.dtype == object or arr.dtype == object:
        index = index.astype(object)
        arr = arr.astype(object)
    return np.asarray(index.get_indexer(arr), dtype=np.int64)


def support_counts(values: Any, support: Sequence[Any]) -> np.ndarray:
    """Counts over the training support PLUS one trailing unexpected-value bin."""
    codes = support_codes(values, support)
    k = len(support)
    codes = np.where(codes < 0, k, codes)
    return np.bincount(codes, minlength=k + 1).astype(np.float64)


# --------------------------------------------------------------------------- histogram bins
def make_hist_edges(train_values: np.ndarray, total_bins: int) -> Tuple[np.ndarray, bool]:
    """
    ``total_bins - 2`` equal-width interior intervals spanning [train min, train max]
    (``total_bins - 1`` edges). A constant training column uses the deterministic
    unit-width interval [v - 0.5, v + 0.5] and returns ``constant=True``.
    """
    x = np.asarray(train_values, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("histogram edges need finite, non-empty training values")
    lo, hi = float(x.min()), float(x.max())
    constant = lo == hi
    if constant:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, total_bins - 1, dtype=np.float64), constant


def hist_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Counts in [underflow | interior_1 .. interior_K | overflow], K = len(edges) - 1.

    Interior bin i is [e_{i-1}, e_i); the last interior bin is closed on both sides,
    so BOTH training extrema are interior. Underflow is strictly below e_0, overflow
    strictly above e_K.
    """
    x = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite value reached the histogram; validity must be checked first")
    edges = np.asarray(edges, dtype=np.float64)
    idx = np.searchsorted(edges, x, side="right")
    idx[x == edges[-1]] = len(edges) - 1
    return np.bincount(idx, minlength=len(edges) + 1).astype(np.float64)


# --------------------------------------------------------------------------- divergences
def _probabilities(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    if np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("malformed counts")
    tot = c.sum(axis=-1, keepdims=True)
    if np.any(tot <= 0):
        raise ValueError("empty sample: probabilities are undefined")
    return c / tot


def js_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> Any:
    """
    Jensen-Shannon DIVERGENCE (natural log, range [0, log 2]) from aligned counts,
    unsmoothed, with 0 log 0 = 0. Works along the last axis (1-D -> float).
    """
    p = _probabilities(p_counts)
    q = _probabilities(q_counts)
    m = 0.5 * (p + q)
    js = 0.5 * rel_entr(p, m).sum(axis=-1) + 0.5 * rel_entr(q, m).sum(axis=-1)
    js = np.clip(js, 0.0, math.log(2.0))      # round-off only; the divergence is within the range analytically
    return float(js) if np.ndim(js) == 0 else js


def smooth_probabilities(counts: np.ndarray, mass: float) -> np.ndarray:
    """(p + mass/B) / (1 + mass) over the B bins."""
    p = _probabilities(counts)
    b = p.shape[-1]
    return (p + mass / b) / (1.0 + mass)


def kl_smoothed(p_counts: np.ndarray, q_counts: np.ndarray, mass: float) -> float:
    """KL(p || q) after the SAME uniform smoothing of both distributions."""
    p = smooth_probabilities(p_counts, mass)
    q = smooth_probabilities(q_counts, mass)
    val = float(rel_entr(p, q).sum())
    if not math.isfinite(val):
        raise ValueError("KL is not finite: malformed probabilities")
    return max(val, 0.0)                       # KL >= 0 analytically; guards -1e-17 round-off


def wd_1d(a: np.ndarray, b: np.ndarray) -> float:
    """First Wasserstein distance between two 1-D empirical samples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("empty sample")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("non-finite value reached the Wasserstein distance; validity must be checked first")
    return float(wasserstein_distance(a, b))


def js_labels(a: Any, b: Any) -> float:
    """JS divergence between two label samples over the union of their labels."""
    _, ca, cb = aligned_label_counts(a, b)
    return js_divergence(ca, cb)


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------------------- real-side guard
def require_clean_real(df: pd.DataFrame, schema, name: str = "real") -> None:
    """Real tables are inputs of the protocol, not something to score: fail loudly."""
    missing = [c for c in schema.column_order if c not in df.columns]
    if missing:
        raise ValueError(f"{name} table is missing schema columns: {missing}")
    if len(df) == 0:
        raise ValueError(f"{name} table has no rows")
    for c in list(schema.continuous) + list(schema.discrete):
        if not np.all(np.isfinite(numeric_array(df[c]))):
            raise ValueError(f"{name} table: non-finite or non-numeric value in numeric column {c!r}")
    for c in schema.categorical:
        if null_mask(df[c]).any():
            raise ValueError(f"{name} table: null label in categorical column {c!r}")
