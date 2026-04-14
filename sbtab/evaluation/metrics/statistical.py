from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy import stats
from typing import Dict, List, Optional

from sbtab.data.schema import TabularSchema

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_proj: int = 256) -> float:
    """
    Cut-off Wasserstein distance (SWD). 
    Evaluates the similarity of the joint distribution of all features.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    # Centering the data
    Xc, Yc = X_t - X_t.mean(0), Y_t - Y_t.mean(0)
    
    # Generating random projection
    thetas = torch.randn(n_proj, X_t.shape[1])
    thetas = thetas / thetas.norm(dim=1, keepdim=True)

    sw2 = 0.0
    for theta in thetas:
        # Projecting multidimensional data onto a line
        x1 = Xc @ theta
        y1 = Yc @ theta
        # We consider 1 day to be a weekend because of the schedules
        x1, _ = torch.sort(x1)
        y1, _ = torch.sort(y1)
        sw2 += F.mse_loss(x1, y1, reduction="mean")
        
    return float(sw2 / n_proj)

def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Mean 1D Wasserstein over the given numeric columns (no scaling)."""
    return float(np.mean([wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols]))


def _hist_prob(values: np.ndarray, bins: np.ndarray, eps: float) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins)
    probs = hist.astype(np.float64) + eps
    probs /= probs.sum()
    return probs


def _sorted_union_categories(r: pd.Series, s: pd.Series) -> List:
    combined = pd.concat([r, s], ignore_index=True).dropna()
    u = combined.unique().tolist()
    return sorted(u, key=lambda x: (str(type(x).__name__), repr(x)))


def category_levels_mixed(
    real: pd.DataFrame, synth: pd.DataFrame, schema: TabularSchema
) -> Dict[str, List]:
    """Category level lists per categorical feature (union of real and synthetic)."""
    return {c: _sorted_union_categories(real[c], synth[c]) for c in schema.categorical_cols}


def mixed_to_numeric_matrix(
    df: pd.DataFrame,
    schema: TabularSchema,
    category_levels: Dict[str, List],
) -> np.ndarray:
    """
    Expand categorical columns to one-hot rows and stack continuous columns for correlation analysis.
    """
    n = len(df)
    blocks: List[np.ndarray] = []
    for c in schema.feature_cols:
        if c in schema.categorical_cols:
            levels = category_levels.get(c, [])
            if not levels:
                continue
            codes = pd.Categorical(df[c], categories=levels).codes
            k = len(levels)
            block = np.zeros((n, k), dtype=np.float64)
            valid = codes >= 0
            if np.any(valid):
                idx = np.flatnonzero(valid)
                block[idx, codes[valid]] = 1.0
            blocks.append(block)
        else:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            x = np.nan_to_num(x, nan=0.0)
            blocks.append(x.reshape(-1, 1))
    if not blocks:
        return np.zeros((n, 0), dtype=np.float64)
    return np.hstack(blocks)


def marginal_kl_mean(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    schema: TabularSchema,
    n_bins: int = 50,
    eps: float = 1e-8,
) -> float:
    """
    Mean marginal KL(real || synth) over all feature columns.

    Categorical columns use discrete distributions over the union of observed categories;
    continuous columns use histogram density on bins spanning the real-data range.
    """
    kl_values: List[float] = []
    for c in schema.feature_cols:
        r, s = real[c], synth[c]

        if c in schema.categorical_cols:
            categories = _sorted_union_categories(r, s)
            if not categories:
                continue
            r_counts = r.value_counts(dropna=False)
            s_counts = s.value_counts(dropna=False)
            r_probs = (
                np.array([float(r_counts.get(cat, 0)) for cat in categories], dtype=np.float64) + eps
            )
            s_probs = (
                np.array([float(s_counts.get(cat, 0)) for cat in categories], dtype=np.float64) + eps
            )
            r_probs /= r_probs.sum()
            s_probs /= s_probs.sum()
            kl_values.append(float(stats.entropy(r_probs, s_probs)))
            continue

        r_num = pd.to_numeric(r, errors="coerce").dropna()
        s_num = pd.to_numeric(s, errors="coerce").dropna()
        if r_num.empty or s_num.empty:
            continue

        r_min, r_max = float(r_num.min()), float(r_num.max())
        if r_max == r_min:
            kl_values.append(0.0)
            continue
        bins = np.linspace(r_min, r_max, n_bins + 1)
        p = _hist_prob(r_num.to_numpy(), bins, eps)
        q = _hist_prob(s_num.to_numpy(), bins, eps)
        kl_values.append(float(stats.entropy(p, q)))

    return float(np.mean(kl_values)) if kl_values else 0.0


def frobenius_corr_diff_mixed(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    schema: TabularSchema,
) -> float:
    """Frobenius norm of (corr(real) - corr(synth)) on a mixed numeric + one-hot feature matrix."""
    levels = category_levels_mixed(real, synth, schema)
    real_mat = mixed_to_numeric_matrix(real, schema, levels)
    synth_mat = mixed_to_numeric_matrix(synth, schema, levels)
    if real_mat.shape[0] == 0 or synth_mat.shape[0] == 0 or real_mat.shape[1] == 0:
        return 0.0
    c_real = np.corrcoef(real_mat, rowvar=False)
    c_syn = np.corrcoef(synth_mat, rowvar=False)
    c_real = np.nan_to_num(c_real, nan=0.0)
    c_syn = np.nan_to_num(c_syn, nan=0.0)
    return float(np.linalg.norm(c_real - c_syn, ord="fro"))


def avg_wasserstein_mean(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    schema: TabularSchema,
    fit_scaler_on: str = "real",
) -> float:
    """
    Mean 1D Wasserstein distance over continuous features in standardized space.

    Categorical features are excluded. Scaler is fit on ``fit_scaler_on`` (``\"real\"`` or ``\"synth\"``).
    """
    cols = schema.continuous_cols
    if not cols:
        raise ValueError(
            "No continuous columns in schema; cannot compute Wasserstein mean for categorical-only data."
        )
    if fit_scaler_on not in ("real", "synth"):
        raise ValueError("fit_scaler_on must be 'real' or 'synth'")

    from sklearn.preprocessing import StandardScaler

    real_num = real[cols].astype(float)
    synth_num = synth[cols].astype(float)
    ref = real_num if fit_scaler_on == "real" else synth_num
    scaler = StandardScaler()
    scaler.fit(ref)
    real_z = scaler.transform(real_num)
    synth_z = scaler.transform(synth_num)
    dists = [
        float(wasserstein_distance(real_z[:, j], synth_z[:, j]))
        for j in range(len(cols))
    ]
    return float(np.mean(dists)) if dists else 0.0

def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """Maximum average discrepancy in RBF units."""
    X, Y = np.asarray(X), np.asarray(Y)
    # Subsampling to calculate sigma (acceleration)
    X_sub = X[:1000]
    if sigma is None:
        dists = cdist(X_sub, X_sub, metric='euclidean')
        sigma = np.median(dists[dists > 0]) or 1.0
            
    def kernel(a, b, s):
        sq_dist = cdist(a, b, metric='sqeuclidean')
        return np.exp(-sq_dist / (2 * s**2))

    k_xx = kernel(X_sub, X_sub, sigma).mean()
    k_yy = kernel(Y[:1000], Y[:1000], sigma).mean()
    k_xy = kernel(X_sub, Y[:1000], sigma).mean()
    return float(k_xx + k_yy - 2 * k_xy)

def calculate_marginal_kl(X_real: np.ndarray, X_syn: np.ndarray, bins: int = 50) -> float:
    """
    Average marginal KL for a fully numeric feature matrix (continuous-only).

    For mixed categorical and continuous columns in DataFrames, use ``marginal_kl_mean``
    with a :class:`~sbtab.data.schema.TabularSchema` instead.
    """
    vals = []
    epsilon = 1e-8 
    for j in range(X_real.shape[1]):
        real_col, syn_col = X_real[:, j], X_syn[:, j]
        low, high = min(real_col.min(), syn_col.min()), max(real_col.max(), syn_col.max())
        if high == low:
            vals.append(0.0)
            continue
        p, _ = np.histogram(real_col, bins=np.linspace(low, high, bins+1), density=True)
        q, _ = np.histogram(syn_col, bins=np.linspace(low, high, bins+1), density=True)
        vals.append(float(stats.entropy(p + epsilon, q + epsilon)))
    return float(np.mean(vals))

def calculate_frobenius_corr_diff(X_real: np.ndarray, X_syn: np.ndarray) -> float:
    """Correlation distance (Frobenius norm of the difference of Pearson matrices)."""
    C_real = np.nan_to_num(np.corrcoef(X_real, rowvar=False))
    C_syn = np.nan_to_num(np.corrcoef(X_syn, rowvar=False))
    return float(np.linalg.norm(C_real - C_syn, ord="fro"))