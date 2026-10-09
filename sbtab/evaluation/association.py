"""
Dependence metrics (spec section 9, first half).

Every association is computed WITHIN one table (real held-out, or generated) and the
two resulting matrices are compared; row vectors of the two tables are never paired.

Blocks
  pearson         continuous x continuous
  spearman        discrete x discrete, on the original ordered values
  nmi             categorical x categorical, arithmetic normalisation
  eta_squared     numerical (continuous + discrete) x nominal conditioning column
  spearman_cross  continuous x discrete

Conventions: diagonal := 1 in both matrices (excluded from the normalised RMSE);
an off-diagonal Pearson/Spearman involving a constant vector := 0 with a flag;
NMI involving a constant column := 0 (also constant/constant); eta^2 := 0 with a
flag when the numerical column has zero total variance. ``diff = synth - real``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ._common import label_array, numeric_array, require_clean_real
from .spec import INSUFFICIENT_DATA, INVALID_GENERATED, METRIC_VERSION, NOT_APPLICABLE, OK
from .validity import check_validity

DIAGONAL_VALUE = 1.0
_DENSE_CONTINGENCY_LIMIT = 4_000_000


# --------------------------------------------------------------------------- building blocks
def _numeric_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.empty((len(df), 0), dtype=np.float64)
    return np.column_stack([numeric_array(df[c]) for c in cols]).astype(np.float64, copy=False)


def _constant_flags(X: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    return X.max(axis=0) == X.min(axis=0)


def _unit_columns(X: np.ndarray, const: np.ndarray) -> np.ndarray:
    """Centred, unit-norm columns; constant columns become exact zero vectors."""
    Xc = X - X.mean(axis=0, keepdims=True)
    norm = np.sqrt((Xc * Xc).sum(axis=0))
    dead = const | (norm == 0)
    U = Xc / np.where(dead, 1.0, norm)
    U[:, dead] = 0.0
    return U


def correlation_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pearson matrix of the columns of X with the constant-column convention."""
    const = _constant_flags(X)
    U = _unit_columns(X, const)
    R = np.clip(U.T @ U, -1.0, 1.0)
    np.fill_diagonal(R, DIAGONAL_VALUE)
    return R, const


def cross_correlation(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy = _constant_flags(X), _constant_flags(Y)
    return np.clip(_unit_columns(X, cx).T @ _unit_columns(Y, cy), -1.0, 1.0), cx, cy


def _ranks(X: np.ndarray) -> np.ndarray:
    return rankdata(X, method="average", axis=0).astype(np.float64) if X.shape[1] else X


def _codes(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(label_array(df[col]), use_na_sentinel=False)
    return np.asarray(codes, dtype=np.int64), int(len(uniques))


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def _mutual_information(ci: np.ndarray, ki: int, cj: np.ndarray, kj: int) -> float:
    n = ci.size
    flat = ci * kj + cj
    if ki * kj <= _DENSE_CONTINGENCY_LIMIT:
        joint = np.bincount(flat, minlength=ki * kj)
        nz = np.flatnonzero(joint)
        nij = joint[nz].astype(np.float64)
    else:                                   # two high-cardinality columns: sparse contingency
        nz, nij = np.unique(flat, return_counts=True)
        nij = nij.astype(np.float64)
    ni = np.bincount(ci, minlength=ki).astype(np.float64)[nz // kj]
    nj = np.bincount(cj, minlength=kj).astype(np.float64)[nz % kj]
    mi = float((nij / n * (np.log(nij) + math.log(n) - np.log(ni) - np.log(nj))).sum())
    return max(mi, 0.0)


def nmi_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise NMI (arithmetic mean of entropies), marginal entropies, constant flags."""
    d = len(cols)
    coded = [_codes(df, c) for c in cols]
    ent = np.array([_entropy(np.bincount(c, minlength=k).astype(np.float64)) for c, k in coded], dtype=np.float64)
    const = np.array([k <= 1 for _, k in coded], dtype=bool)
    M = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            if const[i] or const[j]:
                continue                     # NMI := 0, including constant/constant
            mi = _mutual_information(coded[i][0], coded[i][1], coded[j][0], coded[j][1])
            M[i, j] = M[j, i] = min(max(mi / (0.5 * (ent[i] + ent[j])), 0.0), 1.0)
    np.fill_diagonal(M, DIAGONAL_VALUE)
    return M, ent, const


def eta_squared_matrix(df: pd.DataFrame, numeric_cols: List[str], nominal_cols: List[str]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """eta^2[i, j] = between-group variance of numeric i across the levels of nominal j / total variance."""
    X = _numeric_matrix(df, numeric_cols)
    zero_var = _constant_flags(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    ss_tot = (Xc * Xc).sum(axis=0)
    zero_var = zero_var | (ss_tot == 0)
    E = np.zeros((len(numeric_cols), len(nominal_cols)), dtype=np.float64)
    for j, g in enumerate(nominal_cols):
        codes, k = _codes(df, g)
        n_g = np.bincount(codes, minlength=k).astype(np.float64)
        for i in range(len(numeric_cols)):
            if zero_var[i]:
                continue
            s_g = np.bincount(codes, weights=Xc[:, i], minlength=k)
            E[i, j] = min(max(float((s_g * s_g / n_g).sum() / ss_tot[i]), 0.0), 1.0)
    return E, zero_var


def offdiag_summary(real: np.ndarray, synth: np.ndarray) -> Dict[str, float]:
    """Raw Frobenius distance and sqrt( sum_{i != j} diff_ij^2 / (d (d - 1)) )."""
    diff = synth - real
    d = diff.shape[0]
    off = diff[~np.eye(d, dtype=bool)]
    return {"frobenius": float(np.sqrt((diff * diff).sum())),
            "offdiag_rmse": float(np.sqrt((off * off).sum() / (d * (d - 1)))),
            "max_abs_offdiag_diff": float(np.abs(off).max())}


def _rect_summary(real: np.ndarray, synth: np.ndarray) -> Dict[str, float]:
    err = np.abs(synth - real)
    return {"frobenius": float(np.sqrt((err * err).sum())), "rmse": float(np.sqrt((err * err).mean())),
            "mean_abs_error": float(err.mean()), "max_abs_error": float(err.max())}


def _flagged(cols: List[str], flags: np.ndarray) -> List[str]:
    return [c for c, f in zip(cols, flags) if f]


# --------------------------------------------------------------------------- public entry point
def association_metrics(ctx, real: pd.DataFrame, synth: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    schema = ctx.schema
    require_clean_real(real, schema, "real")
    validity = check_validity(synth, schema, ctx)
    cont, disc, cat = list(schema.continuous), list(schema.discrete), list(schema.categorical)
    nominal_cond = [c["column"] for c in ctx.conditioning["conditioners"] if c["kind"] == "categorical"]
    nominal_excluded = [c["column"] for c in ctx.conditioning["excluded"] if c["kind"] == "categorical"]
    numeric = cont + disc

    summary: Dict[str, Any] = {
        "status": validity["status"], "metric_version": METRIC_VERSION,
        "n_real": int(len(real)), "n_synth": int(validity["n_rows"]),
        "diagonal_value": DIAGONAL_VALUE, "diff_definition": "synth - real",
        "nmi_average_method": "arithmetic", "blocks": {}, "validity": validity,
        "context_spec_hash": ctx.spec_hash(),
    }
    arrays: Dict[str, np.ndarray] = {}
    blocks = summary["blocks"]
    applicable = {
        "pearson": len(cont) >= 2, "spearman": len(disc) >= 2, "nmi": len(cat) >= 2,
        "eta_squared": bool(numeric) and bool(nominal_cond), "spearman_cross": bool(cont) and bool(disc),
    }
    columns = {"pearson": cont, "spearman": disc, "nmi": cat}
    for name, ok in applicable.items():
        blocks[name] = {"status": OK if ok else NOT_APPLICABLE}
        if name in columns:        # square blocks; values stay None unless the block is computed
            blocks[name].update(columns=list(columns[name]), n_columns=len(columns[name]),
                                frobenius=None, offdiag_rmse=None, max_abs_offdiag_diff=None)
        else:
            blocks[name].update(frobenius=None, rmse=None, mean_abs_error=None, max_abs_error=None)
    blocks["eta_squared"].update(numeric_columns=numeric, nominal_columns=nominal_cond,
                                 excluded_high_cardinality_nominal_columns=nominal_excluded)
    blocks["spearman_cross"].update(continuous_columns=cont, discrete_columns=disc)

    bad = validity["status"] != OK
    if not bad and not any(applicable.values()):
        summary["status"] = NOT_APPLICABLE
        return summary, arrays
    short = min(len(real), validity["n_rows"]) < 2
    if bad or short:
        for name, ok in applicable.items():
            if ok:
                blocks[name]["status"] = INVALID_GENERATED if bad else INSUFFICIENT_DATA
        if short and not bad:
            summary["status"] = INSUFFICIENT_DATA
        return summary, arrays

    def _square(name: str, cols: List[str], R: np.ndarray, S: np.ndarray, rc: np.ndarray, sc: np.ndarray) -> None:
        arrays.update({f"{name}_columns": np.array(cols, dtype=str), f"{name}_real": R, f"{name}_synth": S,
                       f"{name}_diff": S - R, f"{name}_real_constant": rc, f"{name}_synth_constant": sc})
        blocks[name].update(offdiag_summary(R, S), normalizer="d*(d-1)",
                            real_constant_columns=_flagged(cols, rc), synth_constant_columns=_flagged(cols, sc))

    if applicable["pearson"]:
        R, rc = correlation_matrix(_numeric_matrix(real, cont))
        S, sc = correlation_matrix(_numeric_matrix(synth, cont))
        _square("pearson", cont, R, S, rc, sc)
    rank_real = rank_synth = None
    if disc:
        rank_real, rank_synth = _ranks(_numeric_matrix(real, disc)), _ranks(_numeric_matrix(synth, disc))
    if applicable["spearman"]:
        R, rc = correlation_matrix(rank_real)
        S, sc = correlation_matrix(rank_synth)
        _square("spearman", disc, R, S, rc, sc)
    if applicable["nmi"]:
        R, r_ent, rc = nmi_matrix(real, cat)
        S, s_ent, sc = nmi_matrix(synth, cat)
        _square("nmi", cat, R, S, rc, sc)
        arrays.update(nmi_real_entropy=r_ent, nmi_synth_entropy=s_ent)
        blocks["nmi"].update(real_entropy=dict(zip(cat, r_ent.tolist())), synth_entropy=dict(zip(cat, s_ent.tolist())))
    if applicable["eta_squared"]:
        R, rz = eta_squared_matrix(real, numeric, nominal_cond)
        S, sz = eta_squared_matrix(synth, numeric, nominal_cond)
        arrays.update(eta2_numeric_columns=np.array(numeric, dtype=str),
                      eta2_nominal_columns=np.array(nominal_cond, dtype=str),
                      eta2_real=R, eta2_synth=S, eta2_abs_error=np.abs(S - R),
                      eta2_real_zero_variance=rz, eta2_synth_zero_variance=sz)
        blocks["eta_squared"].update(_rect_summary(R, S), real_zero_variance_columns=_flagged(numeric, rz),
                                     synth_zero_variance_columns=_flagged(numeric, sz))
    if applicable["spearman_cross"]:
        R, rcx, rcy = cross_correlation(_ranks(_numeric_matrix(real, cont)), rank_real)
        S, scx, scy = cross_correlation(_ranks(_numeric_matrix(synth, cont)), rank_synth)
        arrays.update(spearman_cross_continuous_columns=np.array(cont, dtype=str),
                      spearman_cross_discrete_columns=np.array(disc, dtype=str),
                      spearman_cross_real=R, spearman_cross_synth=S, spearman_cross_abs_error=np.abs(S - R),
                      spearman_cross_real_constant_continuous=rcx, spearman_cross_real_constant_discrete=rcy,
                      spearman_cross_synth_constant_continuous=scx, spearman_cross_synth_constant_discrete=scy)
        blocks["spearman_cross"].update(
            _rect_summary(R, S),
            real_constant_columns=_flagged(cont, rcx) + _flagged(disc, rcy),
            synth_constant_columns=_flagged(cont, scx) + _flagged(disc, scy))
    return summary, arrays
