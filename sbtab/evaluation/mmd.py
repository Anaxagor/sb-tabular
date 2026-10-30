"""
Joint fidelity: characteristic product-kernel MMD in the mixed metric space (spec section 10).

    k(x, x') = exp( -||z - z'||^2 / (2 h^2)  -  (1 / max(C, 1)) * sum_j 1[c_j != c'_j] )

z  = continuous coordinates (already standardised by the common training scaler) and
     discrete coordinates standardised by a METRIC-ONLY scaler fitted on the same
     training rows (zero std -> 1);
c  = nominal columns, equality only (classification target included).
h  = median positive Euclidean distance among at most ``mmd_bandwidth_max_rows``
     deterministically chosen training rows; no positive distance -> h = 1 + flag.
Absent factors are omitted in pure regimes. Nothing here is fitted on E_k or on a
generated table.

The reported estimate is the signed unbiased MMD^2; it can be negative and is never
clipped. ``mmd2_biased`` (V-statistic, >= 0) is a separately named presentation aid.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

from ._common import joint_codes, numeric_array, require_clean_real, to_native
from .spec import INSUFFICIENT_DATA, INVALID_GENERATED, METRIC_VERSION, NOT_APPLICABLE, OK, MetricConfig
from .validity import check_validity

KERNEL_NAMES = ("full", "features")


# --------------------------------------------------------------------------- train-time metadata
def select_bandwidth_rows(n: int, max_rows: int) -> np.ndarray:
    """Deterministic, evenly spaced row positions (all rows when n <= max_rows). No RNG."""
    k = min(int(n), int(max_rows))
    return (np.arange(k, dtype=np.int64) * int(n)) // k


def fit_kernel_metadata(train: pd.DataFrame, schema, columns: Sequence[str], config: MetricConfig,
                        row_ids: np.ndarray) -> Dict[str, Any]:
    cont = [c for c in columns if schema.type_of(c) == "continuous"]
    disc = [c for c in columns if schema.type_of(c) == "discrete"]
    nom = [c for c in columns if schema.type_of(c) == "categorical"]
    if not (cont or disc or nom):
        return {"status": NOT_APPLICABLE, "reason": "no columns"}
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    zero_std: List[str] = []
    for c in disc:
        x = numeric_array(train[c])
        m, s = float(x.mean()), float(x.std(ddof=0))
        if x.min() == x.max() or s == 0.0 or not math.isfinite(s):
            s = 1.0
            zero_std.append(c)
        mean[c], std[c] = m, s
    meta: Dict[str, Any] = {
        "status": OK, "continuous_columns": cont, "discrete_columns": disc, "nominal_columns": nom,
        "discrete_scaler": {"mean": mean, "std": std, "zero_std_columns": zero_std, "ddof": 0},
        "hamming_normalizer": max(len(nom), 1),
    }
    pos = select_bandwidth_rows(len(train), config.mmd_bandwidth_max_rows)
    Z = numeric_matrix(train.iloc[pos], meta)
    bandwidth: Optional[float] = None
    degenerate = False
    if Z.shape[1] > 0:
        d = pdist(Z, metric="euclidean")
        d = d[d > 0]
        if d.size:
            bandwidth = float(np.median(d))
        else:
            bandwidth, degenerate = 1.0, True
    meta.update(bandwidth=bandwidth, bandwidth_degenerate=degenerate,
                bandwidth_rule="median positive Euclidean distance among the selected training rows",
                bandwidth_n_rows=int(len(pos)), bandwidth_row_positions=[int(p) for p in pos],
                bandwidth_row_ids=[to_native(row_ids[p]) for p in pos])
    return meta


def fit_mmd_metadata(train: pd.DataFrame, schema, config: MetricConfig, row_ids: np.ndarray) -> Dict[str, Any]:
    out = {"full": fit_kernel_metadata(train, schema, schema.column_order, config, row_ids)}
    if schema.target is None:
        out["features"] = {"status": NOT_APPLICABLE, "reason": "schema has no target"}
    else:
        out["features"] = fit_kernel_metadata(train, schema, schema.features, config, row_ids)
    return out


# --------------------------------------------------------------------------- representation
def numeric_matrix(df: pd.DataFrame, meta: Dict[str, Any]) -> np.ndarray:
    """z coordinates: continuous as received, discrete through the saved metric-only scaler."""
    cols = [numeric_array(df[c]) for c in meta["continuous_columns"]]
    sc = meta["discrete_scaler"]
    cols += [(numeric_array(df[c]) - sc["mean"][c]) / sc["std"][c] for c in meta["discrete_columns"]]
    if not cols:
        return np.empty((len(df), 0), dtype=np.float64)
    return np.column_stack(cols).astype(np.float64, copy=False)


def nominal_matrices(a: pd.DataFrame, b: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Nominal columns as codes assigned BY LABEL over both tables (only equality is used)."""
    noms = meta["nominal_columns"]
    if not noms:
        return np.empty((len(a), 0), dtype=np.float64), np.empty((len(b), 0), dtype=np.float64)
    pa, pb = [], []
    for c in noms:
        ca, cb, _ = joint_codes(a[c], b[c])
        pa.append(ca)
        pb.append(cb)
    return np.column_stack(pa).astype(np.float64), np.column_stack(pb).astype(np.float64)


# --------------------------------------------------------------------------- kernel and estimator
def kernel_block(ZA: np.ndarray, CA: np.ndarray, ZB: np.ndarray, CB: np.ndarray, bandwidth: Optional[float]
                 ) -> np.ndarray:
    expo = np.zeros((len(ZA), len(ZB)), dtype=np.float64)
    if ZA.shape[1] > 0:
        expo += cdist(ZA, ZB, metric="sqeuclidean") / (2.0 * float(bandwidth) ** 2)
    if CA.shape[1] > 0:
        expo += cdist(CA, CB, metric="hamming")          # = (1/C) * number of differing nominal columns
    return np.exp(-expo)


def _kernel_sum(ZA, CA, ZB, CB, bandwidth, block: int, exclude_diagonal: bool) -> float:
    sums: List[float] = []
    for i in range(0, len(ZA), block):
        for j in range(0, len(ZB), block):
            K = kernel_block(ZA[i:i + block], CA[i:i + block], ZB[j:j + block], CB[j:j + block], bandwidth)
            if exclude_diagonal:
                lo, hi = max(i, j), min(i + block, j + block, len(ZA))
                if lo < hi:
                    idx = np.arange(lo, hi)
                    K[idx - i, idx - j] = 0.0
            sums.append(float(K.sum(dtype=np.float64)))
    return math.fsum(sums)


def mmd2_blockwise(ZA: np.ndarray, CA: np.ndarray, ZB: np.ndarray, CB: np.ndarray, bandwidth: Optional[float],
                   block_size: int = 1024) -> Dict[str, float]:
    """
    Unbiased  sum_{i!=j} k(a_i,a_j)/(m(m-1)) + sum_{i!=j} k(b_i,b_j)/(n(n-1)) - 2 sum_{i,j} k(a_i,b_j)/(mn)
    accumulated block by block in float64; signed, never clipped. Also the biased V-statistic.
    """
    m, n = len(ZA), len(ZB)
    if m < 2 or n < 2:
        raise ValueError("the unbiased MMD needs at least two rows per side")
    block = int(block_size)
    s_aa = _kernel_sum(ZA, CA, ZA, CA, bandwidth, block, exclude_diagonal=True)
    s_bb = _kernel_sum(ZB, CB, ZB, CB, bandwidth, block, exclude_diagonal=True)
    s_ab = _kernel_sum(ZA, CA, ZB, CB, bandwidth, block, exclude_diagonal=False)
    unbiased = s_aa / (m * (m - 1)) + s_bb / (n * (n - 1)) - 2.0 * s_ab / (m * n)
    biased = (s_aa + m) / (m * m) + (s_bb + n) / (n * n) - 2.0 * s_ab / (m * n)      # k(x, x) = 1
    return {"mmd2_unbiased": float(unbiased), "mmd2_biased": float(max(biased, 0.0))}


# --------------------------------------------------------------------------- subsampling
def _subset(n: int, size: int, seed: int, stream: int) -> np.ndarray:
    if size >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng([int(seed), int(stream)]).choice(n, size=size, replace=False))


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def _ids(row_ids: Optional[Sequence[Any]], n: int, name: str) -> np.ndarray:
    if row_ids is None:
        return np.arange(n, dtype=np.int64)
    ids = np.asarray(row_ids)
    if ids.shape != (n,):
        raise ValueError(f"{name}_row_ids must have one id per row")
    return ids


# --------------------------------------------------------------------------- public entry point
def mmd_metrics(ctx, real: pd.DataFrame, synth: pd.DataFrame, real_row_ids=None, synth_row_ids=None
                ) -> Dict[str, Any]:
    schema, config = ctx.schema, ctx.config
    require_clean_real(real, schema, "real")
    validity = check_validity(synth, schema, ctx)
    m_all, n_all = int(len(real)), int(validity["n_rows"])
    seeds = [int(s) for s in config.mmd_seeds]
    size = min(m_all, n_all, int(config.mmd_max_rows))
    floor_size = min(m_all // 2, n_all, int(config.mmd_max_rows))
    out: Dict[str, Any] = {
        "status": validity["status"], "metric_version": METRIC_VERSION,
        "estimator": "unbiased MMD^2 (signed, not clipped)",
        "n_real": m_all, "n_synth": n_all, "max_rows": int(config.mmd_max_rows),
        "subsample_size": size, "seeds": seeds, "block_size": int(config.mmd_block_size),
        "variability_note": "std over saved subsampling seeds = subsampling variability, NOT a generalisation "
                            "standard error; identical uncapped subsets give 0 without implying certainty",
        "subsets": {}, "floor_size": floor_size,
        "floor_size_limited_by_synth": bool(n_all < min(m_all // 2, int(config.mmd_max_rows))),
        "floor_definition": "real-real: two disjoint subsets of E_k of size min(floor(|E_k|/2), |G_k|, max_rows); "
                            "real_synth_matched: the first of them vs a generated subset of the same size",
        "floor_subsets": {}, "kernels": {}, "validity": validity, "context_spec_hash": ctx.spec_hash(),
    }
    for name in KERNEL_NAMES:
        meta = ctx.mmd[name]
        out["kernels"][name] = {"status": meta["status"]} if meta["status"] != OK else {
            "status": OK, "bandwidth": meta["bandwidth"], "bandwidth_degenerate": meta["bandwidth_degenerate"],
            "continuous_columns": meta["continuous_columns"], "discrete_columns": meta["discrete_columns"],
            "nominal_columns": meta["nominal_columns"]}
    if validity["status"] != OK:
        for k in out["kernels"].values():
            if k["status"] == OK:
                k["status"] = INVALID_GENERATED
        return out
    if size < 2:
        out["status"] = INSUFFICIENT_DATA
        for k in out["kernels"].values():
            if k["status"] == OK:
                k["status"] = INSUFFICIENT_DATA
        return out

    rid, sid = _ids(real_row_ids, m_all, "real"), _ids(synth_row_ids, n_all, "synth")
    subsets = {s: (_subset(m_all, size, s, 0), _subset(n_all, size, s, 1)) for s in seeds}
    out["subsets"] = {str(s): {"real_ids": rid[ir].tolist(), "synth_ids": sid[js].tolist()}
                      for s, (ir, js) in subsets.items()}
    floors: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if floor_size >= 2:
        for s in seeds:
            perm = np.random.default_rng([s, 2]).permutation(m_all)
            a, b = np.sort(perm[:floor_size]), np.sort(perm[floor_size:2 * floor_size])
            floors[s] = (a, b, _subset(n_all, floor_size, s, 3))
        out["floor_subsets"] = {str(s): {"real_a_ids": rid[a].tolist(), "real_b_ids": rid[b].tolist(),
                                         "synth_ids": sid[g].tolist()} for s, (a, b, g) in floors.items()}

    block = int(config.mmd_block_size)
    for name in KERNEL_NAMES:
        meta = ctx.mmd[name]
        if meta["status"] != OK:
            continue
        h = meta["bandwidth"]
        ZR, ZS = numeric_matrix(real, meta), numeric_matrix(synth, meta)
        CR, CS = nominal_matrices(real, synth, meta)
        est = [mmd2_blockwise(ZR[ir], CR[ir], ZS[js], CS[js], h, block) for ir, js in subsets.values()]
        unb, bia = [e["mmd2_unbiased"] for e in est], [e["mmd2_biased"] for e in est]
        mean_u, std_u = _mean_std(unb)
        res = out["kernels"][name]
        res.update(mmd2_unbiased=unb, mmd2_unbiased_mean=mean_u, mmd2_unbiased_subsampling_std=std_u,
                   mmd2_biased=bia, mmd2_biased_mean=_mean_std(bia)[0])
        if floors:
            rr = [mmd2_blockwise(ZR[a], CR[a], ZR[b], CR[b], h, block)["mmd2_unbiased"] for a, b, _ in floors.values()]
            rs = [mmd2_blockwise(ZR[a], CR[a], ZS[g], CS[g], h, block)["mmd2_unbiased"] for a, _, g in floors.values()]
            (rr_m, rr_s), (rs_m, rs_s) = _mean_std(rr), _mean_std(rs)
            res["floor"] = {"status": OK, "size": floor_size,
                            "real_real_mmd2_unbiased": rr, "real_real_mean": rr_m, "real_real_subsampling_std": rr_s,
                            "real_synth_matched_mmd2_unbiased": rs, "real_synth_matched_mean": rs_m,
                            "real_synth_matched_subsampling_std": rs_s}
        else:
            res["floor"] = {"status": INSUFFICIENT_DATA, "size": floor_size,
                            "reason": "fewer than two rows per side: real-real reference unavailable"}
    return out
