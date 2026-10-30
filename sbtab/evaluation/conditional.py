"""
Conditional-distribution metrics (spec section 9, second half).

For every conditioner C (a categorical/discrete column with at most
``conditional_max_levels`` training-supported values, classification targets
included, plus train-defined quantile strata of a regression target) and every level
observed in the real held-out rows E_k, the conditional distribution of each OTHER
column is compared: standardised WD for continuous responses, JS over the aligned
label support for discrete/categorical responses. WD and JS are never pooled.

A level that the generated table does not contain has an UNDEFINED score: it is kept
as a row (status "undefined", ``missing_in_synth``), its real mass is reported as
``missing_category_mass`` and the conditioner is flagged
"incomplete_conditional_coverage". Nothing is filled with zero or renormalised away
silently: every mean states the eligible mass it is normalised over.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._common import joint_codes, js_divergence, numeric_array, require_clean_real, support_codes, wd_1d
from .spec import (INCOMPLETE_COVERAGE, INSUFFICIENT_DATA, METRIC_VERSION, NOT_APPLICABLE, OK, UNDEFINED,
                   MetricConfig)
from .validity import check_validity

STRATA_KIND = "regression_target_strata"

_LEVEL_TABLE_DTYPES = {
    "conditioner": object, "conditioner_column": object, "conditioner_kind": object, "level": object,
    "response": object, "response_type": object, "metric": object,
    "n_real": np.int64, "n_synth": np.int64, "real_level_mass": np.float64, "synth_level_mass": np.float64,
    "score": np.float64, "status": object, "missing_in_synth": bool, "synthetic_only": bool,
}
LEVEL_TABLE_COLUMNS = list(_LEVEL_TABLE_DTYPES)


# --------------------------------------------------------------------------- train-time definitions
def fit_conditioning(train: pd.DataFrame, schema, config: MetricConfig, support: Dict[str, List[Any]]
                     ) -> Dict[str, Any]:
    conditioners: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for c in schema.finite_support:
        k = len(support[c])
        entry = {"id": c, "column": c, "kind": schema.type_of(c), "n_levels": int(k)}
        if k <= config.conditional_max_levels:
            conditioners.append(entry)
        else:
            excluded.append({**entry, "reason": "high_cardinality", "max_levels": int(config.conditional_max_levels)})
    if schema.task == "regression" and schema.target is not None:
        t = schema.target
        y = numeric_array(train[t])
        q = int(config.regression_target_strata)
        probs = np.linspace(0.0, 1.0, q + 1)
        bounds = np.unique(np.quantile(y, probs))       # duplicate boundaries dropped
        interior = [float(b) for b in bounds[1:-1]]     # exterior boundaries are replaced by -inf / +inf
        conditioners.append({
            "id": f"{t}::quantile_strata", "column": t, "kind": STRATA_KIND,
            "n_levels": len(interior) + 1, "requested_strata": q, "quantile_probs": [float(p) for p in probs],
            "interior_boundaries": interior, "interval_closure": "right",
            "exterior": ["-inf", "+inf"], "labels": strata_labels(interior),
        })
    return {"max_levels": int(config.conditional_max_levels), "min_rows": int(config.conditional_min_rows),
            "conditioners": conditioners, "excluded": excluded}


def strata_labels(interior: List[float]) -> List[str]:
    b = ["-inf"] + [repr(float(v)) for v in interior] + ["+inf"]
    return [f"({b[i]}, {b[i + 1]}" + ("]" if i + 1 < len(b) - 1 else ")") for i in range(len(b) - 1)]


def strata_codes(values: np.ndarray, interior: List[float]) -> np.ndarray:
    """Right-closed strata (-inf, b1], (b1, b2], ..., (bk, +inf)."""
    return np.searchsorted(np.asarray(interior, dtype=np.float64), np.asarray(values, dtype=np.float64), side="left")


# --------------------------------------------------------------------------- helpers
def _group_slices(codes: np.ndarray, n_levels: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(codes, kind="stable")
    starts = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=n_levels))])
    return order, starts


def _aggregate(scores: np.ndarray, eligible: np.ndarray, real_mass: np.ndarray) -> Dict[str, Any]:
    n_el = int(eligible.sum())
    mass = float(real_mass[eligible].sum())
    if n_el == 0:
        return {"status": INSUFFICIENT_DATA, "macro_mean": None, "weighted_mean": None,
                "eligible_mass": mass, "n_eligible_levels": 0}
    s, w = scores[eligible], real_mass[eligible]
    return {"status": OK, "macro_mean": float(s.mean()), "weighted_mean": float((s * w).sum() / w.sum()),
            "eligible_mass": mass, "n_eligible_levels": n_el}


def _block(responses: Dict[str, Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """Equal column weights over the response columns of ONE metric block (WD or JS)."""
    items = [r for r in responses.values() if r["metric"] == metric]
    good = [r for r in items if r["status"] in (OK, INCOMPLETE_COVERAGE)]
    out = {"status": OK if good else (INSUFFICIENT_DATA if items else NOT_APPLICABLE),
           "n_responses": len(items), "n_eligible_responses": len(good), "macro_mean": None, "weighted_mean": None,
           "eligible_mass": float(np.mean([r["eligible_mass"] for r in good])) if good else None}
    if good:
        if any(r["status"] == INCOMPLETE_COVERAGE for r in good):
            out["status"] = INCOMPLETE_COVERAGE
        out["macro_mean"] = float(np.mean([r["macro_mean"] for r in good]))
        out["weighted_mean"] = float(np.mean([r["weighted_mean"] for r in good]))
    return out


class _TableBuilder:
    """Column-wise accumulation of the per-(conditioner, level, response) rows."""

    def __init__(self) -> None:
        self.parts: Dict[str, List[np.ndarray]] = {c: [] for c in LEVEL_TABLE_COLUMNS}

    def add(self, n: int, **values: Any) -> None:
        for c, dtype in _LEVEL_TABLE_DTYPES.items():
            v = values[c]
            self.parts[c].append(np.full(n, v, dtype=dtype) if np.ndim(v) == 0 else np.asarray(v, dtype=dtype))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({c: (np.concatenate(p) if p else np.empty(0, dtype=_LEVEL_TABLE_DTYPES[c]))
                             for c, p in self.parts.items()}, columns=LEVEL_TABLE_COLUMNS)


def _empty_table() -> pd.DataFrame:
    return _TableBuilder().frame()


# --------------------------------------------------------------------------- public entry point
def conditional_metrics(ctx, real: pd.DataFrame, synth: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    schema, config = ctx.schema, ctx.config
    require_clean_real(real, schema, "real")
    validity = check_validity(synth, schema, ctx)
    cond_defs = ctx.conditioning["conditioners"]
    min_rows = int(config.conditional_min_rows)
    summary: Dict[str, Any] = {
        "status": validity["status"], "metric_version": METRIC_VERSION,
        "n_real": int(len(real)), "n_synth": int(validity["n_rows"]),
        "min_rows": min_rows, "max_levels": int(config.conditional_max_levels),
        "n_conditioners": len(cond_defs),
        "excluded_conditioners": [dict(e) for e in ctx.conditioning["excluded"]],
        "incomplete_conditioners": [], "conditioners": {}, "secondary_across_conditioners": None,
        "pooling": "WD and JS blocks are never pooled; conditioners are reported separately",
        "validity": validity, "context_spec_hash": ctx.spec_hash(),
    }
    if validity["status"] != OK:
        return summary, _empty_table()
    if not cond_defs:
        summary["status"] = NOT_APPLICABLE
        return summary, _empty_table()

    n_real, n_synth = len(real), len(synth)
    numeric_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    label_cache: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}

    def _numeric(col: str) -> Tuple[np.ndarray, np.ndarray]:
        if col not in numeric_cache:
            numeric_cache[col] = (numeric_array(real[col]), numeric_array(synth[col]))
        return numeric_cache[col]

    def _labels(col: str) -> Tuple[np.ndarray, np.ndarray, int]:
        if col not in label_cache:
            cr, cs, labs = joint_codes(real[col], synth[col])
            label_cache[col] = (cr, cs, int(len(labs)))
        return label_cache[col]

    builder = _TableBuilder()
    for cdef in cond_defs:
        cid, ccol, kind = cdef["id"], cdef["column"], cdef["kind"]
        if kind == STRATA_KIND:
            yr, ys = _numeric(ccol)
            code_r, code_s = strata_codes(yr, cdef["interior_boundaries"]), strata_codes(ys, cdef["interior_boundaries"])
            level_names = list(cdef["labels"])
            in_support = np.ones(len(level_names), dtype=bool)
        else:
            code_r, code_s, labels = joint_codes(real[ccol], synth[ccol])
            level_names = [str(v) for v in labels.tolist()]
            in_support = support_codes(labels, ctx.support[ccol]) >= 0
        L = len(level_names)
        cnt_r = np.bincount(code_r, minlength=L).astype(np.int64)
        cnt_s = np.bincount(code_s, minlength=L).astype(np.int64)
        mass_r, mass_s = cnt_r / n_real, cnt_s / n_synth
        observed = np.flatnonzero(cnt_r > 0)                      # levels observed in E_k
        synth_only = np.flatnonzero((cnt_r == 0) & (cnt_s > 0))
        missing = cnt_s[observed] == 0
        eligible = (cnt_r[observed] >= min_rows) & (cnt_s[observed] >= min_rows)
        level_status = np.where(missing, UNDEFINED, np.where(eligible, OK, INSUFFICIENT_DATA))
        obs_mass = mass_r[observed]

        order_r, starts_r = _group_slices(code_r, L)
        order_s, starts_s = _group_slices(code_s, L)
        # compact index of the ELIGIBLE levels, so the level x label contingency table below is
        # bounded by the eligible real levels whatever the generated table contains
        el_levels = observed[eligible]
        remap = np.full(L, -1, dtype=np.int64)
        remap[el_levels] = np.arange(el_levels.size)
        keep_r, keep_s = remap[code_r] >= 0, remap[code_s] >= 0
        el_r, el_s = remap[code_r][keep_r], remap[code_s][keep_s]
        responses: Dict[str, Dict[str, Any]] = {}
        for resp in schema.column_order:
            if resp == ccol:
                continue
            rtype = schema.type_of(resp)
            scores = np.full(len(observed), np.nan, dtype=np.float64)
            if rtype == "continuous":
                metric = "wd"
                xr, xs = _numeric(resp)
                xr, xs = xr[order_r], xs[order_s]
                for k in np.flatnonzero(eligible):
                    lv = observed[k]
                    scores[k] = wd_1d(xr[starts_r[lv]:starts_r[lv + 1]], xs[starts_s[lv]:starts_s[lv + 1]])
            else:
                metric = "js"
                rr, rs, K = _labels(resp)
                E = int(el_levels.size)
                if E:
                    # level x response-label contingency counts; both tables share the label codes
                    joint_r = np.bincount(el_r * K + rr[keep_r], minlength=E * K).reshape(E, K)
                    joint_s = np.bincount(el_s * K + rs[keep_s], minlength=E * K).reshape(E, K)
                    scores[eligible] = np.atleast_1d(js_divergence(joint_r, joint_s))
            agg = _aggregate(scores, eligible, obs_mass)
            if missing.any() and agg["status"] == OK:
                agg["status"] = INCOMPLETE_COVERAGE
            responses[resp] = {"metric": metric, "response_type": rtype, "n_levels": int(len(observed)), **agg}
            builder.add(
                len(observed), conditioner=cid, conditioner_column=ccol, conditioner_kind=kind,
                level=[level_names[i] for i in observed], response=resp, response_type=rtype, metric=metric,
                n_real=cnt_r[observed], n_synth=cnt_s[observed], real_level_mass=obs_mass,
                synth_level_mass=mass_s[observed], score=scores, status=level_status,
                missing_in_synth=missing, synthetic_only=False)
        if synth_only.size:
            # one row per synthetic-only level (no response): unexpected / marginal mass error
            builder.add(
                len(synth_only), conditioner=cid, conditioner_column=ccol, conditioner_kind=kind,
                level=[level_names[i] for i in synth_only], response=None, response_type=None, metric=None,
                n_real=0, n_synth=cnt_s[synth_only], real_level_mass=0.0, synth_level_mass=mass_s[synth_only],
                score=np.nan, status=UNDEFINED, missing_in_synth=False, synthetic_only=True)

        missing_mass = float(obs_mass[missing].sum())
        insufficient = ~missing & ~eligible
        if missing.any():
            status = INCOMPLETE_COVERAGE
            summary["incomplete_conditioners"].append(cid)
        elif not eligible.any():
            status = INSUFFICIENT_DATA
        else:
            status = OK
        summary["conditioners"][cid] = {
            "status": status, "column": ccol, "kind": kind,
            "n_levels_train": int(cdef["n_levels"]), "n_levels_real": int(len(observed)),
            "n_levels_synth": int((cnt_s > 0).sum()),
            "n_eligible_levels": int(eligible.sum()), "eligible_mass": float(obs_mass[eligible].sum()),
            "n_insufficient_levels": int(insufficient.sum()), "insufficient_mass": float(obs_mass[insufficient].sum()),
            "n_missing_levels": int(missing.sum()), "missing_category_mass": missing_mass,
            "missing_levels": [level_names[i] for i in observed[missing]],
            "synthetic_only_mass": float(mass_s[synth_only].sum()),
            "synthetic_only_levels": [
                {"level": level_names[i], "n_synth": int(cnt_s[i]), "synth_mass": float(mass_s[i]),
                 "in_training_support": bool(in_support[i])} for i in synth_only],
            "wd_block": _block(responses, "wd"), "js_block": _block(responses, "js"),
            "responses": responses,
        }

    table = builder.frame()
    conds = summary["conditioners"]
    if summary["incomplete_conditioners"]:
        summary["status"] = INCOMPLETE_COVERAGE
    elif all(c["status"] == INSUFFICIENT_DATA for c in conds.values()):
        summary["status"] = INSUFFICIENT_DATA

    def _across(block: str, key: str) -> Optional[float]:
        vals = [c[block][key] for c in conds.values() if c[block][key] is not None]
        return float(np.mean(vals)) if vals else None

    summary["secondary_across_conditioners"] = {
        "note": "equal-weight mean over conditioners; secondary to the per-conditioner report and only as "
                "complete as the coverage statuses above",
        "wd_macro_mean": _across("wd_block", "macro_mean"), "wd_weighted_mean": _across("wd_block", "weighted_mean"),
        "js_macro_mean": _across("js_block", "macro_mean"), "js_weighted_mean": _across("js_block", "weighted_mean"),
        "n_conditioners_wd": sum(c["wd_block"]["macro_mean"] is not None for c in conds.values()),
        "n_conditioners_js": sum(c["js_block"]["macro_mean"] is not None for c in conds.values()),
        "mean_missing_category_mass": float(np.mean([c["missing_category_mass"] for c in conds.values()])),
        "max_missing_category_mass": float(np.max([c["missing_category_mass"] for c in conds.values()])),
        "mean_eligible_mass": float(np.mean([c["eligible_mass"] for c in conds.values()])),
    }
    return summary, table
