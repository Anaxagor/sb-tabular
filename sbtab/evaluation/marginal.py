"""
Tuning objective and final marginal-fidelity metrics (spec section 8).

``tuning_objective`` and ``marginal_metrics`` share the per-column WD / JS code
path (``_column_wd`` / ``_column_js``) and the objective formula
(``_objective_from_means``); nothing is restated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._common import (aligned_label_counts, hist_counts, js_divergence, kl_smoothed, make_hist_edges,
                      mean_or_none, numeric_array, require_clean_real, support_counts, wd_1d)
from .spec import (INVALID_GENERATED, KL_DIRECTION, LOG_BASE, METRIC_VERSION, NOT_APPLICABLE, OK, MetricConfig)
from .validity import check_validity

PER_FEATURE_COLUMNS = [
    "column", "type", "group", "is_target", "status", "n_real", "n_synth",
    "wd", "kl", "js", "kl_n_bins", "train_constant",
    "real_underflow_mass", "real_overflow_mass", "synth_underflow_mass", "synth_overflow_mass",
    "real_unexpected_mass", "synth_unexpected_mass",
]


# --------------------------------------------------------------------------- train-time fitting
def fit_histograms(train: pd.DataFrame, schema, config: MetricConfig) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in schema.continuous:
        x = numeric_array(train[c])
        edges, constant = make_hist_edges(x, config.kl_total_bins)
        out[c] = {"edges": edges, "constant": bool(constant),
                  "train_min": float(x.min()), "train_max": float(x.max())}
    return out


# --------------------------------------------------------------------------- shared per-column scores
def _column_wd(real: pd.DataFrame, synth: pd.DataFrame, col: str) -> float:
    return wd_1d(numeric_array(real[col]), numeric_array(synth[col]))


def _column_js(real: pd.DataFrame, synth: pd.DataFrame, col: str) -> float:
    _, cr, cs = aligned_label_counts(real[col], synth[col])
    return js_divergence(cr, cs)


def _objective_from_means(mean_wd: Optional[float], mean_js: Optional[float]) -> Optional[float]:
    """mean WD (continuous) + ONE combined mean JS (discrete and categorical together).
    An empty group contributes zero here -- and only here -- which reduces the mixed
    formula to the pure-regime objectives."""
    if mean_wd is None and mean_js is None:
        return None
    return float((mean_wd or 0.0) + (mean_js or 0.0))


def _invalid_columns(validity: Dict[str, Any]) -> set:
    bad = set(validity["missing_columns"])
    bad |= {c for c, r in validity["nonfinite_rate"].items() if r}
    bad |= {c for c, r in validity["null_rate"].items() if r}
    return bad


# --------------------------------------------------------------------------- tuning objective
def tuning_objective(real: pd.DataFrame, synth: pd.DataFrame, schema) -> Dict[str, Any]:
    """
    continuous regime: mean WD;  discrete regime: mean JS over discrete+categorical;
    mixed: mean WD + the single combined mean JS. The target sits in its declared group.
    Invalid generated data -> objective None with status "invalid_generated_data".
    """
    require_clean_real(real, schema, "real")
    validity = check_validity(synth, schema, None)
    cont, fin = list(schema.continuous), list(schema.finite_support)
    out: Dict[str, Any] = {
        "objective": None, "status": validity["status"], "regime": schema.regime,
        "mean_wd": None, "mean_js": None,
        "n_continuous": len(cont), "n_finite": len(fin),
        "per_column": {}, "validity": validity, "metric_version": METRIC_VERSION,
    }
    bad = _invalid_columns(validity)
    if validity["n_rows"] == 0:
        return out
    wds: List[float] = []
    jss: List[float] = []
    for c in schema.column_order:
        is_cont = c in cont
        entry = {"group": "continuous" if is_cont else "finite", "type": schema.type_of(c),
                 "metric": "wd" if is_cont else "js", "value": None,
                 "status": INVALID_GENERATED if c in bad else OK}
        if c not in bad:
            entry["value"] = _column_wd(real, synth, c) if is_cont else _column_js(real, synth, c)
            (wds if is_cont else jss).append(entry["value"])
        out["per_column"][c] = entry
    if validity["status"] != OK:
        return out                      # per-column diagnostics stay; no aggregate, no objective
    out["mean_wd"] = mean_or_none(wds)
    out["mean_js"] = mean_or_none(jss)
    out["objective"] = _objective_from_means(out["mean_wd"], out["mean_js"])
    return out


# --------------------------------------------------------------------------- final marginal metrics
def marginal_metrics(ctx, real: pd.DataFrame, synth: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    schema, config = ctx.schema, ctx.config
    require_clean_real(real, schema, "real")
    validity = check_validity(synth, schema, ctx)
    bad = _invalid_columns(validity)
    n_real, n_synth = int(len(real)), int(validity["n_rows"])
    target = schema.target
    mass = float(config.kl_smoothing_mass)

    rows: List[Dict[str, Any]] = []
    for c in schema.column_order:
        ctype = schema.type_of(c)
        row: Dict[str, Any] = {k: None for k in PER_FEATURE_COLUMNS}
        row.update(column=c, type=ctype, group=ctype, is_target=(c == target), n_real=n_real, n_synth=n_synth,
                   status=OK)
        if c in bad or n_synth == 0:
            row["status"] = INVALID_GENERATED
            rows.append(row)
            continue
        if ctype == "continuous":
            h = ctx.hist[c]
            cr = hist_counts(numeric_array(real[c]), h["edges"])
            cs = hist_counts(numeric_array(synth[c]), h["edges"])
            row.update(wd=_column_wd(real, synth, c), kl=kl_smoothed(cr, cs, mass), kl_n_bins=int(len(cr)),
                       train_constant=bool(h["constant"]),
                       real_underflow_mass=float(cr[0] / cr.sum()), real_overflow_mass=float(cr[-1] / cr.sum()),
                       synth_underflow_mass=float(cs[0] / cs.sum()), synth_overflow_mass=float(cs[-1] / cs.sum()))
        else:
            support = ctx.support[c]
            cr = support_counts(real[c], support)
            cs = support_counts(synth[c], support)
            row.update(kl=kl_smoothed(cr, cs, mass), js=_column_js(real, synth, c), kl_n_bins=int(len(cr)),
                       train_constant=bool(len(support) == 1),
                       real_unexpected_mass=float(cr[-1] / cr.sum()), synth_unexpected_mass=float(cs[-1] / cs.sum()))
        if (ctype == "categorical" and config.categorical_unexpected_is_invalid
                and validity["unexpected_value_rate"].get(c)):
            # the numbers above are diagnostics of an INVALID column (out-of-vocabulary labels)
            row["status"] = INVALID_GENERATED
        rows.append(row)
    table = pd.DataFrame(rows, columns=PER_FEATURE_COLUMNS)
    for col in ("wd", "kl", "js", "real_underflow_mass", "real_overflow_mass", "synth_underflow_mass",
                "synth_overflow_mass", "real_unexpected_mass", "synth_unexpected_mass", "kl_n_bins"):
        table[col] = pd.to_numeric(table[col], errors="coerce").astype(np.float64)

    valid = validity["status"] == OK

    def _group(cols: List[str], metrics: Tuple[str, ...]) -> Dict[str, Any]:
        g: Dict[str, Any] = {"status": OK, "n_columns": len(cols), "columns": list(cols)}
        if not cols:
            g["status"] = NOT_APPLICABLE
        elif not valid:
            g["status"] = INVALID_GENERATED
        sub = table[table["column"].isin(cols)]
        for m in metrics:
            g[f"mean_{m}"] = float(sub[m].mean()) if g["status"] == OK else None
        return g

    groups = {
        "continuous": _group(list(schema.continuous), ("wd", "kl")),
        "discrete": _group(list(schema.discrete), ("kl", "js")),
        "categorical": _group(list(schema.categorical), ("kl", "js")),
        "finite_combined": _group(list(schema.finite_support), ("kl", "js")),
    }
    summary: Dict[str, Any] = {
        "status": validity["status"],
        "metric_version": METRIC_VERSION,
        "regime": schema.regime,
        "n_real": n_real,
        "n_synth": n_synth,
        "kl_direction": KL_DIRECTION,
        "kl_total_bins": int(config.kl_total_bins),
        "kl_smoothing_mass": mass,
        "log_base": LOG_BASE,
        "js_smoothed": False,
        "target": target,
        "groups": groups,
        # the tuning formula re-evaluated on these rows, for traceability only
        "tuning_objective_equivalent": (_objective_from_means(groups["continuous"]["mean_wd"],
                                                              groups["finite_combined"]["mean_js"])
                                        if valid else None),
        "constant_training_columns": ([c for c in schema.continuous if ctx.hist[c]["constant"]]
                                      + [c for c in schema.finite_support if len(ctx.support[c]) == 1]),
        "validity": validity,
        "context_spec_hash": ctx.spec_hash(),
    }
    return summary, table
