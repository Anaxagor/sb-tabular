"""
Validity of a generated table. Invalid cells are COUNTED and reported; they are
never filtered out so that the remaining rows could pass as a successful run.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._common import null_mask, numeric_array, support_codes
from .spec import INVALID_GENERATED, METRIC_VERSION, OK, MetricConfig


def check_validity(synth: pd.DataFrame, schema, ctx=None, *, support=None) -> Dict[str, Any]:
    """
    status = "invalid_generated_data" when any of
      * a schema column is missing, or the table has no rows;
      * a continuous/discrete cell is NaN, +-inf or not a number;
      * a categorical cell is null;
      * (ctx given, config.categorical_unexpected_is_invalid) a categorical label is
        outside the training vocabulary -- an invalid native category id.
    Out-of-support values of DISCRETE columns are reported in
    ``unexpected_value_rate`` but are legal numbers: they do not invalidate the table
    and are scored through the unexpected-value bin / the label union.

    Without ``ctx`` or explicit ``support``, training-support checks are skipped.
    Generation can supply the fitted common preprocessor's support directly;
    no histogram, MMD or held-out data is needed to validate a generated table.
    """
    config: MetricConfig = ctx.config if ctx is not None else MetricConfig()
    if ctx is not None and support is not None:
        raise ValueError("provide either ctx or support, not both")
    support = ctx.support if ctx is not None else support
    n = int(len(synth))
    missing = [c for c in schema.column_order if c not in synth.columns]
    extra = [str(c) for c in synth.columns if c not in set(schema.column_order)]
    numeric_cols = [c for c in list(schema.continuous) + list(schema.discrete) if c not in missing]
    nominal_cols = [c for c in schema.categorical if c not in missing]
    finite_cols = [c for c in schema.finite_support if c not in missing]

    bad_row = np.zeros(n, dtype=bool)
    nonfinite_rate: Dict[str, Optional[float]] = {}
    null_rate: Dict[str, Optional[float]] = {}
    unexpected_rate: Dict[str, Optional[float]] = {}
    reasons: List[str] = []

    for c in numeric_cols:
        bad = ~np.isfinite(numeric_array(synth[c]))
        nonfinite_rate[c] = float(bad.mean()) if n else None
        bad_row |= bad
    for c in nominal_cols:
        bad = null_mask(synth[c])
        null_rate[c] = float(bad.mean()) if n else None
        bad_row |= bad
    unexpected_categorical = False
    for c in finite_cols:
        if support is None or n == 0:
            unexpected_rate[c] = None
            continue
        bad = support_codes(synth[c], support[c]) < 0
        unexpected_rate[c] = float(bad.mean())
        if c in nominal_cols and config.categorical_unexpected_is_invalid and bad.any():
            unexpected_categorical = True
            bad_row |= bad

    if missing:
        reasons.append("missing_columns")
    if n == 0:
        reasons.append("no_rows")
    if any(v for v in nonfinite_rate.values()):
        reasons.append("nonfinite_numeric_values")
    if any(v for v in null_rate.values()):
        reasons.append("null_categorical_labels")
    if unexpected_categorical:
        reasons.append("unexpected_categorical_labels")

    return {
        "status": INVALID_GENERATED if reasons else OK,
        "metric_version": METRIC_VERSION,
        "reasons": reasons,
        "n_rows": n,
        "missing_columns": missing,
        "extra_columns": extra,
        "nonfinite_rate": nonfinite_rate,
        "null_rate": null_rate,
        "unexpected_value_rate": unexpected_rate,
        "support_checked": support is not None,
        "categorical_unexpected_is_invalid": bool(config.categorical_unexpected_is_invalid),
        "n_invalid_rows": int(bad_row.sum()),
        "invalid_row_rate": float(bad_row.mean()) if n else None,
    }


def numerical_diagnostics(synth: pd.DataFrame, schema, train: pd.DataFrame, *, extreme_ratio: float = 100.0
                          ) -> Dict[str, Any]:
    """Describe numeric tails in common units without changing validity or scores.

    Flag |generated| / max(1, max|training|) > 100 by default. This is a
    diagnostic threshold, not a distributional test or a rejection criterion.
    Nearest-order quantiles avoid interpolation overflow for large finite data.
    All reference statistics use training rows only.
    """
    if not np.isfinite(extreme_ratio) or extreme_ratio <= 1:
        raise ValueError("extreme_ratio must be finite and greater than one")
    columns, warnings = {}, []
    for c in list(schema.continuous) + list(schema.discrete):
        if c not in synth:
            continue
        values = numeric_array(synth[c])
        finite = values[np.isfinite(values)]
        reference = numeric_array(train[c])
        if not len(reference) or not np.isfinite(reference).all():
            raise ValueError(f"numerical diagnostics require finite training values in {c!r}")
        low, high = float(reference.min()), float(reference.max())
        scale = max(1.0, float(np.abs(reference).max()))
        magnitudes = np.abs(finite)
        extreme = magnitudes / scale > extreme_ratio
        columns[c] = {
            "n_finite": int(len(finite)), "n_nonfinite": int(len(values) - len(finite)),
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "max_abs": float(magnitudes.max()) if len(finite) else None,
            "median_abs": float(np.quantile(magnitudes, .5, method="nearest")) if len(finite) else None,
            "p99_abs": float(np.quantile(magnitudes, .99, method="nearest")) if len(finite) else None,
            "train_min": low, "train_max": high, "reference_abs_scale": scale,
            "outside_train_range_fraction": float(((finite < low) | (finite > high)).mean()) if len(finite) else None,
            "extreme_finite_fraction": float(extreme.mean()) if len(finite) else None,
        }
        if extreme.any():
            warnings.append({"column": c, "reason": "extreme_finite_values", "count": int(extreme.sum())})
    return {"version": "sbtab.numerical_diagnostics/1", "columns": columns, "warnings": warnings,
            "extreme_ratio": float(extreme_ratio), "affects_validity_or_selection": False,
            "definition": "abs(generated) / max(1, max(abs(training))) > extreme_ratio; fractions over finite cells"}
