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


def check_validity(synth: pd.DataFrame, schema, ctx=None) -> Dict[str, Any]:
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

    ``ctx=None`` (tuning objective) skips the training-support checks; the
    corresponding rates are ``None``.
    """
    config: MetricConfig = ctx.config if ctx is not None else MetricConfig()
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
        if ctx is None or n == 0:
            unexpected_rate[c] = None
            continue
        bad = support_codes(synth[c], ctx.support[c]) < 0
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
        "support_checked": ctx is not None,
        "categorical_unexpected_is_invalid": bool(config.categorical_unexpected_is_invalid),
        "n_invalid_rows": int(bad_row.sum()),
        "invalid_row_rate": float(bad_row.mean()) if n else None,
    }
