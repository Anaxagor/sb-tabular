"""
Dataset eligibility: which rows of a source table form D, the "complete eligible dataset".

Rule ``min_value_count`` (protocol-level, applied BEFORE any split, identically for every model):
a row is removed when its value in a finite-support column occurs in fewer than ``min_value_count``
rows of the table. Removing rows can push another value below the threshold, so the rule is
iterated to a fixed point; every pass evaluates all columns on the same snapshot, which makes the
result independent of column order.

This is a DATASET-DEFINITION step, not a learned transform: it uses value counts of the whole
source table and nothing else (no split, no model, no seed). It does, however, look at the
classification target like at any other finite-support column, so rare CLASSES are removed too —
that changes the task and is reported explicitly (``target_values_removed``).

A count threshold reduces, but cannot guarantee, category-support coverage under a fixed random
split (three rows of a value can still end up 1 in V and 2 in the same CV test fold). The support
validation in ``prepare_splits`` therefore stays in force after this rule.

Original row ids are preserved: the eligible frame keeps the source ``row_id`` index (with gaps).
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from sbtab.data.dataset_schema import CATEGORICAL, MISSING_TOKEN, DatasetSchema

ELIGIBILITY_VERSION = "sbtab.eligibility/1"
COLUMN_SCOPES = ("finite_support", "categorical")


def finite_levels(frame: pd.DataFrame, col: str, schema: DatasetSchema) -> pd.Series:
    """
    Levels of a finite-support column as strings. A missing CATEGORICAL value is the level
    ``__missing__`` (it must be covered like any other level). A missing numeric discrete value is
    not a level: it is imputed from the training rows.
    """
    s = frame[col]
    if schema.type_of(col) == CATEGORICAL:
        return s.astype(object).where(~s.isna(), MISSING_TOKEN).astype(str)
    return s.map(lambda v: None if pd.isna(v) else repr(float(v)))


def validate_rule(rule: dict) -> dict:
    rule = dict(rule or {})
    unknown = sorted(set(rule) - {"min_value_count", "columns", "iterate_to_fixed_point"})
    if unknown:
        raise ValueError(f"unknown eligibility keys {unknown}")
    n = rule.get("min_value_count", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("eligibility.min_value_count must be an integer >= 1")
    scope = rule.get("columns", "finite_support")
    if scope not in COLUMN_SCOPES:
        raise ValueError(f"eligibility.columns must be one of {COLUMN_SCOPES}")
    return {"min_value_count": n, "columns": scope, "iterate_to_fixed_point": bool(rule.get("iterate_to_fixed_point", True))}


def apply_eligibility(frame: pd.DataFrame, schema: DatasetSchema, rule: dict, max_passes: int = 100) -> Tuple[pd.DataFrame, dict]:
    """Returns (eligible frame with the ORIGINAL row ids, JSON-safe report)."""
    rule = validate_rule(rule)
    n_min = rule["min_value_count"]
    cols: List[str] = list(schema.finite_support if rule["columns"] == "finite_support" else schema.categorical)
    keep, removals, passes = frame, [], 0

    while n_min > 1 and cols:
        passes += 1
        if passes > max_passes:
            raise RuntimeError("eligibility rule did not reach a fixed point")
        drop = pd.Series(False, index=keep.index)
        found = []
        for col in cols:
            levels = finite_levels(keep, col, schema)
            counts = levels.value_counts(dropna=True)
            for value, n in counts[counts < n_min].items():
                ids = levels.index[levels == value]
                drop.loc[ids] = True
                found.append({"pass": passes, "column": col, "column_type": schema.type_of(col),
                              "is_target": col == schema.target, "value": str(value), "count": int(n),
                              "row_ids": [int(i) for i in ids]})
        if not found:
            passes -= 1
            break
        removals += found
        keep = keep.loc[~drop]
        if len(keep) == 0:
            raise ValueError("the eligibility rule removed every row")
        if not rule["iterate_to_fixed_point"]:
            break

    removed_ids = sorted(set(frame.index) - set(keep.index))
    per_column = {}
    for r in removals:
        c = per_column.setdefault(r["column"], {"n_values_removed": 0, "values": []})
        c["n_values_removed"] += 1
        c["values"].append({"value": r["value"], "count": r["count"], "pass": r["pass"]})
    target_values = [r["value"] for r in removals if r["is_target"]]
    report = {
        "version": ELIGIBILITY_VERSION, "rule": rule, "applied": n_min > 1,
        "columns_checked": cols if n_min > 1 else [],
        "n_source_rows": int(len(frame)), "n_eligible_rows": int(len(keep)), "n_removed_rows": int(len(removed_ids)),
        "removed_fraction": float(len(removed_ids) / len(frame)) if len(frame) else 0.0,
        "passes": int(passes), "removed_row_ids": [int(i) for i in removed_ids],
        "per_column": per_column, "removals": removals,
        "target_values_removed": target_values,
        "task_changed": bool(target_values),
        "note": "Applied to the whole source table before any split; identical for every model. "
                "A count threshold does not guarantee support coverage; prepare_splits still validates it.",
    }
    return keep, report
