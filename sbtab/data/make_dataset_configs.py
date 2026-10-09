"""
Drafting aid for ``configs/datasets/*.yaml``.

    python -m sbtab.data.make_dataset_configs [--output configs/datasets]

The YAML files are the benchmark's explicit metadata and the ONLY thing the
experiment stages read; they are meant to be reviewed and edited by hand. This
tool merely writes a first draft from a fixed rule plus the named overrides
below, so that the draft is reproducible. It looks at the complete table only to
read dtypes/cardinalities for drafting — never at a split — and it never runs as
part of an experiment.

Drafting rule
  non-numeric                                  -> categorical (nominal)
  numeric with a non-integer value             -> continuous
  integer-valued, <= DISCRETE_MAX_UNIQUE values -> discrete (ordered, finite support)
  integer-valued, more values                   -> continuous
  classification target                         -> categorical, whatever its storage
The ``datasets_continuous_only`` bundle is declared fully continuous by its
authors; that declaration is kept (every column continuous).

The rule is deliberately NOT adjusted to make a dataset pass the category-support
check: heavy-tailed counts typed ``discrete`` may block a dataset, and the
support report is where that must become visible.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sbtab.data.loading import BUNDLE_DIR, load_bundle, normalise_frame

DISCRETE_MAX_UNIQUE = 20

# Integer-coded columns that are NOMINAL (codes carry no order) or binary flags.
NOMINAL_INTEGER = {
    "Online Shoppers": ["OperatingSystems", "Browser", "Region", "TrafficType"],
    "Cardiovascular Disease": ["gender", "smoke", "alco", "active"],
    "Churn Modelling": ["HasCrCard", "IsActiveMember"],
    "House Sales": ["waterfront", "zipcode"],
    "Stroke Prediction": ["hypertension", "heart_disease"],
    "Auto MPG": ["origin"],
    "Eucalyptus": ["Rep", "Frosts"],
}

# Ordinal string columns: declared order, lowest first.
ORDERED = {
    "Car Evaluation": {
        "buying": ["low", "med", "high", "vhigh"],
        "maint": ["low", "med", "high", "vhigh"],
        "doors": ["2", "3", "4", "5more"],
        "persons": ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety": ["low", "med", "high"],
    },
    # "30-39"-style ranges: ordered by their leading integer (filled in below).
    "Breast cancer": {"age": "range", "tumor-size": "range", "inv-nodes": "range"},
}

# Identifier columns removed before splitting.
DROPPED = {"bank_loan": ["ID"]}

# Label clean-up applied before splitting.
VALUE_MAPS = {
    # UCI Adult: the original test partition writes its labels with a trailing dot.
    "Adult": {"income": {"<=50K.": "<=50K", ">50K.": ">50K"}},
}

NOTES = {
    "Breast cancer": (
        "tumor-size and inv-nodes contain Excel date-mangled range labels from the UCI source: '9-May' = 5-9, '14-Oct' = 10-14, '5-Mar' = 3-5, '8-Jun' = 6-8, '11-Sep' = 9-11, '14-Dec' = 12-14. The declared ordinal order was checked against these meanings (the mangled label's leading number is the range's upper bound, so it sorts correctly). Labels are kept verbatim."
    ),
}

SLUG_OVERRIDES = {("datasets_mixed", "Online Shoppers"): "online_shoppers_mixed"}


def slugify(bundle: str, name: str) -> str:
    if (bundle, name) in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[(bundle, name)]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _integer_valued(s: pd.Series) -> bool:
    v = s.dropna().to_numpy(dtype=float)
    return bool(v.size) and bool(np.all(np.abs(v - np.round(v)) < 1e-12))


def draft(bundle: str, name: str, df: pd.DataFrame) -> dict:
    attrs = dict(getattr(df, "attrs", {}) or {})
    target, task = attrs.get("target_variable"), attrs.get("task_type")
    f = normalise_frame(df)
    dropped = [c for c in DROPPED.get(name, []) if c in f.columns]
    nominal = set(NOMINAL_INTEGER.get(name, []))
    ordered = ORDERED.get(name, {})
    all_continuous = bundle == "datasets_continuous_only"

    columns = []
    for col in f.columns:
        if col in dropped:
            continue
        s = f[col]
        entry = {"name": col, "role": "target" if col == target else "feature"}
        if col == target and task == "classification":
            ctype = "categorical"
        elif not pd.api.types.is_numeric_dtype(s.dtype):
            ctype = "categorical"
        elif all_continuous:
            ctype = "continuous"
        elif col in nominal:
            ctype = "categorical"
        elif _integer_valued(s) and s.nunique(dropna=True) <= DISCRETE_MAX_UNIQUE:
            ctype = "discrete"
        else:
            ctype = "continuous"
        entry["type"] = ctype
        if col in ordered and ctype == "categorical" and col != target:
            order = ordered[col]
            if order == "range":
                vals = sorted(s.dropna().unique().tolist(), key=lambda v: int(re.match(r"\d+", v).group()))
                order = vals
            entry["ordered_values"] = [str(v) for v in order]
        columns.append(entry)

    n_missing = int(f[[c["name"] for c in columns]].isna().sum().sum())
    cfg = {
        "name": slugify(bundle, name),
        "source": {"bundle": f"{bundle}.pkl", "key": name},
        "task": task,
        "target": target,
        "missing_policy": "impute" if n_missing else "reject",
        "dropped_columns": dropped,
        "columns": columns,
    }
    if name in VALUE_MAPS:
        cfg["value_maps"] = VALUE_MAPS[name]
    if name in NOTES:
        cfg["notes"] = NOTES[name]          # notes are excluded from the schema hash
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="configs/datasets")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    for bundle in ("datasets_continuous_only", "datasets_categorical", "datasets_mixed"):
        for name, df in load_bundle(BUNDLE_DIR / f"{bundle}.pkl").items():
            cfg = draft(bundle, name, df)
            path = out / f"{cfg['name']}.yaml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# Explicit benchmark metadata. Edit by hand; a change creates a new schema hash.\n")
                yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True, width=120)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
