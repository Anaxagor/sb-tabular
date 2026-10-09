"""
Dataset registry: explicit YAML metadata -> (table, schema, manifest).

A dataset is identified by ``configs/datasets/<name>.yaml``. Loading applies only
schema-driven, outcome-independent steps, all BEFORE any split: identifier
removal, declared label clean-up and dtype normalisation. Rows are never dropped.
Every row keeps a stable ``row_id``: its position in the source table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from sbtab.data.dataset_schema import CATEGORICAL, ColumnSpec, DatasetSchema
from sbtab.data.loading import BUNDLE_DIR, frame_fingerprint, load_bundle, normalise_frame

DEFAULT_CONFIG_DIR = Path("configs/datasets")
ROW_ID = "row_id"

_BUNDLE_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}


class DatasetConfigError(ValueError):
    pass


def available_datasets(config_dir=DEFAULT_CONFIG_DIR) -> List[str]:
    return sorted(p.stem for p in Path(config_dir).glob("*.yaml"))


def load_dataset_config(name: str, config_dir=DEFAULT_CONFIG_DIR) -> dict:
    path = Path(config_dir) / f"{name}.yaml"
    if not path.exists():
        raise DatasetConfigError(f"no dataset config {path}; available: {available_datasets(config_dir)}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg.get("name") != name:
        raise DatasetConfigError(f"{path}: 'name' must equal the file stem")
    return cfg


def schema_from_config(cfg: dict) -> DatasetSchema:
    columns = tuple(
        ColumnSpec(name=str(c["name"]), type=c["type"], role=c.get("role", "feature"),
                   ordered_values=None if c.get("ordered_values") is None else tuple(str(v) for v in c["ordered_values"]))
        for c in cfg["columns"]
    )
    schema = DatasetSchema(name=cfg["name"], columns=columns, task=cfg.get("task"),
                           missing_policy=cfg.get("missing_policy", "reject"),
                           dropped_columns=tuple(cfg.get("dropped_columns", ())), notes=cfg.get("notes", ""))
    if schema.target != cfg.get("target"):
        raise DatasetConfigError("config 'target' does not match the column with role=target")
    return schema


def _source_frame(cfg: dict) -> pd.DataFrame:
    src = cfg["source"]
    if "bundle" in src:
        bundle = src["bundle"]
        if bundle not in _BUNDLE_CACHE:
            _BUNDLE_CACHE[bundle] = load_bundle(BUNDLE_DIR / bundle)
        try:
            return _BUNDLE_CACHE[bundle][src["key"]]
        except KeyError as e:
            raise DatasetConfigError(f"bundle {bundle} has no dataset {src['key']!r}") from e
    if "csv" in src:
        return pd.read_csv(src["csv"])
    if "parquet" in src:
        return pd.read_parquet(src["parquet"])
    raise DatasetConfigError("source must declare one of: bundle, csv, parquet")


def prepare_frame(raw: pd.DataFrame, cfg: dict, schema: DatasetSchema) -> pd.DataFrame:
    """Schema-driven clean-up of a raw table; returns a frame indexed by row_id."""
    f = normalise_frame(raw)
    dropped = [c for c in schema.dropped_columns]
    missing_drop = [c for c in dropped if c not in f.columns]
    if missing_drop:
        raise DatasetConfigError(f"dropped_columns not present in the source: {missing_drop}")
    f = f.drop(columns=dropped)

    declared = schema.column_order
    undeclared = [c for c in f.columns if c not in declared]
    absent = [c for c in declared if c not in f.columns]
    if undeclared or absent:
        raise DatasetConfigError(f"schema/table mismatch: undeclared columns {undeclared}, absent columns {absent}")
    f = f[declared]

    for col, mapping in (cfg.get("value_maps") or {}).items():
        mapping = {str(k): str(v) for k, v in mapping.items()}
        f[col] = f[col].map(lambda v, m=mapping: v if v is None else m.get(str(v), v))

    for spec in schema.columns:
        s = f[spec.name]
        if spec.type == CATEGORICAL:
            # labels are strings; integer-stored nominal codes become "1", "2", ...
            if pd.api.types.is_numeric_dtype(s.dtype):
                def _label(v):
                    if pd.isna(v):
                        return None
                    return str(int(v)) if float(v).is_integer() else str(v)
                f[spec.name] = s.map(_label).astype(object)
            if spec.ordered_values is not None:
                unknown = sorted(set(f[spec.name].dropna().unique()) - set(spec.ordered_values))
                if unknown:
                    raise DatasetConfigError(f"column {spec.name!r}: values {unknown} missing from ordered_values")
        else:
            if not pd.api.types.is_numeric_dtype(s.dtype):
                raise DatasetConfigError(f"column {spec.name!r} is declared {spec.type} but is not numeric")
            f[spec.name] = s.astype(np.float64)

    f.index = pd.RangeIndex(len(f), name=ROW_ID)
    return f


def load_dataset(name: str, config_dir=DEFAULT_CONFIG_DIR) -> Tuple[pd.DataFrame, DatasetSchema, dict]:
    cfg = load_dataset_config(name, config_dir)
    schema = schema_from_config(cfg)
    frame = prepare_frame(_source_frame(cfg), cfg, schema)
    manifest = {
        "dataset": name,
        "source": cfg["source"],
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "fingerprint": frame_fingerprint(frame),
        "schema_hash": schema.hash(),
        "regime": schema.regime,
        "task": schema.task,
        "target": schema.target,
        "dropped_columns": list(schema.dropped_columns),
        "value_maps": cfg.get("value_maps") or {},
        "missing_policy": schema.missing_policy,
        "missing_counts": {c: int(n) for c, n in frame.isna().sum().items() if n},
        "row_filtering": "none",
    }
    return frame, schema, manifest
