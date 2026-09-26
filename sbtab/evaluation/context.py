"""
``MetricContext``: everything a metric LEARNS, learned from the training rows only
and frozen for the real held-out rows and for every generated table of that fold.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._common import json_safe, require_clean_real, sorted_labels
from .conditional import fit_conditioning
from .marginal import fit_histograms
from .mmd import fit_mmd_metadata
from .spec import KL_DIRECTION, LOG_BASE, METRIC_VERSION, MetricConfig


def _row_hash(row_ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(v) for v in row_ids.tolist()).encode()).hexdigest()


class MetricContext:
    def __init__(self, schema, config: MetricConfig, hist: Dict[str, Dict[str, Any]],
                 support: Dict[str, List[Any]], conditioning: Dict[str, Any], mmd: Dict[str, Any],
                 n_train: int, fit_row_hash: str, row_id_source: str):
        self.schema = schema
        self.config = config
        self.hist = hist                    # continuous column -> {"edges", "constant", "train_min", "train_max"}
        self.support = support              # finite column -> sorted training labels
        self.conditioning = conditioning    # {"conditioners": [...], "excluded": [...], ...}
        self.mmd = mmd                      # {"full": {...}, "features": {...}}
        self.n_train = int(n_train)
        self.fit_row_hash = fit_row_hash
        self.row_id_source = row_id_source
        self.metric_version = METRIC_VERSION
        self._spec_hash: Optional[str] = None

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, train: pd.DataFrame, schema, config: MetricConfig = MetricConfig(), train_row_ids=None
            ) -> "MetricContext":
        require_clean_real(train, schema, "train")
        if train_row_ids is None:
            row_ids, source = np.arange(len(train), dtype=np.int64), "row_position"
        else:
            row_ids, source = np.asarray(train_row_ids), "caller"
            if row_ids.shape != (len(train),):
                raise ValueError("train_row_ids must have one id per training row")
        support = {c: sorted_labels(train[c]) for c in schema.finite_support}
        return cls(schema=schema, config=config,
                   hist=fit_histograms(train, schema, config),
                   support=support,
                   conditioning=fit_conditioning(train, schema, config, support),
                   mmd=fit_mmd_metadata(train, schema, config, row_ids),
                   n_train=len(train), fit_row_hash=_row_hash(row_ids), row_id_source=source)

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "metric_version": self.metric_version,
            "schema_hash": self.schema.hash(),
            "schema_name": self.schema.name,
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "n_train": self.n_train,
            "fit_row_hash": self.fit_row_hash,
            "fit_row_hash_definition": "sha256 of the newline-joined str() of the training row ids, in row order",
            "row_id_source": self.row_id_source,
            "kl": {"direction": KL_DIRECTION, "total_bins": int(self.config.kl_total_bins),
                   "interior_bins": int(self.config.kl_total_bins) - 2,
                   "smoothing_mass": float(self.config.kl_smoothing_mass), "log_base": LOG_BASE,
                   "finite_columns": "training support + one unexpected-value bin"},
            "histograms": {c: {"edges": [float(e) for e in h["edges"]], "constant": bool(h["constant"]),
                               "train_min": float(h["train_min"]), "train_max": float(h["train_max"])}
                           for c, h in self.hist.items()},
            "support": {c: list(v) for c, v in self.support.items()},
            "conditioning": self.conditioning,
            "mmd": self.mmd,
        }
        return json_safe(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], schema) -> "MetricContext":
        if d.get("metric_version") != METRIC_VERSION:
            raise ValueError(f"unsupported metric version {d.get('metric_version')!r}")
        if d.get("schema_hash") != schema.hash():
            raise ValueError("the saved metric context belongs to a different schema")
        hist = {c: {"edges": np.asarray(h["edges"], dtype=np.float64), "constant": bool(h["constant"]),
                    "train_min": float(h["train_min"]), "train_max": float(h["train_max"])}
                for c, h in d["histograms"].items()}
        return cls(schema=schema, config=MetricConfig.from_dict(d["config"]), hist=hist,
                   support={c: list(v) for c, v in d["support"].items()},
                   conditioning=copy.deepcopy(d["conditioning"]), mmd=copy.deepcopy(d["mmd"]),
                   n_train=d["n_train"], fit_row_hash=d["fit_row_hash"],
                   row_id_source=d.get("row_id_source", "caller"))

    def spec_hash(self) -> str:
        if self._spec_hash is None:
            payload = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
            self._spec_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self._spec_hash
