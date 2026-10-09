"""
Model adapter contract used by every experiment stage.

    fit(train, schema, config, seed)   train: common representation of the CURRENT fit rows
    sample(n, seed)                    exactly n rows, common representation, schema column order
    save_checkpoint(path)              directory; inference-complete
    load_checkpoint(path)              never refits

The runner owns data splitting. An adapter sees only the rows it is given; it
must not split them, load a dataset itself, or add real rows to its output.
Configs are STRICT: a key the adapter does not consume raises, so a sampled
hyperparameter can never be silently ignored.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sbtab.data.dataset_schema import DatasetSchema
from sbtab.data.preprocessing import row_id_hash

ADAPTER_FORMAT = "sbtab.adapter/1"


class AdapterConfigError(ValueError):
    pass


class UnsupportedRegimeError(ValueError):
    pass


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


@contextmanager
def _measure(timing: dict, key: str):
    """Retain active time even when an operation raises or is interrupted."""
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - start


class ModelAdapter(ABC):
    registry_id: ClassVar[str]
    supported_regimes: ClassVar[Tuple[str, ...]] = ("continuous", "discrete", "mixed")
    # regimes the algorithm models natively; the rest go through a declared representation
    native_regimes: ClassVar[Tuple[str, ...]] = ()
    # config keys -> default. `None` default = required.
    DEFAULTS: ClassVar[Dict[str, Any]] = {}
    checkpoint_kind: ClassVar[str] = "inference"   # "inference" | "resumable"

    def __init__(self):
        self.schema: Optional[DatasetSchema] = None
        self.config: Dict[str, Any] = {}
        self.seed: Optional[int] = None
        self.fit_row_hash: Optional[str] = None
        self.n_fit_rows = 0
        self.fitted = False
        self.decoding_report_: dict = {}
        self.last_sample_timing: dict = {}
        self.last_fit_timing: dict = {}
        self._encoded = None

    # ------------------------------------------------------------------ config
    @classmethod
    def resolve_config(cls, config: Optional[dict]) -> Dict[str, Any]:
        config = dict(config or {})
        unknown = sorted(set(config) - set(cls.DEFAULTS))
        if unknown:
            raise AdapterConfigError(f"{cls.registry_id}: config keys {unknown} are not consumed by this adapter; "
                                     f"accepted keys: {sorted(cls.DEFAULTS)}")
        eff = {**cls.DEFAULTS, **config}
        missing = [k for k, v in eff.items() if v is None and k in cls.REQUIRED]
        if missing:
            raise AdapterConfigError(f"{cls.registry_id}: missing required config keys {missing}")
        return eff

    REQUIRED: ClassVar[Tuple[str, ...]] = ()

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _prepare(self, train: pd.DataFrame) -> None:
        """Fit the model-specific representation and encode the fit rows (counted as fitting)."""

    @abstractmethod
    def _build_model(self) -> None:
        """Construct the model / move it to its device (counted as initialisation)."""

    @abstractmethod
    def _fit_model(self) -> None:
        """Train. Must not keep the training rows on the adapter afterwards."""

    @abstractmethod
    def _generate(self, n: int, seed: int):
        """Sample n rows in the MODEL representation."""

    @abstractmethod
    def _decode(self, generated) -> pd.DataFrame:
        """Model representation -> common schema (sets self.decoding_report_)."""

    @abstractmethod
    def _save_model(self, directory: Path) -> None: ...

    @abstractmethod
    def _load_model(self, directory: Path, state: dict) -> None: ...

    def _state(self) -> dict:
        """Adapter-specific JSON state (representation etc.)."""
        return {}

    def describe(self) -> dict:
        """Orientation, reference, grid, stages, update counts — filled by subclasses."""
        return {}

    @property
    def n_updates(self) -> Optional[int]:
        return None

    # ------------------------------------------------------------------ public API
    def fit(self, train: pd.DataFrame, schema: DatasetSchema, config: Optional[dict], seed: int) -> "ModelAdapter":
        self.fitted = False
        self._encoded = None
        self.last_fit_timing = {}
        if schema.regime not in self.supported_regimes:
            raise UnsupportedRegimeError(f"{self.registry_id} does not support the {schema.regime!r} regime")
        if list(train.columns) != schema.column_order:
            raise ValueError("training frame columns must equal the schema column order")
        if len(train) == 0:
            raise ValueError("cannot fit on zero rows")
        if not np.isfinite(train.to_numpy(dtype=np.float64)).all():
            raise ValueError("training frame (common representation) contains non-finite values")
        self.schema = schema
        self.config = self.resolve_config(config)
        self.seed = int(seed)
        self.fit_row_hash = row_id_hash(train.index)
        self.n_fit_rows = int(len(train))
        try:
            with _measure(self.last_fit_timing, "generator_fit_seconds"):
                self._prepare(train)
            with _measure(self.last_fit_timing, "model_init_seconds"):
                self._build_model()
            with _measure(self.last_fit_timing, "generator_fit_seconds"):
                self._fit_model()
        finally:
            self._encoded = None        # also release training rows after a failed fit
        self.fitted = True
        return self

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        self.last_sample_timing = {}
        if not self.fitted:
            raise RuntimeError("fit() or load_checkpoint() first")
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        with _measure(self.last_sample_timing, "generation_seconds"):
            generated = self._generate(n, int(seed))
        with _measure(self.last_sample_timing, "inverse_transform_seconds"):
            out = self._decode(generated)
        if list(out.columns) != self.schema.column_order:
            raise RuntimeError(f"{self.registry_id}: generated columns do not match the schema")
        if len(out) != n:
            raise RuntimeError(f"{self.registry_id}: generated {len(out)} rows, {n} were requested")
        out = out.reset_index(drop=True)
        out.index.name = "synthetic_id"     # generated ids; never joinable to real row ids
        return out

    def save_checkpoint(self, path) -> Path:
        if not self.fitted:
            raise RuntimeError("nothing to save: the adapter is not fitted")
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        self._save_model(d)
        meta = {
            "format": ADAPTER_FORMAT, "registry_id": self.registry_id, "adapter_class": type(self).__name__,
            "checkpoint_kind": self.checkpoint_kind,
            "schema": self.schema.to_dict(), "effective_config": self.config, "seed": self.seed,
            "fit_row_hash": self.fit_row_hash, "n_fit_rows": self.n_fit_rows, "fit_status": "fitted",
            "n_updates": self.n_updates, "describe": self.describe(), "state": self._state(),
        }
        tmp = d / "adapter.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1, default=_json_default), encoding="utf-8")
        tmp.replace(d / "adapter.json")
        return d

    @classmethod
    def load_checkpoint(cls, path) -> "ModelAdapter":
        d = Path(path)
        meta = json.loads((d / "adapter.json").read_text(encoding="utf-8"))
        if meta.get("format") != ADAPTER_FORMAT:
            raise ValueError(f"unsupported adapter checkpoint format {meta.get('format')!r}")
        if meta["registry_id"] != cls.registry_id:
            raise ValueError(f"checkpoint belongs to {meta['registry_id']!r}, not {cls.registry_id!r}")
        a = cls()
        a.schema = DatasetSchema.from_dict(meta["schema"])
        a.config = meta["effective_config"]
        a.seed = meta["seed"]
        a.fit_row_hash, a.n_fit_rows = meta["fit_row_hash"], meta["n_fit_rows"]
        a._load_model(d, meta["state"])
        a.fitted = True
        return a
