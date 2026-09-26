"""
Versioned, explicit dataset schema for the benchmark protocol.

The schema is benchmark METADATA. It is written down per dataset (see
``configs/datasets/*.yaml``) and frozen across models and folds; nothing here
infers a semantic type from a dtype or from the rows of a particular split.

Feature types
  continuous   real-valued; standardised by the common pipeline.
  discrete     ordered numeric with finite support (ordinal / small count). Keeps
               its original numeric values and ordering; never scaled or recoded.
  categorical  nominal labels. Integer storage does not make a column numeric:
               a classification target is categorical regardless of dtype.

The target is an ordinary column with role="target". It belongs to exactly one
type group, the one its ``type`` declares, so every per-group metric counts it
exactly once.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "sbtab.schema/1"

CONTINUOUS = "continuous"
DISCRETE = "discrete"
CATEGORICAL = "categorical"
FEATURE_TYPES = (CONTINUOUS, DISCRETE, CATEGORICAL)

MISSING_TOKEN = "__missing__"
MISSING_POLICIES = ("reject", "impute")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str
    role: str = "feature"                    # "feature" | "target"
    # Declared order of an ORDINAL categorical column, lowest first. A discrete
    # column is always ordered by its numeric value and must leave this empty.
    ordered_values: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.type not in FEATURE_TYPES:
            raise ValueError(f"column {self.name!r}: unknown type {self.type!r}")
        if self.role not in ("feature", "target"):
            raise ValueError(f"column {self.name!r}: unknown role {self.role!r}")
        if self.ordered_values is not None:
            if self.type != CATEGORICAL:
                raise ValueError(f"column {self.name!r}: ordered_values is only valid for categorical columns")
            object.__setattr__(self, "ordered_values", tuple(str(v) for v in self.ordered_values))

    @property
    def is_ordered(self) -> bool:
        """True when state order is meaningful (discrete, or ordinal categorical)."""
        return self.type == DISCRETE or self.ordered_values is not None


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    columns: Tuple[ColumnSpec, ...]
    task: Optional[str] = None               # "classification" | "regression" | None
    missing_policy: str = "reject"           # "reject" | "impute"
    dropped_columns: Tuple[str, ...] = ()    # identifiers etc. removed BEFORE splitting
    version: str = SCHEMA_VERSION
    notes: str = ""

    def __post_init__(self):
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "dropped_columns", tuple(self.dropped_columns))
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("duplicate column names in schema")
        if self.missing_policy not in MISSING_POLICIES:
            raise ValueError(f"unknown missing_policy {self.missing_policy!r}")
        targets = [c for c in self.columns if c.role == "target"]
        if len(targets) > 1:
            raise ValueError("at most one target column is supported")
        if self.task not in (None, "classification", "regression"):
            raise ValueError(f"unknown task {self.task!r}")
        if (self.task is None) != (len(targets) == 0):
            raise ValueError("task and target column must be declared together")
        if targets:
            t = targets[0]
            if self.task == "classification" and t.type != CATEGORICAL:
                raise ValueError("a classification target is nominal categorical regardless of integer storage")
            if self.task == "regression" and t.type == CATEGORICAL:
                raise ValueError("a regression target must be continuous or a declared discrete type")

    # ------------------------------------------------------------------ views
    @property
    def column_order(self) -> List[str]:
        return [c.name for c in self.columns]

    def _of(self, kind: str, include_target: bool = True) -> List[str]:
        return [c.name for c in self.columns if c.type == kind and (include_target or c.role != "target")]

    @property
    def continuous(self) -> List[str]:
        return self._of(CONTINUOUS)

    @property
    def discrete(self) -> List[str]:
        return self._of(DISCRETE)

    @property
    def categorical(self) -> List[str]:
        return self._of(CATEGORICAL)

    @property
    def finite_support(self) -> List[str]:
        """Columns whose support must be covered by the training rows."""
        return [c.name for c in self.columns if c.type in (DISCRETE, CATEGORICAL)]

    @property
    def target(self) -> Optional[str]:
        for c in self.columns:
            if c.role == "target":
                return c.name
        return None

    @property
    def features(self) -> List[str]:
        return [c.name for c in self.columns if c.role == "feature"]

    def spec(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def type_of(self, name: str) -> str:
        return self.spec(name).type

    @property
    def regime(self) -> str:
        """Regime of the schema ACTUALLY generated, target included."""
        has_cont = len(self.continuous) > 0
        has_fin = len(self.discrete) + len(self.categorical) > 0
        if has_cont and has_fin:
            return "mixed"
        return "continuous" if has_cont else "discrete"

    def without_target(self) -> "DatasetSchema":
        return DatasetSchema(name=self.name, columns=tuple(c for c in self.columns if c.role != "target"),
                             task=None, missing_policy=self.missing_policy,
                             dropped_columns=self.dropped_columns, version=self.version, notes=self.notes)

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "task": self.task,
            "target": self.target,
            "missing_policy": self.missing_policy,
            "dropped_columns": list(self.dropped_columns),
            "regime": self.regime,
            "columns": [
                {"name": c.name, "type": c.type, "role": c.role,
                 "ordered_values": None if c.ordered_values is None else list(c.ordered_values)}
                for c in self.columns
            ],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetSchema":
        if d.get("version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {d.get('version')!r}")
        cols = tuple(
            ColumnSpec(name=str(c["name"]), type=c["type"], role=c.get("role", "feature"),
                       ordered_values=None if c.get("ordered_values") is None else tuple(c["ordered_values"]))
            for c in d["columns"]
        )
        return cls(name=d["name"], columns=cols, task=d.get("task"),
                   missing_policy=d.get("missing_policy", "reject"),
                   dropped_columns=tuple(d.get("dropped_columns", ())), notes=d.get("notes", ""))

    def hash(self) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k != "notes"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
