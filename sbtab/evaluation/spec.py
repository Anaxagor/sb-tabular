"""
Versioned constants of the canonical metric package.

Every number that influences a reported metric lives in ``MetricConfig`` and is
hashed into the saved ``MetricContext``; nothing is tuned per model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple

METRIC_VERSION: str = "sbtab.metrics/2"

STATUSES: Tuple[str, ...] = (
    "ok",
    "not_applicable",
    "insufficient_data",
    "incomplete_conditional_coverage",
    "undefined",
    "invalid_generated_data",
    "training_failed",
    "sampling_failed",
    "utility_fit_failed",
    "blocked_support",
)

OK = "ok"
NOT_APPLICABLE = "not_applicable"
INSUFFICIENT_DATA = "insufficient_data"
INCOMPLETE_COVERAGE = "incomplete_conditional_coverage"
UNDEFINED = "undefined"
INVALID_GENERATED = "invalid_generated_data"
UTILITY_FIT_FAILED = "utility_fit_failed"

KL_DIRECTION = "KL(real||synthetic)"
LOG_BASE = "e"


@dataclass(frozen=True)
class MetricConfig:
    # --- marginal KL ---------------------------------------------------------
    kl_total_bins: int = 50              # interior = total - 2 (one underflow + one overflow bin)
    kl_smoothing_mass: float = 1e-6      # total mass, spread uniformly over the B bins
    # A generated NOMINAL label outside the training vocabulary is an invalid native
    # category id (run-level failure). Out-of-support values of an ordered DISCRETE
    # column are legal numbers: they land in the diagnostic unexpected-value bin.
    categorical_unexpected_is_invalid: bool = True
    # --- conditional distributions -------------------------------------------
    conditional_max_levels: int = 50
    conditional_min_rows: int = 10
    regression_target_strata: int = 5
    # --- mixed-space MMD -----------------------------------------------------
    mmd_bandwidth_max_rows: int = 1024
    mmd_max_rows: int = 2048
    mmd_seeds: Tuple[int, ...] = (0, 1, 2)
    mmd_block_size: int = 1024
    # --- TSTR utility --------------------------------------------------------
    utility_seed: int = 0
    utility_thread_count: int = 4

    def __post_init__(self):
        object.__setattr__(self, "mmd_seeds", tuple(int(s) for s in self.mmd_seeds))
        if self.kl_total_bins < 3:
            raise ValueError("kl_total_bins must leave at least one interior bin")
        if not (self.kl_smoothing_mass >= 0.0):
            raise ValueError("kl_smoothing_mass must be non-negative")
        if self.mmd_block_size < 1 or self.mmd_max_rows < 2 or self.mmd_bandwidth_max_rows < 2:
            raise ValueError("invalid MMD size configuration")
        if self.regression_target_strata < 1 or self.conditional_min_rows < 1:
            raise ValueError("invalid conditional configuration")

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_document(cls, d: Dict[str, Any]) -> "MetricConfig":
        """Validate the YAML document including metadata that controls metric meaning."""
        if d.get("metric_version") != METRIC_VERSION:
            raise ValueError(f"unsupported metric version {d.get('metric_version')!r}")
        if d.get("kl_direction", "real_to_synthetic") != "real_to_synthetic":
            raise ValueError("only kl_direction: real_to_synthetic is implemented")
        return cls.from_dict({k: v for k, v in d.items() if k not in ("metric_version", "kl_direction")})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown MetricConfig keys: {sorted(unknown)}")
        kw = dict(d)
        if "mmd_seeds" in kw:
            kw["mmd_seeds"] = tuple(kw["mmd_seeds"])
        return cls(**kw)

    def hash(self) -> str:
        payload = {"metric_version": METRIC_VERSION, "config": self.to_dict()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
