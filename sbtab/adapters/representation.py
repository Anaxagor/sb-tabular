"""
Reversible, model-specific representations built ON TOP of the common benchmark
representation (continuous standardised, discrete original values, categorical
label codes). They are fitted on the same training rows as the model, serialised
with it, and always decode back to the common schema, where every metric lives.

Nothing here clips continuous outputs to the training range. Decoding steps that
can alter a generated value are declared and measured:
  * one-hot block  -> argmax (a representation decoder)
  * discrete value represented continuously -> nearest TRAINING support value;
    the pre-decoding invalidity rate and the mean rounding distance are recorded.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sbtab.data.dataset_schema import CATEGORICAL, CONTINUOUS, DISCRETE, DatasetSchema


def _nearest_support(values: np.ndarray, support: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Nearest-support decoding with its diagnostics (values must be finite)."""
    support = np.asarray(support, dtype=np.float64)
    idx = np.clip(np.searchsorted(support, values), 1, len(support) - 1) if len(support) > 1 else np.zeros(len(values), int)
    if len(support) > 1:
        left, right = support[idx - 1], support[idx]
        idx = np.where(np.abs(values - left) <= np.abs(right - values), idx - 1, idx)
    decoded = support[idx]
    dist = np.abs(values - decoded)
    report = {
        "n": int(len(values)),
        "pre_decoding_invalid_rate": float(np.mean(dist > 1e-9)) if len(values) else None,
        "mean_rounding_distance": float(dist.mean()) if len(values) else None,
        "max_rounding_distance": float(dist.max()) if len(values) else None,
        "out_of_range_rate": float(np.mean((values < support[0]) | (values > support[-1]))) if len(values) else None,
    }
    return decoded, report


class ContinuousRepresentation:
    """
    Everything as one real vector, for continuous-only generators.

      continuous   unchanged (already standardised)
      discrete     standardised with the training mean/std (zero std -> 1)
      categorical  one-hot over the training vocabulary (includes a classification target)
    """
    kind = "continuous_onehot"

    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.columns: List[str] = []
        self.blocks: List[dict] = []
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> "ContinuousRepresentation":
        self.blocks, self.columns = [], []
        for spec in self.schema.columns:
            s = train[spec.name]
            if spec.type == CONTINUOUS:
                b = {"name": spec.name, "type": CONTINUOUS, "out": [spec.name]}
            elif spec.type == DISCRETE:
                v = s.to_numpy(dtype=np.float64)
                std = float(v.std(ddof=0))
                b = {"name": spec.name, "type": DISCRETE, "out": [spec.name], "mean": float(v.mean()),
                     "scale": std if std > 0 else 1.0, "support": sorted(float(x) for x in np.unique(v))}
            else:
                codes = sorted(int(c) for c in np.unique(s.to_numpy(dtype=np.int64)))
                b = {"name": spec.name, "type": CATEGORICAL, "codes": codes,
                     "out": [f"{spec.name}__oh{c}" for c in codes]}
            self.blocks.append(b)
            self.columns += b["out"]
        self.fitted = True
        return self

    @property
    def dim(self) -> int:
        return len(self.columns)

    def encode(self, common: pd.DataFrame) -> pd.DataFrame:
        parts = {}
        for b in self.blocks:
            s = common[b["name"]]
            if b["type"] == CONTINUOUS:
                parts[b["out"][0]] = s.to_numpy(dtype=np.float64)
            elif b["type"] == DISCRETE:
                parts[b["out"][0]] = (s.to_numpy(dtype=np.float64) - b["mean"]) / b["scale"]
            else:
                codes = s.to_numpy(dtype=np.int64)
                unknown = set(np.unique(codes).tolist()) - set(b["codes"])
                if unknown:
                    raise ValueError(f"column {b['name']!r}: codes {sorted(unknown)} absent from the fitted one-hot block")
                for c, out in zip(b["codes"], b["out"]):
                    parts[out] = (codes == c).astype(np.float64)
        return pd.DataFrame(parts, index=common.index)[self.columns].astype(np.float32)

    def decode(self, x) -> Tuple[pd.DataFrame, dict]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected an array of shape (n, {self.dim})")
        pos = {c: i for i, c in enumerate(self.columns)}
        out, report = {}, {"nonfinite_rate": float(np.mean(~np.isfinite(x))) if x.size else 0.0, "discrete": {}, "one_hot": {}}
        for b in self.blocks:
            cols = [pos[c] for c in b["out"]]
            if b["type"] == CONTINUOUS:
                out[b["name"]] = x[:, cols[0]]
            elif b["type"] == DISCRETE:
                raw = x[:, cols[0]] * b["scale"] + b["mean"]
                finite = np.isfinite(raw)
                decoded = np.full(len(raw), np.nan)
                dec, rep = _nearest_support(raw[finite], np.asarray(b["support"]))
                decoded[finite] = dec           # non-finite values stay NaN and are reported as invalid downstream
                out[b["name"]] = decoded
                report["discrete"][b["name"]] = rep
            else:
                block = x[:, cols]
                finite = np.isfinite(block).all(axis=1)
                codes = np.full(len(block), -1, dtype=np.int64)  # -1: undecodable row, reported as invalid downstream
                if finite.any():
                    codes[finite] = np.asarray(b["codes"], dtype=np.int64)[np.argmax(block[finite], axis=1)]
                out[b["name"]] = codes
                report["one_hot"][b["name"]] = {"undecodable_rate": float(np.mean(~finite)) if len(block) else 0.0}
        return pd.DataFrame(out)[self.schema.column_order], report

    def state(self) -> dict:
        return {"kind": self.kind, "columns": self.columns, "blocks": self.blocks}

    @classmethod
    def from_state(cls, state: dict, schema: DatasetSchema) -> "ContinuousRepresentation":
        r = cls(schema)
        r.columns, r.blocks, r.fitted = list(state["columns"]), list(state["blocks"]), True
        return r


class NativeMixedRepresentation:
    """
    Numerical block + finite-state block, for native mixed / categorical solvers.

      continuous   -> numerical block, unchanged
      discrete     -> contiguous state index over the SORTED training support
                      (a bijection, mapped back exactly); ordered = True
      categorical  -> contiguous state index over the training codes; ordered only
                      for a declared ordinal column. Nominal codes never imply order.
    """
    kind = "native_mixed"

    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.states: Dict[str, List[float]] = {}
        self.is_ordered: List[bool] = []
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> "NativeMixedRepresentation":
        self.num_cols = list(self.schema.continuous)
        self.cat_cols = [c.name for c in self.schema.columns if c.type in (DISCRETE, CATEGORICAL)]
        self.states, self.is_ordered = {}, []
        for name in self.cat_cols:
            self.states[name] = sorted(float(v) for v in np.unique(train[name].to_numpy(dtype=np.float64)))
            self.is_ordered.append(bool(self.schema.spec(name).is_ordered))
        self.fitted = True
        return self

    @property
    def cardinalities(self) -> List[int]:
        return [len(self.states[c]) for c in self.cat_cols]

    def encode(self, common: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        num = common[self.num_cols].to_numpy(dtype=np.float32) if self.num_cols else np.zeros((len(common), 0), np.float32)
        cat = np.zeros((len(common), len(self.cat_cols)), dtype=np.int64)
        for j, name in enumerate(self.cat_cols):
            states = np.asarray(self.states[name])
            v = common[name].to_numpy(dtype=np.float64)
            idx = np.searchsorted(states, v)
            idx = np.clip(idx, 0, len(states) - 1)
            if not np.array_equal(states[idx], v):
                raise ValueError(f"column {name!r}: values outside the fitted state space")
            cat[:, j] = idx
        return num, cat

    def decode(self, num, cat) -> Tuple[pd.DataFrame, dict]:
        num = np.asarray(num, dtype=np.float64).reshape(len(cat) if len(self.cat_cols) else -1, len(self.num_cols))
        cat = np.asarray(cat).reshape(len(num), len(self.cat_cols))
        out = {}
        for j, name in enumerate(self.num_cols):
            out[name] = num[:, j]
        invalid = {}
        for j, name in enumerate(self.cat_cols):
            states = np.asarray(self.states[name])
            raw = np.asarray(cat[:, j], dtype=np.float64)
            valid = np.isfinite(raw) & (raw == np.floor(raw)) & (raw >= 0) & (raw < len(states))
            invalid[name] = float((~valid).mean()) if len(raw) else 0.0
            vals = np.full(len(raw), np.nan)
            vals[valid] = states[raw[valid].astype(np.int64)]
            out[name] = vals
        df = pd.DataFrame(out)[self.schema.column_order]
        for name in self.schema.categorical:
            if not df[name].isna().any():
                df[name] = df[name].astype(np.int64)
        return df, {"nonfinite_rate": float(np.mean(~np.isfinite(num))) if num.size else 0.0,
                    "invalid_state_rate": invalid, "decoding": "exact bijection (no rounding)"}

    def state(self) -> dict:
        return {"kind": self.kind, "num_cols": self.num_cols, "cat_cols": self.cat_cols,
                "states": self.states, "is_ordered": self.is_ordered}

    @classmethod
    def from_state(cls, state: dict, schema: DatasetSchema) -> "NativeMixedRepresentation":
        r = cls(schema)
        r.num_cols, r.cat_cols = list(state["num_cols"]), list(state["cat_cols"])
        r.states = {k: list(v) for k, v in state["states"].items()}
        r.is_ordered, r.fitted = list(state["is_ordered"]), True
        return r
