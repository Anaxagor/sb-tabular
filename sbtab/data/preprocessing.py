"""
Common benchmark preprocessing, fitted on the CURRENT TRAINING ROWS ONLY.

  continuous    StandardScaler (mean, population std). Zero training variance ->
                scale 1 and a constant flag. WD / continuous KL live in this space.
  discrete      untouched: original numeric values and ordering.
  categorical   one LabelEncoder-equivalent vocabulary per column. Codes are
                nominal labels. For a declared ordinal column the vocabulary
                follows the declared order so that code order is meaningful.

A value outside a training vocabulary is an error; the vocabulary is never
expanded by held-out or generated data. Missing values are handled only under an
explicit ``missing_policy: impute`` (train median / train lower median for
discrete / a ``__missing__`` category); every imputation is counted.

This is the representation every generator receives and every metric reads.
Model-specific representations (one-hot, state indices, ...) are built on top of
it by the model adapters and are always mapped back to it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from sbtab.data.dataset_schema import CATEGORICAL, CONTINUOUS, DISCRETE, MISSING_TOKEN, DatasetSchema

PREPROCESSOR_VERSION = "sbtab.preprocessor/1"


class UnseenCategoryError(ValueError):
    pass


class MissingValueError(ValueError):
    pass


def row_id_hash(row_ids: Iterable[int]) -> str:
    ids = np.sort(np.asarray(list(row_ids), dtype=np.int64))
    return hashlib.sha256(ids.tobytes()).hexdigest()


class CommonPreprocessor:
    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.fitted = False
        self.means: Dict[str, float] = {}
        self.scales: Dict[str, float] = {}
        self.constant: Dict[str, bool] = {}
        self.vocab: Dict[str, List[str]] = {}
        self.support: Dict[str, List[float]] = {}
        self.impute_values: Dict[str, float] = {}
        self.fit_row_hash: Optional[str] = None
        self.n_fit_rows = 0
        self.fit_imputed_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "CommonPreprocessor":
        self.fitted = False
        schema = self.schema
        if list(train.columns) != schema.column_order:
            raise ValueError("training frame columns must equal the schema column order")
        if len(train) == 0:
            raise ValueError("cannot fit a preprocessor on zero rows")
        self.fit_row_hash = row_id_hash(train.index)
        self.n_fit_rows = int(len(train))

        n_missing = train.isna().sum()
        if schema.missing_policy == "reject" and int(n_missing.sum()) > 0:
            raise MissingValueError(
                f"missing values in {dict(n_missing[n_missing > 0])} but missing_policy is 'reject'; "
                f"declare an explicit policy in the dataset config")

        for spec in schema.columns:
            s = train[spec.name]
            if spec.type in (CONTINUOUS, DISCRETE):
                obs = s.dropna().to_numpy(dtype=np.float64)
                if obs.size == 0:
                    raise MissingValueError(f"column {spec.name!r} has no observed training value to impute from")
                if not np.isfinite(obs).all():
                    raise ValueError(f"column {spec.name!r}: training values must be finite")
                if spec.type == CONTINUOUS:
                    self.impute_values[spec.name] = float(np.median(obs))
                else:
                    # lower median: the imputed value is a member of the training support
                    self.impute_values[spec.name] = float(np.sort(obs)[(obs.size - 1) // 2])
        filled, self.fit_imputed_counts = self._impute(train)

        for spec in schema.columns:
            s = filled[spec.name]
            if spec.type == CONTINUOUS:
                v = s.to_numpy(dtype=np.float64)
                mean, std = float(v.mean()), float(v.std(ddof=0))
                const = not std > 0
                self.means[spec.name] = mean
                self.scales[spec.name] = 1.0 if const else std
                self.constant[spec.name] = bool(const)
            elif spec.type == DISCRETE:
                self.support[spec.name] = sorted(float(x) for x in pd.unique(s.to_numpy(dtype=np.float64)))
                self.constant[spec.name] = len(self.support[spec.name]) == 1
            else:
                seen = set(str(x) for x in s.tolist())
                if spec.ordered_values is not None:
                    vocab = [v for v in spec.ordered_values if v in seen]
                    if MISSING_TOKEN in seen:
                        vocab.append(MISSING_TOKEN)
                else:
                    vocab = sorted(seen)
                self.vocab[spec.name] = vocab
                self.constant[spec.name] = len(vocab) == 1
        self.fitted = True
        return self

    # ------------------------------------------------------------------ helpers
    def _impute(self, df: pd.DataFrame):
        counts: Dict[str, int] = {}
        if not df.isna().any().any():
            return df, counts
        if self.schema.missing_policy != "impute":
            bad = df.isna().sum()
            raise MissingValueError(f"missing values in {dict(bad[bad > 0])} but missing_policy is 'reject'")
        out = df.copy()
        for spec in self.schema.columns:
            mask = out[spec.name].isna()
            n = int(mask.sum())
            if not n:
                continue
            counts[spec.name] = n
            if spec.type == CATEGORICAL:
                out.loc[mask, spec.name] = MISSING_TOKEN
            else:
                out.loc[mask, spec.name] = self.impute_values[spec.name]
        return out, counts

    def _check(self):
        if not self.fitted:
            raise RuntimeError("preprocessor is not fitted")

    # ------------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, return_imputed_counts: bool = False):
        """Raw table -> common representation (index preserved)."""
        self._check()
        if list(df.columns) != self.schema.column_order:
            raise ValueError("frame columns must equal the schema column order")
        filled, counts = self._impute(df)
        out = pd.DataFrame(index=df.index)
        for spec in self.schema.columns:
            s = filled[spec.name]
            if spec.type == CONTINUOUS:
                out[spec.name] = (s.to_numpy(dtype=np.float64) - self.means[spec.name]) / self.scales[spec.name]
            elif spec.type == DISCRETE:
                out[spec.name] = s.to_numpy(dtype=np.float64)
            else:
                index = {v: i for i, v in enumerate(self.vocab[spec.name])}
                labels = [str(x) for x in s.tolist()]
                unseen = sorted(set(labels) - set(index))
                if unseen:
                    raise UnseenCategoryError(
                        f"column {spec.name!r}: values {unseen[:10]} are not in the training vocabulary; "
                        f"the vocabulary is never expanded by held-out data")
                out[spec.name] = np.fromiter((index[x] for x in labels), dtype=np.int64, count=len(labels))
        return (out, counts) if return_imputed_counts else out

    def inverse_transform(self, common: pd.DataFrame) -> pd.DataFrame:
        """Common representation -> raw units / labels, exact column order."""
        self._check()
        out = pd.DataFrame(index=common.index)
        for spec in self.schema.columns:
            s = common[spec.name]
            if spec.type == CONTINUOUS:
                out[spec.name] = s.to_numpy(dtype=np.float64) * self.scales[spec.name] + self.means[spec.name]
            elif spec.type == DISCRETE:
                out[spec.name] = s.to_numpy(dtype=np.float64)
            else:
                vocab = self.vocab[spec.name]
                codes = s.to_numpy()
                if not np.all(np.isfinite(codes.astype(np.float64))):
                    raise ValueError(f"column {spec.name!r}: non-finite category codes")
                numeric = codes.astype(np.float64)
                if not np.equal(numeric, np.floor(numeric)).all():
                    raise ValueError(f"column {spec.name!r}: category codes must be integers")
                if numeric.size and (numeric.min() < 0 or numeric.max() >= len(vocab)):
                    raise ValueError(f"column {spec.name!r}: category codes outside [0, {len(vocab)})")
                codes = numeric.astype(np.int64)
                if codes.size and (codes.min() < 0 or codes.max() >= len(vocab)):
                    raise ValueError(f"column {spec.name!r}: category codes outside [0, {len(vocab)})")
                out[spec.name] = np.asarray(vocab, dtype=object)[codes]
        return out[self.schema.column_order]

    def cardinalities(self) -> Dict[str, int]:
        return {c: len(v) for c, v in self.vocab.items()}

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        self._check()
        return {
            "version": PREPROCESSOR_VERSION,
            "schema_hash": self.schema.hash(),
            "fit_row_hash": self.fit_row_hash,
            "n_fit_rows": self.n_fit_rows,
            "means": self.means, "scales": self.scales, "constant": self.constant,
            "vocab": self.vocab, "support": self.support,
            "impute_values": self.impute_values,
            "fit_imputed_counts": self.fit_imputed_counts,
            "missing_policy": self.schema.missing_policy,
        }

    @classmethod
    def from_dict(cls, d: dict, schema: DatasetSchema) -> "CommonPreprocessor":
        if d.get("version") != PREPROCESSOR_VERSION:
            raise ValueError(f"unsupported preprocessor version {d.get('version')!r}")
        if d["schema_hash"] != schema.hash():
            raise ValueError("preprocessor was fitted under a different schema")
        p = cls(schema)
        p.means, p.scales = dict(d["means"]), dict(d["scales"])
        p.constant = dict(d["constant"])
        p.vocab = {k: list(v) for k, v in d["vocab"].items()}
        p.support = {k: list(v) for k, v in d["support"].items()}
        p.impute_values = dict(d["impute_values"])
        p.fit_imputed_counts = dict(d.get("fit_imputed_counts", {}))
        p.fit_row_hash, p.n_fit_rows = d["fit_row_hash"], int(d["n_fit_rows"])
        p.fitted = True
        return p

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "preprocessor.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, directory, schema: DatasetSchema) -> "CommonPreprocessor":
        return cls.from_dict(json.loads((Path(directory) / "preprocessor.json").read_text(encoding="utf-8")), schema)
