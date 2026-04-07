from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .base import TransformState
from sbtab.data.schema import TabularSchema


@dataclass
class ContinuousStandardScaler:
    name: str = "continuous_standard_scaler"
    eps: float = 1e-12

    fitted_: bool = False
    continuous_cols_: List[str] = field(default_factory=list)
    means_: Dict[str, float] = field(default_factory=dict)
    stds_: Dict[str, float] = field(default_factory=dict)

    def requires_fit(self) -> bool:
        return True

    def is_invertible(self) -> bool:
        return True

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "ContinuousStandardScaler":
        cols = list(schema.continuous_cols)
        self.continuous_cols_ = cols

        if not cols:
            self.means_ = {}
            self.stds_ = {}
            self.fitted_ = True
            return self

        x = df[cols].astype(float)
        means = x.mean(axis=0, skipna=True)
        stds = x.std(axis=0, ddof=0, skipna=True)

        self.means_ = {col: float(means[col]) for col in cols}
        self.stds_ = {col: float(max(float(stds[col]), self.eps)) for col in cols}
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("ContinuousStandardScaler must be fitted before transform().")

        out = df.copy()
        for col in self.continuous_cols_:
            if col not in out.columns:
                raise ValueError(f"Continuous column {col!r} is missing during scaling.")
            out[col] = (out[col].astype(float) - self.means_[col]) / self.stds_[col]
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("ContinuousStandardScaler must be fitted before inverse_transform().")

        out = df.copy()
        for col in self.continuous_cols_:
            if col not in out.columns:
                raise ValueError(f"Continuous column {col!r} is missing during inverse scaling.")
            out[col] = out[col].astype(float) * self.stds_[col] + self.means_[col]
        return out

    def get_state(self) -> TransformState:
        return TransformState(
            name=self.name,
            params={
                "eps": self.eps,
                "fitted": self.fitted_,
                "continuous_cols": self.continuous_cols_,
                "means": self.means_,
                "stds": self.stds_,
            },
        )

    @classmethod
    def from_state(cls, state: TransformState) -> "ContinuousStandardScaler":
        obj = cls(eps=float(state.params.get("eps", 1e-12)))
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.continuous_cols_ = list(state.params.get("continuous_cols", []))
        obj.means_ = dict(state.params.get("means", {}))
        obj.stds_ = dict(state.params.get("stds", {}))
        return obj
