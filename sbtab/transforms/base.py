from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, TYPE_CHECKING, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from sbtab.data.schema import TabularSchema


@dataclass(frozen=True)
class TransformState:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformPipelineState:
    transforms: List[TransformState] = field(default_factory=list)


@runtime_checkable
class BaseTransform(Protocol):
    name: str

    def requires_fit(self) -> bool:
        ...

    def is_invertible(self) -> bool:
        ...

    def fit(self, df: pd.DataFrame, schema: "TabularSchema") -> "BaseTransform":
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def get_state(self) -> TransformState:
        ...

    @classmethod
    def from_state(cls, state: TransformState) -> "BaseTransform":
        ...
