from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import pandas as pd

from .base import BaseTransform, TransformPipelineState
from .categorical import CategoricalRepresentationTransform
from .continuous import ContinuousStandardScaler
from .missing import DropMissingRows, TypeAwareImputer
from sbtab.data.schema import TabularSchema


_TRANSFORM_REGISTRY: Dict[str, Type[object]] = {
    "drop_missing_rows": DropMissingRows,
    "type_aware_imputer": TypeAwareImputer,
    "continuous_standard_scaler": ContinuousStandardScaler,
    "categorical_representation_transform": CategoricalRepresentationTransform,
}


@dataclass
class TransformPipeline:
    transforms: List[BaseTransform] = field(default_factory=list)
    name: str = "transform_pipeline"

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "TransformPipeline":
        x = df.copy()
        for transform in self.transforms:
            transform.fit(x, schema)
            x = transform.transform(x)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        for transform in self.transforms:
            if transform.requires_fit() and not _is_fitted(transform):
                raise RuntimeError(
                    f"Transform {transform.__class__.__name__} requires fit() before transform()."
                )
            x = transform.transform(x)
        return x

    def transform_global(self, df: pd.DataFrame, schema: TabularSchema) -> pd.DataFrame:
        """
        Apply only stateless transforms that are safe before splitting.

        This is intended for operations such as dropping rows with missing values.
        Any transform that requires train-time fitting is skipped here to avoid leakage.
        """
        x = df.copy()
        for transform in self.transforms:
            if transform.requires_fit():
                continue
            transform.fit(x, schema)
            x = transform.transform(x)
        return x

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        for transform in reversed(self.transforms):
            x = transform.inverse_transform(x)
        return x

    def get_state(self) -> TransformPipelineState:
        return TransformPipelineState(transforms=[transform.get_state() for transform in self.transforms])

    @classmethod
    def from_state(cls, state: TransformPipelineState) -> "TransformPipeline":
        transforms: List[BaseTransform] = []
        for transform_state in state.transforms:
            if transform_state.name not in _TRANSFORM_REGISTRY:
                raise KeyError(
                    f"Unknown transform {transform_state.name!r}. "
                    f"Register it in _TRANSFORM_REGISTRY first."
                )
            transform_cls = _TRANSFORM_REGISTRY[transform_state.name]
            transforms.append(transform_cls.from_state(transform_state))  # type: ignore[attr-defined]
        return cls(transforms=transforms)

    @classmethod
    def default_dropna_and_scale(cls) -> "TransformPipeline":
        return cls(transforms=[DropMissingRows(), ContinuousStandardScaler()])

    @classmethod
    def default_impute_and_scale(cls) -> "TransformPipeline":
        return cls(transforms=[TypeAwareImputer(), ContinuousStandardScaler()])

    @classmethod
    def default_impute_scale_encode(cls) -> "TransformPipeline":
        return cls(
            transforms=[
                TypeAwareImputer(),
                ContinuousStandardScaler(),
                CategoricalRepresentationTransform(
                    representation_name="one_hot_representation",
                    representation_kwargs={"handle_unknown": "ignore"},
                ),
            ]
        )


def _is_fitted(transform: BaseTransform) -> bool:
    return bool(getattr(transform, "fitted_", True))
