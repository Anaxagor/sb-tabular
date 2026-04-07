from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .base import TransformState
from sbtab.data.schema import TabularSchema


@dataclass
class OneHotRepresentation:
    handle_unknown: str = "ignore"
    drop: Any = None
    dtype: Any = float

    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    passthrough_cols_: List[str] = field(default_factory=list)
    encoded_cols_: List[str] = field(default_factory=list)
    original_col_order_: List[str] = field(default_factory=list)
    categories_: Dict[str, List[Any]] = field(default_factory=dict)
    encoder_: Optional[OneHotEncoder] = field(default=None, repr=False)

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "OneHotRepresentation":
        self.categorical_cols_ = list(schema.categorical_cols)
        self.passthrough_cols_ = [col for col in df.columns if col not in self.categorical_cols_]
        self.original_col_order_ = list(df.columns)

        if not self.categorical_cols_:
            self.encoded_cols_ = []
            self.categories_ = {}
            self.encoder_ = None
            self.fitted_ = True
            return self

        encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            drop=self.drop,
            sparse_output=False,
            dtype=self.dtype,
        )
        encoder.fit(df[self.categorical_cols_].astype("object"))

        self.encoder_ = encoder
        self.encoded_cols_ = encoder.get_feature_names_out(self.categorical_cols_).tolist()
        self.categories_ = {
            col: list(encoder.categories_[idx]) for idx, col in enumerate(self.categorical_cols_)
        }
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("OneHotRepresentation must be fitted before transform().")

        if not self.categorical_cols_:
            return df.copy()

        missing = [col for col in self.categorical_cols_ if col not in df.columns]
        if missing:
            raise ValueError(f"Categorical columns are missing during encoding: {missing}")

        transformed = self.encoder_.transform(df[self.categorical_cols_].astype("object"))  # type: ignore[union-attr]
        encoded_df = pd.DataFrame(transformed, columns=self.encoded_cols_, index=df.index)
        passthrough_df = df[self.passthrough_cols_].copy()
        return pd.concat([passthrough_df, encoded_df], axis=1)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("OneHotRepresentation must be fitted before inverse_transform().")

        if not self.categorical_cols_:
            return df.copy()

        missing = [col for col in self.encoded_cols_ if col not in df.columns]
        if missing:
            raise ValueError(f"Encoded categorical columns are missing during inverse_transform: {missing}")

        recovered = self.encoder_.inverse_transform(df[self.encoded_cols_])  # type: ignore[union-attr]
        recovered_df = pd.DataFrame(recovered, columns=self.categorical_cols_, index=df.index)
        passthrough_df = df[self.passthrough_cols_].copy()
        out = pd.concat([passthrough_df, recovered_df], axis=1)

        ordered_cols = [col for col in self.original_col_order_ if col in out.columns]
        remaining_cols = [col for col in out.columns if col not in ordered_cols]
        return out[ordered_cols + remaining_cols]

    def get_state(self) -> Dict[str, Any]:
        return {
            "handle_unknown": self.handle_unknown,
            "drop": self.drop,
            "dtype": self.dtype,
            "fitted": self.fitted_,
            "categorical_cols": self.categorical_cols_,
            "passthrough_cols": self.passthrough_cols_,
            "encoded_cols": self.encoded_cols_,
            "original_col_order": self.original_col_order_,
            "categories": self.categories_,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "OneHotRepresentation":
        obj = cls(
            handle_unknown=state.get("handle_unknown", "ignore"),
            drop=state.get("drop", None),
            dtype=state.get("dtype", float),
        )
        obj.fitted_ = bool(state.get("fitted", False))
        obj.categorical_cols_ = list(state.get("categorical_cols", []))
        obj.passthrough_cols_ = list(state.get("passthrough_cols", []))
        obj.encoded_cols_ = list(state.get("encoded_cols", []))
        obj.original_col_order_ = list(state.get("original_col_order", []))
        obj.categories_ = dict(state.get("categories", {}))

        if obj.fitted_ and obj.categorical_cols_:
            synthetic = _build_synthetic_categorical_frame(obj.categories_)
            encoder = OneHotEncoder(
                handle_unknown=obj.handle_unknown,
                drop=obj.drop,
                sparse_output=False,
                dtype=obj.dtype,
            )
            encoder.fit(synthetic.astype("object"))
            obj.encoder_ = encoder
            obj.encoded_cols_ = encoder.get_feature_names_out(obj.categorical_cols_).tolist()

        return obj


_REPRESENTATION_REGISTRY: Dict[str, Type[OneHotRepresentation]] = {
    "one_hot_representation": OneHotRepresentation,
}


@dataclass
class CategoricalRepresentationTransform:
    name: str = "categorical_representation_transform"
    representation_name: str = "one_hot_representation"
    representation_kwargs: Dict[str, Any] = field(default_factory=dict)

    fitted_: bool = False
    repr_: Optional[OneHotRepresentation] = None

    def requires_fit(self) -> bool:
        return True

    def is_invertible(self) -> bool:
        return True

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "CategoricalRepresentationTransform":
        if self.representation_name not in _REPRESENTATION_REGISTRY:
            raise KeyError(
                f"Unknown representation {self.representation_name!r}. "
                f"Available representations: {sorted(_REPRESENTATION_REGISTRY)}"
            )

        rep_cls = _REPRESENTATION_REGISTRY[self.representation_name]
        rep = rep_cls(**self.representation_kwargs)
        rep.fit(df, schema)

        self.repr_ = rep
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.repr_ is None:
            raise RuntimeError("CategoricalRepresentationTransform must be fitted before transform().")
        return self.repr_.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.repr_ is None:
            raise RuntimeError("CategoricalRepresentationTransform must be fitted before inverse_transform().")
        return self.repr_.inverse_transform(df)

    def get_state(self) -> TransformState:
        rep_state = None if self.repr_ is None else self.repr_.get_state()
        return TransformState(
            name=self.name,
            params={
                "representation_name": self.representation_name,
                "representation_kwargs": self.representation_kwargs,
                "fitted": self.fitted_,
                "rep_state": rep_state,
            },
        )

    @classmethod
    def from_state(cls, state: TransformState) -> "CategoricalRepresentationTransform":
        obj = cls(
            representation_name=str(state.params.get("representation_name", "one_hot_representation")),
            representation_kwargs=dict(state.params.get("representation_kwargs", {})),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        rep_state = state.params.get("rep_state")
        if obj.fitted_ and rep_state is not None:
            rep_cls = _REPRESENTATION_REGISTRY[obj.representation_name]
            obj.repr_ = rep_cls.from_state(rep_state)
        return obj


def register_representation(name: str, cls: Type[OneHotRepresentation]) -> None:
    _REPRESENTATION_REGISTRY[name] = cls


def _build_synthetic_categorical_frame(categories_by_col: Dict[str, List[Any]]) -> pd.DataFrame:
    if not categories_by_col:
        return pd.DataFrame()

    max_len = max(max(len(values), 1) for values in categories_by_col.values())
    data: Dict[str, List[Any]] = {}
    for col, values in categories_by_col.items():
        if not values:
            data[col] = [None] * max_len
            continue
        padded = list(values) + [values[-1]] * max(0, max_len - len(values))
        data[col] = padded[:max_len]
    return pd.DataFrame(data)
