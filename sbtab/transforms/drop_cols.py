from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
from sbtab.data.schema import TabularSchema
from sbtab.transforms.base import TransformState


@dataclass
class DropDataCols:
    name: str = "drop_data_cols"
    datetime_cols: Optional[List[str]] = None

    dropped_cols_: Optional[List[str]] = field(default=None, init=False)

    def requires_fit(self) -> bool:
        return False

    def is_invertible(self) -> bool:
        return False

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "DropDataCols":
        if self.datetime_cols is not None:
            self.dropped_cols_ = list(self.datetime_cols)
        else:
            self.dropped_cols_ = [
                col for col in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[col])
            ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = self.dropped_cols_ if self.dropped_cols_ is not None else self.datetime_cols

        if cols_to_drop:
            existing_cols = [col for col in cols_to_drop if col in df.columns]
            return df.drop(existing_cols, axis=1)

        return df.copy()

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_state(self):
        return TransformState(
            name=self.name,
            params={
                "datetime_cols": self.datetime_cols,
            },
        )

    @classmethod
    def from_state(cls, state) -> "DropDataCols":
        return cls(
            datetime_cols=state.params.get("datetime_cols"),
        )