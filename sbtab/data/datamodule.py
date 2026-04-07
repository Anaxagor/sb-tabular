from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .schema import TabularSchema
from .splits import (
    HoldoutSplit,
    KFoldSplit,
    SplitConfigHoldout,
    SplitConfigKFold,
    make_holdout_split,
    make_kfold_splits,
)


@dataclass
class FoldData:
    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    transforms: Optional[Any] = None


@dataclass
class HoldoutData:
    train: pd.DataFrame
    val: pd.DataFrame
    transforms: Optional[Any] = None


class TabularDataModule:
    """
    Data module for mixed-type tabular data.

    Design choices
    --------------
    1. The schema validates the raw input.
    2. Only stateless/global-safe transforms are applied before splitting.
    3. Train-time preprocessing is fit on each split's training subset only.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: TabularSchema,
        transforms: Optional[Any] = None,
        reset_index: bool = True,
        validate: bool = True,
    ) -> None:
        self.schema = schema
        self.transforms = transforms

        if validate:
            schema.validate(df)

        df0 = df.copy()
        if self.transforms is not None:
            global_transforms = self._clone_transforms(self.transforms)
            if hasattr(global_transforms, "transform_global"):
                df0 = global_transforms.transform_global(df0, self.schema)

        if reset_index:
            df0 = df0.reset_index(drop=True)

        self.df_clean = df0
        self.n_samples = len(df0)
        self._kfold_splits: Optional[list[KFoldSplit]] = None
        self._holdout_split: Optional[HoldoutSplit] = None

    def prepare_kfold(self, cfg: SplitConfigKFold) -> None:
        self._kfold_splits = make_kfold_splits(self.n_samples, cfg)

    def get_fold(self, fold_id: int) -> FoldData:
        if self._kfold_splits is None:
            raise RuntimeError("K-fold splits are not prepared. Call prepare_kfold(cfg) first.")
        if fold_id < 0 or fold_id >= len(self._kfold_splits):
            raise IndexError(f"fold_id={fold_id} is out of range for {len(self._kfold_splits)} folds.")

        fold = self._kfold_splits[fold_id]
        train_raw = self.df_clean.iloc[fold.train_idx].copy()
        test_raw = self.df_clean.iloc[fold.test_idx].copy()

        if self.transforms is None:
            return FoldData(fold_id=fold_id, train=train_raw, test=test_raw, transforms=None)

        pipe = self._clone_transforms(self.transforms)
        pipe.fit(train_raw, self.schema)
        train = pipe.transform(train_raw)
        test = pipe.transform(test_raw)

        self._validate_post_transform(train, context=f"fold={fold_id} train")
        self._validate_post_transform(test, context=f"fold={fold_id} test")
        self._validate_matching_columns(train, test, left_name="train", right_name="test", context=f"fold={fold_id}")

        return FoldData(fold_id=fold_id, train=train, test=test, transforms=pipe)

    def get_all_folds(self) -> Dict[int, FoldData]:
        if self._kfold_splits is None:
            raise RuntimeError("K-fold splits are not prepared. Call prepare_kfold(cfg) first.")
        return {split.fold_id: self.get_fold(split.fold_id) for split in self._kfold_splits}

    def prepare_holdout(self, cfg: SplitConfigHoldout) -> None:
        self._holdout_split = make_holdout_split(self.n_samples, cfg)

    def get_holdout(self) -> HoldoutData:
        if self._holdout_split is None:
            raise RuntimeError("Holdout split is not prepared. Call prepare_holdout(cfg) first.")

        split = self._holdout_split
        train_raw = self.df_clean.iloc[split.train_idx].copy()
        val_raw = self.df_clean.iloc[split.val_idx].copy()

        if self.transforms is None:
            return HoldoutData(train=train_raw, val=val_raw, transforms=None)

        pipe = self._clone_transforms(self.transforms)
        pipe.fit(train_raw, self.schema)
        train = pipe.transform(train_raw)
        val = pipe.transform(val_raw)

        self._validate_post_transform(train, context="holdout train")
        self._validate_post_transform(val, context="holdout val")
        self._validate_matching_columns(train, val, left_name="train", right_name="val", context="holdout")

        return HoldoutData(train=train, val=val, transforms=pipe)

    def get_clean_df(self) -> pd.DataFrame:
        return self.df_clean.copy()

    def _validate_post_transform(self, df: pd.DataFrame, *, context: str) -> None:
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"[{context}] Duplicate columns found after preprocessing: {duplicates}")

        required_passthrough = [
            col
            for col in (self.schema.id_col, self.schema.target_col)
            if col is not None
        ]
        missing_passthrough = [col for col in required_passthrough if col not in df.columns]
        if missing_passthrough:
            raise ValueError(
                f"[{context}] Preprocessing removed required non-feature columns: {missing_passthrough}"
            )

    def _validate_matching_columns(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        left_name: str,
        right_name: str,
        context: str,
    ) -> None:
        if list(left.columns) != list(right.columns):
            left_cols = list(left.columns)
            right_cols = list(right.columns)
            raise ValueError(
                f"[{context}] Transformed {left_name}/{right_name} columns do not match.\n"
                f"{left_name}: {left_cols}\n{right_name}: {right_cols}"
            )

    @staticmethod
    def _clone_transforms(transforms: Any) -> Any:
        if hasattr(transforms, "get_state") and hasattr(transforms.__class__, "from_state"):
            state = transforms.get_state()
            return transforms.__class__.from_state(state)

        import copy

        return copy.deepcopy(transforms)
