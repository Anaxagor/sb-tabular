"""Shared validation for decoded raw tables entering evaluation."""

from __future__ import annotations

import pandas as pd

from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_tabular_dataset,
)


def validate_raw_table(
    dataset: TabularDataset,
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Validate one decoded modeled table without repairing its values."""

    if not isinstance(frame, pd.DataFrame):
        raise ContractViolation(f"{label} must be a pandas DataFrame.")
    actual_columns = tuple(frame.columns.tolist())
    if actual_columns != dataset.column_order:
        raise ContractViolation(
            f"{label} columns must match canonical modeled order; "
            f"actual={actual_columns!r}, expected={dataset.column_order!r}."
        )
    if frame.empty:
        raise ContractViolation(f"{label} must contain at least one row.")
    missing = {
        name: int(frame[name].isna().sum())
        for name in dataset.column_order
        if frame[name].isna().any()
    }
    if missing:
        raise ContractViolation(f"{label} contains missing values: {missing!r}.")

    validate_tabular_dataset(
        TabularDataset(
            name=f"{dataset.name}:{label}",
            frame=frame,
            columns=dataset.columns,
            target=dataset.target,
            task=dataset.task,
        )
    )
