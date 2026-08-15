"""Explicit dataset declarations for the unified benchmark."""

from __future__ import annotations

from sbtab.benchmark.datasets.acquisition import (
    fetch_all_mixed_datasets,
    fetch_mixed_dataset,
)
from sbtab.benchmark.datasets.mixed import (
    MIXED_DATASET_COLUMNS,
    MIXED_DATASET_KEYS,
    make_mixed_dataset,
)
from sbtab.benchmark.datasets.online_shoppers import (
    ONLINE_SHOPPERS_COLUMNS,
    ONLINE_SHOPPERS_TARGET,
    ONLINE_SHOPPERS_UCI_ID,
    fetch_online_shoppers_frame,
    make_online_shoppers_dataset,
)

__all__ = [
    "MIXED_DATASET_COLUMNS",
    "MIXED_DATASET_KEYS",
    "ONLINE_SHOPPERS_COLUMNS",
    "ONLINE_SHOPPERS_TARGET",
    "ONLINE_SHOPPERS_UCI_ID",
    "fetch_all_mixed_datasets",
    "fetch_mixed_dataset",
    "fetch_online_shoppers_frame",
    "make_mixed_dataset",
    "make_online_shoppers_dataset",
]
