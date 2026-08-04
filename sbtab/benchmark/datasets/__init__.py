"""Explicit dataset declarations for the unified benchmark."""

from __future__ import annotations

from sbtab.benchmark.datasets.online_shoppers import (
    ONLINE_SHOPPERS_COLUMNS,
    ONLINE_SHOPPERS_TARGET,
    ONLINE_SHOPPERS_UCI_ID,
    make_online_shoppers_dataset,
)

__all__ = [
    "ONLINE_SHOPPERS_COLUMNS",
    "ONLINE_SHOPPERS_TARGET",
    "ONLINE_SHOPPERS_UCI_ID",
    "make_online_shoppers_dataset",
]
