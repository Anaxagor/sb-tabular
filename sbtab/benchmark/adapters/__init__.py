"""Model-specific translations from canonical tables to native generators."""

from __future__ import annotations

from sbtab.benchmark.adapters.msbm import MSBMAdapter
from sbtab.benchmark.adapters.tabddpm import TabDDPMAdapter

__all__ = [
    "MSBMAdapter",
    "TabDDPMAdapter",
]
