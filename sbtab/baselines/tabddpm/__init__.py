"""TabDDPM native API with a lazy legacy DataFrame wrapper."""

from __future__ import annotations

from sbtab.baselines.tabddpm.native import TabDDPMConfig, TabDDPMSolver

__all__ = ["TabDDPMConfig", "TabDDPMSolver", "TabDDPMWrapper"]


def __getattr__(name: str) -> object:
    """Load the legacy schema-aware wrapper only when explicitly requested."""

    if name == "TabDDPMWrapper":
        from sbtab.baselines.tabddpm.model import TabDDPMWrapper

        return TabDDPMWrapper
    raise AttributeError(name)
