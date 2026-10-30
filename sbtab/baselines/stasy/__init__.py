"""
Simplified VE score-SDE baseline.  NOT a faithful STaSy implementation - see ``FAITHFULNESS``.
``STaSyConfig`` / ``STaSyGenerative`` are deprecated aliases that warn on use.
"""
from sbtab.baselines.stasy.model import (
    FAITHFULNESS,
    VARIANT_ID,
    STaSyConfig,
    STaSyGenerative,
    VEScoreSDEBaseline,
    VEScoreSDEConfig,
)

__all__ = [
    "FAITHFULNESS",
    "VARIANT_ID",
    "VEScoreSDEBaseline",
    "VEScoreSDEConfig",
    "STaSyConfig",
    "STaSyGenerative",
]
