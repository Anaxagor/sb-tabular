"""
Non-SB baselines behind one interface (see ``sbtab.baselines.base``).

Importing this package never imports a third-party generator library: ``sdv`` (CTGAN) and
``tabpfgen`` / ``tabpfn`` (TabPFGen) are imported lazily inside ``fit`` / ``load_checkpoint``.
"""
from sbtab.baselines.base import (
    ArrayLike,
    BaselineFitInfo,
    BaselineGenerativeModel,
    ColumnRoles,
    largest_remainder_allocation,
    resolve_column_roles,
)

__all__ = [
    "ArrayLike",
    "BaselineFitInfo",
    "BaselineGenerativeModel",
    "ColumnRoles",
    "largest_remainder_allocation",
    "resolve_column_roles",
]
