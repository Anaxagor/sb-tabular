"""Canonical semantic declaration for UCI Online Shoppers dataset 468.

This module contains no acquisition or preprocessing logic. A caller supplies
the raw 17-feature table with its ``Revenue`` target attached, and the factory
returns a validated :class:`~sbtab.benchmark.contracts.TabularDataset`.

The three page-visit counts are numeric discrete values. Durations, rates,
page value, and special-day proximity are continuous. The remaining source
fields are nominal categories even when UCI stores them as integer codes or
booleans. ``Month`` is cyclic rather than ordinal, so it has no linear
``ordered_values`` declaration.
"""

from __future__ import annotations

import pandas as pd

from sbtab.benchmark.contracts import (
    ColumnKind,
    ColumnSpec,
    TabularDataset,
    TaskType,
)
from sbtab.benchmark.validation import validate_tabular_dataset


ONLINE_SHOPPERS_UCI_ID = 468
ONLINE_SHOPPERS_TARGET = "Revenue"

# UCI feature order followed by the separate source target. This tuple is the
# canonical generated-table order; callers are not required to rely on the
# physical order of columns in their input DataFrame.
ONLINE_SHOPPERS_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("Administrative", ColumnKind.DISCRETE),
    ColumnSpec("Administrative_Duration", ColumnKind.CONTINUOUS),
    ColumnSpec("Informational", ColumnKind.DISCRETE),
    ColumnSpec("Informational_Duration", ColumnKind.CONTINUOUS),
    ColumnSpec("ProductRelated", ColumnKind.DISCRETE),
    ColumnSpec("ProductRelated_Duration", ColumnKind.CONTINUOUS),
    ColumnSpec("BounceRates", ColumnKind.CONTINUOUS),
    ColumnSpec("ExitRates", ColumnKind.CONTINUOUS),
    ColumnSpec("PageValues", ColumnKind.CONTINUOUS),
    ColumnSpec("SpecialDay", ColumnKind.CONTINUOUS),
    ColumnSpec("Month", ColumnKind.CATEGORICAL),
    ColumnSpec("OperatingSystems", ColumnKind.CATEGORICAL),
    ColumnSpec("Browser", ColumnKind.CATEGORICAL),
    ColumnSpec("Region", ColumnKind.CATEGORICAL),
    ColumnSpec("TrafficType", ColumnKind.CATEGORICAL),
    ColumnSpec("VisitorType", ColumnKind.CATEGORICAL),
    ColumnSpec("Weekend", ColumnKind.CATEGORICAL),
    ColumnSpec(ONLINE_SHOPPERS_TARGET, ColumnKind.CATEGORICAL),
)


def make_online_shoppers_dataset(
    frame: pd.DataFrame,
    *,
    name: str = "online_shoppers_uci_468",
) -> TabularDataset:
    """Attach the approved Online Shoppers semantics to one raw table.

    Parameters
    ----------
    frame:
        Raw UCI 468 data with all 17 features and the ``Revenue`` target. The
        function retains this object and does not download, filter, reorder,
        or copy its rows.
    name:
        Stable artifact label only. Benchmark behavior must not dispatch on
        this value.

    Returns
    -------
    TabularDataset
        Validated declaration with ``Revenue`` retained as the final modeled
        column and classification target.

    Raises
    ------
    ContractViolation
        If the supplied frame contradicts the declared names or value
        semantics, including missing, duplicate, or undeclared columns.
    """

    dataset = TabularDataset(
        name=name,
        frame=frame,
        columns=ONLINE_SHOPPERS_COLUMNS,
        target=ONLINE_SHOPPERS_TARGET,
        task=TaskType.CLASSIFICATION,
    )
    validate_tabular_dataset(dataset)
    return dataset
