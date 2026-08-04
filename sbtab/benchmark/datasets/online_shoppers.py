"""Canonical declaration of the approved MSBM pilot dataset.

The module contains no download logic and does not infer semantics from pandas
dtypes. A caller supplies the raw UCI Online Shoppers frame, including the
``Revenue`` target, and receives a validated :class:`TabularDataset`.

MSBM treats the three page-count columns as ordered numeric discrete states.
The remaining finite-state columns are nominal. In particular, ``Month`` is
cyclic while MSBM's ordered reference is linear, so declaring it ordered would
encode the wrong neighbourhood.
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
    """Build and validate the approved Online Shoppers dataset declaration.

    Parameters
    ----------
    frame:
        Raw UCI 468 table with all 17 features and the ``Revenue`` target. The
        function does not download, filter, reorder, or copy rows.
    name:
        Artifact label only. Benchmark behavior must not dispatch on it.

    Returns
    -------
    TabularDataset
        Validated dataset with target retained in canonical modeled order.

    Raises
    ------
    ContractViolation
        If required columns are absent, duplicated, or accompanied by
        undeclared raw columns.
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
