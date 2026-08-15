"""Explicit declarations for the fourteen published mixed datasets.

The upstream ``dataset_update`` work stored inferred feature groups in pickle
``DataFrame.attrs``.  This module materializes those groups as reviewable
``ColumnSpec`` values and includes every target in the modeled table.  It does
not download data, inspect pandas dtypes, or depend on the legacy
``TabularSchema``.

``make_mixed_dataset`` is an acquisition boundary, not benchmark dispatch.
The selected declaration is resolved before the runner receives the resulting
``TabularDataset``; runners and adapters must never branch on its name.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import pandas as pd

from sbtab.benchmark.contracts import (
    ColumnKind,
    ColumnSpec,
    TabularDataset,
    TaskType,
)
from sbtab.benchmark.datasets.online_shoppers import ONLINE_SHOPPERS_COLUMNS
from sbtab.benchmark.validation import validate_tabular_dataset


CONTINUOUS = ColumnKind.CONTINUOUS
DISCRETE = ColumnKind.DISCRETE
CATEGORICAL = ColumnKind.CATEGORICAL

MIXED_DATASET_KEYS: tuple[str, ...] = (
    "adult",
    "credit_approval",
    "online_shoppers",
    "eucalyptus",
    "forest_fires",
    "insurance",
    "house_sales",
    "cardiovascular_disease",
    "churn_modelling",
    "auto_mpg",
    "diamonds",
    "real_estate",
    "stroke_prediction",
    "palmer_penguins",
)
"""Stable acquisition keys for the published fourteen-dataset collection."""


_ADULT_COLUMNS = (
    ColumnSpec("age", CONTINUOUS),
    ColumnSpec("workclass", CATEGORICAL),
    ColumnSpec("fnlwgt", CONTINUOUS),
    ColumnSpec("education", CATEGORICAL),
    ColumnSpec("education-num", DISCRETE),
    ColumnSpec("marital-status", CATEGORICAL),
    ColumnSpec("occupation", CATEGORICAL),
    ColumnSpec("relationship", CATEGORICAL),
    ColumnSpec("race", CATEGORICAL),
    ColumnSpec("sex", CATEGORICAL),
    ColumnSpec("capital-gain", CONTINUOUS),
    ColumnSpec("capital-loss", CONTINUOUS),
    ColumnSpec("hours-per-week", CONTINUOUS),
    ColumnSpec("native-country", CATEGORICAL),
    ColumnSpec("income", CATEGORICAL),
)

_CREDIT_APPROVAL_COLUMNS = (
    ColumnSpec("A15", CONTINUOUS),
    ColumnSpec("A14", CONTINUOUS),
    ColumnSpec("A13", CATEGORICAL),
    ColumnSpec("A12", CATEGORICAL),
    ColumnSpec("A11", CONTINUOUS),
    ColumnSpec("A10", CATEGORICAL),
    ColumnSpec("A9", CATEGORICAL),
    ColumnSpec("A8", CONTINUOUS),
    ColumnSpec("A7", CATEGORICAL),
    ColumnSpec("A6", CATEGORICAL),
    ColumnSpec("A5", CATEGORICAL),
    ColumnSpec("A4", CATEGORICAL),
    ColumnSpec("A3", CONTINUOUS),
    ColumnSpec("A2", CONTINUOUS),
    ColumnSpec("A1", CATEGORICAL),
    ColumnSpec("A16", CATEGORICAL),
)

_EUCALYPTUS_COLUMNS = (
    ColumnSpec("Abbrev", CATEGORICAL),
    ColumnSpec("Rep", DISCRETE),
    ColumnSpec("Locality", CATEGORICAL),
    ColumnSpec("Map_Ref", CATEGORICAL),
    ColumnSpec("Latitude", CATEGORICAL),
    ColumnSpec("Altitude", DISCRETE),
    ColumnSpec("Rainfall", DISCRETE),
    ColumnSpec("Frosts", DISCRETE),
    ColumnSpec("Year", DISCRETE),
    ColumnSpec("Sp", CATEGORICAL),
    ColumnSpec("PMCno", CONTINUOUS),
    ColumnSpec("DBH", CONTINUOUS),
    ColumnSpec("Ht", CONTINUOUS),
    ColumnSpec("Surv", CONTINUOUS),
    ColumnSpec("Vig", CONTINUOUS),
    ColumnSpec("Ins_res", CONTINUOUS),
    ColumnSpec("Stem_Fm", CONTINUOUS),
    ColumnSpec("Crown_Fm", CONTINUOUS),
    ColumnSpec("Brnch_Fm", CONTINUOUS),
    ColumnSpec("Utility", CATEGORICAL),
)

_FOREST_FIRES_COLUMNS = (
    ColumnSpec("X", DISCRETE),
    ColumnSpec("Y", DISCRETE),
    ColumnSpec("month", CATEGORICAL),
    ColumnSpec("day", CATEGORICAL),
    ColumnSpec("FFMC", CONTINUOUS),
    ColumnSpec("DMC", CONTINUOUS),
    ColumnSpec("DC", CONTINUOUS),
    ColumnSpec("ISI", CONTINUOUS),
    ColumnSpec("temp", CONTINUOUS),
    ColumnSpec("RH", CONTINUOUS),
    ColumnSpec("wind", CONTINUOUS),
    ColumnSpec("rain", CONTINUOUS),
    ColumnSpec("area", CONTINUOUS),
)

_INSURANCE_COLUMNS = (
    ColumnSpec("age", CONTINUOUS),
    ColumnSpec("sex", CATEGORICAL),
    ColumnSpec("bmi", CONTINUOUS),
    ColumnSpec("children", DISCRETE),
    ColumnSpec("smoker", CATEGORICAL),
    ColumnSpec("region", CATEGORICAL),
    ColumnSpec("charges", CONTINUOUS),
)

_HOUSE_SALES_COLUMNS = (
    ColumnSpec("date", CATEGORICAL),
    ColumnSpec("price", CONTINUOUS),
    ColumnSpec("bedrooms", DISCRETE),
    ColumnSpec("bathrooms", CONTINUOUS),
    ColumnSpec("sqft_living", CONTINUOUS),
    ColumnSpec("sqft_lot", CONTINUOUS),
    ColumnSpec("floors", CONTINUOUS),
    ColumnSpec("waterfront", DISCRETE),
    ColumnSpec("view", DISCRETE),
    ColumnSpec("condition", DISCRETE),
    ColumnSpec("grade", DISCRETE),
    ColumnSpec("sqft_above", CONTINUOUS),
    ColumnSpec("sqft_basement", CONTINUOUS),
    ColumnSpec("yr_built", CONTINUOUS),
    ColumnSpec("yr_renovated", CONTINUOUS),
    ColumnSpec("zipcode", CONTINUOUS),
    ColumnSpec("lat", CONTINUOUS),
    ColumnSpec("long", CONTINUOUS),
    ColumnSpec("sqft_living15", CONTINUOUS),
    ColumnSpec("sqft_lot15", CONTINUOUS),
)

_CARDIOVASCULAR_DISEASE_COLUMNS = (
    ColumnSpec("age", CONTINUOUS),
    ColumnSpec("gender", DISCRETE),
    ColumnSpec("height", CONTINUOUS),
    ColumnSpec("weight", CONTINUOUS),
    ColumnSpec("ap_hi", CONTINUOUS),
    ColumnSpec("ap_lo", CONTINUOUS),
    ColumnSpec("cholesterol", DISCRETE),
    ColumnSpec("gluc", DISCRETE),
    ColumnSpec("smoke", DISCRETE),
    ColumnSpec("alco", DISCRETE),
    ColumnSpec("active", DISCRETE),
    ColumnSpec("cardio", CATEGORICAL),
)

_CHURN_MODELLING_COLUMNS = (
    ColumnSpec("CreditScore", CONTINUOUS),
    ColumnSpec("Geography", CATEGORICAL),
    ColumnSpec("Gender", CATEGORICAL),
    ColumnSpec("Age", CONTINUOUS),
    ColumnSpec("Tenure", DISCRETE),
    ColumnSpec("Balance", CONTINUOUS),
    ColumnSpec("NumOfProducts", DISCRETE),
    ColumnSpec("HasCrCard", DISCRETE),
    ColumnSpec("IsActiveMember", DISCRETE),
    ColumnSpec("EstimatedSalary", CONTINUOUS),
    ColumnSpec("Exited", CATEGORICAL),
)

_AUTO_MPG_COLUMNS = (
    ColumnSpec("displacement", CONTINUOUS),
    ColumnSpec("cylinders", DISCRETE),
    ColumnSpec("horsepower", CONTINUOUS),
    ColumnSpec("weight", CONTINUOUS),
    ColumnSpec("acceleration", CONTINUOUS),
    ColumnSpec("model_year", DISCRETE),
    ColumnSpec("origin", DISCRETE),
    ColumnSpec("mpg", CONTINUOUS),
)

_DIAMONDS_COLUMNS = (
    ColumnSpec("carat", CONTINUOUS),
    ColumnSpec("cut", CATEGORICAL),
    ColumnSpec("color", CATEGORICAL),
    ColumnSpec("clarity", CATEGORICAL),
    ColumnSpec("depth", CONTINUOUS),
    ColumnSpec("table", CONTINUOUS),
    ColumnSpec("price", CONTINUOUS),
    ColumnSpec("x", CONTINUOUS),
    ColumnSpec("y", CONTINUOUS),
    ColumnSpec("z", CONTINUOUS),
)

_REAL_ESTATE_COLUMNS = (
    ColumnSpec("X1 transaction date", CONTINUOUS),
    ColumnSpec("X2 house age", CONTINUOUS),
    ColumnSpec("X3 distance to the nearest MRT station", CONTINUOUS),
    ColumnSpec("X4 number of convenience stores", DISCRETE),
    ColumnSpec("X5 latitude", CONTINUOUS),
    ColumnSpec("X6 longitude", CONTINUOUS),
    ColumnSpec("Y house price of unit area", CONTINUOUS),
)

_STROKE_PREDICTION_COLUMNS = (
    ColumnSpec("gender", CATEGORICAL),
    ColumnSpec("age", CONTINUOUS),
    ColumnSpec("hypertension", DISCRETE),
    ColumnSpec("heart_disease", DISCRETE),
    ColumnSpec("ever_married", CATEGORICAL),
    ColumnSpec("work_type", CATEGORICAL),
    ColumnSpec("Residence_type", CATEGORICAL),
    ColumnSpec("avg_glucose_level", CONTINUOUS),
    ColumnSpec("bmi", CONTINUOUS),
    ColumnSpec("smoking_status", CATEGORICAL),
    ColumnSpec("stroke", CATEGORICAL),
)

_PALMER_PENGUINS_COLUMNS = (
    ColumnSpec("Species", CATEGORICAL),
    ColumnSpec("Island", CATEGORICAL),
    ColumnSpec("Clutch Completion", CATEGORICAL),
    ColumnSpec("Date Egg", CATEGORICAL),
    ColumnSpec("Culmen Length (mm)", CONTINUOUS),
    ColumnSpec("Culmen Depth (mm)", CONTINUOUS),
    ColumnSpec("Flipper Length (mm)", CONTINUOUS),
    ColumnSpec("Body Mass (g)", CONTINUOUS),
    ColumnSpec("Sex", CATEGORICAL),
    ColumnSpec("Delta 15 N (o/oo)", CONTINUOUS),
    ColumnSpec("Delta 13 C (o/oo)", CONTINUOUS),
)


MIXED_DATASET_COLUMNS: Mapping[str, tuple[ColumnSpec, ...]] = MappingProxyType(
    {
        "adult": _ADULT_COLUMNS,
        "credit_approval": _CREDIT_APPROVAL_COLUMNS,
        "online_shoppers": ONLINE_SHOPPERS_COLUMNS,
        "eucalyptus": _EUCALYPTUS_COLUMNS,
        "forest_fires": _FOREST_FIRES_COLUMNS,
        "insurance": _INSURANCE_COLUMNS,
        "house_sales": _HOUSE_SALES_COLUMNS,
        "cardiovascular_disease": _CARDIOVASCULAR_DISEASE_COLUMNS,
        "churn_modelling": _CHURN_MODELLING_COLUMNS,
        "auto_mpg": _AUTO_MPG_COLUMNS,
        "diamonds": _DIAMONDS_COLUMNS,
        "real_estate": _REAL_ESTATE_COLUMNS,
        "stroke_prediction": _STROKE_PREDICTION_COLUMNS,
        "palmer_penguins": _PALMER_PENGUINS_COLUMNS,
    }
)
"""Read-only explicit ``ColumnSpec`` sequence for every published key."""

_TARGET_BY_KEY: Mapping[str, str] = MappingProxyType(
    {
        "adult": "income",
        "credit_approval": "A16",
        "online_shoppers": "Revenue",
        "eucalyptus": "Utility",
        "forest_fires": "area",
        "insurance": "charges",
        "house_sales": "price",
        "cardiovascular_disease": "cardio",
        "churn_modelling": "Exited",
        "auto_mpg": "mpg",
        "diamonds": "price",
        "real_estate": "Y house price of unit area",
        "stroke_prediction": "stroke",
        "palmer_penguins": "Species",
    }
)

_TASK_BY_KEY: Mapping[str, TaskType] = MappingProxyType(
    {
        "adult": TaskType.CLASSIFICATION,
        "credit_approval": TaskType.CLASSIFICATION,
        "online_shoppers": TaskType.CLASSIFICATION,
        "eucalyptus": TaskType.CLASSIFICATION,
        "forest_fires": TaskType.REGRESSION,
        "insurance": TaskType.REGRESSION,
        "house_sales": TaskType.REGRESSION,
        "cardiovascular_disease": TaskType.CLASSIFICATION,
        "churn_modelling": TaskType.CLASSIFICATION,
        "auto_mpg": TaskType.REGRESSION,
        "diamonds": TaskType.REGRESSION,
        "real_estate": TaskType.REGRESSION,
        "stroke_prediction": TaskType.CLASSIFICATION,
        "palmer_penguins": TaskType.CLASSIFICATION,
    }
)


def make_mixed_dataset(key: str, frame: pd.DataFrame) -> TabularDataset:
    """Build one validated new-format dataset from an acquired raw frame.

    Parameters
    ----------
    key:
        Exact member of :data:`MIXED_DATASET_KEYS`.  The key selects an
        acquisition-time declaration; it is not inspected by benchmark
        runners, codecs, evaluators, or model adapters.
    frame:
        Raw table after source-specific non-modeled columns have been removed.
        The function neither copies nor reorders rows and never infers column
        semantics from values or pandas dtypes.

    Returns
    -------
    TabularDataset
        Validated dataset whose target remains a normal modeled column.

    Raises
    ------
    KeyError
        If ``key`` is not one of the fourteen published dataset keys.
    ContractViolation
        If the raw frame contradicts the selected explicit declaration.
    """

    if key not in MIXED_DATASET_COLUMNS:
        raise KeyError(
            f"Unknown mixed dataset key {key!r}; expected one of "
            f"{MIXED_DATASET_KEYS!r}."
        )
    dataset = TabularDataset(
        name=key,
        frame=frame,
        columns=MIXED_DATASET_COLUMNS[key],
        target=_TARGET_BY_KEY[key],
        task=_TASK_BY_KEY[key],
    )
    validate_tabular_dataset(dataset)
    return dataset
