"""Lazy acquisition for the fourteen published mixed datasets.

Network clients are imported only when a caller explicitly fetches data.
Source-specific removal of identifiers and unused descriptive columns happens
here, before the raw frame crosses into the semantic declaration boundary.
The resulting datasets are validated by :func:`make_mixed_dataset`.

The upstream loader selected the largest CSV in each Kaggle download.  That is
not reproducible when a dataset contains several tables, so every Kaggle
source below names the exact CSV used by the published collection.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from sbtab.benchmark.contracts import TabularDataset
from sbtab.benchmark.datasets.mixed import (
    MIXED_DATASET_KEYS,
    make_mixed_dataset,
)
from sbtab.benchmark.validation import ContractViolation


def _fetch_uci_frame(dataset_id: int, target: str) -> pd.DataFrame:
    """Return UCI features and one named target in source-provided order."""

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as error:
        raise RuntimeError(
            "Fetching UCI benchmark data requires the optional "
            "'ucimlrepo' package."
        ) from error

    repository = fetch_ucirepo(id=dataset_id)
    features = repository.data.features.copy().reset_index(drop=True)
    targets = repository.data.targets
    if targets is None:
        raise ContractViolation(f"UCI {dataset_id} returned no target table.")
    if isinstance(targets, pd.Series):
        target_frame = targets.to_frame()
    elif isinstance(targets, pd.DataFrame):
        target_frame = targets.copy()
    else:
        target_frame = pd.DataFrame(targets)
    target_frame = target_frame.reset_index(drop=True)
    if target not in target_frame.columns:
        raise ContractViolation(
            f"UCI {dataset_id} target table lacks {target!r}."
        )
    if target in features.columns:
        raise ContractViolation(
            f"UCI {dataset_id} features unexpectedly contain target {target!r}."
        )
    return pd.concat((features, target_frame[[target]]), axis=1)


def _fetch_openml_frame(data_id: int) -> pd.DataFrame:
    """Return one OpenML frame while keeping acquisition optional at import."""

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as error:
        raise RuntimeError(
            "Fetching OpenML benchmark data requires scikit-learn."
        ) from error

    result = fetch_openml(data_id=data_id, as_frame=True)
    if result.frame is None:
        raise ContractViolation(f"OpenML {data_id} returned no pandas frame.")
    return result.frame.copy().reset_index(drop=True)


def _fetch_kaggle_csv(
    handle: str,
    filename: str,
    *,
    separator: str = ",",
) -> pd.DataFrame:
    """Download and read one exact CSV from a public Kaggle dataset."""

    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Fetching Kaggle benchmark data requires the optional "
            "'kagglehub' package. Install it and authenticate if the source "
            "requires user consent."
        ) from error

    downloaded_path = Path(kagglehub.dataset_download(handle, path=filename))
    if not downloaded_path.is_file():
        raise FileNotFoundError(
            f"Kaggle source {handle!r} did not provide {filename!r}; "
            f"received {str(downloaded_path)!r}."
        )
    return pd.read_csv(downloaded_path, sep=separator)


def _drop_required_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    source: str,
) -> pd.DataFrame:
    """Remove known non-modeled columns and fail on source-version drift."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractViolation(
            f"{source} lacks expected non-modeled columns {missing!r}."
        )
    return frame.drop(columns=list(columns))


def _fetch_adult() -> pd.DataFrame:
    return _fetch_uci_frame(2, "income")


def _fetch_credit_approval() -> pd.DataFrame:
    return _fetch_uci_frame(27, "A16")


def _fetch_online_shoppers() -> pd.DataFrame:
    return _fetch_uci_frame(468, "Revenue")


def _fetch_eucalyptus() -> pd.DataFrame:
    return _fetch_openml_frame(188)


def _fetch_forest_fires() -> pd.DataFrame:
    return _fetch_uci_frame(162, "area")


def _fetch_insurance() -> pd.DataFrame:
    return _fetch_kaggle_csv("mirichoi0218/insurance", "insurance.csv")


def _fetch_house_sales() -> pd.DataFrame:
    frame = _fetch_kaggle_csv(
        "harlfoxem/housesalesprediction",
        "kc_house_data.csv",
    )
    return _drop_required_columns(frame, ("id",), source="House Sales")


def _fetch_cardiovascular_disease() -> pd.DataFrame:
    frame = _fetch_kaggle_csv(
        "sulianova/cardiovascular-disease-dataset",
        "cardio_train.csv",
        separator=";",
    )
    return _drop_required_columns(
        frame,
        ("id",),
        source="Cardiovascular Disease",
    )


def _fetch_churn_modelling() -> pd.DataFrame:
    frame = _fetch_kaggle_csv(
        "shrutimechlearn/churn-modelling",
        "Churn_Modelling.csv",
    )
    return _drop_required_columns(
        frame,
        ("RowNumber", "CustomerId", "Surname"),
        source="Churn Modelling",
    )


def _fetch_auto_mpg() -> pd.DataFrame:
    return _fetch_uci_frame(9, "mpg")


def _fetch_diamonds() -> pd.DataFrame:
    frame = _fetch_kaggle_csv("shivam2503/diamonds", "diamonds.csv")
    return _drop_required_columns(
        frame,
        ("Unnamed: 0",),
        source="Diamonds",
    )


def _fetch_real_estate() -> pd.DataFrame:
    frame = _fetch_kaggle_csv(
        "quantbruce/real-estate-price-prediction",
        "Real estate.csv",
    )
    return _drop_required_columns(frame, ("No",), source="Real Estate")


def _fetch_stroke_prediction() -> pd.DataFrame:
    frame = _fetch_kaggle_csv(
        "fedesoriano/stroke-prediction-dataset",
        "healthcare-dataset-stroke-data.csv",
    )
    return _drop_required_columns(frame, ("id",), source="Stroke Prediction")


def _fetch_palmer_penguins() -> pd.DataFrame:
    frame = _fetch_kaggle_csv(
        "parulpandey/palmer-archipelago-antarctica-penguin-data",
        "penguins_lter.csv",
    )
    return _drop_required_columns(
        frame,
        (
            "studyName",
            "Sample Number",
            "Individual ID",
            "Region",
            "Stage",
            "Comments",
        ),
        source="Palmer Penguins",
    )


_FETCHER_BY_KEY: Mapping[str, Callable[[], pd.DataFrame]] = MappingProxyType(
    {
        "adult": _fetch_adult,
        "credit_approval": _fetch_credit_approval,
        "online_shoppers": _fetch_online_shoppers,
        "eucalyptus": _fetch_eucalyptus,
        "forest_fires": _fetch_forest_fires,
        "insurance": _fetch_insurance,
        "house_sales": _fetch_house_sales,
        "cardiovascular_disease": _fetch_cardiovascular_disease,
        "churn_modelling": _fetch_churn_modelling,
        "auto_mpg": _fetch_auto_mpg,
        "diamonds": _fetch_diamonds,
        "real_estate": _fetch_real_estate,
        "stroke_prediction": _fetch_stroke_prediction,
        "palmer_penguins": _fetch_palmer_penguins,
    }
)


def fetch_mixed_dataset(key: str) -> TabularDataset:
    """Fetch and validate one published mixed dataset by stable key.

    Network and source-cache side effects occur only when this function is
    called.  Unknown keys fail before any optional dependency is imported.
    """

    if key not in _FETCHER_BY_KEY:
        raise KeyError(
            f"Unknown mixed dataset key {key!r}; expected one of "
            f"{MIXED_DATASET_KEYS!r}."
        )
    return make_mixed_dataset(key, _FETCHER_BY_KEY[key]())


def fetch_all_mixed_datasets() -> tuple[TabularDataset, ...]:
    """Fetch all fourteen datasets in the published canonical order.

    The function intentionally runs sequentially and fails at the first bad
    source instead of returning a silently incomplete benchmark collection.
    Individual acquisition is available through :func:`fetch_mixed_dataset`.
    """

    return tuple(fetch_mixed_dataset(key) for key in MIXED_DATASET_KEYS)
