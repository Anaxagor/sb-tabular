"""Train-on-synthetic, test-on-real utility with one fixed CatBoost protocol.

The evaluator trains the same downstream predictor twice and changes only its
training table. It consumes decoded raw data and does not reuse or fit the
generator's model codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score

from sbtab.benchmark.contracts import (
    ColumnKind,
    TabularDataset,
    TaskType,
)
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_tabular_dataset,
)
from sbtab.evaluation._validation import validate_raw_table


class UtilityMetric(str, Enum):
    """Downstream score selected by declared task semantics."""

    MACRO_F1 = "macro_f1"
    R2 = "r2"


@dataclass(frozen=True)
class UtilityScore:
    """TSTR comparison for one benchmark fold.

    Parameters
    ----------
    metric:
        Macro-F1 for classification or R² for regression.
    real_score:
        Score of CatBoost trained on the real training partition.
    synthetic_score:
        Score of CatBoost trained on the decoded synthetic table.
    absolute_change:
        ``synthetic_score - real_score``. A negative value is degradation.
    relative_change_percent:
        Signed change divided by ``abs(real_score)``. It is ``None`` when the
        real baseline is exactly zero.
    """

    metric: UtilityMetric
    real_score: float
    synthetic_score: float
    absolute_change: float
    relative_change_percent: float | None


def _make_predictor(task: TaskType, seed: int) -> object:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as error:
        raise RuntimeError(
            "Final TSTR evaluation requires the catboost package."
        ) from error

    common = {
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
    }
    if task is TaskType.CLASSIFICATION:
        return CatBoostClassifier(
            **common,
            allow_const_label=True,
        )
    return CatBoostRegressor(**common)


def _score_predictions(
    task: TaskType,
    target: pd.Series,
    predictions: np.ndarray,
) -> tuple[UtilityMetric, float]:
    if task is TaskType.CLASSIFICATION:
        return (
            UtilityMetric.MACRO_F1,
            float(
                f1_score(
                    target,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
        )
    return UtilityMetric.R2, float(r2_score(target, predictions))


def _utility_semantics(
    dataset: TabularDataset,
    seed: int,
) -> tuple[str, TaskType]:
    if dataset.target is None or dataset.task is None:
        raise ContractViolation(
            "TSTR utility requires a declared dataset target and task."
        )
    target_kind = dataset.column(dataset.target).kind
    if (
        dataset.task is TaskType.CLASSIFICATION
        and target_kind is ColumnKind.CONTINUOUS
    ):
        raise ContractViolation(
            "Classification target must be discrete or categorical."
        )
    if (
        dataset.task is TaskType.REGRESSION
        and target_kind is ColumnKind.CATEGORICAL
    ):
        raise ContractViolation(
            "Regression target must be continuous or numeric discrete."
        )
    if len(dataset.column_order) < 2:
        raise ContractViolation(
            "TSTR utility requires at least one modeled feature besides target."
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractViolation("TSTR seed must be an integer.")
    if not 0 <= seed < 2**32:
        raise ContractViolation("TSTR seed must be in [0, 2**32).")
    return dataset.target, dataset.task


def evaluate_utility(
    dataset: TabularDataset,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    seed: int,
) -> UtilityScore:
    """Evaluate one real-trained and one synthetic-trained CatBoost predictor.

    The two predictors use identical settings and the same seed. A degenerate
    single-class synthetic target is accepted by CatBoost and produces its
    naturally poor constant prediction instead of aborting evaluation.
    """

    validate_tabular_dataset(dataset)
    target_name, task = _utility_semantics(dataset, seed)
    validate_raw_table(dataset, real_train, label="real_train")
    validate_raw_table(dataset, real_test, label="real_test")
    validate_raw_table(dataset, synthetic, label="synthetic")

    feature_names = tuple(
        name for name in dataset.column_order if name != target_name
    )
    categorical_features = tuple(
        name
        for name in dataset.categorical_columns
        if name != target_name
    )

    real_predictor = _make_predictor(task, seed)
    real_predictor.fit(
        real_train.loc[:, list(feature_names)],
        real_train[target_name],
        cat_features=list(categorical_features),
    )
    real_predictions = np.asarray(
        real_predictor.predict(real_test.loc[:, list(feature_names)])
    ).reshape(-1)
    metric, real_score = _score_predictions(
        task,
        real_test[target_name],
        real_predictions,
    )
    if not np.isfinite(real_score):
        raise ContractViolation(
            f"Real-trained utility score is not finite: {real_score!r}."
        )

    synthetic_predictor = _make_predictor(task, seed)
    synthetic_predictor.fit(
        synthetic.loc[:, list(feature_names)],
        synthetic[target_name],
        cat_features=list(categorical_features),
    )
    synthetic_predictions = np.asarray(
        synthetic_predictor.predict(real_test.loc[:, list(feature_names)])
    ).reshape(-1)
    synthetic_metric, synthetic_score = _score_predictions(
        task,
        real_test[target_name],
        synthetic_predictions,
    )
    if synthetic_metric is not metric:
        raise ContractViolation("Utility predictors produced different metrics.")
    if not np.isfinite(synthetic_score):
        raise ContractViolation(
            "Synthetic-trained utility score is not finite: "
            f"{synthetic_score!r}."
        )

    absolute_change = synthetic_score - real_score
    relative_change_percent = (
        None
        if real_score == 0.0
        else 100.0 * absolute_change / abs(real_score)
    )
    if not np.isfinite(absolute_change) or (
        relative_change_percent is not None
        and not np.isfinite(relative_change_percent)
    ):
        raise ContractViolation("Utility change is not finite.")
    return UtilityScore(
        metric=metric,
        real_score=real_score,
        synthetic_score=synthetic_score,
        absolute_change=absolute_change,
        relative_change_percent=relative_change_percent,
    )
