"""Model-independent tuning objective for decoded generator samples.

The objective follows the mixed-data experiment protocol after shared codec
decoding. Continuous columns use one-dimensional Wasserstein distance after a
train-fitted standardization, so raw measurement units cannot dominate model
selection. Discrete and categorical columns use empirical Jensen--Shannon
divergence. The target participates according to its declared
:class:`ColumnKind`, exactly like every other modeled column.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from sbtab.benchmark.contracts import ColumnKind, TabularDataset
from sbtab.benchmark.validation import validate_tabular_dataset
from sbtab.evaluation._validation import validate_raw_table


class TuningMetric(str, Enum):
    """Per-column distance used by the reference tuning objective."""

    STANDARDIZED_WASSERSTEIN = "standardized_wasserstein"
    JENSEN_SHANNON = "jensen_shannon"


@dataclass(frozen=True)
class ColumnTuningScore:
    """One raw column's contribution before semantic-group averaging.

    Parameters
    ----------
    column:
        Declared modeled column name, including target when applicable.
    kind:
        Raw semantic kind that selected the distance formula.
    metric:
        Wasserstein for continuous data or Jensen--Shannon for finite data.
    value:
        Non-negative distance after the metric-specific transformation.
    reference_scale:
        Population standard deviation fitted on holdout-train for a continuous
        column. A constant train column uses ``1.0``. Finite columns use
        ``None`` because Jensen--Shannon compares exact decoded states.
    """

    column: str
    kind: ColumnKind
    metric: TuningMetric
    value: float
    reference_scale: float | None


@dataclass(frozen=True)
class TuningScore:
    """Composite value minimized by every model-owned tuning study.

    Parameters
    ----------
    total:
        Sum of the semantic-group means that exist in the dataset.
    mean_wasserstein:
        Mean train-standardized continuous-column distance, or ``None`` when
        there are no continuous modeled columns.
    mean_jensen_shannon:
        Mean across all discrete and categorical columns, or ``None`` when
        there are no finite modeled columns.
    columns:
        Per-column evidence in canonical modeled order.
    """

    total: float
    mean_wasserstein: float | None
    mean_jensen_shannon: float | None
    columns: tuple[ColumnTuningScore, ...]


def _jensen_shannon_divergence(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    combined = pd.concat(
        (real.reset_index(drop=True), synthetic.reset_index(drop=True)),
        ignore_index=True,
    )
    codes, observed = pd.factorize(combined, sort=False)
    cardinality = len(observed)
    real_size = len(real)
    p = np.bincount(codes[:real_size], minlength=cardinality).astype(np.float64)
    q = np.bincount(codes[real_size:], minlength=cardinality).astype(np.float64)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    p_positive = p > 0.0
    q_positive = q > 0.0
    divergence = 0.5 * np.sum(
        p[p_positive] * np.log(p[p_positive] / midpoint[p_positive])
    )
    divergence += 0.5 * np.sum(
        q[q_positive] * np.log(q[q_positive] / midpoint[q_positive])
    )
    return float(divergence)


def evaluate_tuning_score(
    dataset: TabularDataset,
    real_train: pd.DataFrame,
    real_validation: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> TuningScore:
    """Calculate the scale-balanced objective for one tuning trial.

    Each continuous column is standardized with population mean and standard
    deviation fitted only on ``real_train``; a zero train deviation maps to
    scale ``1.0``. The same transform is applied to validation and synthetic
    values before Wasserstein distance. Jensen--Shannon uses natural logarithms
    and exact decoded finite values, without rounding or coercing supports. The
    returned ``total`` is minimized.
    """

    validate_tabular_dataset(dataset)
    validate_raw_table(dataset, real_train, label="real_train")
    validate_raw_table(dataset, real_validation, label="real_validation")
    validate_raw_table(dataset, synthetic, label="synthetic")

    continuous_scores: list[float] = []
    finite_scores: list[float] = []
    column_scores: list[ColumnTuningScore] = []
    for column in dataset.columns:
        if column.kind is ColumnKind.CONTINUOUS:
            train_values = real_train[column.name].to_numpy(dtype=np.float64)
            reference_mean = float(np.mean(train_values))
            observed_scale = float(np.std(train_values, ddof=0))
            reference_scale = observed_scale if observed_scale > 0.0 else 1.0
            real_values = (
                real_validation[column.name].to_numpy(dtype=np.float64)
                - reference_mean
            ) / reference_scale
            synthetic_values = (
                synthetic[column.name].to_numpy(dtype=np.float64)
                - reference_mean
            ) / reference_scale
            value = float(
                wasserstein_distance(
                    real_values,
                    synthetic_values,
                )
            )
            metric = TuningMetric.STANDARDIZED_WASSERSTEIN
            continuous_scores.append(value)
        else:
            reference_scale = None
            value = _jensen_shannon_divergence(
                real_validation[column.name],
                synthetic[column.name],
            )
            metric = TuningMetric.JENSEN_SHANNON
            finite_scores.append(value)
        column_scores.append(
            ColumnTuningScore(
                column=column.name,
                kind=column.kind,
                metric=metric,
                value=value,
                reference_scale=reference_scale,
            )
        )

    mean_wasserstein = (
        float(np.mean(continuous_scores)) if continuous_scores else None
    )
    mean_jensen_shannon = (
        float(np.mean(finite_scores)) if finite_scores else None
    )
    total = sum(
        value
        for value in (mean_wasserstein, mean_jensen_shannon)
        if value is not None
    )
    return TuningScore(
        total=float(total),
        mean_wasserstein=mean_wasserstein,
        mean_jensen_shannon=mean_jensen_shannon,
        columns=tuple(column_scores),
    )
