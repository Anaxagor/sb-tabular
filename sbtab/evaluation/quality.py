"""Model-independent final quality metrics for decoded raw tables.

The formulas implement :mod:`docs/benchmark-metrics.md`. Marginal distances
compare a held-out real fold with a train-sized synthetic sample. Association
distances compare within-table relationship matrices; they never align
unrelated real and synthetic rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import normalized_mutual_info_score

from sbtab.benchmark.contracts import ColumnKind, TabularDataset
from sbtab.benchmark.validation import validate_tabular_dataset
from sbtab.evaluation._validation import validate_raw_table


CONTINUOUS_KL_BINS = 50
KL_PSEUDOCOUNT = 1e-12


@dataclass(frozen=True)
class ContinuousColumnQuality:
    """Marginal distances for one decoded continuous column.

    Parameters
    ----------
    column:
        Canonical modeled column name.
    wasserstein:
        One-dimensional Wasserstein distance in decoded raw units.
    kl:
        Histogram ``KL(real || synthetic)`` over 50 shared bins.
    """

    column: str
    wasserstein: float
    kl: float


@dataclass(frozen=True)
class FiniteColumnQuality:
    """Marginal divergence for one decoded finite-support column.

    Parameters
    ----------
    column:
        Canonical modeled column name.
    kind:
        Numeric discrete or categorical semantics used for group aggregation.
    kl:
        Empirical ``KL(real || synthetic)`` over the exact union support.
    """

    column: str
    kind: ColumnKind
    kl: float


@dataclass(frozen=True)
class ContinuousQuality:
    """Fold-level continuous marginals and Pearson association distance.

    Parameters
    ----------
    mean_wasserstein, mean_kl:
        Arithmetic means across ``columns``.
    pearson_frobenius:
        Frobenius distance between off-diagonal Pearson matrices. ``None``
        means that fewer than two continuous columns exist.
    columns:
        Per-column evidence in canonical continuous-column order.
    """

    mean_wasserstein: float
    mean_kl: float
    pearson_frobenius: float | None
    columns: tuple[ContinuousColumnQuality, ...]


@dataclass(frozen=True)
class DiscreteQuality:
    """Fold-level discrete marginals and Spearman association distance.

    Parameters
    ----------
    mean_kl:
        Arithmetic mean over exact-support per-column KL values.
    spearman_frobenius:
        Frobenius distance between off-diagonal rank-correlation matrices.
        ``None`` means that fewer than two discrete columns exist.
    columns:
        Per-column evidence in canonical discrete-column order.
    """

    mean_kl: float
    spearman_frobenius: float | None
    columns: tuple[FiniteColumnQuality, ...]


@dataclass(frozen=True)
class CategoricalQuality:
    """Fold-level categorical marginals and NMI association distance.

    Parameters
    ----------
    mean_kl:
        Arithmetic mean over exact-support per-column KL values.
    nmi_frobenius:
        Frobenius distance between off-diagonal pairwise NMI matrices. ``None``
        means that fewer than two categorical columns exist.
    columns:
        Per-column evidence in canonical categorical-column order.
    """

    mean_kl: float
    nmi_frobenius: float | None
    columns: tuple[FiniteColumnQuality, ...]


@dataclass(frozen=True)
class QualityScore:
    """All applicable statistical-quality metrics for one benchmark fold.

    A semantic group is ``None`` only when the dataset has no modeled columns
    of that kind. The target remains in its declared group.
    """

    continuous: ContinuousQuality | None
    discrete: DiscreteQuality | None
    categorical: CategoricalQuality | None


def _kl_from_counts(real_counts: np.ndarray, synthetic_counts: np.ndarray) -> float:
    real_probability = real_counts.astype(np.float64) + KL_PSEUDOCOUNT
    synthetic_probability = (
        synthetic_counts.astype(np.float64) + KL_PSEUDOCOUNT
    )
    real_probability /= real_probability.sum()
    synthetic_probability /= synthetic_probability.sum()
    return float(
        np.sum(
            real_probability
            * np.log(real_probability / synthetic_probability)
        )
    )


def _continuous_kl(real: pd.Series, synthetic: pd.Series) -> float:
    real_values = real.to_numpy(dtype=np.float64)
    synthetic_values = synthetic.to_numpy(dtype=np.float64)
    lower = float(min(real_values.min(), synthetic_values.min()))
    upper = float(max(real_values.max(), synthetic_values.max()))
    if lower == upper:
        return 0.0
    bins = np.linspace(lower, upper, CONTINUOUS_KL_BINS + 1)
    real_counts, _ = np.histogram(real_values, bins=bins)
    synthetic_counts, _ = np.histogram(synthetic_values, bins=bins)
    return _kl_from_counts(real_counts, synthetic_counts)


def _finite_kl(real: pd.Series, synthetic: pd.Series) -> float:
    combined = pd.concat(
        (real.reset_index(drop=True), synthetic.reset_index(drop=True)),
        ignore_index=True,
    )
    codes, support = pd.factorize(combined, sort=False)
    real_size = len(real)
    real_counts = np.bincount(
        codes[:real_size],
        minlength=len(support),
    )
    synthetic_counts = np.bincount(
        codes[real_size:],
        minlength=len(support),
    )
    return _kl_from_counts(real_counts, synthetic_counts)


def _correlation_frobenius(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    method: str,
) -> float | None:
    if len(columns) < 2:
        return None
    real_matrix = (
        real.loc[:, list(columns)]
        .corr(method=method)
        .to_numpy(dtype=np.float64)
    )
    synthetic_matrix = (
        synthetic.loc[:, list(columns)]
        .corr(method=method)
        .to_numpy(dtype=np.float64)
    )
    real_matrix = np.nan_to_num(real_matrix, nan=0.0)
    synthetic_matrix = np.nan_to_num(synthetic_matrix, nan=0.0)
    np.fill_diagonal(real_matrix, 0.0)
    np.fill_diagonal(synthetic_matrix, 0.0)
    return float(np.linalg.norm(real_matrix - synthetic_matrix, ord="fro"))


def _nmi_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    matrix = np.zeros((len(columns), len(columns)), dtype=np.float64)
    encoded = {
        name: pd.factorize(frame[name], sort=False)[0]
        for name in columns
    }
    for left_index, left_name in enumerate(columns):
        for right_index in range(left_index + 1, len(columns)):
            right_name = columns[right_index]
            value = normalized_mutual_info_score(
                encoded[left_name],
                encoded[right_name],
                average_method="arithmetic",
            )
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
    return matrix


def _nmi_frobenius(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: tuple[str, ...],
) -> float | None:
    if len(columns) < 2:
        return None
    return float(
        np.linalg.norm(
            _nmi_matrix(real, columns) - _nmi_matrix(synthetic, columns),
            ord="fro",
        )
    )


def evaluate_quality(
    dataset: TabularDataset,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> QualityScore:
    """Evaluate final marginal and association quality for one fold.

    Both tables are decoded raw modeled tables in canonical order. They may
    have different positive row counts and are never aligned row by row.
    """

    validate_tabular_dataset(dataset)
    validate_raw_table(dataset, real_test, label="real_test")
    validate_raw_table(dataset, synthetic, label="synthetic")

    continuous_columns = tuple(
        ContinuousColumnQuality(
            column=name,
            wasserstein=float(
                wasserstein_distance(real_test[name], synthetic[name])
            ),
            kl=_continuous_kl(real_test[name], synthetic[name]),
        )
        for name in dataset.continuous_columns
    )
    continuous = (
        ContinuousQuality(
            mean_wasserstein=float(
                np.mean([score.wasserstein for score in continuous_columns])
            ),
            mean_kl=float(
                np.mean([score.kl for score in continuous_columns])
            ),
            pearson_frobenius=_correlation_frobenius(
                real_test,
                synthetic,
                dataset.continuous_columns,
                method="pearson",
            ),
            columns=continuous_columns,
        )
        if continuous_columns
        else None
    )

    discrete_columns = tuple(
        FiniteColumnQuality(
            column=name,
            kind=ColumnKind.DISCRETE,
            kl=_finite_kl(real_test[name], synthetic[name]),
        )
        for name in dataset.discrete_columns
    )
    discrete = (
        DiscreteQuality(
            mean_kl=float(np.mean([score.kl for score in discrete_columns])),
            spearman_frobenius=_correlation_frobenius(
                real_test,
                synthetic,
                dataset.discrete_columns,
                method="spearman",
            ),
            columns=discrete_columns,
        )
        if discrete_columns
        else None
    )

    categorical_columns = tuple(
        FiniteColumnQuality(
            column=name,
            kind=ColumnKind.CATEGORICAL,
            kl=_finite_kl(real_test[name], synthetic[name]),
        )
        for name in dataset.categorical_columns
    )
    categorical = (
        CategoricalQuality(
            mean_kl=float(
                np.mean([score.kl for score in categorical_columns])
            ),
            nmi_frobenius=_nmi_frobenius(
                real_test,
                synthetic,
                dataset.categorical_columns,
            ),
            columns=categorical_columns,
        )
        if categorical_columns
        else None
    )
    return QualityScore(
        continuous=continuous,
        discrete=discrete,
        categorical=categorical,
    )
