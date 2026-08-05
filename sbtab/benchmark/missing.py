"""One model-independent missing-value policy applied before splitting.

The policy sees the complete raw :class:`TabularDataset` and is applied once
for an experiment. It considers every modeled column, including target, but
ignores the optional identifier because identifiers never enter a model.

This module does not impute values, create splits, or expose model-specific
fallbacks. Its result contains both the retained raw dataset and immutable
audit evidence describing the policy's effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from sbtab.benchmark.contracts import TabularDataset, TaskType
from sbtab.benchmark.validation import ContractViolation, validate_tabular_dataset


class MissingPolicy(str, Enum):
    """Global action for missing values in modeled raw columns.

    ``ERROR`` preserves every row and raises :class:`MissingValuesError` when
    at least one modeled value is missing. ``COMPLETE_CASE`` removes each row
    missing any modeled value. No enum member delegates handling to a model.
    """

    ERROR = "error"
    COMPLETE_CASE = "complete_case"


@dataclass(frozen=True)
class ClassCount:
    """One raw classification-target value and its number of rows.

    ``label`` is retained in its raw scalar representation. A missing target
    is therefore visible as a missing label in the pre-policy distribution.
    ``count`` is the number of source or retained rows carrying that value.
    """

    label: object
    count: int


@dataclass(frozen=True)
class MissingReport:
    """Audit evidence produced by one missing-policy application.

    Parameters
    ----------
    policy:
        Explicit policy used for this result.
    rows_before, rows_after:
        Raw row counts before and after applying the policy. ``ERROR`` reports
        identical values even when it raises.
    dropped_count, dropped_fraction:
        Number and fraction of source rows removed. The fraction is zero for
        an empty source table rather than undefined.
    missing_by_column:
        Pre-policy missing count for every modeled column in canonical order.
        The optional identifier is never included. The mapping is snapshotted
        and exposed read-only.
    class_counts_before, class_counts_after:
        Raw target counts for classification datasets. They are ``None`` for
        regression and datasets without a target.
    """

    policy: MissingPolicy
    rows_before: int
    rows_after: int
    dropped_count: int
    dropped_fraction: float
    missing_by_column: Mapping[str, int] = field(default_factory=dict)
    class_counts_before: tuple[ClassCount, ...] | None = None
    class_counts_after: tuple[ClassCount, ...] | None = None

    def __post_init__(self) -> None:
        """Detach missing counts from the mutable builder dictionary."""

        object.__setattr__(
            self,
            "missing_by_column",
            MappingProxyType(dict(self.missing_by_column)),
        )


@dataclass(frozen=True)
class MissingPolicyResult:
    """Post-policy raw dataset paired with its immutable audit report.

    ``dataset`` is the original object for a successful ``ERROR`` check and a
    new object with a copied filtered frame for ``COMPLETE_CASE``. ``report``
    always describes the original input and the returned dataset.
    """

    dataset: TabularDataset
    report: MissingReport


class MissingValuesError(ContractViolation):
    """Failure raised by ``ERROR`` with a machine-readable report attached."""

    def __init__(self, report: MissingReport):
        columns_with_missing = {
            name: count
            for name, count in report.missing_by_column.items()
            if count > 0
        }
        super().__init__(
            "Modeled columns contain missing values while MissingPolicy.ERROR "
            f"is active: {columns_with_missing!r}."
        )
        self.report = report


def _class_counts(dataset: TabularDataset) -> tuple[ClassCount, ...] | None:
    """Return raw classification counts, including a missing-target group."""

    if dataset.task is not TaskType.CLASSIFICATION or dataset.target is None:
        return None
    counts = dataset.frame[dataset.target].value_counts(dropna=False, sort=False)
    return tuple(
        ClassCount(label=label, count=int(count))
        for label, count in counts.items()
        if int(count) > 0
    )


def _build_report(
    *,
    source: TabularDataset,
    result: TabularDataset,
    policy: MissingPolicy,
    missing_by_column: Mapping[str, int],
) -> MissingReport:
    """Derive report fields from source and result instead of accepting them."""

    rows_before = len(source.frame)
    rows_after = len(result.frame)
    dropped_count = rows_before - rows_after
    return MissingReport(
        policy=policy,
        rows_before=rows_before,
        rows_after=rows_after,
        dropped_count=dropped_count,
        dropped_fraction=(
            dropped_count / rows_before if rows_before else 0.0
        ),
        missing_by_column=missing_by_column,
        class_counts_before=_class_counts(source),
        class_counts_after=_class_counts(result),
    )


def apply_missing_policy(
    dataset: TabularDataset,
    policy: MissingPolicy,
) -> MissingPolicyResult:
    """Apply one explicit policy across all modeled columns before splitting.

    The input dataset is validated and never mutated. ``ERROR`` returns the
    original object when modeled values are complete; otherwise it raises with
    the report it would have returned. ``COMPLETE_CASE`` returns a new dataset
    whose copied frame retains only rows complete across ``dataset.columns``.

    The caller must select a :class:`MissingPolicy` explicitly. This prevents a
    string typo or a model-local fallback from changing benchmark rows.
    """

    if not isinstance(policy, MissingPolicy):
        raise ContractViolation(
            f"policy must be MissingPolicy, got {policy!r}."
        )
    validate_tabular_dataset(dataset)

    modeled_columns = dataset.column_order
    missing_by_column = {
        name: int(dataset.frame[name].isna().sum()) for name in modeled_columns
    }

    if policy is MissingPolicy.ERROR:
        report = _build_report(
            source=dataset,
            result=dataset,
            policy=policy,
            missing_by_column=missing_by_column,
        )
        if any(count > 0 for count in missing_by_column.values()):
            raise MissingValuesError(report)
        return MissingPolicyResult(dataset=dataset, report=report)

    complete_rows = ~dataset.frame.loc[
        :,
        list(modeled_columns),
    ].isna().any(axis=1)
    filtered_dataset = TabularDataset(
        name=dataset.name,
        frame=dataset.frame.loc[complete_rows].copy(),
        columns=dataset.columns,
        target=dataset.target,
        task=dataset.task,
        identifier=dataset.identifier,
    )
    validate_tabular_dataset(filtered_dataset)
    return MissingPolicyResult(
        dataset=filtered_dataset,
        report=_build_report(
            source=dataset,
            result=filtered_dataset,
            policy=policy,
            missing_by_column=missing_by_column,
        ),
    )
