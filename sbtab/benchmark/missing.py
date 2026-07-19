"""One model-independent missing-value policy applied before splitting.

The benchmark applies this module once to a raw :class:`TabularDataset`. An
adapter never receives missing-policy configuration and cannot override the
result. Identifier values are deliberately excluded from row filtering because
identifiers never enter the modeled table.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from sbtab.benchmark.contracts import TabularDataset, TaskType
from sbtab.benchmark.validation import ContractViolation, validate_tabular_dataset


class MissingPolicy(str, Enum):
    """Global action taken when modeled raw columns contain missing values.

    ``ERROR`` is the safe default for future benchmark configuration and stops
    with a complete report. ``COMPLETE_CASE`` removes every row missing at
    least one modeled value before any train/test split is created.
    """

    ERROR = "error"
    COMPLETE_CASE = "complete_case"


@dataclass(frozen=True)
class ClassCount:
    """Count of one raw classification-target value in a missing-data report."""

    label: object
    count: int

    def __post_init__(self) -> None:
        """Reject impossible artifact counts at construction."""

        if not isinstance(self.label, Hashable):
            raise ContractViolation("ClassCount.label must be hashable.")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise ContractViolation("ClassCount.count must be an integer.")
        if self.count < 0:
            raise ContractViolation("ClassCount.count must be non-negative.")


@dataclass(frozen=True)
class MissingReport:
    """Auditable effect of one global missing-value policy application.

    Parameters
    ----------
    policy:
        Policy that produced the report.
    rows_before, rows_after:
        Raw row counts before and after applying the policy.
    dropped_count, dropped_fraction:
        Absolute and relative number of removed rows. ``ERROR`` never removes
        rows, including when it raises :class:`MissingValuesError`.
    missing_by_column:
        Pre-policy missing counts for every modeled column in canonical order.
        The optional identifier is not included.
    class_counts_before, class_counts_after:
        Raw classification-target counts when the dataset declares a
        classification task; otherwise ``None``.
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
        """Validate and snapshot report evidence at construction."""

        if not isinstance(self.policy, MissingPolicy):
            raise ContractViolation("MissingReport.policy must be MissingPolicy.")
        for field_name in ("rows_before", "rows_after", "dropped_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractViolation(
                    f"MissingReport.{field_name} must be an integer."
                )
            if value < 0:
                raise ContractViolation(
                    f"MissingReport.{field_name} must be non-negative."
                )
        if self.rows_after > self.rows_before:
            raise ContractViolation(
                "MissingReport.rows_after cannot exceed rows_before."
            )
        expected_dropped = self.rows_before - self.rows_after
        if self.dropped_count != expected_dropped:
            raise ContractViolation(
                "MissingReport.dropped_count must equal rows_before - rows_after."
            )
        if isinstance(self.dropped_fraction, bool) or not isinstance(
            self.dropped_fraction,
            (int, float),
        ):
            raise ContractViolation(
                "MissingReport.dropped_fraction must be a finite number."
            )
        expected_fraction = (
            expected_dropped / self.rows_before if self.rows_before else 0.0
        )
        if not math.isfinite(float(self.dropped_fraction)) or not math.isclose(
            float(self.dropped_fraction),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ContractViolation(
                "MissingReport.dropped_fraction is inconsistent with row counts."
            )
        if not isinstance(self.missing_by_column, Mapping):
            raise ContractViolation(
                "MissingReport.missing_by_column must be a mapping."
            )
        missing_by_column = dict(self.missing_by_column)
        for name, count in missing_by_column.items():
            if not isinstance(name, str) or not name:
                raise ContractViolation(
                    "MissingReport missing-count keys must be column names."
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ContractViolation(
                    f"Missing count for {name!r} must be a non-negative integer."
                )
            if count > self.rows_before:
                raise ContractViolation(
                    f"Missing count for {name!r} cannot exceed rows_before."
                )

        if self.policy is MissingPolicy.ERROR and self.dropped_count:
            raise ContractViolation(
                "MissingPolicy.ERROR reports must not contain removed rows."
            )
        if self.policy is MissingPolicy.COMPLETE_CASE:
            total_missing = sum(missing_by_column.values())
            largest_column_count = max(missing_by_column.values(), default=0)
            if not largest_column_count <= self.dropped_count <= total_missing:
                raise ContractViolation(
                    "COMPLETE_CASE dropped_count must be consistent with "
                    "modeled-column missing counts."
                )

        if (self.class_counts_before is None) != (self.class_counts_after is None):
            raise ContractViolation(
                "MissingReport class counts must be both present or both absent."
            )
        for field_name, counts, expected_rows in (
            ("class_counts_before", self.class_counts_before, self.rows_before),
            ("class_counts_after", self.class_counts_after, self.rows_after),
        ):
            if counts is None:
                continue
            if not isinstance(counts, tuple) or not all(
                isinstance(count, ClassCount) for count in counts
            ):
                raise ContractViolation(
                    f"MissingReport.{field_name} must be a tuple of ClassCount."
                )
            if sum(count.count for count in counts) != expected_rows:
                raise ContractViolation(
                    f"MissingReport.{field_name} must count every row."
                )

        object.__setattr__(
            self,
            "missing_by_column",
            MappingProxyType(missing_by_column),
        )


@dataclass(frozen=True)
class MissingPolicyResult:
    """Filtered dataset paired with the report proving how it was obtained."""

    dataset: TabularDataset
    report: MissingReport

    def __post_init__(self) -> None:
        """Ensure the returned dataset agrees with its audit report."""

        if not isinstance(self.dataset, TabularDataset):
            raise ContractViolation(
                "MissingPolicyResult.dataset must be TabularDataset."
            )
        if not isinstance(self.report, MissingReport):
            raise ContractViolation(
                "MissingPolicyResult.report must be MissingReport."
            )
        if len(self.dataset.frame) != self.report.rows_after:
            raise ContractViolation(
                "MissingPolicyResult dataset row count disagrees with report."
            )


class MissingValuesError(ContractViolation):
    """Raised by ``ERROR`` with a machine-readable missing-data report."""

    def __init__(self, report: MissingReport):
        if not isinstance(report, MissingReport):
            raise TypeError("report must be MissingReport.")
        columns = {
            name: count
            for name, count in report.missing_by_column.items()
            if count > 0
        }
        super().__init__(
            "Modeled columns contain missing values while MissingPolicy.ERROR "
            f"is active: {columns!r}."
        )
        self.report: MissingReport = report


def _class_counts(dataset: TabularDataset) -> tuple[ClassCount, ...] | None:
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
    rows_before = len(source.frame)
    rows_after = len(result.frame)
    dropped_count = rows_before - rows_after
    dropped_fraction = dropped_count / rows_before if rows_before else 0.0
    return MissingReport(
        policy=policy,
        rows_before=rows_before,
        rows_after=rows_after,
        dropped_count=dropped_count,
        dropped_fraction=dropped_fraction,
        missing_by_column=missing_by_column,
        class_counts_before=_class_counts(source),
        class_counts_after=_class_counts(result),
    )


def apply_missing_policy(
    dataset: TabularDataset,
    policy: MissingPolicy,
) -> MissingPolicyResult:
    """Apply one explicit missing policy across all modeled columns.

    The input dataset is validated and never mutated. ``COMPLETE_CASE`` returns
    a new dataset containing a copied filtered frame. ``ERROR`` returns the
    original dataset when no modeled value is missing and otherwise raises with
    a complete :class:`MissingReport`.
    """

    validate_tabular_dataset(dataset)
    if not isinstance(policy, MissingPolicy):
        raise ContractViolation(
            f"policy must be MissingPolicy, got {policy!r}."
        )

    modeled_columns = dataset.column_order
    missing_by_column = {
        name: int(dataset.frame[name].isna().sum()) for name in modeled_columns
    }
    has_missing = any(count > 0 for count in missing_by_column.values())

    if policy is MissingPolicy.ERROR:
        report = _build_report(
            source=dataset,
            result=dataset,
            policy=policy,
            missing_by_column=missing_by_column,
        )
        if has_missing:
            raise MissingValuesError(report)
        return MissingPolicyResult(dataset=dataset, report=report)

    keep_mask = ~dataset.frame.loc[:, list(modeled_columns)].isna().any(axis=1)
    filtered_frame = dataset.frame.loc[keep_mask].copy()
    filtered_dataset = TabularDataset(
        name=dataset.name,
        frame=filtered_frame,
        columns=dataset.columns,
        target=dataset.target,
        task=dataset.task,
        identifier=dataset.identifier,
    )
    validate_tabular_dataset(filtered_dataset)
    report = _build_report(
        source=dataset,
        result=filtered_dataset,
        policy=policy,
        missing_by_column=missing_by_column,
    )
    return MissingPolicyResult(dataset=filtered_dataset, report=report)
