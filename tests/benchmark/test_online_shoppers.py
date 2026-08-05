"""Tests for the explicit UCI Online Shoppers dataset declaration."""

from __future__ import annotations

import unittest

import pandas as pd

from sbtab.benchmark import ColumnKind, ContractViolation, TaskType
from sbtab.benchmark.datasets import (
    ONLINE_SHOPPERS_COLUMNS,
    ONLINE_SHOPPERS_TARGET,
    ONLINE_SHOPPERS_UCI_ID,
    make_online_shoppers_dataset,
)


def _online_shoppers_frame() -> pd.DataFrame:
    """Return source names with representative source-compatible values."""

    return pd.DataFrame(
        {
            "Administrative": [0, 2],
            "Administrative_Duration": [0.0, 64.5],
            "Informational": [0, 1],
            "Informational_Duration": [0.0, 12.0],
            "ProductRelated": [1, 18],
            "ProductRelated_Duration": [0.0, 740.2],
            "BounceRates": [0.2, 0.01],
            "ExitRates": [0.2, 0.03],
            "PageValues": [0.0, 12.4],
            "SpecialDay": [0.0, 0.4],
            "Month": ["Feb", "May"],
            "OperatingSystems": [1, 2],
            "Browser": [1, 2],
            "Region": [1, 3],
            "TrafficType": [1, 4],
            "VisitorType": ["Returning_Visitor", "New_Visitor"],
            "Weekend": [False, True],
            "Revenue": [False, True],
        }
    )


class OnlineShoppersDeclarationTests(unittest.TestCase):
    """Verify source identity and approved semantics without network access."""

    def test_source_order_and_revenue_target_are_explicit(self) -> None:
        frame = _online_shoppers_frame()

        dataset = make_online_shoppers_dataset(frame)

        self.assertEqual(ONLINE_SHOPPERS_UCI_ID, 468)
        self.assertEqual(ONLINE_SHOPPERS_TARGET, "Revenue")
        self.assertIs(dataset.frame, frame)
        self.assertEqual(dataset.column_order, tuple(frame.columns))
        self.assertEqual(dataset.target, ONLINE_SHOPPERS_TARGET)
        self.assertEqual(dataset.task, TaskType.CLASSIFICATION)
        self.assertIsNone(dataset.identifier)
        self.assertEqual(dataset.column("Revenue").kind, ColumnKind.CATEGORICAL)

    def test_semantic_groups_follow_meaning_not_storage_dtype(self) -> None:
        dataset = make_online_shoppers_dataset(_online_shoppers_frame())

        self.assertEqual(
            dataset.continuous_columns,
            (
                "Administrative_Duration",
                "Informational_Duration",
                "ProductRelated_Duration",
                "BounceRates",
                "ExitRates",
                "PageValues",
                "SpecialDay",
            ),
        )
        self.assertEqual(
            dataset.discrete_columns,
            ("Administrative", "Informational", "ProductRelated"),
        )
        self.assertEqual(
            dataset.categorical_columns,
            (
                "Month",
                "OperatingSystems",
                "Browser",
                "Region",
                "TrafficType",
                "VisitorType",
                "Weekend",
                "Revenue",
            ),
        )
        self.assertTrue(
            all(column.ordered_values is None for column in dataset.columns)
        )

    def test_canonical_order_does_not_depend_on_dataframe_column_order(self) -> None:
        frame = _online_shoppers_frame()
        reversed_frame = frame.loc[:, list(reversed(frame.columns))]

        dataset = make_online_shoppers_dataset(reversed_frame)

        self.assertEqual(
            dataset.column_order,
            tuple(column.name for column in ONLINE_SHOPPERS_COLUMNS),
        )
        self.assertNotEqual(dataset.column_order, tuple(reversed_frame.columns))

    def test_source_schema_drift_is_not_silently_accepted(self) -> None:
        cases = (
            (
                _online_shoppers_frame().drop(columns=["Revenue"]),
                "Revenue",
            ),
            (
                _online_shoppers_frame().assign(legacy_target=[1, 2]),
                "legacy_target",
            ),
        )
        for frame, expected_column in cases:
            with self.subTest(expected_column=expected_column):
                with self.assertRaisesRegex(ContractViolation, expected_column):
                    make_online_shoppers_dataset(frame)


if __name__ == "__main__":
    unittest.main()
