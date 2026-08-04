"""Tests for the explicit Online Shoppers pilot declaration."""

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
    """Return a tiny raw fixture with the real UCI column names and semantics."""

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
    """Verify the approved pilot schema without downloading data."""

    def test_declaration_preserves_real_column_order_and_target(self) -> None:
        frame = _online_shoppers_frame()

        dataset = make_online_shoppers_dataset(frame)

        self.assertEqual(ONLINE_SHOPPERS_UCI_ID, 468)
        self.assertEqual(dataset.column_order, tuple(frame.columns))
        self.assertEqual(dataset.target, ONLINE_SHOPPERS_TARGET)
        self.assertEqual(dataset.task, TaskType.CLASSIFICATION)
        self.assertEqual(dataset.column("Revenue").kind, ColumnKind.CATEGORICAL)

    def test_page_counts_are_discrete_and_nominal_columns_have_no_order(self) -> None:
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
        self.assertIsNone(dataset.column("Month").ordered_values)
        self.assertIsNone(dataset.column("Revenue").ordered_values)

    def test_static_specs_include_every_feature_and_target_once(self) -> None:
        names = tuple(column.name for column in ONLINE_SHOPPERS_COLUMNS)

        self.assertEqual(len(names), 18)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[-1], ONLINE_SHOPPERS_TARGET)

    def test_missing_source_column_is_rejected_with_name(self) -> None:
        frame = _online_shoppers_frame().drop(columns=["Revenue"])

        with self.assertRaisesRegex(ContractViolation, "Revenue"):
            make_online_shoppers_dataset(frame)

    def test_unapproved_extra_column_is_not_silently_ignored(self) -> None:
        frame = _online_shoppers_frame().assign(legacy_target=[1, 2])

        with self.assertRaisesRegex(ContractViolation, "legacy_target"):
            make_online_shoppers_dataset(frame)


if __name__ == "__main__":
    unittest.main()
