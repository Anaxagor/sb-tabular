"""Tests for clean benchmark-owned holdout and K-fold strategies."""

from __future__ import annotations

import unittest

import pandas as pd

from sbtab.benchmark import (
    ColumnKind,
    ColumnSpec,
    ContractViolation,
    HoldoutConfig,
    KFoldConfig,
    MissingPolicy,
    StratifiedHoldoutConfig,
    StratifiedKFoldConfig,
    TabularDataset,
    TaskType,
    apply_missing_policy,
    make_holdout,
    make_splits,
)


def _classification_dataset() -> TabularDataset:
    labels = ["no", "yes"] * 10
    return TabularDataset(
        name="balanced",
        frame=pd.DataFrame(
            {
                "value": [float(index) for index in range(20)],
                "label": labels,
            }
        ),
        columns=(
            ColumnSpec("value", ColumnKind.CONTINUOUS),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
    )


class SplitStrategyTests(unittest.TestCase):
    """Verify reproducibility, partition integrity, and target stratification."""

    def test_stratified_folds_are_deterministic_and_balanced(self) -> None:
        dataset = _classification_dataset()
        config = StratifiedKFoldConfig(n_splits=5, seed=42)

        first = make_splits(dataset, config)
        second = make_splits(dataset, config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        for fold in first:
            labels = dataset.frame.iloc[list(fold.test_positions)]["label"]
            self.assertEqual(labels.value_counts().to_dict(), {"no": 2, "yes": 2})

    def test_holdout_uses_reference_defaults_and_is_deterministic(self) -> None:
        dataset = _classification_dataset()

        first = make_holdout(dataset, HoldoutConfig())
        second = make_holdout(dataset, HoldoutConfig())

        self.assertEqual(first, second)
        self.assertEqual(len(first.train_positions), 16)
        self.assertEqual(len(first.validation_positions), 4)
        self.assertTrue(
            set(first.train_positions).isdisjoint(first.validation_positions)
        )
        self.assertEqual(
            sorted(first.train_positions + first.validation_positions),
            list(range(20)),
        )

    def test_stratified_holdout_represents_each_class_in_validation(self) -> None:
        dataset = _classification_dataset()

        split = make_holdout(dataset, StratifiedHoldoutConfig())

        validation_labels = dataset.frame.iloc[
            list(split.validation_positions)
        ]["label"]
        self.assertEqual(
            validation_labels.value_counts().to_dict(),
            {"no": 2, "yes": 2},
        )

    def test_holdout_stores_positions_instead_of_dataframe_index_labels(self) -> None:
        dataset = _classification_dataset()
        frame = dataset.frame.copy()
        frame.index = range(100, 120)
        indexed = TabularDataset(
            name=dataset.name,
            frame=frame,
            columns=dataset.columns,
            target=dataset.target,
            task=dataset.task,
        )

        split = make_holdout(indexed, HoldoutConfig())

        self.assertEqual(
            sorted(split.train_positions + split.validation_positions),
            list(range(20)),
        )

    def test_test_folds_partition_every_position_exactly_once(self) -> None:
        dataset = _classification_dataset()

        folds = make_splits(dataset, KFoldConfig(n_splits=5, seed=7))

        all_test_positions = sorted(
            position for fold in folds for position in fold.test_positions
        )
        self.assertEqual(all_test_positions, list(range(len(dataset.frame))))
        for fold in folds:
            self.assertTrue(
                set(fold.train_positions).isdisjoint(fold.test_positions)
            )

    def test_splits_store_positions_instead_of_dataframe_index_labels(self) -> None:
        dataset = _classification_dataset()
        frame = dataset.frame.copy()
        frame.index = range(100, 120)
        indexed = TabularDataset(
            name=dataset.name,
            frame=frame,
            columns=dataset.columns,
            target=dataset.target,
            task=dataset.task,
        )

        folds = make_splits(indexed, KFoldConfig(n_splits=5, seed=42))

        positions = sorted(
            position for fold in folds for position in fold.test_positions
        )
        self.assertEqual(positions, list(range(20)))

    def test_unused_pandas_category_is_not_treated_as_observed_class(self) -> None:
        dataset = _classification_dataset()
        frame = dataset.frame.copy()
        frame["label"] = pd.Categorical(
            frame["label"],
            categories=["no", "yes", "unused"],
        )
        categorical = TabularDataset(
            name=dataset.name,
            frame=frame,
            columns=dataset.columns,
            target=dataset.target,
            task=dataset.task,
        )

        folds = make_splits(
            categorical,
            StratifiedKFoldConfig(n_splits=5, seed=42),
        )

        self.assertEqual(len(folds), 5)

    def test_discrete_non_integer_class_values_are_factorized_for_sklearn(self) -> None:
        dataset = TabularDataset(
            name="float-classes",
            frame=pd.DataFrame({"label": [0.1, 0.2] * 4}),
            columns=(ColumnSpec("label", ColumnKind.DISCRETE),),
            target="label",
            task=TaskType.CLASSIFICATION,
        )

        folds = make_splits(
            dataset,
            StratifiedKFoldConfig(n_splits=2, seed=42),
        )

        self.assertEqual(len(folds), 2)

    def test_mixed_hashable_category_labels_are_factorized_for_sklearn(self) -> None:
        dataset = TabularDataset(
            name="mixed-classes",
            frame=pd.DataFrame({"label": [1, "yes"] * 4}),
            columns=(ColumnSpec("label", ColumnKind.CATEGORICAL),),
            target="label",
            task=TaskType.CLASSIFICATION,
        )

        folds = make_splits(
            dataset,
            StratifiedKFoldConfig(n_splits=2, seed=42),
        )

        self.assertEqual(len(folds), 2)

    def test_stratification_rejects_single_observed_class(self) -> None:
        dataset = TabularDataset(
            name="single-class",
            frame=pd.DataFrame({"label": ["only"] * 6}),
            columns=(ColumnSpec("label", ColumnKind.CATEGORICAL),),
            target="label",
            task=TaskType.CLASSIFICATION,
        )

        with self.assertRaisesRegex(ContractViolation, "at least two"):
            make_splits(
                dataset,
                StratifiedKFoldConfig(n_splits=3, seed=42),
            )

    def test_stratification_requires_classification_target(self) -> None:
        dataset = TabularDataset(
            name="regression",
            frame=pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]}),
            columns=(ColumnSpec("target", ColumnKind.CONTINUOUS),),
            target="target",
            task=TaskType.REGRESSION,
        )

        with self.assertRaisesRegex(ContractViolation, "classification target"):
            make_splits(
                dataset,
                StratifiedKFoldConfig(n_splits=2, seed=42),
            )

        with self.assertRaisesRegex(ContractViolation, "classification target"):
            make_holdout(dataset, StratifiedHoldoutConfig())

    def test_each_class_must_represent_every_stratified_fold(self) -> None:
        dataset = TabularDataset(
            name="rare-class",
            frame=pd.DataFrame(
                {
                    "value": [0.0, 1.0, 2.0, 3.0],
                    "label": ["common", "common", "common", "rare"],
                }
            ),
            columns=(
                ColumnSpec("value", ColumnKind.CONTINUOUS),
                ColumnSpec("label", ColumnKind.CATEGORICAL),
            ),
            target="label",
            task=TaskType.CLASSIFICATION,
        )

        with self.assertRaisesRegex(ContractViolation, "rare"):
            make_splits(
                dataset,
                StratifiedKFoldConfig(n_splits=2, seed=42),
            )

        with self.assertRaisesRegex(ContractViolation, "at least two rows"):
            make_holdout(dataset, StratifiedHoldoutConfig())

    def test_split_rejects_missing_modeled_values_before_sklearn(self) -> None:
        dataset = _classification_dataset()
        frame = dataset.frame.copy()
        frame.loc[0, "value"] = None
        incomplete = TabularDataset(
            name=dataset.name,
            frame=frame,
            columns=dataset.columns,
            target=dataset.target,
            task=dataset.task,
        )

        with self.assertRaisesRegex(ContractViolation, "MissingPolicy"):
            make_splits(incomplete, KFoldConfig(n_splits=2, seed=42))

    def test_split_explicitly_rejects_dataset_emptied_by_complete_case(self) -> None:
        incomplete = TabularDataset(
            name="all-missing",
            frame=pd.DataFrame({"value": [None, None]}, dtype="float64"),
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
        )
        result = apply_missing_policy(incomplete, MissingPolicy.COMPLETE_CASE)

        self.assertEqual(result.report.rows_after, 0)
        with self.assertRaisesRegex(ContractViolation, "exceeds row count 0"):
            make_splits(result.dataset, KFoldConfig(n_splits=2, seed=42))

    def test_split_config_rejects_boolean_seed(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "seed must be an integer"):
            make_splits(
                _classification_dataset(),
                KFoldConfig(n_splits=2, seed=True),  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(ContractViolation, "seed must be an integer"):
            make_holdout(
                _classification_dataset(),
                HoldoutConfig(seed=True),  # type: ignore[arg-type]
            )

    def test_split_config_rejects_negative_seed_before_sklearn(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "range"):
            make_splits(
                _classification_dataset(),
                KFoldConfig(n_splits=2, seed=-1),
            )

    def test_stratification_rejects_continuous_classification_target(self) -> None:
        dataset = TabularDataset(
            name="continuous-class-label",
            frame=pd.DataFrame({"label": [0.0, 0.5, 1.0, 1.5]}),
            columns=(ColumnSpec("label", ColumnKind.CONTINUOUS),),
            target="label",
            task=TaskType.CLASSIFICATION,
        )

        with self.assertRaisesRegex(ContractViolation, "discrete or categorical"):
            make_splits(
                dataset,
                StratifiedKFoldConfig(n_splits=2, seed=42),
            )

        with self.assertRaisesRegex(ContractViolation, "discrete or categorical"):
            make_holdout(dataset, StratifiedHoldoutConfig())

    def test_holdout_rejects_invalid_validation_fraction(self) -> None:
        for fraction in (0.0, 1.0, -0.1):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(
                    ContractViolation,
                    "strictly between zero and one",
                ):
                    make_holdout(
                        _classification_dataset(),
                        HoldoutConfig(validation_fraction=fraction),
                    )

    def test_stratified_holdout_requires_room_for_each_class(self) -> None:
        dataset = _classification_dataset()

        with self.assertRaisesRegex(ContractViolation, "one row per class"):
            make_holdout(
                dataset,
                StratifiedHoldoutConfig(validation_fraction=0.05),
            )


if __name__ == "__main__":
    unittest.main()
