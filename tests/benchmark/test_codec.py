"""Tests for the fold-local model codec and its leakage boundaries."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sbtab.benchmark import (
    CategoricalView,
    ColumnKind,
    ColumnSpec,
    ContinuousView,
    ContractViolation,
    DiscreteView,
    InputSpec,
    PreparedSchema,
    PreparedTable,
    TabularDataset,
    TaskType,
    compile_codec,
)


def _mixed_dataset() -> TabularDataset:
    return TabularDataset(
        name="mixed-codec",
        frame=pd.DataFrame(
            {
                "row_id": ["a", "b", "c", "d", "held-out"],
                "amount": [0, 2, 4, 6, 1_000],
                "count": [10, 0, 10, 1, 5],
                "grade": ["high", "low", "medium", "high", "very-high"],
                "segment": ["beta", "alpha", "beta", "alpha", "held-out"],
                "label": ["yes", "no", "yes", "no", "yes"],
            }
        ),
        columns=(
            ColumnSpec("amount", ColumnKind.CONTINUOUS),
            ColumnSpec("count", ColumnKind.DISCRETE),
            ColumnSpec(
                "grade",
                ColumnKind.CATEGORICAL,
                ordered_values=("low", "medium", "high", "very-high"),
            ),
            ColumnSpec("segment", ColumnKind.CATEGORICAL),
            ColumnSpec("label", ColumnKind.CATEGORICAL),
        ),
        target="label",
        task=TaskType.CLASSIFICATION,
        identifier="row_id",
    )


def _msbm_spec() -> InputSpec:
    return InputSpec(
        continuous_view=ContinuousView.STANDARD,
        discrete_view=DiscreteView.FINITE_STATE_CODES,
        categorical_view=CategoricalView.FINITE_STATE_CODES,
    )


class ModelCodecTests(unittest.TestCase):
    """Verify train-only fitting, reversible state spaces, and strict output."""

    def test_msbm_views_are_fitted_on_train_and_keep_canonical_table(self) -> None:
        dataset = _mixed_dataset()
        codec = compile_codec(dataset, _msbm_spec())

        prepared = codec.fit_transform(dataset.frame.iloc[:4].copy())

        self.assertEqual(tuple(prepared.frame.columns), dataset.column_order)
        self.assertNotIn("row_id", prepared.frame.columns)
        self.assertEqual(prepared.frame.index.tolist(), [0, 1, 2, 3])
        np.testing.assert_allclose(
            prepared.frame["amount"].to_numpy(),
            np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(5.0),
        )
        self.assertEqual(prepared.frame["count"].tolist(), [2, 0, 2, 1])
        self.assertEqual(prepared.frame["grade"].tolist(), [2, 0, 1, 2])
        self.assertEqual(prepared.frame["segment"].tolist(), [0, 1, 0, 1])
        self.assertEqual(prepared.frame["label"].tolist(), [0, 1, 0, 1])
        self.assertIs(prepared.schema, codec.schema)
        self.assertEqual(prepared.schema.target_col, "label")

    def test_state_metadata_uses_train_support_and_declared_order(self) -> None:
        dataset = _mixed_dataset()
        prepared = compile_codec(dataset, _msbm_spec()).fit_transform(
            dataset.frame.iloc[:4].copy()
        )

        states = prepared.schema.state_columns
        self.assertEqual(states["count"].cardinality, 3)
        self.assertTrue(states["count"].ordered)
        self.assertEqual(states["grade"].cardinality, 3)
        self.assertTrue(states["grade"].ordered)
        self.assertEqual(states["segment"].cardinality, 2)
        self.assertFalse(states["segment"].ordered)
        self.assertEqual(states["label"].cardinality, 2)
        self.assertFalse(states["label"].ordered)

    def test_compiled_codec_does_not_retain_the_complete_dataset_frame(self) -> None:
        codec = compile_codec(_mixed_dataset(), _msbm_spec())

        retained_frames = [
            value for value in vars(codec).values() if isinstance(value, pd.DataFrame)
        ]
        self.assertEqual(retained_frames, [])

    def test_fit_transform_round_trip_restores_raw_semantic_values(self) -> None:
        dataset = _mixed_dataset()
        train = dataset.frame.iloc[:4].copy()
        codec = compile_codec(dataset, _msbm_spec())
        prepared = codec.fit_transform(train)

        decoded = codec.inverse_transform(prepared)

        self.assertEqual(tuple(decoded.columns), dataset.column_order)
        np.testing.assert_allclose(decoded["amount"], train["amount"])
        for name in ("count", "grade", "segment", "label"):
            self.assertEqual(decoded[name].tolist(), train[name].tolist())

    def test_invalid_generated_state_code_is_rejected_not_clipped(self) -> None:
        dataset = _mixed_dataset()
        codec = compile_codec(dataset, _msbm_spec())
        prepared = codec.fit_transform(dataset.frame.iloc[:4].copy())
        invalid_frame = prepared.frame.copy()
        invalid_frame.loc[0, "segment"] = 2

        with self.assertRaisesRegex(ContractViolation, "invalid codes"):
            codec.inverse_transform(
                PreparedTable(frame=invalid_frame, schema=prepared.schema)
            )

    def test_inverse_requires_the_exact_fold_schema_object(self) -> None:
        dataset = _mixed_dataset()
        codec = compile_codec(dataset, _msbm_spec())
        prepared = codec.fit_transform(dataset.frame.iloc[:4].copy())
        equivalent_schema = PreparedSchema(
            column_order=prepared.schema.column_order,
            continuous_columns=prepared.schema.continuous_columns,
            discrete_columns=prepared.schema.discrete_columns,
            categorical_columns=prepared.schema.categorical_columns,
            target_col=prepared.schema.target_col,
            task_type=prepared.schema.task_type,
            state_columns=prepared.schema.state_columns,
        )

        with self.assertRaisesRegex(ContractViolation, "exact PreparedSchema"):
            codec.inverse_transform(
                PreparedTable(frame=prepared.frame, schema=equivalent_schema)
            )

    def test_raw_finite_views_reject_values_absent_from_train_support(self) -> None:
        dataset = _mixed_dataset()
        raw_spec = InputSpec(
            continuous_view=ContinuousView.RAW,
            discrete_view=DiscreteView.RAW_VALUES,
            categorical_view=CategoricalView.RAW_VALUES,
        )
        codec = compile_codec(dataset, raw_spec)
        prepared = codec.fit_transform(dataset.frame.iloc[:4].copy())
        generated = prepared.frame.copy()
        generated.loc[0, "segment"] = "never-seen"

        with self.assertRaisesRegex(ContractViolation, "absent from train support"):
            codec.inverse_transform(
                PreparedTable(frame=generated, schema=prepared.schema)
            )

    def test_datetime_like_categories_round_trip_in_both_finite_views(self) -> None:
        cases = (
            pd.Series(
                [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-01-01")],
                name="value",
            ),
            pd.Series(
                [pd.Timedelta("2 days"), pd.Timedelta("1 day")],
                name="value",
            ),
        )
        for source in cases:
            with self.subTest(dtype=str(source.dtype)):
                dataset = TabularDataset(
                    name="datetime-like-category",
                    frame=source.to_frame(),
                    columns=(ColumnSpec("value", ColumnKind.CATEGORICAL),),
                )
                for view in (
                    CategoricalView.FINITE_STATE_CODES,
                    CategoricalView.RAW_VALUES,
                ):
                    with self.subTest(view=view):
                        codec = compile_codec(
                            dataset,
                            InputSpec(
                                continuous_view=ContinuousView.UNSUPPORTED,
                                discrete_view=DiscreteView.UNSUPPORTED,
                                categorical_view=view,
                            ),
                        )
                        prepared = codec.fit_transform(dataset.frame.copy())
                        decoded = codec.inverse_transform(prepared)
                        self.assertEqual(decoded["value"].tolist(), source.tolist())

    def test_unsupported_rejects_only_a_non_empty_semantic_group(self) -> None:
        dataset = _mixed_dataset()
        invalid_spec = InputSpec(
            continuous_view=ContinuousView.UNSUPPORTED,
            discrete_view=DiscreteView.FINITE_STATE_CODES,
            categorical_view=CategoricalView.FINITE_STATE_CODES,
        )

        with self.assertRaisesRegex(ContractViolation, "non-empty continuous"):
            compile_codec(dataset, invalid_spec)

        categorical_only = TabularDataset(
            name="categorical-only",
            frame=pd.DataFrame({"value": ["a", "b"]}),
            columns=(ColumnSpec("value", ColumnKind.CATEGORICAL),),
        )
        codec = compile_codec(categorical_only, invalid_spec)
        prepared = codec.fit_transform(categorical_only.frame.copy())
        self.assertEqual(prepared.frame["value"].tolist(), [0, 1])

    def test_codec_is_single_use_and_has_no_held_out_transform_method(self) -> None:
        dataset = _mixed_dataset()
        codec = compile_codec(dataset, _msbm_spec())

        with self.assertRaisesRegex(ContractViolation, "not fitted"):
            _ = codec.schema
        self.assertFalse(hasattr(codec, "transform"))
        codec.fit_transform(dataset.frame.iloc[:4].copy())
        with self.assertRaisesRegex(ContractViolation, "only once"):
            codec.fit_transform(dataset.frame.iloc[:4].copy())

    def test_fit_rejects_missing_values_even_if_policy_was_skipped(self) -> None:
        dataset = _mixed_dataset()
        train = dataset.frame.iloc[:4].copy()
        train.loc[0, "amount"] = None
        codec = compile_codec(dataset, _msbm_spec())

        with self.assertRaisesRegex(ContractViolation, "MissingPolicy"):
            codec.fit_transform(train)

    def test_standard_decode_keeps_real_values_instead_of_rounding_to_raw_dtype(
        self,
    ) -> None:
        dataset = TabularDataset(
            name="integer-storage-continuous-semantics",
            frame=pd.DataFrame({"value": [0, 2]}),
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
        )
        codec = compile_codec(
            dataset,
            InputSpec(
                continuous_view=ContinuousView.STANDARD,
                discrete_view=DiscreteView.UNSUPPORTED,
                categorical_view=CategoricalView.UNSUPPORTED,
            ),
        )
        prepared = codec.fit_transform(dataset.frame.copy())
        generated = PreparedTable(
            frame=pd.DataFrame({"value": [0.5]}),
            schema=prepared.schema,
        )

        decoded = codec.inverse_transform(generated)

        self.assertEqual(decoded["value"].tolist(), [1.5])
        self.assertTrue(pd.api.types.is_float_dtype(decoded["value"].dtype))

    def test_constant_standard_column_uses_unit_scale(self) -> None:
        dataset = TabularDataset(
            name="constant-continuous",
            frame=pd.DataFrame({"value": [5.0, 5.0]}),
            columns=(ColumnSpec("value", ColumnKind.CONTINUOUS),),
        )
        codec = compile_codec(
            dataset,
            InputSpec(
                continuous_view=ContinuousView.STANDARD,
                discrete_view=DiscreteView.UNSUPPORTED,
                categorical_view=CategoricalView.UNSUPPORTED,
            ),
        )
        prepared = codec.fit_transform(dataset.frame.copy())
        self.assertEqual(prepared.frame["value"].tolist(), [0.0, 0.0])
        generated = PreparedTable(
            frame=pd.DataFrame({"value": [2.0]}),
            schema=prepared.schema,
        )

        decoded = codec.inverse_transform(generated)

        self.assertEqual(decoded["value"].tolist(), [7.0])

    def test_inverse_supports_empty_finite_state_sample(self) -> None:
        dataset = _mixed_dataset()
        codec = compile_codec(dataset, _msbm_spec())
        prepared = codec.fit_transform(dataset.frame.iloc[:4].copy())
        empty_sample = PreparedTable(
            frame=prepared.frame.iloc[:0].copy(),
            schema=prepared.schema,
        )

        decoded = codec.inverse_transform(empty_sample)

        self.assertEqual(len(decoded), 0)
        self.assertEqual(tuple(decoded.columns), dataset.column_order)
        self.assertTrue(pd.api.types.is_numeric_dtype(decoded["count"].dtype))


if __name__ == "__main__":
    unittest.main()
