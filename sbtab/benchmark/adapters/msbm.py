"""Thin canonical/native adapter for the mixed-state bridge model.

Canonical -> native mapping
---------------------------
``PreparedSchema.continuous_columns`` become one ``float32`` Torch tensor.
Every named state column, selected in canonical table order, becomes one
``int64`` tensor; the same names provide native cardinalities and ordered flags.
The target is never separated from its semantic block.

Native -> canonical mapping
---------------------------
The native continuous and state samples are checked before DataFrame assembly,
labeled with the fitted block names, and reassembled in exact canonical order.
The returned :class:`PreparedTable` carries the identical fitted schema object.

The adapter does not change MSBM priors, reference transitions, loss weights,
time grid, integration, direction sequence, optimizer, snapshots, or sampling
direction. See ``docs/model-migrations/msbm.md`` for characterization evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sbtab.benchmark.adapter import (
    RunContext,
    validate_sample_request,
)
from sbtab.benchmark.contracts import (
    CategoricalView,
    ContinuousView,
    DiscreteView,
    InputSpec,
    PreparedSchema,
    PreparedTable,
)
from sbtab.benchmark.validation import (
    ContractViolation,
    validate_prepared_table,
)
from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver


class MSBMCompatibilityError(ContractViolation):
    """Raised when a valid prepared table cannot be modeled by current MSBM."""


class MSBMAdapter:
    """Single-fold adapter for the current mixed continuous/state solver.

    The pilot intentionally uses native ``MixedSBMConfig`` defaults. Only
    ``device`` and training ``seed`` are supplied from :class:`RunContext`.
    A separate typed adapter config should be introduced only when a reviewed
    benchmark profile needs non-default model mathematics.
    """

    def __init__(self) -> None:
        self._schema: PreparedSchema | None = None
        self._continuous_names: tuple[str, ...] = ()
        self._state_names: tuple[str, ...] = ()
        self._solver: MixedSBMSolver | None = None

    @property
    def name(self) -> str:
        """Return the stable model-family label used in benchmark artifacts."""

        return "msbm"

    @property
    def input_spec(self) -> InputSpec:
        """Request standard continuous values and codes for every finite state."""

        return InputSpec(
            continuous_view=ContinuousView.STANDARD,
            discrete_view=DiscreteView.FINITE_STATE_CODES,
            categorical_view=CategoricalView.FINITE_STATE_CODES,
        )

    @staticmethod
    def _block_names(
        schema: PreparedSchema,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        continuous_names = schema.continuous_columns
        state_name_set = set(schema.state_columns)
        state_names = tuple(
            name for name in schema.column_order if name in state_name_set
        )
        if not continuous_names:
            raise MSBMCompatibilityError(
                "Current MSBM requires at least one continuous prepared column."
            )
        if not state_names:
            raise MSBMCompatibilityError(
                "Current MSBM requires at least one finite-state prepared column."
            )

        finite_names = set(schema.discrete_columns) | set(
            schema.categorical_columns
        )
        expected_state_names = tuple(
            name
            for name in schema.column_order
            if name in finite_names
        )
        if state_names != expected_state_names:
            raise MSBMCompatibilityError(
                "MSBM requires FINITE_STATE_CODES for every discrete and "
                f"categorical column; states={state_names!r}, "
                f"expected={expected_state_names!r}."
            )
        singleton_nominal = tuple(
            name
            for name in state_names
            if schema.state_columns[name].cardinality == 1
            and not schema.state_columns[name].ordered
        )
        if singleton_nominal:
            raise MSBMCompatibilityError(
                "Current default MSBM reference does not support singleton "
                f"nominal states: {singleton_nominal!r}."
            )
        return continuous_names, state_names

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        """Validate, translate, and fit one native MSBM on a prepared train fold."""

        if self._schema is not None:
            raise ContractViolation("One MSBMAdapter instance can be fitted only once.")
        if not isinstance(context, RunContext):
            raise ContractViolation("context must be RunContext.")
        validate_prepared_table(train)
        if train.frame.empty:
            raise MSBMCompatibilityError("MSBM cannot fit an empty train table.")

        continuous_names, state_names = self._block_names(train.schema)
        with np.errstate(over="ignore", invalid="ignore"):
            continuous_array = train.frame.loc[:, continuous_names].to_numpy(
                dtype=np.float32,
                copy=True,
            )
        unrepresentable = tuple(
            name
            for index, name in enumerate(continuous_names)
            if not bool(np.isfinite(continuous_array[:, index]).all())
        )
        if unrepresentable:
            raise MSBMCompatibilityError(
                "Prepared continuous values cannot be represented as native "
                f"float32 without becoming non-finite: {unrepresentable!r}."
            )
        int64_max = int(np.iinfo(np.int64).max)
        oversized_cardinalities = tuple(
            name
            for name in state_names
            if train.schema.state_columns[name].cardinality > int64_max
        )
        maximum_codes = {
            name: int(train.frame[name].max()) for name in state_names
        }
        oversized_codes = {
            name: code for name, code in maximum_codes.items() if code > int64_max
        }
        if oversized_cardinalities or oversized_codes:
            raise MSBMCompatibilityError(
                "Prepared states cannot be represented by native torch.int64; "
                f"oversized_cardinalities={oversized_cardinalities!r}, "
                f"oversized_codes={oversized_codes!r}."
            )
        cardinalities_exceeding_rows: dict[str, int] = {}
        missing_train_codes: dict[str, tuple[int, ...]] = {}
        for name in state_names:
            cardinality = train.schema.state_columns[name].cardinality
            if cardinality > len(train.frame):
                cardinalities_exceeding_rows[name] = cardinality
                continue
            observed_codes = {int(value) for value in train.frame[name].tolist()}
            missing_codes = tuple(
                sorted(set(range(cardinality)) - observed_codes)
            )
            if missing_codes:
                missing_train_codes[name] = missing_codes
        if cardinalities_exceeding_rows or missing_train_codes:
            raise MSBMCompatibilityError(
                "Prepared train states must realize every code in their "
                "train-observed cardinality; cardinalities_exceeding_rows="
                f"{cardinalities_exceeding_rows!r}, "
                f"missing_codes={missing_train_codes!r}."
            )
        state_array = train.frame.loc[:, state_names].to_numpy(
            dtype=np.int64,
            copy=True,
        )
        ordered_array = np.asarray(
            [train.schema.state_columns[name].ordered for name in state_names],
            dtype=np.bool_,
        )
        cardinalities = [
            train.schema.state_columns[name].cardinality for name in state_names
        ]

        native_config = MixedSBMConfig(
            device=context.device,
            seed=context.seed,
        )
        device = torch.device(context.device)
        train_num = torch.as_tensor(
            continuous_array,
            dtype=torch.float32,
            device=device,
        )
        train_cat = torch.as_tensor(
            state_array,
            dtype=torch.int64,
            device=device,
        )
        is_ordered = torch.as_tensor(
            ordered_array,
            dtype=torch.bool,
            device=device,
        )
        solver = MixedSBMSolver(
            continuous_dim=len(continuous_names),
            cardinalities=cardinalities,
            is_ordered=is_ordered,
            cfg=native_config,
        )
        solver.fit(train_num, train_cat)

        self._schema = train.schema
        self._continuous_names = continuous_names
        self._state_names = state_names
        self._solver = solver

    def sample(self, n: int, seed: int) -> PreparedTable:
        """Generate and validate a complete canonical prepared MSBM sample."""

        validate_sample_request(n, seed)
        if self._schema is None or self._solver is None:
            raise ContractViolation("Call MSBMAdapter.fit() before sample().")
        if n == 0:
            return self._empty_sample()

        generated_num, generated_state = self._solver.sample(
            n_samples=n,
            seed=seed,
        )
        self._validate_native_output(
            generated_num,
            expected_shape=(n, len(self._continuous_names)),
            expected_dtype=torch.float32,
            block_name="continuous",
        )
        self._validate_native_output(
            generated_state,
            expected_shape=(n, len(self._state_names)),
            expected_dtype=torch.int64,
            block_name="state",
        )

        continuous_array = generated_num.detach().cpu().numpy()
        state_array = generated_state.detach().cpu().numpy()
        blocks = {
            **{
                name: continuous_array[:, index]
                for index, name in enumerate(self._continuous_names)
            },
            **{
                name: state_array[:, index]
                for index, name in enumerate(self._state_names)
            },
        }
        frame = pd.DataFrame(blocks).loc[:, self._schema.column_order]
        sample = PreparedTable(frame=frame, schema=self._schema)
        validate_prepared_table(sample, expected_rows=n)
        return sample

    @staticmethod
    def _validate_native_output(
        tensor: torch.Tensor,
        *,
        expected_shape: tuple[int, int],
        expected_dtype: torch.dtype,
        block_name: str,
    ) -> None:
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise ContractViolation(
                f"Native MSBM {block_name} sample has shape {actual_shape!r}, "
                f"expected {expected_shape!r}."
            )
        if tensor.dtype != expected_dtype:
            raise ContractViolation(
                f"Native MSBM {block_name} sample has dtype {tensor.dtype!r}, "
                f"expected {expected_dtype!r}."
            )

    def _empty_sample(self) -> PreparedTable:
        if self._schema is None:
            raise ContractViolation("Call MSBMAdapter.fit() before sample().")
        continuous_set = set(self._continuous_names)
        frame = pd.DataFrame(
            {
                name: pd.Series(
                    dtype="float32" if name in continuous_set else "int64"
                )
                for name in self._schema.column_order
            }
        )
        sample = PreparedTable(frame=frame, schema=self._schema)
        validate_prepared_table(sample, expected_rows=0)
        return sample
