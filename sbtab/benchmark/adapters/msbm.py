"""Thin canonical/native adapter for the mixed-state bridge model.

Canonical -> native mapping
---------------------------
``PreparedSchema.continuous_columns`` become one ``float32`` Torch tensor.
Every named state column, selected in canonical table order, becomes one
``int64`` tensor; the same names provide native cardinalities and ordered flags.
The target is never separated from its semantic block.

Native -> canonical mapping
---------------------------
The native continuous and state samples are labeled with the fitted block names
and reassembled in exact canonical order. The returned :class:`PreparedTable`
carries the identical fitted schema object; shared decoding validates it.

The adapter does not change MSBM priors, reference transitions, loss weights,
time grid, integration, direction sequence, optimizer, snapshots, or sampling
direction. See ``docs/model-migrations/msbm.md`` for characterization evidence.
"""

from __future__ import annotations

import pandas as pd
import torch

from sbtab.benchmark.adapter import RunContext
from sbtab.benchmark.contracts import (
    CategoricalView,
    ContinuousView,
    DiscreteView,
    InputSpec,
    PreparedSchema,
    PreparedTable,
)
from sbtab.benchmark.validation import ContractViolation
from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver


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

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        """Translate one prepared train fold and fit a native MSBM."""

        continuous_names = train.schema.continuous_columns
        state_names = tuple(
            name for name in train.schema.column_order
            if name in train.schema.state_columns
        )
        cardinalities = [
            train.schema.state_columns[name].cardinality for name in state_names
        ]

        native_config = MixedSBMConfig(
            device=context.device,
            seed=context.seed,
        )
        device = torch.device(context.device)
        train_num = torch.tensor(
            train.frame.loc[:, continuous_names].to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        train_cat = torch.tensor(
            train.frame.loc[:, state_names].to_numpy(),
            dtype=torch.int64,
            device=device,
        )
        is_ordered = torch.tensor(
            [train.schema.state_columns[name].ordered for name in state_names],
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
        """Generate and reassemble one complete canonical prepared sample."""

        if self._schema is None or self._solver is None:
            raise ContractViolation("Call MSBMAdapter.fit() before sample().")

        generated_num, generated_state = self._solver.sample(
            n_samples=n,
            seed=seed,
        )
        native_frame = pd.concat(
            (
                pd.DataFrame(
                    generated_num.detach().cpu().numpy(),
                    columns=self._continuous_names,
                ),
                pd.DataFrame(
                    generated_state.detach().cpu().numpy(),
                    columns=self._state_names,
                ),
            ),
            axis=1,
        )
        return PreparedTable(
            frame=native_frame.loc[:, self._schema.column_order],
            schema=self._schema,
        )
