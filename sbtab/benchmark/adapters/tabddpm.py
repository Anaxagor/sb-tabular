"""Thin canonical/native adapter for the repository's TabDDPM baseline.

Canonical -> native mapping
---------------------------
Standardized ``PreparedSchema.continuous_columns`` become one ``float32``
Torch tensor for the Gaussian diffusion block. Every name in
``PreparedSchema.column_order`` that has ``state_columns`` metadata becomes one
``int64`` tensor column for the multinomial block. The same ordered names
provide real per-column cardinalities. TabDDPM treats those states
symmetrically and does not consume the metadata's ordinal flag. Target remains
in whichever semantic block its column kind selected; it is never separated as
a conditioning label.

Native -> canonical mapping
---------------------------
The native solver returns separate numerical and state tensors. The adapter
labels them with the fitted block names and reassembles a DataFrame in exact
``PreparedSchema.column_order``. It returns the identical fitted schema object.

The adapter does not change the native denoiser, Gaussian or multinomial loss,
beta schedule, optimizer, fixed-step learning-rate annealing, EMA timing, or
ancestral sampling. It performs no generic preprocessing, raw decoding,
clipping, rounding, identifier handling, splitting, tuning, or evaluation.
"""

from __future__ import annotations

import copy

import pandas as pd
import torch

from sbtab.baselines.tabddpm.native import TabDDPMConfig, TabDDPMSolver
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


class TabDDPMAdapter:
    """Single-fold adapter for unconditional joint TabDDPM generation.

    ``config`` is a fixed native configuration selected before final
    cross-validation. Per-fold ``device`` and training ``seed`` always come
    from :class:`RunContext`. ``use_ema_for_sampling`` remains a typed field on
    :class:`TabDDPMConfig` because the shared sampling protocol intentionally
    has no model-specific keyword arguments.
    """

    def __init__(self, config: TabDDPMConfig | None = None) -> None:
        self._config = copy.deepcopy(
            config if config is not None else TabDDPMConfig()
        )
        self._schema: PreparedSchema | None = None
        self._continuous_names: tuple[str, ...] = ()
        self._state_names: tuple[str, ...] = ()
        self._solver: TabDDPMSolver | None = None

    @property
    def name(self) -> str:
        """Return the stable model-family label used in benchmark artifacts."""

        return "tabddpm"

    @property
    def input_spec(self) -> InputSpec:
        """Request benchmark-standard values and finite-state codes."""

        return InputSpec(
            continuous_view=ContinuousView.STANDARD,
            discrete_view=DiscreteView.FINITE_STATE_CODES,
            categorical_view=CategoricalView.FINITE_STATE_CODES,
        )

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        """Translate one prepared train fold and fit native TabDDPM."""

        continuous_names = train.schema.continuous_columns
        state_names = tuple(
            name
            for name in train.schema.column_order
            if name in train.schema.state_columns
        )
        cardinalities = [
            train.schema.state_columns[name].cardinality for name in state_names
        ]

        native_config = copy.deepcopy(self._config)
        native_config.device = context.device
        native_config.seed = context.seed
        device = torch.device(context.device)
        train_num = torch.tensor(
            train.frame.loc[:, continuous_names].to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        train_state = torch.tensor(
            train.frame.loc[:, state_names].to_numpy(),
            dtype=torch.int64,
            device=device,
        )
        solver = TabDDPMSolver(
            num_numerical_features=len(continuous_names),
            cardinalities=cardinalities,
            cfg=native_config,
        )
        solver.fit(train_num, train_state)

        self._schema = train.schema
        self._continuous_names = continuous_names
        self._state_names = state_names
        self._solver = solver

    def sample(self, n: int, seed: int) -> PreparedTable:
        """Generate and reassemble one complete canonical prepared sample."""

        if self._schema is None or self._solver is None:
            raise ContractViolation("Call TabDDPMAdapter.fit() before sample().")

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
