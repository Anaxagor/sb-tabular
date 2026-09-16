"""Thin canonical/native adapter for the repository's TabDDPM baseline.

Canonical -> native mapping
---------------------------
Standardized continuous columns and raw integer-valued discrete columns form
one ``float32`` tensor for Gaussian diffusion, in canonical column order.
Only categorical columns become ``int64`` codes for multinomial diffusion;
their named metadata supplies per-column cardinalities. Categorical states
are symmetric. Target remains in the block selected by its declared kind.

Native -> canonical mapping
---------------------------
The native solver returns separate numerical and categorical tensors. Discrete
Gaussian outputs are quantized with ``np.rint`` (ties to even), without range
clipping or projection onto train support. They remain floating-point numbers
so non-finite output reaches shared validation unchanged. Continuous outputs
and categorical codes are not rounded. The adapter reassembles the exact
canonical order and returns the identical fitted schema object. Fractional
discrete training values are unsupported by this integer-output convention.

The adapter does not change the native denoiser, Gaussian or multinomial loss,
beta schedule, optimizer, fixed-step learning-rate annealing, EMA timing, or
ancestral sampling. Routing and discrete quantization are explicit benchmark
integration choices. The adapter fits no generic preprocessing and performs
no raw decoding, clipping, splitting, tuning, or evaluation.
"""

from __future__ import annotations

import copy

import numpy as np
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
        self._numeric_names: tuple[str, ...] = ()
        self._state_names: tuple[str, ...] = ()
        self._solver: TabDDPMSolver | None = None

    @property
    def name(self) -> str:
        """Return the stable model-family label used in benchmark artifacts."""

        return "tabddpm"

    @property
    def input_spec(self) -> InputSpec:
        """Request standard continuous, raw discrete, and categorical codes."""

        return InputSpec(
            continuous_view=ContinuousView.STANDARD,
            discrete_view=DiscreteView.RAW_VALUES,
            categorical_view=CategoricalView.FINITE_STATE_CODES,
        )

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        """Translate one prepared train fold and fit native TabDDPM."""

        numeric_names = tuple(
            name
            for name in train.schema.column_order
            if name in train.schema.continuous_columns
            or name in train.schema.discrete_columns
        )
        state_names = train.schema.categorical_columns
        cardinalities = [
            train.schema.state_columns[name].cardinality for name in state_names
        ]

        # This is a model-specific support restriction: generic discrete data
        # may be fractional, but this adapter's output uses integer rounding.
        for name in train.schema.discrete_columns:
            values = train.frame[name].to_numpy(dtype=np.float64)
            if not np.equal(values, np.rint(values)).all():
                raise ContractViolation(
                    f"TabDDPM integer rounding requires integer-valued discrete "
                    f"train column {name!r}; fractional values are unsupported."
                )

        native_config = copy.deepcopy(self._config)
        native_config.device = context.device
        native_config.seed = context.seed
        device = torch.device(context.device)
        train_num = torch.tensor(
            train.frame.loc[:, numeric_names].to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        train_state = torch.tensor(
            train.frame.loc[:, state_names].to_numpy(),
            dtype=torch.int64,
            device=device,
        )
        solver = TabDDPMSolver(
            num_numerical_features=len(numeric_names),
            cardinalities=cardinalities,
            cfg=native_config,
        )
        solver.fit(train_num, train_state)

        self._schema = train.schema
        self._numeric_names = numeric_names
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
                    columns=self._numeric_names,
                ),
                pd.DataFrame(
                    generated_state.detach().cpu().numpy(),
                    columns=self._state_names,
                ),
            ),
            axis=1,
        )
        for name in self._schema.discrete_columns:
            native_frame[name] = np.rint(native_frame[name])
        return PreparedTable(
            frame=native_frame.loc[:, self._schema.column_order],
            schema=self._schema,
        )
