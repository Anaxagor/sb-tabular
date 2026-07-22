"""Native MSBM input invariants required independently of the benchmark."""

from __future__ import annotations

import unittest

import torch

from sbtab.bridge.reference import CategoricalReference
from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver


def _small_config() -> MixedSBMConfig:
    return MixedSBMConfig(
        fb_sequence=("b",),
        cat_emb_dim=2,
        hidden_dim=4,
        time_dim=4,
        n_layers=1,
        dropout=0.0,
        num_steps=2,
        batch_size=2,
        epochs_per_direction=0,
        device="cpu",
        seed=7,
    )


class MSBMNativeInputTests(unittest.TestCase):
    """Keep mathematical input limitations with the native implementation."""

    def test_solver_requires_a_continuous_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous block"):
            MixedSBMSolver(
                continuous_dim=0,
                cardinalities=[2],
                is_ordered=torch.tensor([False]),
                cfg=_small_config(),
            )

    def test_reference_requires_a_finite_state_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite-state block"):
            CategoricalReference(
                cardinalities=[],
                is_ordered=torch.empty(0, dtype=torch.bool),
                total_number_of_q_powers=2,
            )

    def test_reference_rejects_singleton_nominal_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "singleton nominal"):
            CategoricalReference(
                cardinalities=[1],
                is_ordered=torch.tensor([False]),
                total_number_of_q_powers=2,
            )

    def test_solver_rejects_non_finite_continuous_training_data(self) -> None:
        solver = MixedSBMSolver(
            continuous_dim=1,
            cardinalities=[2],
            is_ordered=torch.tensor([False]),
            cfg=_small_config(),
        )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            solver.fit(
                train_num=torch.tensor([[0.0], [torch.inf]]),
                train_cat=torch.tensor([[0], [1]], dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
