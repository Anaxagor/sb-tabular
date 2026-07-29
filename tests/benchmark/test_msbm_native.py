"""Native MSBM input invariants required independently of the benchmark."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from sbtab.bridge.losses import MixedSBMLoss
from sbtab.bridge.reference import CategoricalReference
from sbtab.solvers.msbm import (
    CategoricalLossNormalization,
    MixedSBMConfig,
    MixedSBMSolver,
)


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

    def test_solver_uses_configured_categorical_reference_alpha(self) -> None:
        config = _small_config()
        config.alpha = 0.798

        solver = MixedSBMSolver(
            continuous_dim=1,
            cardinalities=[2],
            is_ordered=torch.tensor([False]),
            cfg=config,
        )

        self.assertEqual(solver.ref_cat.alpha, 0.798)

    def test_categorical_loss_column_normalization_is_an_explicit_choice(
        self,
    ) -> None:
        reference = CategoricalReference(
            cardinalities=[2, 2, 2],
            is_ordered=torch.tensor([False, False, False]),
            total_number_of_q_powers=2,
        )

        def evaluate(
            normalization: CategoricalLossNormalization,
        ) -> float:
            loss = MixedSBMLoss(
                reference=reference,
                lambda_num=0.0,
                lambda_cat=1.0,
                categorical_normalization=normalization,
            )
            loss.cat_loss_fn.forward_loss = MagicMock(
                return_value=torch.tensor(12.0)
            )
            value = loss(
                pred_num=torch.zeros((2, 1)),
                target_num=torch.zeros((2, 1)),
                pred_logits_cat=torch.zeros((2, 3, 2)),
                true_cat=torch.zeros((2, 3), dtype=torch.int64),
                x_t_cat=torch.zeros((2, 3), dtype=torch.int64),
                n=torch.ones(2, dtype=torch.int64),
                K=2,
                direction="forward",
            )
            return float(value)

        self.assertEqual(
            evaluate(CategoricalLossNormalization.NONE),
            12.0,
        )
        self.assertEqual(
            evaluate(CategoricalLossNormalization.BY_NUM_COLUMNS),
            4.0,
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
