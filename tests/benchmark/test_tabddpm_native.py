"""Characterization tests for the schema-independent TabDDPM native API."""

from __future__ import annotations

import unittest

import pandas as pd
import torch

from sbtab.baselines.tabddpm.model import TabDDPMWrapper
from sbtab.baselines.tabddpm.native import TabDDPMConfig, TabDDPMSolver
from sbtab.data.schema import TabularSchema


def _tiny_config(*, seed: int = 7) -> TabDDPMConfig:
    """Return the smallest useful CPU configuration for native smoke tests."""

    return TabDDPMConfig(
        steps=1,
        num_timesteps=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        d_layers=[4],
        dropout=0.0,
        gaussian_loss_type="mse",
        scheduler="cosine",
        ema_decay=0.9,
        device="cpu",
        seed=seed,
    )


def _mixed_blocks() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [
                [-1.0, 0.25],
                [-0.2, -0.75],
                [0.4, 1.25],
                [1.1, -0.5],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0, 1],
                [2, 0],
                [1, 1],
                [0, 0],
            ],
            dtype=torch.int64,
        ),
    )


class TabDDPMNativeTests(unittest.TestCase):
    """Keep model mechanics testable without legacy table preprocessing."""

    def test_mixed_solver_returns_separate_native_blocks(self) -> None:
        train_numerical, train_states = _mixed_blocks()
        solver = TabDDPMSolver(
            num_numerical_features=2,
            cardinalities=[3, 2],
            cfg=_tiny_config(),
        )

        solver.fit(train_numerical, train_states)
        generated_numerical, generated_states = solver.sample(
            n_samples=3,
            seed=19,
        )

        self.assertEqual(tuple(generated_numerical.shape), (3, 2))
        self.assertEqual(tuple(generated_states.shape), (3, 2))
        self.assertEqual(generated_numerical.dtype, torch.float32)
        self.assertEqual(generated_states.dtype, torch.int64)
        first_state_valid = (
            (0 <= generated_states[:, 0]) & (generated_states[:, 0] < 3)
        )
        second_state_valid = (
            (0 <= generated_states[:, 1]) & (generated_states[:, 1] < 2)
        )
        self.assertTrue(bool(first_state_valid.all()))
        self.assertTrue(bool(second_state_valid.all()))

    def test_training_seed_controls_native_model_state(self) -> None:
        train_numerical, train_states = _mixed_blocks()

        def fit_state(seed: int) -> dict[str, torch.Tensor]:
            solver = TabDDPMSolver(
                num_numerical_features=2,
                cardinalities=[3, 2],
                cfg=_tiny_config(seed=seed),
            )
            solver.fit(train_numerical, train_states)
            assert solver.diffusion is not None
            return {
                name: value.detach().clone()
                for name, value in solver.diffusion.state_dict().items()
            }

        first = fit_state(11)
        repeated = fit_state(11)
        different = fit_state(12)

        self.assertEqual(first.keys(), repeated.keys())
        for name in first:
            torch.testing.assert_close(first[name], repeated[name])
        self.assertTrue(
            any(
                not torch.equal(first[name], different[name])
                for name in first
                if first[name].is_floating_point()
            )
        )

    def test_configured_gaussian_loss_reaches_diffusion(self) -> None:
        train_numerical, train_states = _mixed_blocks()
        config = _tiny_config()
        config.gaussian_loss_type = "kl"
        solver = TabDDPMSolver(
            num_numerical_features=2,
            cardinalities=[3, 2],
            cfg=config,
        )

        solver.fit(train_numerical, train_states)

        assert solver.diffusion is not None
        self.assertEqual(solver.diffusion.gaussian_loss_type, "kl")

    def test_numeric_only_and_state_only_layouts_fit(self) -> None:
        cases = (
            (
                torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32),
                torch.empty((3, 0), dtype=torch.int64),
                1,
                [],
            ),
            (
                torch.empty((3, 0), dtype=torch.float32),
                torch.tensor([[0], [1], [0]], dtype=torch.int64),
                0,
                [2],
            ),
        )
        for train_numerical, train_states, numerical_width, cardinalities in cases:
            with self.subTest(
                numerical_width=numerical_width,
                cardinalities=cardinalities,
            ):
                solver = TabDDPMSolver(
                    num_numerical_features=numerical_width,
                    cardinalities=cardinalities,
                    cfg=_tiny_config(),
                )
                solver.fit(train_numerical, train_states)
                generated_numerical, generated_states = solver.sample(
                    n_samples=2,
                    seed=23,
                )
                self.assertEqual(
                    tuple(generated_numerical.shape),
                    (2, numerical_width),
                )
                self.assertEqual(
                    tuple(generated_states.shape),
                    (2, len(cardinalities)),
                )

    def test_legacy_wrapper_delegates_and_preserves_table_shape(self) -> None:
        frame = pd.DataFrame(
            {
                "amount": [-1.0, -0.25, 0.5, 1.25],
                "label": ["no", "yes", "yes", "no"],
            }
        )
        schema = TabularSchema(
            continuous_cols=["amount"],
            discrete_cols=[],
            categorical_cols=["label"],
        )
        wrapper = TabDDPMWrapper(_tiny_config())

        wrapper.fit(frame, schema=schema)
        generated = wrapper.sample(n=3, seed=29)

        self.assertEqual(list(generated.columns), list(frame.columns))
        self.assertEqual(len(generated), 3)
        self.assertTrue(set(generated["label"]).issubset({"no", "yes"}))


if __name__ == "__main__":
    unittest.main()
