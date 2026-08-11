"""Characterization tests for the schema-independent TabDDPM native API."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from sbtab.baselines.tabddpm.model import TabDDPMWrapper
from sbtab.baselines.tabddpm.native import (
    TabDDPMConfig,
    TabDDPMSolver,
    _is_compatible_device,
)
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

    def test_unindexed_config_device_accepts_resolved_accelerator_index(
        self,
    ) -> None:
        self.assertTrue(
            _is_compatible_device(
                torch.device("mps:0"),
                torch.device("mps"),
            )
        )
        self.assertFalse(
            _is_compatible_device(
                torch.device("cuda:1"),
                torch.device("cuda:0"),
            )
        )

    def test_progress_reports_training_and_sampling_without_changing_output(
        self,
    ) -> None:
        train_numerical, train_states = _mixed_blocks()
        config = _tiny_config()
        config.show_progress = True
        solver = TabDDPMSolver(
            num_numerical_features=2,
            cardinalities=[3, 2],
            cfg=config,
        )
        progress_output = io.StringIO()

        with redirect_stderr(progress_output):
            solver.fit(train_numerical, train_states)
            generated_numerical, generated_states = solver.sample(
                n_samples=3,
                seed=19,
            )
        solver.cfg.show_progress = False
        repeated_numerical, repeated_states = solver.sample(
            n_samples=3,
            seed=19,
        )

        rendered = progress_output.getvalue()
        self.assertIn("Training TabDDPM", rendered)
        self.assertIn("Sampling TabDDPM", rendered)
        self.assertEqual(tuple(generated_numerical.shape), (3, 2))
        self.assertEqual(tuple(generated_states.shape), (3, 2))
        torch.testing.assert_close(
            generated_numerical,
            repeated_numerical,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            generated_states,
            repeated_states,
            rtol=0,
            atol=0,
        )

    def test_config_keeps_legacy_positional_device_and_seed_slots(self) -> None:
        config = TabDDPMConfig(
            1,
            None,
            2,
            2,
            1e-3,
            0.0,
            [4],
            0.0,
            "mse",
            "cosine",
            0.9,
            "cpu",
            17,
        )

        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.seed, 17)
        self.assertTrue(config.use_ema_for_sampling)

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

    def test_sample_seed_repeats_the_native_sample(self) -> None:
        train_numerical, train_states = _mixed_blocks()
        solver = TabDDPMSolver(
            num_numerical_features=2,
            cardinalities=[3, 2],
            cfg=_tiny_config(),
        )
        solver.fit(train_numerical, train_states)

        first = solver.sample(n_samples=3, seed=19)
        repeated = solver.sample(n_samples=3, seed=19)

        torch.testing.assert_close(first[0], repeated[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], repeated[1], rtol=0, atol=0)

    def test_typed_ema_setting_selects_and_restores_the_denoiser(self) -> None:
        train_numerical, train_states = _mixed_blocks()
        config = _tiny_config()
        config.use_ema_for_sampling = True
        solver = TabDDPMSolver(
            num_numerical_features=2,
            cardinalities=[3, 2],
            cfg=config,
        )
        solver.fit(train_numerical, train_states)
        assert solver.diffusion is not None
        assert solver.ema_model is not None
        fitted_denoiser = solver.diffusion._denoise_fn
        active_denoisers: list[torch.nn.Module] = []

        def fake_sample_all(
            num_samples: int,
            batch_size: int,
            y_distribution: torch.Tensor,
            *,
            show_progress: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del batch_size, y_distribution, show_progress
            active_denoisers.append(solver.diffusion._denoise_fn)
            width = solver.num_numerical_features + len(solver.cardinalities)
            return (
                torch.zeros((num_samples, width), device=solver.device),
                torch.zeros((num_samples,), device=solver.device),
            )

        for use_ema, expected_denoiser in (
            (True, solver.ema_model),
            (False, fitted_denoiser),
        ):
            with self.subTest(use_ema=use_ema):
                active_denoisers.clear()
                solver.cfg.use_ema_for_sampling = use_ema
                with patch.object(
                    solver.diffusion,
                    "sample_all",
                    side_effect=fake_sample_all,
                ):
                    solver.sample(n_samples=2, seed=23)

                self.assertEqual(len(active_denoisers), 1)
                self.assertIs(active_denoisers[0], expected_denoiser)
                self.assertIs(solver.diffusion._denoise_fn, fitted_denoiser)

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

    def test_linear_schedule_rejects_invalid_short_horizon(self) -> None:
        config = _tiny_config()
        config.scheduler = "linear"
        config.num_timesteps = 20

        with self.assertRaisesRegex(ValueError, "strictly below 1"):
            TabDDPMSolver(
                num_numerical_features=1,
                cardinalities=[],
                cfg=config,
            )

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
