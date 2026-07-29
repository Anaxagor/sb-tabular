"""Oracle regression tests for MSBM sampling-time conditioning."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from sbtab.bridge.pathsampler import MixedPathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.timegrid import TimeGrid


class _TimeOracle(nn.Module):
    """Record the scalar time supplied to the native mixed-model boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.observed_times: list[float] = []

    def forward(
        self,
        continuous: torch.Tensor,
        states: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.observed_times.append(float(time[0, 0]))
        logits = torch.zeros(
            (len(states), states.shape[1], 2),
            dtype=continuous.dtype,
            device=continuous.device,
        )
        return torch.zeros_like(continuous), logits


class MSBMSamplingTimeTests(unittest.TestCase):
    """Keep sampling time on the same normalized scale used for training."""

    def _observe(self, direction: str) -> list[float]:
        steps = 4
        sampler = MixedPathSampler(
            timegrid=TimeGrid(
                num_steps=steps,
                gamma_min=0.1,
                gamma_max=0.4,
                schedule="linear",
            ),
            reference=CategoricalReference(
                cardinalities=[2],
                is_ordered=torch.tensor([False]),
                total_number_of_q_powers=steps,
                alpha=0.01,
            ),
            integrator=EulerMaruyama(noise=False),
        )
        oracle = _TimeOracle()

        sampler.simulate(
            x_cont_init=torch.zeros((2, 1)),
            x_cat_init=torch.zeros((2, 1), dtype=torch.int64),
            model=oracle,
            direction=direction,
            seed=7,
        )
        return oracle.observed_times

    def test_forward_sampling_uses_left_normalized_step_boundaries(self) -> None:
        self.assertEqual(self._observe("forward"), [0.0, 0.25, 0.5, 0.75])

    def test_backward_sampling_uses_right_normalized_step_boundaries(self) -> None:
        self.assertEqual(self._observe("backward"), [1.0, 0.75, 0.5, 0.25])


if __name__ == "__main__":
    unittest.main()
