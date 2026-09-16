from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sbtab.solvers import (  # noqa: E402
    IMFDSBMConfig,
    IMFDSBMSolver,
    IPFDSBConfig,
    IPFDSBSolver,
)


@pytest.fixture
def tiny_data() -> np.ndarray:
    return np.random.default_rng(42).normal(size=(12, 2)).astype(np.float32)


def _assert_valid_samples(samples: np.ndarray) -> None:
    assert samples.shape == (4, 2)
    assert np.isfinite(samples).all()


def test_dsb_fit_sample_smoke(tiny_data: np.ndarray) -> None:
    cfg = IPFDSBConfig(
        ipf_iters=1,
        num_steps=2,
        batch_size=8,
        cache_batches=1,
        steps_per_phase=1,
        hidden_units=8,
        time_features=4,
        noise=False,
        device="cpu",
        seed=42,
    )
    solver = IPFDSBSolver(dim=2, cfg=cfg).fit(tiny_data)
    _assert_valid_samples(solver.sample(4, seed=43))


def test_dsbm_fit_sample_with_oversized_batch(tiny_data: np.ndarray) -> None:
    cfg = IMFDSBMConfig(
        fb_sequence=("b",),
        num_steps=2,
        inner_iters=1,
        batch_size=64,
        noise=False,
        device="cpu",
        seed=42,
    )
    solver = IMFDSBMSolver(dim=2, cfg=cfg).fit(tiny_data)
    _assert_valid_samples(solver.sample(4, seed=43))
