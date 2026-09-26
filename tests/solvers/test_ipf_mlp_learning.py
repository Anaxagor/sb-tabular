"""
Bounded learning checks for the MLP IPF-DSB solvers, and the per-step contract of
dsb_dt_joint_mlp (every edge network is trained, and is evaluated with the index
of its own edge, in both directions).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver import (
    IPFDSBConfig as CTConfig,
    IPFDSBSolver as CTSolver,
)
from sbtab.solvers.discrete_time.joint_distribution.mlp.ipf_dsb.solver import (
    IPFDSBConfig as DTConfig,
    IPFDSBSolver as DTSolver,
)
from sbtab.models.neural.mlp import StepIndexedMLP

TARGET = np.array([3.0, -3.0])


def shifted_gaussian(n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(size=(n, 2)) * 0.2 + TARGET).astype(np.float32)


# Grid shared by both checks: 16 geometric steps 2e-3..0.3, T = sum(gamma) ~= 1.05, OU reference
# (alpha 1, sigma sqrt 2): the reference carries the data mean from 3 to ~1 and the std to ~1,
# i.e. into the bulk of the N(0, I) prior, so two IPF iterations suffice.
GRID = dict(num_steps=16, gamma_min=2e-3, gamma_max=0.3, schedule="geom")

LEARNING_CASES = {
    # 2 IPF iterations x 2 half-steps x 150 updates = 600 updates (~1 s on CPU)
    "ct": (CTSolver, CTConfig(ipf_iters=2, batch_size=256, cache_batches=150, lr=2e-3, hidden_units=64,
                              n_layers=3, time_features=16, seed=0, **GRID)),
    # 2 x 2 x 200 updates, each training all 16 edge networks (~5 s on CPU)
    "dt": (DTSolver, DTConfig(ipf_iters=2, batch_size=256, cache_batches=200, lr=2e-3, hidden_units=32,
                              n_layers=2, seed=0, **GRID)),
}


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_short_fit_moves_the_samples_onto_the_data(kind):
    """
    Regression test for "the solver ignores the data": Gaussian data with mean
    (3, -3), std 0.2 (2000 rows, numpy seed 0); solver seed 0; 4000 samples drawn
    with seed 1. The legacy solver returned mean ~(0, 0), std ~1.1, i.e. it sat
    at distance ~4.24 from the data mean.

    Thresholds: the sample mean must be within 0.75 of (3, -3) - less than a fifth
    of the 4.24 that separates the data mean from the prior mean - hence more than
    3.4 from the origin. (Observed over solver seeds 0-4: 0.01-0.25.) The median is
    checked too because rare per-step extrapolation outliers can move a mean, and
    95% of the samples must lie within 1.5 of the data mean.
    """
    solver_cls, cfg = LEARNING_CASES[kind]
    solver = solver_cls(2, cfg).fit(shifted_gaussian())
    out = solver.sample(4000, seed=1)

    assert np.isfinite(out).all()
    dist_to_data = np.linalg.norm(out.mean(0) - TARGET)
    dist_to_prior = np.linalg.norm(out.mean(0))
    assert dist_to_data < 0.75, f"sample mean {out.mean(0)} is not on the data"
    assert dist_to_prior > 3.4 and dist_to_data < dist_to_prior / 4
    assert np.linalg.norm(np.median(out, axis=0) - TARGET) < 0.5
    assert (np.linalg.norm(out - TARGET, axis=1) < 1.5).mean() > 0.95

    assert [e["simulated_with"] for e in solver.stage_log] == ["reference", "net_b", "net_f", "net_b"]
    assert all(np.isfinite(e["last_loss"]) and e["n_updates"] > 0 for e in solver.stage_log)


# --------------------------------------------------------------------------- per-step contract (DT)
K = 5


def small_dt(**kw) -> DTSolver:
    base = dict(num_steps=K, gamma_min=1e-2, gamma_max=0.2, hidden_units=8, n_layers=2, batch_size=16,
                cache_batches=3, ipf_iters=1, lr=1e-2, seed=0)
    base.update(kw)
    return DTSolver(2, DTConfig(**base))


def test_dt_has_one_independent_network_per_edge_and_direction():
    solver = small_dt()
    for net in (solver.net_f, solver.net_b):
        assert isinstance(net, StepIndexedMLP) and len(net.steps) == K
        ids = {id(p) for step in net.steps for p in step.parameters()}
        assert len(ids) == sum(len(list(step.parameters())) for step in net.steps)  # no sharing
    assert not any(p is q for p in solver.net_f.parameters() for q in solver.net_b.parameters())
    with pytest.raises(IndexError):
        solver.net_b(torch.zeros(1, 2), K)


def test_dt_every_edge_network_is_trained_in_both_directions():
    data = shifted_gaussian(100)
    fresh, solver = small_dt(), small_dt().fit(data)   # same seed -> identical initialisation
    for name in ("net_f", "net_b"):
        for k in range(K):
            before = torch.cat([p.flatten() for p in getattr(fresh, name).steps[k].parameters()])
            after = torch.cat([p.flatten() for p in getattr(solver, name).steps[k].parameters()])
            assert not torch.equal(before, after), f"{name}.steps[{k}] was never updated"
    for entry in solver.stage_log:
        assert len(entry["edge_updates"]) == K and min(entry["edge_updates"]) > 0


def test_dt_untrained_edge_is_reported(monkeypatch):
    solver = small_dt()

    def skip_edge_zero(direction, batch):
        for k in range(1, K):
            solver._edge_updates[k] += 1

    monkeypatch.setattr(solver, "_record_update", skip_edge_zero)
    with pytest.raises(AssertionError, match=r"edge networks \[0\] received no update"):
        solver.fit(shifted_gaussian(50))


def _hook_steps(net, log):
    for k, step in enumerate(net.steps):
        step.register_forward_pre_hook(lambda mod, args, k=k: log.append((k, args[0].detach().clone())))


def test_dt_backward_sampling_calls_edge_k_on_the_state_at_index_k_plus_1():
    solver = small_dt().fit(shifted_gaussian(100))
    calls = []
    _hook_steps(solver.net_b, calls)
    paths = solver.sample_paths(7, seed=4)
    assert [k for k, _ in calls] == list(range(K - 1, -1, -1))           # every edge, once, in order
    for k, x_in in calls:
        assert np.array_equal(x_in.numpy(), paths[:, k + 1])              # B_k is evaluated at X_{k+1}


def test_dt_forward_simulation_calls_edge_k_on_the_state_at_index_k():
    solver = small_dt().fit(shifted_gaussian(100))
    calls = []
    _hook_steps(solver.net_f, calls)
    x0 = torch.from_numpy(shifted_gaussian(11))
    cache = solver._make_cache("backward", "net_f", x0, solver._generator(0), keep_path=True)
    assert [k for k, _ in calls] == [k for k in range(K) for _ in range(2)]
    for i, (k, x_in) in enumerate(calls):
        expected = cache.path[k] if i % 2 == 0 else cache.path[k + 1]     # F_k(X_k), then F_k(X_{k+1})
        assert torch.equal(x_in, expected)
