"""
IPF caches of the MLP IPF-DSB solvers (dsb_ct_joint_mlp / dsb_dt_joint_mlp).

The legacy implementation never advanced the state: every cached input was an
endpoint sample plus ONE step. These tests check, against closed forms and
oracle networks, that

  * cached inputs of edge k are states of the ACTUAL opposite-process trajectory
    at the matching grid index,
  * the mean map, the noise scale sqrt(gamma_k), the cache slot and (CT) the clock
    label all belong to the same edge k, in both directions,
  * targets are the DSB mean-matching targets (Prop. 3), in displacement units,
  * IPF iteration 0 simulates the declared reference, never an untrained net,
  * caches stay stochastic when cfg.noise is False.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver import (
    IPFCache,
    IPFDSBConfig as CTConfig,
    IPFDSBSolver as CTSolver,
)
from sbtab.solvers.discrete_time.joint_distribution.mlp.ipf_dsb.solver import (
    IPFDSBConfig as DTConfig,
    IPFDSBSolver as DTSolver,
)

KINDS = {"ct": (CTSolver, CTConfig), "dt": (DTSolver, DTConfig)}
K = 8
SIGMA = 1.3


def make_solver(kind: str, **kw):
    solver_cls, cfg_cls = KINDS[kind]
    base = dict(num_steps=K, gamma_min=1e-2, gamma_max=0.3, schedule="geom", sigma=SIGMA, alpha_ou=1.0,
                hidden_units=16, n_layers=2, batch_size=32, cache_batches=4, ipf_iters=1, seed=0)
    if kind == "ct":
        base["time_features"] = 16
    base.update(kw)
    return solver_cls(2, cfg_cls(**base))


class OracleTimeNet(nn.Module):
    """CT oracle: d(x, t) = -c_k x, where k is recovered from the (exact) clock label t."""

    def __init__(self, clock: torch.Tensor, coeffs: torch.Tensor):
        super().__init__()
        self.clock, self.coeffs, self.calls = clock, coeffs, []

    def forward(self, x, t):
        assert t.shape == (x.shape[0], 1)
        k = torch.argmin((t - self.clock[None, :]).abs(), dim=1)
        assert torch.equal(self.clock[k].unsqueeze(1), t), "time label is not a clock value of this direction"
        self.calls.append((int(k[0]), float(t[0, 0])))
        return -self.coeffs[k].unsqueeze(1) * x


class OracleStepNet(nn.Module):
    """DT oracle: d(x, k) = -c_k x."""

    def __init__(self, coeffs: torch.Tensor):
        super().__init__()
        self.coeffs, self.calls = coeffs, []

    def forward(self, x, k):
        self.calls.append((int(k), None))
        return -self.coeffs[int(k)] * x


def install_oracle(solver, kind: str, which: str, coeffs: torch.Tensor):
    direction = "forward" if which == "net_f" else "backward"
    net = OracleTimeNet(solver._clock[direction], coeffs) if kind == "ct" else OracleStepNet(coeffs)
    setattr(solver, which, net)
    return net


def gaussian_data(m: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn((m, 2), generator=g) * 0.2 + torch.tensor([3.0, -3.0])


# --------------------------------------------------------------------------- reference trajectories
@pytest.mark.parametrize("alpha", [1.0, 0.0], ids=["ou", "brownian"])
@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_backward_cache_inputs_are_states_of_the_reference_trajectory(kind, alpha):
    """
    Data N((3,-3), 0.2^2 I), reference X_{k+1} = (1 - gamma_k a) X_k + sigma sqrt(gamma_k) Z.
    The marginals are exactly Gaussian with  m_{k+1} = (1 - gamma_k a) m_k,
    v_{k+1} = (1 - gamma_k a)^2 v_k + sigma^2 gamma_k.  The input of edge k must be
    X_{k+1}. M = 40000 trajectories, seeds 0 / 1; tolerances are 5 standard errors:
    mean: 5 sqrt(v / M); variance (Gaussian): 5 v sqrt(2 / (M - 1)).
    The legacy cache had data variance 0.04 (+ one step) at EVERY k.
    """
    M = 40_000
    solver = make_solver(kind, alpha_ou=alpha)
    cache = solver._make_cache("backward", "reference", gaussian_data(M), solver._generator(1))
    assert cache.x.shape == (K, M, 2) and cache.y.shape == (K, M, 2)

    gam = solver._gamma.double().tolist()
    m, v = np.array([3.0, -3.0]), 0.04
    for k in range(K):
        m, v = (1 - gam[k] * alpha) * m, (1 - gam[k] * alpha) ** 2 * v + SIGMA ** 2 * gam[k]
        x = cache.x[k].double().numpy()
        assert np.all(np.abs(x.mean(0) - m) < 5 * math.sqrt(v / M)), f"edge {k}: mean is not that of X_{k + 1}"
        assert np.all(np.abs(x.var(0, ddof=1) - v) < 5 * v * math.sqrt(2 / (M - 1))), f"edge {k}: variance"
    # power of the test: at the last edge the reference variance is far from the data variance
    assert v > 10 * 0.04
    assert float(cache.x[K - 1].var(dim=0).min()) > 5 * 0.04


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_reference_cache_rows_align_with_path_edge_and_gamma(kind):
    M = 20_000
    solver = make_solver(kind)
    x0 = gaussian_data(M)
    cache = solver._make_cache("backward", "reference", x0, solver._generator(2), keep_path=True)
    path, gam = cache.path, solver._gamma
    assert path.shape == (K + 1, M, 2)
    assert torch.equal(path[0], x0)
    for k in range(K):
        a = 1.0 - float(gam[k]) * 1.0
        assert torch.equal(cache.x[k], path[k + 1])                      # input of edge k is X_{k+1}
        # target F_k(X_k) - F_k(X_{k+1}) with F_k(x) = (1 - gamma_k alpha) x: displacement units
        assert torch.allclose(cache.y[k], a * (path[k] - path[k + 1]), atol=1e-5)
        # the step k -> k+1 used gamma_k: residual std == sigma sqrt(gamma_k) (5 s.e., 2M normal entries)
        resid = (path[k + 1] - a * path[k]).double()
        expect = SIGMA * math.sqrt(float(gam[k]))
        assert abs(float(resid.std()) - expect) < 5 * expect / math.sqrt(2 * resid.numel())
    # power: a step simulated with a NEIGHBOURING gamma would miss by > 10 tolerances
    # (sqrt(gamma_{k+1} / gamma_k) - 1 ~= 0.27  vs  relative tolerance 5 / sqrt(2 * 2M) ~= 0.018)
    ratio = math.sqrt(float(gam[1] / gam[0]))
    assert ratio - 1 > 10 * 5 / math.sqrt(2 * 2 * M)


# --------------------------------------------------------------------------- network-simulated trajectories
@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_forward_cache_is_a_backward_chain_with_matching_edges(kind):
    """
    net_b is an oracle with d_b(x, k) = -c_k x (distinct c_k). The forward cache
    must be the BACKWARD chain X_k = (1 - c_k) X_{k+1} + sigma sqrt(gamma_k) Z from
    the prior, input X_k, target B_k(X_{k+1}) - B_k(X_k) = (1 - c_k)(X_{k+1} - X_k).
    """
    M = 20_000
    solver = make_solver(kind)
    coeffs = torch.linspace(0.05, 0.4, K)
    oracle = install_oracle(solver, kind, "net_b", coeffs)
    g = torch.Generator().manual_seed(3)
    x_prior = torch.randn((M, 2), generator=g)

    cache = solver._make_cache("forward", "net_b", x_prior, solver._generator(4), keep_path=True)
    path, gam = cache.path, solver._gamma
    assert torch.equal(path[K], x_prior)
    for k in range(K):
        b = 1.0 - float(coeffs[k])
        assert torch.equal(cache.x[k], path[k])                          # input of edge k is X_k
        assert torch.allclose(cache.y[k], b * (path[k + 1] - path[k]), atol=1e-5)
        resid = (path[k] - b * path[k + 1]).double()                     # uses c_k AND gamma_k of edge k
        expect = SIGMA * math.sqrt(float(gam[k]))
        assert abs(float(resid.std()) - expect) < 5 * expect / math.sqrt(2 * resid.numel())

    # sweep order K-1..0, two evaluations per edge (old and new state)
    assert [c[0] for c in oracle.calls] == [k for k in range(K - 1, -1, -1) for _ in range(2)]
    if kind == "ct":  # backward label of edge k: time_scale * t_{k+1} / T
        grid, T = solver.timegrid.grid(), solver.timegrid.T
        for k, label in oracle.calls:
            assert label == pytest.approx(solver.cfg.time_scale * float(grid[k + 1]) / T, rel=1e-6)


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_backward_cache_from_net_f_is_a_forward_chain_with_matching_edges(kind):
    M = 20_000
    solver = make_solver(kind)
    coeffs = torch.linspace(0.4, 0.05, K)
    oracle = install_oracle(solver, kind, "net_f", coeffs)
    x0 = gaussian_data(M)

    cache = solver._make_cache("backward", "net_f", x0, solver._generator(5), keep_path=True)
    path, gam = cache.path, solver._gamma
    assert torch.equal(path[0], x0)
    for k in range(K):
        f = 1.0 - float(coeffs[k])
        assert torch.equal(cache.x[k], path[k + 1])
        assert torch.allclose(cache.y[k], f * (path[k] - path[k + 1]), atol=1e-5)
        resid = (path[k + 1] - f * path[k]).double()
        expect = SIGMA * math.sqrt(float(gam[k]))
        assert abs(float(resid.std()) - expect) < 5 * expect / math.sqrt(2 * resid.numel())

    assert [c[0] for c in oracle.calls] == [k for k in range(K) for _ in range(2)]
    if kind == "ct":  # forward label of edge k: time_scale * t_k / T
        grid, T = solver.timegrid.grid(), solver.timegrid.T
        for k, label in oracle.calls:
            assert label == pytest.approx(solver.cfg.time_scale * float(grid[k]) / T, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_a_network_is_never_trained_on_its_own_process(kind):
    solver = make_solver(kind)
    x = gaussian_data(16)
    for direction, sim in [("backward", "net_b"), ("forward", "net_f"), ("forward", "reference")]:
        with pytest.raises(ValueError, match="opposite process"):
            solver._make_cache(direction, sim, x, solver._generator(0))


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_edge_sweep_assertion_detects_a_mismatched_index(kind, monkeypatch):
    solver = make_solver(kind)
    original = solver._mean_map

    def shifted(sim, x, k):  # evaluates the mean map of the WRONG edge
        return original(sim, x, (k + 1) % K)

    monkeypatch.setattr(solver, "_mean_map", shifted)
    with pytest.raises(AssertionError, match="edge sweep mismatch"):
        solver._make_cache("backward", "reference", gaussian_data(16), solver._generator(0))


# --------------------------------------------------------------------------- training rows and clock (CT)
def test_ct_training_rows_carry_the_clock_of_their_edge():
    solver = make_solver("ct", batch_size=32)
    M = 13  # K * M = 104 rows -> 3 full batches + a partial batch of 8, which must be kept
    x = torch.arange(K, dtype=torch.float32)[:, None, None].expand(K, M, 2).clone()
    cache = IPFCache("backward", "reference", x=x, y=torch.zeros_like(x))

    seen, sizes = [], []

    class Recorder(nn.Module):
        def forward(self, xb, tb):
            seen.append((xb.clone(), tb.clone()))
            return torch.zeros_like(xb)

    for xb, kb, yb in solver._iter_batches(cache, solver._generator(0)):
        assert torch.equal(xb[:, 0], kb.float())                         # rows keep their edge index
        solver._batch_loss(Recorder(), "backward", (xb, kb, yb))
        sizes.append(xb.shape[0])
    assert sizes == [32, 32, 32, 8]
    clock_b = solver._clock["backward"]
    for xb, tb in seen:  # the label given to the net is the backward clock of the row's edge
        assert torch.equal(tb, clock_b[xb[:, 0].long()].unsqueeze(1))


def test_ct_clock_is_rescaled_and_separates_neighbouring_edges():
    """
    Defaults: geometric grid, T ~= 0.046. Raw times cannot be told apart by the
    sinusoidal embedding (max_period 1e4); the rescaled clock time_scale * t / T can.
    """
    solver = CTSolver(2, CTConfig(hidden_units=16, n_layers=2, time_features=32))
    grid, T = solver.timegrid.grid(), solver.timegrid.T
    assert torch.allclose(solver._clock["forward"], grid[:-1] * (1000.0 / T))
    assert torch.allclose(solver._clock["backward"], grid[1:] * (1000.0 / T))
    assert float(solver._clock["forward"][0]) == 0.0
    assert float(solver._clock["backward"][-1]) == pytest.approx(1000.0, rel=1e-6)

    emb = solver.net_b.time_emb
    raw = emb(grid[1:].unsqueeze(1))
    rescaled = emb(solver._clock["backward"].unsqueeze(1))
    gap = lambda e: (e[1:] - e[:-1]).norm(dim=1)  # noqa: E731
    assert float(gap(raw).max()) < 0.05          # the defect: neighbouring raw times are indistinguishable
    assert float(gap(rescaled).min()) > 0.5      # every pair of neighbouring edges is separated


# --------------------------------------------------------------------------- fit: declared reference, stochastic caches
@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_iteration_zero_simulates_the_reference_and_never_the_untrained_net_f(kind, monkeypatch):
    solver = make_solver(kind, ipf_iters=2)
    sims, net_f_calls = [], []

    original_map = solver._mean_map
    monkeypatch.setattr(solver, "_mean_map",
                        lambda sim, x, k: (sims.append((len(solver.stage_log), sim)), original_map(sim, x, k))[1])
    original_build = solver._build_networks

    def build_and_hook():
        original_build()
        solver.net_f.register_forward_pre_hook(lambda mod, args: net_f_calls.append(len(solver.stage_log)))

    monkeypatch.setattr(solver, "_build_networks", build_and_hook)
    solver.fit(gaussian_data(200).numpy())

    expected = {0: "reference", 1: "net_b", 2: "net_f", 3: "net_b"}
    assert {s for s, _ in sims} == set(expected)
    assert all(sim == expected[stage] for stage, sim in sims)
    assert net_f_calls and min(net_f_calls) >= 1, "net_f was evaluated before it was ever trained"

    assert [(e["iteration"], e["trained"], e["simulated_with"]) for e in solver.stage_log] == [
        (0, "backward", "reference"), (0, "forward", "net_b"), (1, "backward", "net_f"), (1, "forward", "net_b")]
    assert all(e["n_updates"] > 0 for e in solver.stage_log)
    assert solver.n_updates == sum(e["n_updates"] for e in solver.stage_log)


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_caches_stay_stochastic_when_sampler_noise_is_off(kind):
    solver = make_solver(kind, noise=False)
    assert solver.variant_id == f"{solver.canonical_id}_noiseless_heuristic"
    assert solver.describe()["cache_noise"] is True and solver.describe()["sampler_noise"] is False

    M = 5000
    cache = solver._make_cache("backward", "reference", gaussian_data(M), solver._generator(0), keep_path=True)
    gam = solver._gamma
    for k in range(K):
        resid = (cache.path[k + 1] - (1 - float(gam[k])) * cache.path[k]).double()
        expect = SIGMA * math.sqrt(float(gam[k]))
        assert abs(float(resid.std()) - expect) < 5 * expect / math.sqrt(2 * resid.numel())

    assert make_solver(kind, noise=True).variant_id == solver.canonical_id
