"""
Sampler units, seeding, batch sizes, update counting and checkpoints of the MLP
IPF-DSB solvers (dsb_ct_joint_mlp / dsb_dt_joint_mlp).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver import (
    IPFDSBConfig as CTConfig,
    IPFDSBSolver as CTSolver,
)
from sbtab.solvers.discrete_time.joint_distribution.mlp.ipf_dsb.solver import (
    IPFDSBConfig as DTConfig,
    IPFDSBSolver as DTSolver,
)

KINDS = {"ct": (CTSolver, CTConfig), "dt": (DTSolver, DTConfig)}
K = 6
SIGMA = 1.1
BATCH = 16


def make_solver(kind: str, **kw):
    solver_cls, cfg_cls = KINDS[kind]
    base = dict(num_steps=K, gamma_min=1e-2, gamma_max=0.2, sigma=SIGMA, hidden_units=16, n_layers=2,
                time_features=16, batch_size=BATCH, cache_batches=3, ipf_iters=1, lr=1e-3, seed=0)
    base.update(kw)
    return solver_cls(2, cfg_cls(**base))


def train_data(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(size=(n, 2)) * 0.5 + np.array([1.0, -1.0])).astype(np.float32)


class ConstantDisplacement(nn.Module):
    """Oracle network for both solvers: returns the same known displacement for every row."""

    def __init__(self, d):
        super().__init__()
        self.d = torch.tensor(d, dtype=torch.float32)

    def forward(self, x, _time_or_edge):
        return self.d.expand_as(x)


def with_oracle(kind: str, d, **kw):
    solver = make_solver(kind, **kw)
    solver.net_b = ConstantDisplacement(d)
    solver._fitted = True
    return solver


@pytest.fixture(scope="module", params=["ct", "dt"])
def fitted(request):
    return request.param, make_solver(request.param).fit(train_data())


# --------------------------------------------------------------------------- units
@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_sampler_step_adds_the_displacement_exactly(kind):
    """
    The networks are displacements (units of x). With an oracle returning d, one
    noiseless step moves x by exactly d - not by d * gamma_k (the legacy bug, a
    factor 1e-2..2e-1 here).
    """
    d = [0.37, -1.25]
    solver = with_oracle(kind, d, noise=False)
    x = torch.tensor([[0.5, 2.0], [-1.0, 0.25], [3.0, -3.0]])
    for k in range(K):
        moved = solver._sampler_step(x, k, None) - x
        assert torch.allclose(moved, torch.tensor(d).expand_as(x), atol=1e-6)
        assert not torch.allclose(moved, torch.tensor(d).expand_as(x) * solver._gamma[k], atol=1e-3)


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_sampler_step_noise_is_sigma_sqrt_gamma_of_the_same_edge(kind):
    d = [0.37, -1.25]
    solver = with_oracle(kind, d, noise=True)
    x = torch.zeros(4, 2)
    for k in range(K):
        eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(11))
        expected = x + torch.tensor(d) + SIGMA * math.sqrt(float(solver._gamma[k])) * eps
        got = solver._sampler_step(x, k, torch.Generator().manual_seed(11))
        assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_noiseless_chain_is_the_prior_draw_plus_all_displacements(kind):
    d = [0.1, -0.2]
    solver = with_oracle(kind, d, noise=False)
    out = solver.sample(5, seed=3)
    x_T = torch.randn((5, 2), generator=torch.Generator().manual_seed(3)).numpy()
    assert np.allclose(out, x_T + K * np.array(d), atol=1e-5)


# --------------------------------------------------------------------------- seeds
@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_first_increment_is_not_a_replay_of_the_start_sample(kind):
    """
    Legacy bug: the start draw and the path noise used two generators with the same
    seed, so the first noise increment EQUALLED x_T (correlation 1). With one
    generator the increment is independent: n * D = 4000 pairs, |corr| < 5 / sqrt(4000).
    """
    solver = with_oracle(kind, [0.0, 0.0], noise=True)
    paths = solver.sample_paths(2000, seed=7)
    assert paths.shape == (2000, K + 1, 2)
    start = paths[:, K]
    assert np.array_equal(start, torch.randn((2000, 2), generator=torch.Generator().manual_seed(7)).numpy())
    first_eps = (paths[:, K - 1] - start) / (SIGMA * math.sqrt(float(solver._gamma[K - 1])))
    assert not np.allclose(first_eps, start, atol=1e-3)
    corr = np.corrcoef(first_eps.ravel(), start.ravel())[0, 1]
    assert abs(corr) < 5 / math.sqrt(start.size)
    assert abs(first_eps.std() - 1.0) < 5 / math.sqrt(2 * start.size)   # it IS a unit normal draw
    assert np.array_equal(paths[:, 0], solver.sample(2000, seed=7))


def test_sampling_is_reproducible_with_a_seed_and_differs_without(fitted):
    _, solver = fitted
    a, b = solver.sample(50, seed=123), solver.sample(50, seed=123)
    assert np.array_equal(a, b)
    assert not np.allclose(a, solver.sample(50, seed=124))
    u1, u2 = solver.sample(50), solver.sample(50)
    assert not np.allclose(u1, u2)
    # unseeded sampling follows the caller's global RNG instead of a hidden fixed seed
    torch.manual_seed(5); v1 = solver.sample(50)
    torch.manual_seed(5); v2 = solver.sample(50)
    assert np.array_equal(v1, v2)


def test_construction_fit_and_seeded_sampling_leave_the_global_rng_alone():
    torch.manual_seed(99)
    expected = torch.rand(3)
    for kind in KINDS:
        torch.manual_seed(99)
        solver = make_solver(kind).fit(train_data(50))
        solver.sample(4, seed=1)
        assert torch.equal(torch.rand(3), expected)


# --------------------------------------------------------------------------- sizes
@pytest.mark.parametrize("n", [1, BATCH - 1, BATCH + 1])
def test_sample_sizes_around_the_batch_size(fitted, n):
    _, solver = fitted
    whole = solver.sample(n, seed=2)
    chunked = solver.sample(n, seed=2, batch_size=BATCH)
    assert whole.shape == chunked.shape == (n, 2)
    assert np.isfinite(whole).all() and np.isfinite(chunked).all()
    # the first chunk consumes the generator exactly like an unchunked call of its size ...
    first = min(n, BATCH)
    assert np.array_equal(chunked[:first], solver.sample(first, seed=2))
    if n > BATCH:  # ... and later chunks continue the stream instead of replaying it
        assert not np.allclose(chunked[BATCH:], chunked[: n - BATCH])
    assert solver.sample_paths(n, seed=2, batch_size=BATCH).shape == (n, K + 1, 2)


def test_sample_before_fit_and_bad_sizes_are_rejected():
    solver = make_solver("ct")
    with pytest.raises(RuntimeError):
        solver.sample(3)
    solver._fitted = True
    with pytest.raises(ValueError):
        solver.sample(0)


# --------------------------------------------------------------------------- updates
@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_small_training_set_still_gets_positive_updates(kind):
    """N = 7 rows < batch_size = 16: trajectories re-use the rows, nothing is dropped."""
    solver = make_solver(kind, ipf_iters=2).fit(train_data(7))
    assert len(solver.stage_log) == 4
    assert all(e["n_updates"] > 0 for e in solver.stage_log)
    assert solver.n_updates == sum(e["n_updates"] for e in solver.stage_log) > 0
    assert np.isfinite(solver.sample(5, seed=0)).all()


def test_ct_partial_batches_are_kept():
    # cache_batches=1, batch_size=16, K=6: M = ceil(16 / 6) = 3 trajectories -> 18 rows per cache
    # = one full batch + a PARTIAL batch of 2 rows -> 2 updates per epoch, 3 epochs (3 fresh caches).
    solver = make_solver("ct", cache_batches=1, epochs_per_phase=3).fit(train_data(40))
    assert [e["n_updates"] for e in solver.stage_log] == [6, 6]
    assert [e["cache_rows"] for e in solver.stage_log] == [54, 54]
    assert [e["epochs"] for e in solver.stage_log] == [3, 3]


def test_dt_updates_train_every_edge_each_time():
    solver = make_solver("dt", cache_batches=3, epochs_per_phase=2).fit(train_data(40))
    for e in solver.stage_log:
        assert e["n_updates"] == 6                      # cache_batches * epochs
        assert e["edge_updates"] == [6] * K             # every edge network, every update
        assert e["cache_rows"] == 2 * 3 * BATCH * K


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_steps_per_phase_is_the_exact_number_of_updates(kind):
    solver = make_solver(kind, steps_per_phase=7, epochs_per_phase=1).fit(train_data(40))
    assert [e["n_updates"] for e in solver.stage_log] == [7, 7]
    assert solver.n_updates == 14


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_invalid_configs_are_rejected(kind):
    with pytest.raises(ValueError, match="alpha_ou"):
        make_solver(kind, alpha_ou=6.0)             # 6 * gamma_max = 1.2 >= 1: unstable Euler OU step
    with pytest.raises(ValueError):
        make_solver(kind, sigma=0.0)
    with pytest.raises(ValueError):
        make_solver(kind, steps_per_phase=0)
    with pytest.raises(ValueError):
        make_solver(kind).fit(np.full((5, 2), np.nan, dtype=np.float32))


# --------------------------------------------------------------------------- metadata
def test_describe_states_the_time_parameterization_honestly():
    ct, dt = make_solver("ct").describe(), make_solver("dt").describe()
    assert (ct["solver_id"], ct["time_parameterization"]) == ("dsb_ct_joint_mlp", "time_conditioned")
    assert (dt["solver_id"], dt["time_parameterization"]) == ("dsb_dt_joint_mlp", "per_step")
    assert ct["clock"]["time_scale"] == 1000.0 and dt["clock"] is None
    for d in (ct, dt):
        assert d["reference"]["kind"] == "ou" and d["reference"]["sigma"] == pytest.approx(SIGMA)
        assert d["reference"]["horizon_T"] == pytest.approx(float(make_solver("ct").timegrid.T))
        assert "displacement" in d["network_output"]
    assert make_solver("ct", alpha_ou=0.0).describe()["reference"]["kind"] == "brownian"


# --------------------------------------------------------------------------- checkpoints
def test_checkpoint_reload_reproduces_samples_exactly(fitted, tmp_path):
    kind, solver = fitted
    path = tmp_path / f"{kind}.pt"
    solver.save_checkpoint(path)
    reloaded = KINDS[kind][0].load_checkpoint(path)

    assert reloaded._fitted is True
    assert reloaded.cfg == solver.cfg
    assert reloaded.sigma == solver.sigma and reloaded.alpha_ou == solver.alpha_ou
    assert torch.equal(reloaded.timegrid.grid(), solver.timegrid.grid())
    assert torch.equal(reloaded._gamma, solver._gamma)
    assert reloaded.stage_log == solver.stage_log and reloaded.n_updates == solver.n_updates
    assert reloaded.variant_id == solver.variant_id and reloaded.describe() == solver.describe()
    for name in ("net_f", "net_b"):                    # BOTH directions are restored
        a, b = getattr(solver, name).state_dict(), getattr(reloaded, name).state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

    for n, seed in [(1, 0), (33, 5)]:
        assert np.array_equal(reloaded.sample(n, seed=seed), solver.sample(n, seed=seed))
    assert np.array_equal(reloaded.sample(33, seed=5, batch_size=BATCH), solver.sample(33, seed=5, batch_size=BATCH))
    assert np.array_equal(reloaded.sample_paths(9, seed=1), solver.sample_paths(9, seed=1))


def test_checkpoint_of_the_other_solver_is_rejected(fitted, tmp_path):
    kind, solver = fitted
    path = tmp_path / "x.pt"
    solver.save_checkpoint(path)
    other = KINDS["dt" if kind == "ct" else "ct"][0]
    with pytest.raises(ValueError, match="checkpoint belongs to"):
        other.load_checkpoint(path)


def test_unfitted_checkpoint_stays_unfitted(tmp_path):
    solver = make_solver("ct")
    solver.save_checkpoint(tmp_path / "u.pt")
    reloaded = CTSolver.load_checkpoint(tmp_path / "u.pt")
    assert reloaded._fitted is False
    with pytest.raises(RuntimeError):
        reloaded.sample(2)
