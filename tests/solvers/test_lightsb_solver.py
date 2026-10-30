"""
Solver-level regression tests for LightSB (sbtab.solvers.light_sb): seeding,
column handling, global-RNG hygiene, time-shape validation, removed APIs,
update counting and checkpoints.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

import sbtab.models.sb.light_sb as light_sb_module
from sbtab.models.sb.light_sb import LightSBPotential, LightSBPotentialConfig
from sbtab.solvers.light_sb import LightSBConfig, LightSBSolver

EPS = 0.5
MAX_ITER = 60


def make_cfg(**kw) -> LightSBConfig:
    base = dict(potential=LightSBPotentialConfig(n_potentials=4, epsilon=EPS, sampling_batch_size=64),
                lr=1e-2, batch_size=64, max_iter=MAX_ITER, verbose_every=0, seed=0)
    base.update(kw)
    return LightSBConfig(**base)


def train_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(n, 2)) * 0.2 + np.array([3.0, -3.0])).astype(np.float32)
    return pd.DataFrame(x, columns=["a", "b"])


@pytest.fixture(scope="module")
def fitted() -> LightSBSolver:
    return LightSBSolver(2, make_cfg()).fit(train_frame())


# --------------------------------------------------------------------------- L1 seeding
def test_seeded_sde_sample_equals_the_end_of_the_seeded_path(fitted):
    m = 7
    paths = fitted.sample_paths(40, seed=3, n_euler_steps=m)
    sample = fitted.sample(40, seed=3, use_sde_sampling=True, n_euler_steps=m)
    assert paths.shape == (40, m + 1, 2)
    assert np.array_equal(sample, paths[:, -1])


def test_first_brownian_increment_is_not_a_replay_of_x0(fitted):
    """
    Legacy bug: sample() drew x0 from Generator(seed) and transport() built a second
    Generator(seed), so the first increment equalled x0. n * D = 4000 pairs:
    |corr| < 5 / sqrt(4000) for independent draws (the bug gives exactly 1).
    """
    m, n = 5, 2000
    paths = fitted.sample_paths(n, seed=11, n_euler_steps=m)
    sample = fitted.sample(n, seed=11, use_sde_sampling=True, n_euler_steps=m)
    assert np.array_equal(sample, paths[:, -1])                # sample() walks the same path
    x0 = torch.from_numpy(paths[:, 0])
    assert torch.equal(x0, torch.randn((n, 2), generator=torch.Generator().manual_seed(11)))
    dt = 1.0 / m
    drift0 = fitted.model.get_drift(x0, torch.zeros(n))
    z = (torch.from_numpy(paths[:, 1]) - x0 - drift0 * dt) / math.sqrt(dt * float(fitted.model.epsilon))
    assert not torch.allclose(z, x0, atol=1e-3)
    corr = np.corrcoef(z.numpy().ravel(), x0.numpy().ravel())[0, 1]
    assert abs(corr) < 5 / math.sqrt(z.numel())
    assert abs(float(z.std()) - 1.0) < 5 / math.sqrt(2 * z.numel())


def test_transport_continues_a_caller_generator(fitted):
    gen = torch.Generator().manual_seed(21)
    x0 = torch.randn((30, 2), generator=gen)
    via_transport = fitted.transport(x0, use_sde_sampling=True, n_euler_steps=4, generator=gen)
    assert np.array_equal(via_transport, fitted.sample(30, seed=21, use_sde_sampling=True, n_euler_steps=4))


@pytest.mark.parametrize("sde", [False, True], ids=["direct", "sde"])
def test_sampling_is_reproducible_with_a_seed_and_differs_without(fitted, sde):
    kw = dict(use_sde_sampling=sde, n_euler_steps=4)
    assert np.array_equal(fitted.sample(25, seed=5, **kw), fitted.sample(25, seed=5, **kw))
    assert not np.allclose(fitted.sample(25, seed=5, **kw), fitted.sample(25, seed=6, **kw))
    assert not np.allclose(fitted.sample(25, **kw), fitted.sample(25, **kw))
    x0 = np.zeros((25, 2), dtype=np.float32)
    assert np.array_equal(fitted.transport(x0, seed=9, **kw), fitted.transport(x0, seed=9, **kw))
    assert not np.allclose(fitted.transport(x0, seed=9, **kw), fitted.transport(x0, seed=10, **kw))


def test_seeded_direct_sample_does_not_reuse_the_x0_stream(fitted):
    """The forked RNG of the GMM sampler is seeded with a value DERIVED from the generator, not with `seed`."""
    n, seed = 2000, 17
    out = fitted.sample(n, seed=seed)
    x0 = torch.randn((n, 2), generator=torch.Generator().manual_seed(seed))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)                                # what a naive implementation would do
        naive = fitted.model(x0).numpy()
    assert not np.allclose(out, naive)


@pytest.mark.parametrize("n", [1, 63, 65])                     # sampling_batch_size = 64
def test_sample_sizes_around_the_sampling_batch_size(fitted, n):
    for sde in (False, True):
        out = fitted.sample(n, seed=1, use_sde_sampling=sde, n_euler_steps=3)
        assert out.shape == (n, 2) and np.isfinite(out).all()


# --------------------------------------------------------------------------- L3 columns
def test_transport_of_a_dataframe_keeps_the_training_columns(fitted):
    other = pd.DataFrame(np.zeros((5, 2), dtype=np.float32), columns=["c", "d"])
    fitted.transport(other, seed=1)
    fitted.transport(other, seed=1, use_sde_sampling=True, n_euler_steps=2)
    assert list(fitted.sample_df(3, seed=1).columns) == ["a", "b"]


def test_columns_come_from_fit_only():
    solver = LightSBSolver(2, make_cfg(max_iter=2)).fit(train_frame(50).to_numpy())
    assert solver._columns is None
    assert list(solver.sample_df(2, seed=0).columns) == [0, 1]


# --------------------------------------------------------------------------- L4 global RNG
def test_nothing_clobbers_the_callers_global_rng():
    torch.manual_seed(1234)
    np.random.seed(1234)
    expected_t, expected_n = torch.rand(4), np.random.rand(4)

    torch.manual_seed(1234)
    np.random.seed(1234)
    solver = LightSBSolver(2, make_cfg(max_iter=5)).fit(train_frame(50))      # construction + fit
    solver.sample(8, seed=1)
    solver.sample(8, seed=1, use_sde_sampling=True, n_euler_steps=3)
    solver.sample_paths(8, seed=1, n_euler_steps=3)
    solver.transport(np.zeros((8, 2), dtype=np.float32), seed=1)
    solver.transport(np.zeros((8, 2), dtype=np.float32), seed=1, use_sde_sampling=True, n_euler_steps=3)
    assert torch.equal(torch.rand(4), expected_t)
    assert np.array_equal(np.random.rand(4), expected_n)


def test_fit_is_deterministic_given_the_config_seed_whatever_the_global_rng():
    torch.manual_seed(1)
    a = LightSBSolver(2, make_cfg(max_iter=10)).fit(train_frame(80))
    torch.manual_seed(2)
    b = LightSBSolver(2, make_cfg(max_iter=10)).fit(train_frame(80))
    for (ka, va), (kb, vb) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    c = LightSBSolver(2, make_cfg(max_iter=10, seed=1)).fit(train_frame(80))
    assert not torch.equal(a.model.r, c.model.r)


def test_r_init_with_fewer_rows_than_potentials_is_seeded():
    tiny = train_frame(3)                                       # N = 3 < n_potentials = 4
    a = LightSBSolver(2, make_cfg(max_iter=1)).fit(tiny)
    b = LightSBSolver(2, make_cfg(max_iter=1)).fit(tiny)
    assert torch.equal(a.model.r, b.model.r) and a.n_updates == 1


# --------------------------------------------------------------------------- L5 time shape
def test_sample_at_time_moment_validates_the_time_shape(fitted):
    model = fitted.model
    x0 = torch.randn(5, 2)
    for t in (torch.tensor(0.3), torch.full((5,), 0.3), torch.full((5, 1), 0.3), 0.3):
        assert model.sample_at_time_moment(x0, t).shape == (5, 2)
    assert torch.equal(model.sample_at_time_moment(x0, torch.zeros(5)), x0)   # t = 0 returns x0
    with pytest.raises(ValueError, match="scalar or have shape"):
        model.sample_at_time_moment(x0, torch.tensor([0.0, 1.0]))             # (D,): used to broadcast over features
    with pytest.raises(ValueError):
        model.sample_at_time_moment(x0, torch.full((5, 2), 0.3))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.sample_at_time_moment(x0, torch.tensor(1.5))


def test_per_row_times_are_applied_per_row_not_per_feature(fitted):
    # B == D == 2 is the ambiguous case: t has one entry PER ROW. Row 0 at t = 0 must equal x0[0].
    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = fitted.model.sample_at_time_moment(x0, torch.tensor([0.0, 1.0]))
    assert torch.equal(out[0], x0[0]) and not torch.allclose(out[1], x0[1])


# --------------------------------------------------------------------------- L2 / L6 removed APIs
def test_lightsb_m_alias_and_set_epsilon_are_gone():
    assert not hasattr(light_sb_module, "LightSBM")
    assert "LightSB-M" in light_sb_module.__doc__ and "NOT implemented" in light_sb_module.__doc__
    assert not hasattr(LightSBPotential, "set_epsilon")
    assert "LightSB-M" in LightSBSolver(2, make_cfg()).describe()["algorithm"]


# --------------------------------------------------------------------------- updates, learning
def test_n_updates_counts_optimizer_steps(fitted):
    assert fitted.n_updates == MAX_ITER
    with pytest.raises(ValueError, match="max_iter"):
        LightSBSolver(2, make_cfg(max_iter=0)).fit(train_frame(20))
    with pytest.raises(RuntimeError):
        LightSBSolver(2, make_cfg()).sample(3)


def test_short_fit_puts_the_samples_on_the_data():
    """
    Data N((3,-3), 0.2^2 I), 300 rows (numpy seed 0); 300 Adam steps, lr 1e-2, solver
    seed 0; 4000 samples with seed 1. The sample mean must be within 0.75 of the data
    mean (4.24 away from the N(0, I) reference mean), for both samplers.
    """
    solver = LightSBSolver(2, make_cfg(max_iter=300)).fit(train_frame())
    for kw in (dict(), dict(use_sde_sampling=True, n_euler_steps=20)):
        out = solver.sample(4000, seed=1, **kw)
        assert np.linalg.norm(out.mean(0) - np.array([3.0, -3.0])) < 0.75
        assert np.linalg.norm(out.mean(0)) > 3.4


# --------------------------------------------------------------------------- checkpoints
def test_checkpoint_reload_reproduces_direct_and_sde_samples_exactly(fitted, tmp_path):
    path = tmp_path / "lightsb.pt"
    fitted.save_checkpoint(path)
    reloaded = LightSBSolver.load_checkpoint(path)

    assert reloaded.dim == 2 and reloaded._fitted is True
    assert reloaded.cfg == fitted.cfg and reloaded.cfg.potential == fitted.cfg.potential   # nested config
    assert reloaded._columns == ["a", "b"] and reloaded.n_updates == fitted.n_updates
    for (ka, va), (kb, vb) in zip(fitted.model.state_dict().items(), reloaded.model.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)

    for n in (1, 65):
        assert np.array_equal(reloaded.sample(n, seed=4), fitted.sample(n, seed=4))
        assert np.array_equal(reloaded.sample(n, seed=4, use_sde_sampling=True, n_euler_steps=6),
                              fitted.sample(n, seed=4, use_sde_sampling=True, n_euler_steps=6))
    assert np.array_equal(reloaded.sample_paths(9, seed=2, n_euler_steps=4), fitted.sample_paths(9, seed=2, n_euler_steps=4))
    assert list(reloaded.sample_df(2, seed=0).columns) == ["a", "b"]


def test_checkpoint_restores_the_unseeded_reference_stream(tmp_path):
    solver = LightSBSolver(2, make_cfg(max_iter=3)).fit(train_frame(40))
    solver.save_checkpoint(tmp_path / "s.pt")
    reloaded = LightSBSolver.load_checkpoint(tmp_path / "s.pt")
    # unseeded SDE sampling: x0 comes from the solver's reference generator, the rest from the global RNG
    torch.manual_seed(0); a = solver.sample(5, use_sde_sampling=True, n_euler_steps=2)
    torch.manual_seed(0); b = reloaded.sample(5, use_sde_sampling=True, n_euler_steps=2)
    assert np.array_equal(a, b)


def test_checkpoint_with_mismatching_epsilon_is_rejected(fitted, tmp_path):
    state = fitted.state_dict()
    state["config"]["potential"]["epsilon"] = 0.25               # buffer still holds 0.5
    torch.save(state, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="epsilon"):
        LightSBSolver.load_checkpoint(tmp_path / "bad.pt")
    state = fitted.state_dict()
    state["format"] = "something-else"
    torch.save(state, tmp_path / "fmt.pt")
    with pytest.raises(ValueError, match="format"):
        LightSBSolver.load_checkpoint(tmp_path / "fmt.pt")


def test_checkpoint_keeps_a_float64_model(tmp_path):
    solver = LightSBSolver(2, make_cfg(max_iter=3)).fit(train_frame(40))
    solver.model.double()
    solver.save_checkpoint(tmp_path / "d.pt")
    reloaded = LightSBSolver.load_checkpoint(tmp_path / "d.pt")
    assert reloaded.model.r.dtype == torch.float64
    assert np.array_equal(reloaded.sample(7, seed=3), solver.sample(7, seed=3))
