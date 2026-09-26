"""
Behavioural tests of the five IMF-DSBM solvers: IMF stage bookkeeping, seeding,
small samples, checkpoints, effective / rejected configuration and the structure
handling of the feature-wise solver. Everything is tiny (<= 80 rows, dim 2,
<= 8 steps, <= 3 stages, CatBoost <= 20 iterations of depth <= 2, MLP width <= 32).
"""
from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from sbtab.models.boosted.catboost_continuous_joint import CatBoostContinuousFieldConfig
from sbtab.models.boosted.catboost_discrete_joint import CatBoostDiscreteFieldConfig
from sbtab.models.boosted.catboost_discrete_scalar import CatBoostScalarConfig
from sbtab.models.neural.mlp_discrete_joint import StepMLPJointConfig
from sbtab.solvers.continuous_time.joint_distribution.boosting.imf_dsbm import solver as mod_b
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm import solver as mod_a
from sbtab.solvers.discrete_time.feature_wise.boosting.imf_dsbm_featurewise_boost import solver as mod_e
from sbtab.solvers.discrete_time.joint_distribution.boosting.imf_dsbm_boost import solver as mod_d
from sbtab.solvers.discrete_time.joint_distribution.mlp.imf_dsbm import solver as mod_c
from sbtab.solvers.structure import LearnedDAG

KINDS = ["A", "B", "C", "D", "E"]
CANONICAL = {
    "A": "dsbm_ct_joint_mlp",
    "B": "dsbm_ct_joint_gbt",
    "C": "dsbm_dt_joint_mlp",
    "D": "dsbm_dt_joint_gbt",
    "E": "dsbm_dt_structural_gbt",
}
SOLVER_CLS = {
    "A": mod_a.IMFDSBMSolver,
    "B": mod_b.IMFDSBMContinuousJointCatBoostSolver,
    "C": mod_c.IMFDSBMDiscreteJointMLPSolver,
    "D": mod_d.IMFDSBMBoostSolver,
    "E": mod_e.FeaturewiseDSBMBoostSolver,
}
CONFIG_CLS = {
    "A": mod_a.IMFDSBMConfig,
    "B": mod_b.IMFDSBMContinuousJointCatBoostConfig,
    "C": mod_c.IMFDSBMDiscreteJointMLPConfig,
    "D": mod_d.IMFDSBMBoostConfig,
    "E": mod_e.FeaturewiseDSBMBoostConfig,
}
BATCH = 8
GBT = dict(iterations=8, depth=2, thread_count=1)


def make_data(n: int = 40, dim: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32) * 0.5 + 1.0
    if dim > 1:
        x[:, 1] += 0.8 * x[:, 0]
    return pd.DataFrame(x, columns=[f"c{i}" for i in range(dim)])


def make_config(kind: str, **over):
    base = dict(fb_sequence=("b", "f", "b"), num_steps=4, sigma=0.5, seed=3)
    model_over = over.pop("model", {})
    if kind == "A":
        base.update(inner_iters=6, batch_size=BATCH, hidden_dim=16, n_layers=2, time_emb_dim=8, lr=1e-2)
    elif kind == "B":
        base.update(field=CatBoostContinuousFieldConfig(**{**GBT, **model_over}))
    elif kind == "C":
        base.update(field=StepMLPJointConfig(**{**dict(hidden_dim=16, n_layers=2, n_epochs=2,
                                                       batch_size=BATCH, lr=1e-2), **model_over}))
    elif kind == "D":
        base.update(catboost=CatBoostDiscreteFieldConfig(**{**GBT, **model_over}))
    elif kind == "E":
        base.update(catboost=CatBoostScalarConfig(**{**GBT, **model_over}))
    base.update(over)
    return CONFIG_CLS[kind](**base)


def make_solver(kind: str, dim: int = 2, **over):
    cfg = make_config(kind, **over)
    return SOLVER_CLS[kind](cfg) if kind == "E" else SOLVER_CLS[kind](dim, cfg)


@pytest.fixture(scope="module")
def fitted():
    """One tiny fit per solver; tests must not mutate these."""
    data = make_data()
    return data, {kind: make_solver(kind).fit(data) for kind in KINDS}


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _simulate(solver, kind: str, direction: str, zstart, seed: int, noise: bool = True) -> np.ndarray:
    gen = solver._make_generator(seed)
    if kind == "A":
        net = solver.model.net(direction)
        net.eval()
        return _np(solver._sample_sde(net=net, fb=direction, zstart=zstart, generator=gen, noise=noise))
    return _np(solver._sample_with_direction(zstart, direction, generator=gen, noise=noise))


# ---------------------------------------------------------------------------
# 4. IMF iteration (and S1: fit-time couplings are always noisy)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("noise", [True, False])
def test_imf_stages_train_on_previous_opposite_model_coupling(kind, noise, monkeypatch):
    data = make_data()
    X = data.to_numpy(dtype=np.float32)
    solver = make_solver(kind, noise=noise)
    original = solver._generate_coupling
    params = list(inspect.signature(original).parameters)
    records = []

    def spy(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs).arguments
        x_data, x_prior, prev_fb, seed = (bound[p] for p in params)
        z0, z1, source = original(*args, **kwargs)
        rec = dict(prev_fb=prev_fb, seed=int(seed), source=source, z0=_np(z0).copy(), z1=_np(z1).copy(),
                   x_data=_np(x_data).copy(), x_prior=_np(x_prior).copy())
        if prev_fb is not None:
            # re-simulate the CURRENT latest model of the previous direction, WITH noise
            zstart = x_data if prev_fb == "f" else x_prior
            rec["resim"] = _simulate(solver, kind, prev_fb, zstart, seed, noise=True)
        records.append(rec)
        return z0, z1, source

    monkeypatch.setattr(solver, "_generate_coupling", spy)
    solver.fit(data)

    log = solver.stage_log
    assert [s["stage"] for s in log] == [0, 1, 2]
    assert [s["direction"] for s in log] == ["b", "f", "b"]
    assert [s["coupling_source"] for s in log] == ["independent", "backward_model", "forward_model"]
    assert [s["anchored_endpoint"] for s in log] == ["both", "prior", "data"]
    assert [s["coupling_seed"] for s in log] == [r["seed"] for r in records]
    assert [r["source"] for r in records] == [s["coupling_source"] for s in log]
    for s in log:
        assert s.get("n_updates", 0) > 0 or s.get("n_models_fitted", 0) > 0

    prior = records[0]["x_prior"]
    assert prior.shape == X.shape
    for r in records:
        np.testing.assert_array_equal(r["x_data"], X)          # the real training rows
        np.testing.assert_array_equal(r["x_prior"], prior)     # one fixed set of real prior rows

    # stage 0: independent coupling data (x) prior -- both endpoints are real rows
    np.testing.assert_array_equal(records[0]["z0"], X)
    order = lambda a: a[np.lexsort(a.T[::-1])]
    np.testing.assert_array_equal(order(records[0]["z1"]), order(prior))

    # stage 1 ('f'): coupling of the stage-0 BACKWARD model, anchored at the real prior rows
    np.testing.assert_array_equal(records[1]["z1"], prior)
    np.testing.assert_array_equal(records[1]["z0"], records[1]["resim"])
    assert not np.allclose(records[1]["z0"], X)

    # stage 2 ('b'): coupling of the stage-1 FORWARD model, anchored at the real data rows
    np.testing.assert_array_equal(records[2]["z0"], X)
    np.testing.assert_array_equal(records[2]["z1"], records[2]["resim"])
    assert not np.allclose(records[2]["z1"], prior)

    # S1: whatever cfg.noise says, the simulated endpoint is the NOISY simulation above and
    # not the noiseless one ('f' is trained once, so the forward model is still the same)
    start = torch.from_numpy(X) if kind == "A" else X
    noiseless = _simulate(solver, kind, "f", start, records[2]["seed"], noise=False)
    assert not np.allclose(records[2]["z1"], noiseless, atol=1e-3)


@pytest.mark.parametrize("kind", KINDS)
def test_variant_id_and_noiseless_heuristic_label(kind, fitted):
    assert make_solver(kind, noise=True).variant_id == CANONICAL[kind]
    assert make_solver(kind, noise=False).variant_id == CANONICAL[kind] + "_noiseless_heuristic"
    assert SOLVER_CLS[kind].canonical_id == CANONICAL[kind]
    doc = " ".join(SOLVER_CLS[kind].__doc__.split())
    assert "heuristic" in doc and "NOT a probability-flow ODE" in doc


@pytest.mark.parametrize("kind", KINDS)
def test_noise_flag_changes_sampling_but_not_training(kind):
    """Same seed, noise on/off: identical learned drifts, different samplers."""
    data = make_data()
    on = make_solver(kind, noise=True).fit(data)
    off = make_solver(kind, noise=False).fit(data)

    zstart = np.random.default_rng(1).normal(size=(16, 2)).astype(np.float32)
    start = torch.from_numpy(zstart) if kind == "A" else zstart
    for direction in ("f", "b"):   # noiseless simulation == a pure function of the learned drift
        np.testing.assert_array_equal(_simulate(on, kind, direction, start, 0, noise=False),
                                      _simulate(off, kind, direction, start, 0, noise=False))

    a, b = on.sample(16, seed=5), off.sample(16, seed=5)
    assert a.shape == b.shape == (16, 2)
    assert not np.allclose(a, b)
    np.testing.assert_array_equal(off.sample(16, seed=5), b)


# ---------------------------------------------------------------------------
# S2. first coupling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_default_first_coupling_is_independent(kind):
    assert CONFIG_CLS[kind]().first_coupling == "ind"


@pytest.mark.parametrize("kind", KINDS)
def test_first_coupling_marginals(kind):
    """
    "ind": the t=1 endpoint IS the prior sample. "ref": z1 = z0 + sigma * noise, so
    (z1 - z0)/sigma is standard normal (n = 4000 x 2 scalars, 5 SE on mean and std)
    and the t=1 marginal is data * N(0, sigma^2), not the prior.
    """
    n, sigma = 4000, 0.5
    rng = np.random.default_rng(0)
    x_data = (rng.normal(size=(n, 2)) * 0.3 + 4.0).astype(np.float32)
    x_prior = rng.normal(size=(n, 2)).astype(np.float32)
    if kind == "A":
        x_data, x_prior = torch.from_numpy(x_data), torch.from_numpy(x_prior)

    z0, z1, source = make_solver(kind, first_coupling="ind", sigma=sigma)._generate_coupling(x_data, x_prior, None, 1)
    assert source == "independent"
    np.testing.assert_array_equal(_np(z0), _np(x_data))
    order = lambda a: a[np.lexsort(a.T[::-1])]
    np.testing.assert_array_equal(order(_np(z1)), order(_np(x_prior)))

    z0, z1, source = make_solver(kind, first_coupling="ref", sigma=sigma)._generate_coupling(x_data, x_prior, None, 1)
    assert source == "reference"
    np.testing.assert_array_equal(_np(z0), _np(x_data))
    u = ((_np(z1) - _np(z0)) / sigma).astype(np.float64)
    M = u.size
    assert abs(u.mean()) < 5.0 / math.sqrt(M)
    assert abs(u.std() - 1.0) < 5.0 / math.sqrt(2.0 * M)
    assert abs(_np(z1).mean() - 4.0) < 0.1          # sits on the data, far from the N(0, I) prior


def test_reference_coupling_is_logged(fitted):
    data, _ = fitted
    solver = make_solver("D", first_coupling="ref", fb_sequence=("b",)).fit(data)
    assert solver.stage_log[0]["coupling_source"] == "reference"
    assert solver.stage_log[0]["anchored_endpoint"] == "data"


# ---------------------------------------------------------------------------
# 8b. rejected configurations (S2, S6, C1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bad", [
    dict(first_coupling="bogus"),
    dict(first_coupling="independent"),
    dict(fb_sequence=()),
    dict(fb_sequence=("b", "b")),
    dict(fb_sequence=("b", "f", "f")),
    dict(fb_sequence=("b", "x")),
    dict(fb_sequence=("forward",)),
])
def test_invalid_config_is_rejected(kind, bad):
    with pytest.raises(ValueError):
        make_solver(kind, **bad)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seq", [("b",), ("f", "b"), ("b", "f", "b", "f")])
def test_alternating_sequences_with_either_start_are_accepted(kind, seq):
    make_solver(kind, fb_sequence=seq)


@pytest.mark.parametrize("kind", ["B", "D", "E"])
def test_residual_catboost_field_is_rejected(kind):
    with pytest.raises(ValueError, match="residual"):
        make_solver(kind, model=dict(residual=True))


@pytest.mark.parametrize("mode", ["x_x0", "x_x0_t", "nonsense"])
def test_x0_feature_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="feature_mode"):
        make_solver("C", model=dict(feature_mode=mode))


@pytest.mark.parametrize("mode", ["x", "x_t"])
def test_markov_feature_modes_train_and_sample(mode):
    solver = make_solver("C", fb_sequence=("b",), model=dict(feature_mode=mode)).fit(make_data())
    assert solver.sample(5, seed=0).shape == (5, 2)
    first = solver.field_b.models[0].net[0]
    assert first.in_features == (2 if mode == "x" else 3)


def test_forward_only_fit_cannot_sample():
    solver = make_solver("D", fb_sequence=("f",)).fit(make_data())
    with pytest.raises(RuntimeError, match="backward"):
        solver.sample(3)


# ---------------------------------------------------------------------------
# 5. small samples and shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["A", "C"])
def test_fewer_rows_than_batch_size_still_trains(kind):
    data = make_data(n=BATCH - 3)
    solver = make_solver(kind, fb_sequence=("b", "f")).fit(data)
    assert all(s["n_updates"] > 0 for s in solver.stage_log)
    assert solver.n_updates == sum(s["n_updates"] for s in solver.stage_log) > 0
    if kind == "A":
        assert solver.n_updates == 2 * solver.cfg.inner_iters
    assert np.isfinite(solver.sample(4, seed=0)).all()


def test_partial_batches_are_kept():
    """n = batch + 1: the 1-row remainder is a batch of its own, not dropped."""
    data = make_data(n=BATCH + 1)
    seen = []
    solver = make_solver("A", fb_sequence=("b",), inner_iters=4)
    original = solver._dsbm_train_tuple
    solver._dsbm_train_tuple = lambda z_pairs, **kw: (seen.append(z_pairs.shape[0]), original(z_pairs, **kw))[1]
    solver.fit(data)
    assert seen == [BATCH, 1, BATCH, 1]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n", [1, BATCH - 1, BATCH + 1])
def test_sample_returns_exactly_n_rows(kind, n, fitted):
    _, solvers = fitted
    out = solvers[kind].sample(n, seed=1)
    assert isinstance(out, np.ndarray) and out.shape == (n, 2) and out.dtype == np.float32
    assert np.isfinite(out).all()
    df = solvers[kind].sample_df(n, seed=1)
    assert list(df.columns) == ["c0", "c1"] and len(df) == n
    np.testing.assert_array_equal(df.to_numpy(), out)


@pytest.mark.parametrize("kind", KINDS)
def test_sample_rejects_non_positive_n(kind, fitted):
    with pytest.raises(ValueError):
        fitted[1][kind].sample(0)


@pytest.mark.parametrize("kind", ["B", "D"])
def test_dim_one(kind):
    data = make_data(dim=1)
    solver = make_solver(kind, dim=1).fit(data)
    for n in (1, 5):
        out = solver.sample(n, seed=0)
        assert out.shape == (n, 1) and np.isfinite(out).all()


@pytest.mark.parametrize("kind", ["B", "E"])
def test_ignored_steps_argument_was_removed(kind, fitted):
    assert "steps" not in inspect.signature(SOLVER_CLS[kind].sample).parameters
    with pytest.raises(TypeError):
        fitted[1][kind].sample(3, seed=0, steps=2)


def test_ct_mlp_steps_argument_is_honoured(fitted):
    solver = fitted[1]["A"]
    calls = []
    hook = solver.model.net_b.register_forward_hook(lambda *_: calls.append(1))
    try:
        solver.sample(3, seed=0, steps=7)
    finally:
        hook.remove()
    assert len(calls) == 7


# ---------------------------------------------------------------------------
# 6. seeds (S4, A2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_sample_seed_semantics(kind, fitted):
    solver = fitted[1][kind]
    torch.manual_seed(0)
    a = solver.sample(32, seed=11)
    torch.manual_seed(999)                      # the global RNG must not matter
    np.random.seed(999)
    b = solver.sample(32, seed=11)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, solver.sample(32, seed=12))

    torch.manual_seed(0)                        # seed=None: fresh entropy, even under a fixed global seed
    c = solver.sample(32)
    torch.manual_seed(0)
    d = solver.sample(32)
    assert not np.allclose(c, d)


@pytest.mark.parametrize("kind", KINDS)
def test_fit_is_reproducible_from_cfg_seed_alone(kind):
    data = make_data()
    torch.manual_seed(1)
    np.random.seed(1)
    first = make_solver(kind, fb_sequence=("b", "f")).fit(data)
    torch.manual_seed(2)
    np.random.seed(2)
    second = make_solver(kind, fb_sequence=("b", "f")).fit(data)
    np.testing.assert_array_equal(first.sample(16, seed=4), second.sample(16, seed=4))

    other = make_solver(kind, fb_sequence=("b", "f"), seed=99).fit(data)
    assert not np.allclose(first.sample(16, seed=4), other.sample(16, seed=4))


def test_fit_does_not_touch_the_global_torch_rng():
    data = make_data()
    for kind in ("A", "C"):
        torch.manual_seed(1234)
        expected = torch.rand(3)
        torch.manual_seed(1234)
        make_solver(kind, fb_sequence=("b",)).fit(data).sample(4, seed=0)
        assert torch.equal(torch.rand(3), expected)


class _ZeroDrift(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_input = None

    def forward(self, z, t):
        if self.first_input is None:
            self.first_input = z.clone()
        return torch.zeros_like(z)


def test_ct_mlp_first_brownian_increment_is_independent_of_start():
    """
    A2. One step, zero drift: out = zstart + sigma * eps_1. With the start and the
    SDE noise drawn from two generators seeded alike, eps_1 == zstart (corr = 1).
    n = 5000 x 2 scalars: |corr| < 5/sqrt(M) = 0.05 for independent draws.
    """
    sigma, n = 0.5, 5000
    solver = make_solver("A", fb_sequence=("b",), sigma=sigma).fit(make_data())
    stub = _ZeroDrift()
    solver.model.net_b = stub

    out = solver.sample(n, seed=21, steps=1)
    zstart = stub.first_input.numpy()
    eps1 = (out - zstart) / sigma

    assert not np.allclose(eps1, zstart, atol=1e-3)
    corr = np.corrcoef(eps1.ravel(), zstart.ravel())[0, 1]
    assert abs(corr) < 5.0 / math.sqrt(eps1.size)
    assert abs(eps1.std() - 1.0) < 5.0 / math.sqrt(2.0 * eps1.size)
    assert abs(zstart.std() - 1.0) < 5.0 / math.sqrt(2.0 * zstart.size)


class _ZeroStepField:
    def __init__(self):
        self.first_input = None

    def _drift(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.first_input is None:
            self.first_input = x.copy()
        return np.zeros_like(x)

    def predict(self, x, t=0.0):
        return self._drift(x)

    def predict_step(self, k, x, **_):
        return self._drift(x)


@pytest.mark.parametrize("kind", ["B", "C", "D"])
def test_numpy_solvers_first_increment_is_independent_of_start(kind):
    sigma, n = 0.5, 5000
    solver = make_solver(kind, fb_sequence=("b",), sigma=sigma, num_steps=1).fit(make_data())
    stub = _ZeroStepField()
    solver.field_b = stub

    out = solver.sample(n, seed=21)
    eps1 = (out - stub.first_input) / sigma
    corr = np.corrcoef(eps1.ravel(), stub.first_input.ravel())[0, 1]
    assert abs(corr) < 5.0 / math.sqrt(eps1.size)
    assert abs(eps1.std() - 1.0) < 5.0 / math.sqrt(2.0 * eps1.size)


# ---------------------------------------------------------------------------
# 7. checkpoints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_checkpoint_round_trip_reproduces_samples(kind, fitted, tmp_path):
    data, solvers = fitted
    solver = solvers[kind]
    path = tmp_path / f"{kind}.ckpt"
    solver.save_checkpoint(path)
    loaded = SOLVER_CLS[kind].load_checkpoint(path)

    assert loaded is not solver
    np.testing.assert_array_equal(loaded.sample(33, seed=17), solver.sample(33, seed=17))
    assert loaded.stage_log == solver.stage_log
    assert loaded.variant_id == solver.variant_id
    assert dataclasses.asdict(loaded.cfg) == dataclasses.asdict(solver.cfg)
    assert list(loaded.columns_) == list(data.columns)
    assert list(loaded.sample_df(2, seed=0).columns) == list(data.columns)
    if kind in ("C", "D"):
        np.testing.assert_array_equal(loaded.t_grid_f, solver.t_grid_f)
        np.testing.assert_array_equal(loaded.t_grid_b, solver.t_grid_b)
    if kind == "E":
        assert loaded.feature_order_ == solver.feature_order_
        assert loaded.parents_ == solver.parents_
        assert loaded.context_idx_ == solver.context_idx_

    # the forward model is restored as well, not only the generator
    zstart = data.to_numpy(dtype=np.float32)[:8]
    start = torch.from_numpy(zstart) if kind == "A" else zstart
    np.testing.assert_array_equal(_simulate(loaded, kind, "f", start, 3), _simulate(solver, kind, "f", start, 3))


@pytest.mark.parametrize("kind", KINDS)
def test_training_rows_are_neither_retained_nor_checkpointed(kind, fitted, tmp_path):
    data, solvers = fitted
    solver = solvers[kind]
    X = data.to_numpy(dtype=np.float32)

    def holds_rows(obj) -> bool:
        arr = _np(obj) if isinstance(obj, (np.ndarray, torch.Tensor)) else None
        return arr is not None and arr.shape == X.shape and np.array_equal(arr.astype(np.float32), X)

    for name, value in vars(solver).items():
        assert not holds_rows(value), f"solver.{name} retains the training rows"
    assert not hasattr(solver, "_x_data")

    path = tmp_path / f"{kind}.ckpt"
    solver.save_checkpoint(path)
    blob = path.read_bytes()
    for row in X[:5]:                       # raw float32 bytes of whole training rows
        assert row.tobytes() not in blob


def test_checkpoint_of_wrong_solver_is_rejected(fitted, tmp_path):
    path = tmp_path / "d.ckpt"
    fitted[1]["D"].save_checkpoint(path)
    with pytest.raises(ValueError, match="checkpoint"):
        SOLVER_CLS["B"].load_checkpoint(path)


def test_unfitted_checkpoint_stays_unfitted(tmp_path):
    for kind in KINDS:
        path = tmp_path / f"{kind}.ckpt"
        make_solver(kind).save_checkpoint(path)
        with pytest.raises(RuntimeError):
            SOLVER_CLS[kind].load_checkpoint(path).sample(2)


# ---------------------------------------------------------------------------
# 8a. effective parameters of the CT MLP solver (A5)
# ---------------------------------------------------------------------------
def _linears(net):
    return [m for m in net.modules() if isinstance(m, nn.Linear)]


@pytest.mark.parametrize("fb", ["f", "b"])
def test_ct_mlp_architecture_follows_config(fb):
    solver = make_solver("A", dim=3, hidden_dim=24, n_layers=3, dropout=0.25, time_emb_dim=6,
                         time_emb_max_period=50.0, time_emb_learnable_scale=True)
    net = solver.model.net(fb)
    lin = _linears(net)
    assert len(lin) == 3 + 1
    assert lin[0].in_features == 3 + 6
    assert [l.out_features for l in lin] == [24, 24, 24, 3]
    drops = [m for m in net.modules() if isinstance(m, nn.Dropout)]
    assert len(drops) == 3 and all(d.p == 0.25 for d in drops)
    assert net.time_emb.dim == 6 and net.time_emb.max_period == 50.0
    assert isinstance(net.time_emb.scale, nn.Parameter)

    default = make_solver("A", dim=3)
    assert [l.out_features for l in _linears(default.model.net(fb))] == [16, 16, 3]
    assert not any(isinstance(m, nn.Dropout) for m in default.model.net(fb).modules())
    assert not isinstance(default.model.net(fb).time_emb.scale, nn.Parameter)


def test_ct_mlp_max_period_changes_the_embedding():
    t = torch.full((1, 1), 0.37)
    a = make_solver("A", time_emb_max_period=10.0).model.net_b.time_emb(t)
    b = make_solver("A", time_emb_max_period=10_000.0).model.net_b.time_emb(t)
    assert not torch.allclose(a, b)


def test_ct_mlp_training_parameters_are_effective():
    data = make_data()
    init = make_solver("A").model.net_b.state_dict()

    frozen = make_solver("A", fb_sequence=("b",), lr=0.0).fit(data)
    for key, value in frozen.model.net_b.state_dict().items():
        assert torch.equal(value, init[key]), "lr=0 must leave the weights at their seeded init"

    trained = make_solver("A", fb_sequence=("b",), lr=1e-2).fit(data)
    assert any(not torch.equal(v, init[k]) for k, v in trained.model.net_b.state_dict().items())
    # the untrained direction keeps its init
    for key, value in trained.model.net_f.state_dict().items():
        assert torch.equal(value, make_solver("A").model.net_f.state_dict()[key])

    assert make_solver("A", fb_sequence=("b",), inner_iters=5).fit(data).n_updates == 5
    assert make_solver("A", fb_sequence=("b", "f"), inner_iters=3).fit(data).n_updates == 6


def _flat_weights(solver) -> torch.Tensor:
    return torch.cat([v.flatten() for v in solver.model.net_b.state_dict().values()])


def test_ct_mlp_loss_weight_decay_and_batch_size_are_effective():
    data = make_data()
    base = _flat_weights(make_solver("A", fb_sequence=("b",)).fit(data))
    # same seed and data: any difference is caused by the one field that changed
    for change in (dict(loss_kind="huber"), dict(loss_reduction="sum"), dict(weight_decay=0.5),
                   dict(batch_size=BATCH // 2), dict(sigma=0.9), dict(eps=0.2)):
        other = _flat_weights(make_solver("A", fb_sequence=("b",), **change).fit(data))
        assert not torch.equal(base, other), f"{change} had no effect on training"
    assert torch.equal(base, _flat_weights(make_solver("A", fb_sequence=("b",)).fit(data)))

    with pytest.raises(ValueError, match="loss kind"):
        make_solver("A", fb_sequence=("b",), loss_kind="nope").fit(data)


def test_ct_mlp_has_no_unused_trainer_and_refit_resets_history():
    solver = make_solver("A")
    for name in ("trainer", "trainer_cfg", "integrator"):
        assert not hasattr(solver, name)

    data = make_data()
    solver.fit(data)
    first = solver.sample(8, seed=2)
    solver.fit(data)
    assert len(solver.snapshots) == len(solver.cfg.fb_sequence) == len(solver.stage_log)
    assert [s["fb"] for s in solver.snapshots] == list(solver.cfg.fb_sequence)
    np.testing.assert_array_equal(solver.sample(8, seed=2), first)      # a refit is a fresh fit

    # the last 'b' snapshot is the generator that sample() uses
    last_b = [s for s in solver.snapshots if s["fb"] == "b"][-1]["state"]
    for key, value in solver.model.net_b.state_dict().items():
        assert torch.equal(value.cpu(), last_b[key])


@pytest.mark.parametrize("kind", ["B", "C", "D", "E"])
def test_refit_resets_stage_log(kind):
    data = make_data()
    solver = make_solver(kind, fb_sequence=("b", "f")).fit(data)
    first = solver.sample(8, seed=2)
    solver.fit(data)
    assert len(solver.stage_log) == 2
    np.testing.assert_array_equal(solver.sample(8, seed=2), first)


# ---------------------------------------------------------------------------
# E2. structure of the feature-wise solver
# ---------------------------------------------------------------------------
def make_data3(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    a = rng.normal(size=n)
    b = a + 0.1 * rng.normal(size=n)
    c = rng.normal(size=n)
    return pd.DataFrame(np.stack([a, b, c], axis=1).astype(np.float32), columns=["a", "b", "c"])


def test_autoregressive_structure_is_the_full_chain():
    data = make_data3()
    solver = make_solver("E", fb_sequence=("b",), feature_order=["c", "a", "b"]).fit(data)
    assert solver.feature_order_ == ["c", "a", "b"]
    assert solver.parents_ == {"c": [], "a": ["c"], "b": ["c", "a"]}
    assert solver.context_idx_ == {2: [], 0: [2], 1: [2, 0]}
    assert solver.sample(4, seed=0).shape == (4, 3)

    default = make_solver("E", fb_sequence=("b",)).fit(data)
    assert default.cfg.structure == "autoregressive"
    assert default.parents_ == {"a": [], "b": ["a"], "c": ["a", "b"]}


def test_feature_order_must_be_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        make_solver("E", feature_order=["a", "b"]).fit(make_data3())
    with pytest.raises(ValueError, match="permutation"):
        make_solver("E", feature_order=["a", "b", "b"]).fit(make_data3())


def test_map_structure_keeps_explicit_parent_order():
    data = make_data3()
    cmap = {"a": [], "b": [], "c": ["b", "a"]}
    solver = make_solver("E", fb_sequence=("b",), structure="map", context_cols_map=cmap).fit(data)
    assert solver.parents_ == cmap
    assert solver.context_idx_[2] == [1, 0]                 # verbatim order b, a
    assert solver.feature_order_.index("c") == 2
    # the fitted model of 'c' has exactly [x_c, b, a] as features
    assert solver.fields_b_[2].models[0].feature_names_ == ["0", "1", "2"]
    assert solver.fields_b_[0].models[0].feature_names_ == ["0"]


@pytest.mark.parametrize("cmap,order,match", [
    ({"a": [], "b": ["a"]}, None, "missing"),                          # no silent fallback for 'c'
    ({"a": [], "b": [], "c": [], "zzz": ["a"]}, None, "not dataset columns"),
    ({"a": [], "b": ["nope"], "c": []}, None, "unknown columns"),
    ({"a": ["a"], "b": [], "c": []}, None, "own parent"),
    ({"a": ["b"], "b": ["a"], "c": []}, None, "cyclic"),
    ({"a": [], "b": ["c"], "c": []}, ["a", "b", "c"], "before its parents"),
])
def test_map_structure_rejects_bad_maps(cmap, order, match):
    solver = make_solver("E", structure="map", context_cols_map=cmap, feature_order=order)
    with pytest.raises(ValueError, match=match):
        solver.fit(make_data3())


def test_structure_options_that_would_be_ignored_are_rejected():
    with pytest.raises(ValueError, match="context_cols_map"):
        make_solver("E", structure="autoregressive", context_cols_map={"a": []})
    with pytest.raises(ValueError, match="context_cols_map"):
        make_solver("E", structure="map")
    with pytest.raises(ValueError, match="feature_order"):
        make_solver("E", structure="learned", feature_order=["a", "b", "c"])
    with pytest.raises(ValueError, match="structure"):
        make_solver("E", structure="dag")


def test_learned_structure_is_fitted_on_the_fit_rows_only(monkeypatch):
    data = make_data3()
    seen = []

    def fake_learn_dag(df, n_bins=5):
        seen.append((df.copy(), n_bins))
        return LearnedDAG(order=["c", "b", "a"], parents={"c": [], "b": ["c"], "a": ["c", "b"]},
                          fit_row_count=len(df), n_bins=n_bins)

    monkeypatch.setattr(mod_e, "learn_dag", fake_learn_dag)
    solver = make_solver("E", fb_sequence=("b",), structure="learned", structure_n_bins=4).fit(data)

    assert len(seen) == 1
    df_seen, n_bins = seen[0]
    assert n_bins == 4
    assert list(df_seen.columns) == ["a", "b", "c"]
    np.testing.assert_array_equal(df_seen.to_numpy(dtype=np.float32), data.to_numpy(dtype=np.float32))
    assert solver.feature_order_ == ["c", "b", "a"]
    assert solver.parents_ == {"c": [], "b": ["c"], "a": ["c", "b"]}
    assert solver.context_idx_ == {2: [], 1: [2], 0: [2, 1]}
    assert solver.dag_.fit_row_count == len(data)


def test_learned_structure_end_to_end(tmp_path):
    data = make_data3(n=80)
    solver = make_solver("E", fb_sequence=("b",), structure="learned").fit(data)
    assert solver.dag_ is not None and solver.dag_.fit_row_count == 80
    assert sorted(solver.feature_order_) == ["a", "b", "c"]
    # a and b are almost collinear: the search must link them, in one direction
    assert ("a" in solver.parents_["b"]) != ("b" in solver.parents_["a"])
    placed = set()
    for col in solver.feature_order_:            # topological
        assert set(solver.parents_[col]) <= placed
        placed.add(col)

    path = tmp_path / "e.ckpt"
    solver.save_checkpoint(path)
    loaded = mod_e.FeaturewiseDSBMBoostSolver.load_checkpoint(path)
    assert loaded.dag_.state() == solver.dag_.state()
    assert loaded.parents_ == solver.parents_
    np.testing.assert_array_equal(loaded.sample(9, seed=1), solver.sample(9, seed=1))


class _RowTaggedField:
    """
    Exact backward drift towards a per-row target; records the parent context it is
    given. Used to prove parents are read from the SAME generated row, in the stored
    parent order.
    """

    def __init__(self, targets: np.ndarray, N: int):
        self.targets = targets.reshape(-1, 1).astype(np.float64)
        self.N = N
        self.contexts = []

    def predict_step(self, k, X_feat, **_):
        X_feat = np.asarray(X_feat, dtype=np.float64)
        self.contexts.append(X_feat[:, 1:].copy())
        return (self.targets - X_feat[:, :1]) / ((k + 1) / self.N)


def test_generation_reads_parents_from_the_same_generated_row():
    n, N = 6, 4
    cmap = {"a": [], "b": [], "c": ["b", "a"]}
    solver = make_solver("E", fb_sequence=("b",), structure="map", context_cols_map=cmap, num_steps=N)
    solver._setup(make_data3())
    stubs = {0: _RowTaggedField(np.arange(n) * 1.0, N),
             1: _RowTaggedField(np.arange(n) * 10.0 + 100.0, N),
             2: _RowTaggedField(np.zeros(n), N)}
    solver.fields_b_ = stubs

    out = solver._sample_with_direction(np.zeros((n, 3), dtype=np.float32), "b", noise=False)

    np.testing.assert_allclose(out[:, 0], np.arange(n), atol=1e-4)
    np.testing.assert_allclose(out[:, 1], np.arange(n) * 10.0 + 100.0, atol=1e-4)
    assert len(stubs[2].contexts) == N
    for ctx in stubs[2].contexts:                   # [b, a] of the SAME row, at every step
        np.testing.assert_array_equal(ctx[:, 0], out[:, 1])
        np.testing.assert_array_equal(ctx[:, 1], out[:, 0])
    assert all(ctx.shape[1] == 0 for ctx in stubs[0].contexts + stubs[1].contexts)
