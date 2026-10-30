"""
Mathematical contract of the five IMF-DSBM solvers, checked against oracles that
do not share code (or formulas) with the implementation.

Reference: unit-time Brownian bridge with diffusion sigma, x0 = data (t=0),
x1 = prior (t=1):
    X_t = (1-t) x0 + t x1 + sigma sqrt(t(1-t)) eps
    forward  drift target  u_f = (x1 - X_t) / (1 - t)
    backward drift target  u_b = (x0 - X_t) / t          (drift in s = 1 - t)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from sbtab.models.boosted.catboost_continuous_joint import CatBoostContinuousFieldConfig
from sbtab.models.boosted.catboost_discrete_joint import CatBoostDiscreteFieldConfig, CatBoostDiscreteJoint
from sbtab.models.boosted.catboost_discrete_scalar import CatBoostDiscreteScalar, CatBoostScalarConfig
from sbtab.models.neural.mlp_discrete_joint import MLPTimeDiscretizedField, StepMLPJointConfig
from sbtab.solvers.continuous_time.joint_distribution.boosting.imf_dsbm.solver import (
    IMFDSBMContinuousJointCatBoostConfig,
    IMFDSBMContinuousJointCatBoostSolver,
)
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import IMFDSBMConfig, IMFDSBMSolver
from sbtab.solvers.discrete_time.feature_wise.boosting.imf_dsbm_featurewise_boost.solver import (
    FeaturewiseDSBMBoostConfig,
    FeaturewiseDSBMBoostSolver,
)
from sbtab.solvers.discrete_time.joint_distribution.boosting.imf_dsbm_boost.solver import (
    IMFDSBMBoostConfig,
    IMFDSBMBoostSolver,
)
from sbtab.solvers.discrete_time.joint_distribution.mlp.imf_dsbm.solver import (
    IMFDSBMDiscreteJointMLPConfig,
    IMFDSBMDiscreteJointMLPSolver,
)

DIM = 2
COLS = ["c0", "c1"]
CT_KINDS = ["A", "B"]
DT_KINDS = ["C", "D", "E"]


def make_solver(kind: str, *, sigma: float, num_steps: int = 4, eps: float = 1e-3, fb_sequence=("b", "f")):
    """Unfitted solver with a tiny model config (no test here trains a real model)."""
    common = dict(fb_sequence=fb_sequence, num_steps=num_steps, sigma=sigma, eps=eps, seed=7)
    if kind == "A":
        return IMFDSBMSolver(DIM, IMFDSBMConfig(inner_iters=2, batch_size=8, hidden_dim=8, n_layers=1,
                                                time_emb_dim=4, **common))
    if kind == "B":
        return IMFDSBMContinuousJointCatBoostSolver(
            DIM, IMFDSBMContinuousJointCatBoostConfig(field=CatBoostContinuousFieldConfig(iterations=2, depth=1),
                                                      **common))
    if kind == "C":
        return IMFDSBMDiscreteJointMLPSolver(
            DIM, IMFDSBMDiscreteJointMLPConfig(field=StepMLPJointConfig(hidden_dim=8, n_layers=1), **common))
    if kind == "D":
        return IMFDSBMBoostSolver(
            DIM, IMFDSBMBoostConfig(catboost=CatBoostDiscreteFieldConfig(iterations=2, depth=1), **common))
    if kind == "E":
        return FeaturewiseDSBMBoostSolver(
            FeaturewiseDSBMBoostConfig(catboost=CatBoostScalarConfig(iterations=2, depth=1), **common))
    raise KeyError(kind)


# ---------------------------------------------------------------------------
# 1. Brownian target algebra
# ---------------------------------------------------------------------------
def _bridge_oracle(x0, x1, eps, t, sigma):
    """float64 oracle in the CONDITIONAL-DRIFT form (the solvers use the eps form)."""
    xt = (1.0 - t) * x0 + t * x1 + sigma * np.sqrt(t * (1.0 - t)) * eps
    return xt, (x1 - xt) / (1.0 - t), (x0 - xt) / t


def _endpoints(n=256, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, DIM)) * 1.5 + 0.5
    x1 = rng.normal(size=(n, DIM))
    eps = rng.normal(size=(n, DIM))
    return x0, x1, eps


# Tolerance: values are O(15) at most (|x1-x0| <= ~8, sigma sqrt(19) |eps| <= ~10);
# float32 carries ~6e-8 relative error per operation and the eps form uses ~5
# operations, i.e. <= ~5e-6 absolute. rtol = atol = 1e-5 leaves a factor 2-4.
TUPLE_TOL = dict(rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", CT_KINDS)
@pytest.mark.parametrize("fb", ["f", "b"])
def test_ct_target_equals_conditional_bridge_drift(kind, fb):
    sigma = 0.7
    solver = make_solver(kind, sigma=sigma)
    x0, x1, eps = _endpoints()
    t = np.random.default_rng(1).uniform(0.05, 0.95, size=(len(x0), 1))
    xt_ref, uf_ref, ub_ref = _bridge_oracle(x0, x1, eps, t, sigma)
    u_ref = uf_ref if fb == "f" else ub_ref

    if kind == "A":
        pairs = torch.stack([torch.tensor(x0, dtype=torch.float32), torch.tensor(x1, dtype=torch.float32)], dim=1)
        xt, t_out, target = solver._dsbm_train_tuple(
            pairs, fb, t=torch.tensor(t, dtype=torch.float32), noise=torch.tensor(eps, dtype=torch.float32))
        xt, t_out, target = xt.numpy(), t_out.numpy(), target.numpy()
    else:
        xt, t_out, target = solver._dsbm_train_tuple(
            x0.astype(np.float32), x1.astype(np.float32), fb, t=t, noise=eps)

    np.testing.assert_allclose(t_out, t, rtol=1e-6)
    np.testing.assert_allclose(xt, xt_ref, **TUPLE_TOL)
    np.testing.assert_allclose(target, u_ref, **TUPLE_TOL)


@pytest.mark.parametrize("kind", DT_KINDS)
@pytest.mark.parametrize("fb", ["f", "b"])
@pytest.mark.parametrize("t", [0.0, 0.125, 0.5, 0.9, 1.0])
def test_dt_target_equals_conditional_bridge_drift(kind, fb, t):
    sigma = 0.7
    solver = make_solver(kind, sigma=sigma)
    x0, x1, eps = _endpoints()
    if kind == "E":  # scalar bridges, one column at a time
        x0, x1, eps = x0[:, 0], x1[:, 0], eps[:, 0]

    if (fb == "f" and t == 1.0) or (fb == "b" and t == 0.0):
        # the drift of this direction is singular here; the builder must refuse, not emit inf/nan
        with pytest.raises(ValueError):
            solver._dsbm_train_tuple(x0.astype(np.float32), x1.astype(np.float32), t, fb, noise=eps)
        return

    xt, target = solver._dsbm_train_tuple(x0.astype(np.float32), x1.astype(np.float32), t, fb, noise=eps)
    assert np.isfinite(xt).all() and np.isfinite(target).all()

    # at the closed endpoint of each direction the conditional form is 0/0-free:
    # forward t=0 -> X_t = x0, u_f = x1 - x0; backward t=1 -> X_t = x1, u_b = x0 - x1
    xt_ref = (1.0 - t) * x0 + t * x1 + sigma * math.sqrt(t * (1.0 - t)) * eps
    u_ref = (x1 - xt_ref) / (1.0 - t) if fb == "f" else (x0 - xt_ref) / t
    np.testing.assert_allclose(xt, xt_ref, **TUPLE_TOL)
    np.testing.assert_allclose(target, u_ref, **TUPLE_TOL)


@pytest.mark.parametrize("kind", CT_KINDS)
@pytest.mark.parametrize("fb", ["f", "b"])
def test_ct_internal_draws_use_same_t_and_noise_for_state_and_target(kind, fb):
    """
    With internally drawn (t, eps) the returned triple must satisfy the conditional
    identity. eps=0.05 keeps t in [0.05, 0.95].

    Tolerance: recomputing (x_end - X_t)/(1-t) from float32 outputs amplifies the
    rounding of X_t (~2.4e-7 |X_t|, |X_t| <= ~8) by 1/min(t, 1-t) <= 20, i.e.
    <= ~4e-5; the target itself carries <= ~5e-6. atol = 2e-4 is a factor ~4 above
    that, while independent draws for X_t and the target would be off by O(1).
    """
    sigma = 0.7
    solver = make_solver(kind, sigma=sigma, eps=0.05)
    x0, x1, _ = _endpoints(n=512, seed=3)
    x0 = x0.astype(np.float32)
    x1 = x1.astype(np.float32)

    if kind == "A":
        pairs = torch.stack([torch.from_numpy(x0), torch.from_numpy(x1)], dim=1)
        xt, t, target = solver._dsbm_train_tuple(pairs, fb, generator=solver._make_generator(11))
        xt, t, target = xt.numpy(), t.numpy(), target.numpy()
    else:
        xt, t, target = solver._dsbm_train_tuple(x0, x1, fb, generator=solver._make_generator(11))

    assert t.shape == (len(x0), 1)
    assert 0.05 <= float(t.min()) and float(t.max()) <= 0.95
    assert float(t.std()) > 0.2          # U(0.05, 0.95) has std 0.26: t is really random per row
    xt64, t64 = xt.astype(np.float64), t.astype(np.float64)
    u_ref = (x1 - xt64) / (1.0 - t64) if fb == "f" else (x0 - xt64) / t64
    np.testing.assert_allclose(target, u_ref, rtol=0, atol=2e-4)
    # and the state really is noisy (sigma > 0): residual to the chord has the bridge std
    resid = xt64 - ((1.0 - t64) * x0 + t64 * x1)
    expected = sigma * np.sqrt(np.mean(t64 * (1.0 - t64)))
    assert abs(resid.std() - expected) < 0.1 * expected


# ---------------------------------------------------------------------------
# 2. Analytic process: independent N(0,1) endpoints, sigma = sqrt(2)
# ---------------------------------------------------------------------------
# Var X_t = (1-t)^2 + t^2 + 2 t (1-t) = 1 for all t and Cov(x1, X_t) = t, so
# E[x1 | X_t] = t X_t and the exact forward drift is (t x - x)/(1-t) = -x.
# By symmetry the backward drift (in s = 1-t) is -x as well: both directions are the
# stationary OU process dX = -X dt + sqrt(2) dW with law N(0,1).
#
# Euler-Maruyama with step h: x' = (1-h) x + sqrt(2h) xi, so the variance obeys
#   v' = (1-h)^2 v + 2h,  fixed point v* = 1 / (1 - h/2),
#   v_N = v* + (1 - v*) (1-h)^(2N)   from v_0 = 1.
# The deterministic Euler bias v_N - 1 is ~ (h/2)(1 - e^-2): 0.1285 for N=4 and
# 0.0588 for N=8, i.e. halving the step roughly halves it.
#
# Monte Carlo: n = 100_000 rows x 2 columns = M = 200_000 iid scalars (seed 123).
# The scheme is linear in Gaussians, so its output is exactly Gaussian and
#   SE(mean) = sqrt(v/M) ~ 2.4e-3,  SE(s^2) = v sqrt(2/(M-1)) ~ 3.6e-3.
# Tolerances are 5 SE (two-sided miss probability < 1e-6 per check).
OU_SIGMA = math.sqrt(2.0)
OU_ROWS = 100_000


def _euler_variance(N: int) -> float:
    h = 1.0 / N
    v_star = 1.0 / (1.0 - h / 2.0)
    return v_star + (1.0 - v_star) * (1.0 - h) ** (2 * N)


class _OUFieldStub:
    """Stand-in for a fitted CatBoost CT field: the exact drift -x at every t."""

    def predict(self, x, t=0.0):
        return -np.asarray(x, dtype=np.float32)


def _simulate_ou(kind: str, direction: str, N: int) -> np.ndarray:
    solver = make_solver(kind, sigma=OU_SIGMA, num_steps=N)
    gen = solver._make_generator(123)
    zstart = torch.randn((OU_ROWS, DIM), generator=gen)
    if kind == "A":
        out = solver._sample_sde(net=lambda z, t: -z, fb=direction, zstart=zstart, steps=N,
                                 generator=gen, noise=True)
        return out.numpy().astype(np.float64)
    setattr(solver, "field_f" if direction == "f" else "field_b", _OUFieldStub())
    out = solver._sample_with_direction(zstart.numpy(), direction, generator=gen, noise=True)
    return out.astype(np.float64)


@pytest.mark.parametrize("kind", CT_KINDS)
@pytest.mark.parametrize("direction", ["f", "b"])
def test_ct_integrator_reproduces_stationary_ou(kind, direction):
    M = OU_ROWS * DIM
    biases = {}
    for N in (4, 8):
        x = _simulate_ou(kind, direction, N)
        v_ref = _euler_variance(N)
        assert abs(x.mean()) < 5.0 * math.sqrt(v_ref / M)
        assert abs(x.var() - v_ref) < 5.0 * v_ref * math.sqrt(2.0 / (M - 1))
        biases[N] = x.var() - 1.0

    # deterministic Euler bias: positive, and halving the step reduces it. The
    # analytic gap is 0.1285 - 0.0588 = 0.0697 ~ 14 SE of the difference.
    assert biases[4] > biases[8] > 0.0
    assert biases[4] - biases[8] > 0.0697 - 5.0 * math.sqrt(2) * 3.6e-3


def test_ct_noiseless_sampler_is_not_marginal_preserving():
    """
    Same oracle drift without noise: x_N = (1-h)^N x_0, variance (1-h)^(2N) ~ e^-2,
    far from the stationary 1. This is why noise=False is labelled a heuristic.
    """
    solver = make_solver("A", sigma=OU_SIGMA, num_steps=8)
    gen = solver._make_generator(123)
    zstart = torch.randn((20_000, DIM), generator=gen)
    out = solver._sample_sde(net=lambda z, t: -z, fb="b", zstart=zstart, steps=8, generator=gen, noise=False)
    expected = (1.0 - 1.0 / 8) ** 16 * float(zstart.var())
    assert abs(float(out.var()) - expected) < 1e-4
    assert float(out.var()) < 0.2


# ---------------------------------------------------------------------------
# A4. the time fed to a CT model stays inside the trained range [eps, 1-eps]
# ---------------------------------------------------------------------------
class _RecordingCTField:
    def __init__(self):
        self.times = []

    def predict(self, x, t=0.0):
        self.times.append(float(np.asarray(t).reshape(-1)[0]))
        return np.ones_like(np.asarray(x, dtype=np.float32))


@pytest.mark.parametrize("kind", CT_KINDS)
@pytest.mark.parametrize("direction", ["f", "b"])
@pytest.mark.parametrize("N,eps", [(4, 0.01), (8, 0.3)])   # 1/N > eps and 1/N < eps
def test_ct_model_time_is_clamped_but_state_uses_true_dt(kind, direction, N, eps):
    solver = make_solver(kind, sigma=0.5, num_steps=N, eps=eps)
    zstart = np.zeros((3, DIM), dtype=np.float32)
    times = []

    if kind == "A":
        def net(z, t):
            times.append(float(t[0, 0]))
            return torch.ones_like(z)
        out = solver._sample_sde(net=net, fb=direction, zstart=torch.from_numpy(zstart), steps=N, noise=False)
        out = out.numpy()
    else:
        stub = _RecordingCTField()
        setattr(solver, "field_f" if direction == "f" else "field_b", stub)
        out = solver._sample_with_direction(zstart, direction, noise=False)
        times = stub.times

    state_times = [i / N if direction == "f" else 1.0 - i / N for i in range(N)]
    np.testing.assert_allclose(times, np.clip(state_times, eps, 1.0 - eps), atol=1e-6)
    assert min(times) >= eps - 1e-6 and max(times) <= 1.0 - eps + 1e-6
    # unit drift for N steps of the TRUE dt = 1/N moves the state by exactly 1
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. DT state-time oracle (S3)
# ---------------------------------------------------------------------------
# A perfectly trained model k, fitted at time t_k on a coupling whose target endpoint
# is the point mass at m, is the exact regression function
#     backward: x -> (m - x) / t_k         forward: x -> (m - x) / (1 - t_k).
# The sampler applies model k with a full step 1/N to the state at (k+1)/N
# (backward) or k/N (forward). Only if t_k is that state time does the last step
# land exactly on m. With the midpoint grid t_k = (k+0.5)/N the result is
# m - (x_start - m)/(2N-1) (5.263 from x_start = 0, m = 5, N = 10).
#
# The training times are RECORDED from a real fit() call (with the model fitting
# itself stubbed out), not read from a solver attribute.
POINT_MASS = 5.0


class _ExactPointMassField:
    """Exact drift of model k for a point-mass target, at the time model k was trained at."""

    def __init__(self, direction: str, train_times: dict, scalar: bool = False):
        self.direction = direction
        self.train_times = train_times
        self.scalar = scalar            # feature-wise models see [x_j, parents...] and return (n, 1)

    def predict_step(self, k, X_feat, **_):
        x = np.asarray(X_feat, dtype=np.float64)
        if self.scalar:
            x = x[:, :1]
        t = self.train_times[k]
        return (POINT_MASS - x) / t if self.direction == "b" else (POINT_MASS - x) / (1.0 - t)


def _record_training_times(kind: str, direction: str, N: int, monkeypatch, sigma: float):
    """fit() a single-stage solver with model fitting stubbed; return (solver, {k: t})."""
    solver = make_solver(kind, sigma=sigma, num_steps=N, fb_sequence=(direction,))
    state = {"t": None, "fb": None}
    times: dict = {}

    original_tuple = solver._dsbm_train_tuple

    def spy_tuple(z0, z1, t, fb, **kw):
        state["t"], state["fb"] = float(t), fb
        return original_tuple(z0, z1, t, fb, **kw)

    def fake_fit_step(self, k, *args, **kwargs):
        assert state["fb"] == direction
        assert times.setdefault(int(k), state["t"]) == state["t"]   # all columns share one grid
        self.models[k] = object()
        return 1

    monkeypatch.setattr(solver, "_dsbm_train_tuple", spy_tuple)
    for cls in (MLPTimeDiscretizedField, CatBoostDiscreteJoint, CatBoostDiscreteScalar):
        monkeypatch.setattr(cls, "fit_step", fake_fit_step)

    rng = np.random.default_rng(0)
    solver.fit(pd.DataFrame(rng.normal(size=(12, DIM)).astype(np.float32), columns=COLS))
    assert sorted(times) == list(range(N))
    return solver, times


def _install_exact_field(solver, kind: str, direction: str, times: dict) -> None:
    if kind == "E":
        stub = _ExactPointMassField(direction, times, scalar=True)
        setattr(solver, "fields_f_" if direction == "f" else "fields_b_", {j: stub for j in range(DIM)})
    else:
        setattr(solver, "field_f" if direction == "f" else "field_b", _ExactPointMassField(direction, times))


@pytest.mark.parametrize("kind", DT_KINDS)
@pytest.mark.parametrize("direction", ["f", "b"])
@pytest.mark.parametrize("N", [4, 10, 32])
def test_dt_exact_drift_hits_point_mass(kind, direction, N, monkeypatch):
    solver, times = _record_training_times(kind, direction, N, monkeypatch, sigma=0.5)

    _install_exact_field(solver, kind, direction, times)
    zstart = np.array([[0.0, -3.0], [1.5, 4.0], [7.0, 0.25], [-1.0, 2.0]], dtype=np.float32)
    out = solver._sample_with_direction(zstart, direction, noise=False)

    # float32 state: the last step computes x + (5-x) with |5-x| <= ~3 -> error <= ~5e-7
    assert out.shape == zstart.shape
    np.testing.assert_allclose(out, POINT_MASS, rtol=0, atol=1e-6)

    # the contract behind it: model k is trained at the time of the state it is applied to
    expected = [(k + 1) / N if direction == "b" else k / N for k in range(N)]
    np.testing.assert_allclose([times[k] for k in range(N)], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", DT_KINDS)
@pytest.mark.parametrize("direction", ["f", "b"])
def test_dt_exact_drift_with_noise_leaves_only_last_increment(kind, direction, monkeypatch):
    """
    With noise on, the exact last step maps ANY state onto m, so the output is
    m + sigma sqrt(dt) xi exactly. n = 4000 rows x 2 cols = 8000 scalars (seed 5):
    SE(mean) = s/sqrt(8000), SE(std) ~ s/sqrt(2*8000), s = sigma sqrt(dt); 5 SE each.
    A midpoint grid shifts the mean by (m - x_start)/(2N-1) = 0.26 ~ 150 SE.
    """
    N, sigma, n = 10, 0.5, 4000
    solver, times = _record_training_times(kind, direction, N, monkeypatch, sigma=sigma)
    _install_exact_field(solver, kind, direction, times)

    out = solver._sample_with_direction(np.zeros((n, DIM), dtype=np.float32), direction,
                                        generator=solver._make_generator(5), noise=True).astype(np.float64)
    s = sigma * math.sqrt(1.0 / N)
    M = n * DIM
    assert abs(out.mean() - POINT_MASS) < 5.0 * s / math.sqrt(M)
    assert abs(out.std() - s) < 5.0 * s / math.sqrt(2.0 * M)
