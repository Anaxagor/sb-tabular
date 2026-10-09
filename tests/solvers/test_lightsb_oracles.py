"""
Independent oracles for the LightSB model (sbtab.models.sb.light_sb).

Everything runs in float64 (``model.double()``) and uses ``eps = float(model.epsilon)``.
The oracles are written in numpy from the formulas of the LightSB paper and do not
call the code under test:

  potential          v(y)      = sum_k alpha_k N(y | r_k, eps S_k)            (normalised Gaussians)
  normaliser         C(x)      = int exp(<x, y> / eps) v(y) dy
  conditional plan   pi(y | x) = sum_k w_k(x) N(y | r_k + S_k x, eps S_k),
                     w_k(x) ∝ alpha_k exp((x' S_k x + 2 r_k' x) / (2 eps))
  drift              g(x, t)   = eps grad_x log int N(y | x, eps (1 - t)) exp(|y|^2 / (2 eps)) v(y) dy
                               = (E[y | x_t = x] - x) / (1 - t)
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from sbtab.models.sb.light_sb import LightSBPotential, LightSBPotentialConfig


# --------------------------------------------------------------------------- helpers / numpy oracles
def make_model(r, S, log_alpha, eps=0.5, sampling_batch_size=100_000):
    r, S, log_alpha = np.asarray(r, float), np.asarray(S, float), np.asarray(log_alpha, float)
    K, D = r.shape
    cfg = LightSBPotentialConfig(n_potentials=K, epsilon=eps, is_diagonal=True, sampling_batch_size=sampling_batch_size)
    model = LightSBPotential(D, cfg).double().eval()
    eps64 = float(model.epsilon)
    with torch.no_grad():
        model.r.copy_(torch.tensor(r))
        model.S_log_diagonal_matrix.copy_(torch.tensor(np.log(S)))
        model.log_alpha_raw.copy_(torch.tensor(eps64 * log_alpha))   # the model stores eps * log(alpha)
    return model, eps64


def oracle_log_potential(y, r, S, log_alpha, eps, with_log_det=True):
    """log v(y) for diagonal covariances eps * S_k, in closed form. y: (N, D)."""
    quad = ((y[:, None, :] - r[None]) ** 2 / (eps * S[None])).sum(-1)                 # (N, K)
    log_norm = -0.5 * r.shape[1] * math.log(2 * math.pi)
    log_det = -0.5 * np.log(eps * S).sum(-1) if with_log_det else 0.0                # (K,)
    return logsumexp(log_alpha[None] + log_norm + log_det - 0.5 * quad, axis=1)


def oracle_conditional(x, r, S, log_alpha, eps):
    """Weights, mean and covariance of pi(. | x) for one point x: (D,)."""
    logits = ((x * S * x).sum(-1) + 2 * r @ x) / (2 * eps) + log_alpha
    w = np.exp(logits - logsumexp(logits))
    means = r + S * x
    mu = w @ means
    cov = sum(w[k] * (np.diag(eps * S[k]) + np.outer(means[k], means[k])) for k in range(len(w))) - np.outer(mu, mu)
    return w, mu, cov, means


def balanced_log_alpha(x, r, S, eps, weights):
    """log_alpha that gives the mixture pi(. | x) exactly the requested weights at this x."""
    r, S = np.asarray(r, float), np.asarray(S, float)
    return np.log(weights) - ((x * S * x).sum(-1) + 2 * r @ x) / (2 * eps)


R2 = np.array([[1.5, 0.5], [-1.0, 1.0], [0.3, -1.6]])
S2 = np.array([[0.6, 1.2], [1.0, 0.5], [0.4, 0.9]])
X_STAR = np.array([0.7, -0.4])
WEIGHTS = np.array([0.5, 0.3, 0.2])


@pytest.fixture(scope="module")
def balanced():
    model, eps = make_model(R2, S2, balanced_log_alpha(X_STAR, R2, S2, 0.5, WEIGHTS))
    log_alpha = (model.log_alpha_raw / model.epsilon).detach().numpy()
    w, mu, cov, means = oracle_conditional(X_STAR, R2, S2, log_alpha, eps)
    assert np.allclose(w, WEIGHTS, atol=1e-12) and w.max() <= 0.6          # no dominating component
    # 4th central moments of the ORACLE mixture (numpy sampler, seed 123) size the covariance tolerances
    rng = np.random.default_rng(123)
    comp = rng.choice(3, size=400_000, p=w)
    y = means[comp] + np.sqrt(eps * S2[comp]) * rng.normal(size=(400_000, 2))
    c = y - mu
    m4 = np.einsum("ni,nj->ij", c ** 2, c ** 2) / len(c)
    return dict(model=model, eps=eps, log_alpha=log_alpha, mu=mu, cov=cov, m4=m4)


# --------------------------------------------------------------------------- (i) conditional sampler
def test_conditional_sampler_matches_the_analytic_mixture_moments(balanced):
    """
    N = 200000 draws of model(x) at one x, torch seed 0, weights (0.5, 0.3, 0.2).
    Mean tolerance 5 sqrt(max diag cov / N); covariance entries 5 sqrt(m4_ij / N)
    (standard error of a sample covariance entry, m4 from the numpy oracle mixture).
    """
    N = 200_000
    model, mu, cov, m4 = balanced["model"], balanced["mu"], balanced["cov"], balanced["m4"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        y = model(torch.tensor(X_STAR).repeat(N, 1)).numpy()
    assert y.shape == (N, 2)
    assert np.all(np.abs(y.mean(0) - mu) < 5 * math.sqrt(cov.diagonal().max() / N))
    assert np.all(np.abs(np.cov(y.T) - cov) < 5 * np.sqrt(m4 / N))
    # power: the mixture structure matters - a single Gaussian with the component-averaged
    # covariance would be off by far more than the tolerance
    within = sum(w * np.diag(balanced["eps"] * s) for w, s in zip(WEIGHTS, S2))
    assert np.abs(within - cov).max() > 20 * 5 * np.sqrt(m4 / N).max()


# --------------------------------------------------------------------------- (ii) potential and normaliser
def test_log_potential_is_the_normalised_mixture_with_the_log_det_term():
    rng = np.random.default_rng(0)
    log_alpha = np.log([0.2, 0.5, 0.3]) + 0.7                 # unnormalised on purpose
    model, eps = make_model(R2, S2, log_alpha)
    y = rng.normal(size=(500, 2)) * 2.0
    got = model.get_log_potential(torch.tensor(y)).detach().numpy()
    assert np.abs(got - oracle_log_potential(y, R2, S2, log_alpha, eps)).max() < 1e-10
    # without -0.5 log det(eps S_k) the components (which have DIFFERENT S_k) are re-weighted:
    # the discrepancy is large and not a constant
    diff = got - oracle_log_potential(y, R2, S2, log_alpha, eps, with_log_det=False)
    assert np.abs(diff).min() > 0.1 and diff.std() > 1e-2


def test_log_C_quadrature_identity_1d():
    """log C(x) = log int exp(x y / eps) v(y) dy; rectangle rule, h = 0.01 on [-14, 14] (spectrally accurate)."""
    r, S = np.array([[-1.0], [0.4], [1.5]]), np.array([[0.5], [1.0], [0.3]])
    log_alpha = np.log([0.2, 0.5, 0.3]) + 0.7
    model, eps = make_model(r, S, log_alpha)
    y = torch.arange(-14.0, 14.0 + 1e-9, 0.01, dtype=torch.float64).unsqueeze(1)
    log_v = model.get_log_potential(y).detach()
    assert np.abs(log_v.numpy() - oracle_log_potential(y.numpy(), r, S, log_alpha, eps)).max() < 1e-10
    for x in (-1.2, 0.0, 0.8, 2.0):
        quad = torch.logsumexp(x * y[:, 0] / eps + log_v, dim=0) + math.log(0.01)
        got = model.get_log_C(torch.tensor([[x]], dtype=torch.float64))[0]
        assert abs(float(got - quad)) <= 1e-8
    # x = 0: C(0) = int v = sum_k alpha_k, i.e. the Gaussians are normalised
    got0 = float(model.get_log_C(torch.zeros(1, 1, dtype=torch.float64))[0])
    assert abs(got0 - logsumexp(log_alpha)) < 1e-12


def test_log_C_quadrature_identity_2d():
    """Tensor grid [-10, 10]^2 with h = 0.05 (cell volume h^2); tails beyond the box are < 1e-15."""
    log_alpha = np.log([0.2, 0.5, 0.3]) - 0.4
    model, eps = make_model(R2, S2, log_alpha)
    h = 0.05
    axis = torch.arange(-10.0, 10.0 + 1e-9, h, dtype=torch.float64)
    y = torch.cartesian_prod(axis, axis)
    log_v = model.get_log_potential(y).detach()
    xs = torch.tensor([[0.0, 0.0], [0.7, -0.4], [-1.5, 1.0], [1.2, 1.4]], dtype=torch.float64)
    got = model.get_log_C(xs).detach()
    for i, x in enumerate(xs):
        quad = torch.logsumexp((y @ x) / eps + log_v, dim=0) + math.log(h * h)
        assert abs(float(got[i] - quad)) <= 1e-8


def test_loss_is_invariant_to_a_common_shift_of_log_alpha():
    rng = np.random.default_rng(1)
    log_alpha = np.log([0.2, 0.5, 0.3])
    x0, x1 = torch.tensor(rng.normal(size=(64, 2))), torch.tensor(rng.normal(size=(64, 2)) + 1.0)

    def loss(shift):
        model, _ = make_model(R2, S2, log_alpha + shift)
        return float(model.get_log_C(x0).mean() - model.get_log_potential(x1).mean())

    assert abs(loss(0.0) - loss(1.7)) < 1e-10 and abs(loss(0.0) - loss(-3.2)) < 1e-10


# --------------------------------------------------------------------------- (iii) drift and SDE
def test_drift_at_time_zero_is_conditional_mean_minus_x(balanced):
    model, eps, log_alpha = balanced["model"], balanced["eps"], balanced["log_alpha"]
    X = np.random.default_rng(2).normal(size=(50, 2)) * 1.5
    expected = np.stack([oracle_conditional(x, R2, S2, log_alpha, eps)[1] - x for x in X])
    got = model.get_drift(torch.tensor(X), torch.zeros(50, dtype=torch.float64)).numpy()
    assert np.abs(got - expected).max() < 1e-10


@pytest.mark.parametrize("t", [0.0, 0.35, 0.8])
def test_drift_matches_1d_quadrature(t):
    """g(x, t) = (E[y | x_t = x] - x) / (1 - t), posterior ∝ N(y | x, eps (1 - t)) exp(y^2 / (2 eps)) v(y)."""
    r, S = np.array([[-1.0], [0.4], [1.5]]), np.array([[0.5], [1.0], [0.3]])
    log_alpha = np.log([0.2, 0.5, 0.3]) + 0.7
    model, eps = make_model(r, S, log_alpha)
    y = np.arange(-14.0, 14.0 + 1e-9, 0.01)
    log_v = oracle_log_potential(y[:, None], r, S, log_alpha, eps)
    xs = np.array([-1.5, -0.3, 0.0, 0.9, 2.2])
    expected = []
    for x in xs:
        log_w = -(y - x) ** 2 / (2 * eps * (1 - t)) + y ** 2 / (2 * eps) + log_v
        w = np.exp(log_w - logsumexp(log_w))
        expected.append(((w * y).sum() - x) / (1 - t))
    got = model.get_drift(torch.tensor(xs)[:, None], torch.full((len(xs),), t, dtype=torch.float64)).numpy()[:, 0]
    assert np.abs(got - np.array(expected)).max() < 1e-8


def test_refined_sde_converges_to_the_conditional_plan(balanced):
    """
    Euler-Maruyama from the fixed start x* (B = 40000 copies, generator seed 0) must
    reproduce pi(. | x*), whose mean / covariance are analytic. Error e(n) =
    ||Cov_n - Cov||_F for n = 5, 20, 100 steps.

    Bounds: EM has weak order 1, e(n) ~ C / n, so e(20) ~ e(5) / 4 and
    e(100) ~ e(20) / 5; we require a factor 2 each. At n = 100:
        e(100) <= 2 * (5 / 100) * e(5)   [first-order extrapolation of the n = 5 bias, safety factor 2]
                  + 5 * sqrt(sum_ij m4_ij / B)   [5 sigma Monte-Carlo error of the Frobenius norm]
    with m4 the 4th central moments of the oracle mixture. A wrong noise scale or
    drift leaves an O(1) error that does not decay with n.
    """
    B = 40_000
    model, mu, cov, m4 = balanced["model"], balanced["mu"], balanced["cov"], balanced["m4"]
    x0 = torch.tensor(X_STAR).repeat(B, 1)
    err_cov, err_mean = {}, {}
    for n in (5, 20, 100):
        gen = torch.Generator().manual_seed(0)
        traj = model.sample_euler_maruyama(x0, n, generator=gen)
        assert traj.shape == (B, n + 1, 2) and torch.equal(traj[:, 0], x0)
        x1 = traj[:, -1].numpy()
        err_cov[n] = np.linalg.norm(np.cov(x1.T) - cov)
        err_mean[n] = np.linalg.norm(x1.mean(0) - mu)

    assert err_cov[20] < err_cov[5] / 2 and err_cov[100] < err_cov[20] / 2
    mc_cov = 5 * math.sqrt(m4.sum() / B)
    assert err_cov[100] < 2 * (5 / 100) * err_cov[5] + mc_cov
    assert err_mean[100] < 2 * (5 / 100) * err_mean[5] + 5 * math.sqrt(np.trace(cov) / B)
    # power: the coarse discretisation violates the n = 100 bound, so the bound is not vacuous
    assert err_cov[5] > 2 * (5 / 100) * err_cov[5] + mc_cov


# --------------------------------------------------------------------------- full covariance (optional dependency)
def test_full_covariance_without_geotorch_fails_with_an_explicit_message():
    try:
        import geotorch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="geotorch.*NOT installed"):
            LightSBPotential(2, LightSBPotentialConfig(n_potentials=2, is_diagonal=False))
    else:
        pytest.skip("geotorch is installed; the missing-dependency message cannot be exercised")


def test_full_covariance_quadrature_identity_2d():
    pytest.importorskip(
        "geotorch",
        reason="geotorch is not installed: the FULL-COVARIANCE LightSB path (is_diagonal=False) is NOT validated "
               "by this test-suite. This skip is not a validation.",
    )
    cfg = LightSBPotentialConfig(n_potentials=3, epsilon=0.5, is_diagonal=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LightSBPotential(2, cfg).double().eval()
    eps = float(model.epsilon)
    with torch.no_grad():
        model.r.copy_(torch.tensor(R2))
        model.S_log_diagonal_matrix.copy_(torch.tensor(np.log(S2)))
    h = 0.05
    axis = torch.arange(-10.0, 10.0 + 1e-9, h, dtype=torch.float64)
    y = torch.cartesian_prod(axis, axis)
    log_v = model.get_log_potential(y).detach()
    xs = torch.tensor([[0.0, 0.0], [0.7, -0.4], [-1.0, 1.0]], dtype=torch.float64)
    got = model.get_log_C(xs).detach()
    for i, x in enumerate(xs):
        quad = torch.logsumexp((y @ x) / eps + log_v, dim=0) + math.log(h * h)
        assert abs(float(got[i] - quad)) <= 1e-8
    # drift at t = 0 equals the conditional mean minus x (Monte-Carlo free: autograd of log C)
    xs_g = xs.clone().requires_grad_(True)
    grad = torch.autograd.grad(model.get_log_C(xs_g).sum(), xs_g)[0]
    drift0 = model.get_drift(xs, torch.zeros(len(xs), dtype=torch.float64))
    assert torch.allclose(drift0, eps * grad - xs, atol=1e-9)
