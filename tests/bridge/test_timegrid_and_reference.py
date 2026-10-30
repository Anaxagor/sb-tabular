"""
Grid/clock and reference-semigroup acceptance tests.

Oracles are independent of the implementation: scipy.linalg.expm of an explicitly
built generator, numpy matrix products, and brute-force enumeration.
"""
import numpy as np
import pytest
import torch
from scipy.linalg import expm

from sbtab.bridge.reference import CategoricalReference, GaussianReference, IncompatibleBridgeError
from sbtab.bridge.timegrid import TimeGrid

ATOL = RTOL = 1e-10


# --------------------------------------------------------------------------- grid
@pytest.mark.parametrize("grid", [
    TimeGrid.uniform(1), TimeGrid.uniform(7), TimeGrid.uniform(100), TimeGrid.uniform(10, horizon=2.5),
    TimeGrid(num_steps=100, horizon=1.0), TimeGrid(num_steps=50, schedule="linear", horizon=1.0),
    TimeGrid.from_points([0.0, 0.01, 0.3, 0.31, 0.9, 1.0]),
])
def test_grid_has_exact_endpoints_positive_steps_and_horizon(grid):
    g = grid._grid64().numpy()
    dt = grid.dt().double().numpy()
    assert len(g) == grid.num_steps + 1 and len(dt) == grid.num_steps
    assert g[0] == 0.0                                   # exact, not approximately
    assert g[-1] == grid.T
    assert (np.diff(g) > 0).all()
    assert float(grid.grid()[0]) == 0.0 and float(grid.grid()[-1]) == pytest.approx(grid.T, rel=1e-7)
    assert dt.sum() == pytest.approx(grid.T, rel=1e-6)


def test_legacy_geometric_grid_is_not_unit_horizon_but_can_be_normalised():
    # The reproducer for the mixed-time-grid defect: the old 100-step default totals ~0.217914.
    legacy = TimeGrid(num_steps=100)
    assert legacy.T == pytest.approx(0.217914, abs=1e-5)
    assert TimeGrid(num_steps=100, horizon=1.0).T == 1.0
    # the legacy API is the inclusive cumulative sum = grid()[1:]
    assert torch.equal(legacy.times(), legacy.grid()[1:])
    assert torch.equal(legacy.gammas(), legacy.dt())
    # relative step sizes are preserved by normalisation
    a, b = legacy.dt().double(), TimeGrid(num_steps=100, horizon=1.0).dt().double()
    assert torch.allclose(a / a.sum(), b / b.sum(), rtol=1e-6, atol=0)


def test_explicit_grid_validation():
    with pytest.raises(ValueError):
        TimeGrid.from_points([0.1, 0.5, 1.0])            # must start at exactly 0
    with pytest.raises(ValueError):
        TimeGrid.from_points([0.0, 0.5, 0.5, 1.0])       # strictly increasing


def test_gaussian_reference_unseeded_calls_differ():
    # A fresh unseeded torch.Generator has a FIXED default seed: the old code
    # returned identical "random" draws on every unseeded call.
    ref = GaussianReference(dim=3)
    assert not torch.equal(ref.sample(16), ref.sample(16))
    assert torch.equal(ref.sample(16, seed=4), ref.sample(16, seed=4))
    g = torch.Generator().manual_seed(1)
    assert not torch.equal(ref.sample(16, generator=g), ref.sample(16, generator=g))


# --------------------------------------------------------------------------- oracle generators
def ordered_generator(S, rate, bandwidth):
    """Independent numpy construction of R = rate (P - I), P the D3PM discretised Gaussian."""
    i = np.arange(S)
    scale = (bandwidth * (S - 1)) ** 2
    n = np.arange(-(S - 1), S)
    Z = np.exp(-4.0 * n ** 2 / scale).sum()
    P = np.exp(-4.0 * (i[:, None] - i[None, :]) ** 2 / scale) / Z
    np.fill_diagonal(P, 0.0)
    P = P + np.diag(1.0 - P.sum(1))
    return rate * (P - np.eye(S))


def uniform_generator(S, rate):
    return rate * (np.ones((S, S)) / S - np.eye(S))


GRIDS = [TimeGrid.uniform(100), TimeGrid(num_steps=40, horizon=1.0), TimeGrid.from_points([0, .01, .3, .31, .9, 1.0])]


@pytest.mark.parametrize("S", [2, 20, 50])
@pytest.mark.parametrize("grid", GRIDS)
def test_transitions_match_independent_matrix_exponential(S, grid):
    rate, bw = 3.0, 0.2
    ref = CategoricalReference([S, S], torch.tensor([True, False]), grid, mixing_rate=rate, ordered_bandwidth=bw)
    t = grid._grid64().numpy()
    N = grid.num_steps
    for d, R in ((0, ordered_generator(S, rate, bw)), (1, uniform_generator(S, rate))):
        for n in sorted({0, 1, N // 2, N - 1, N}):
            np.testing.assert_allclose(ref.transition(d, "from_start", n).numpy(), expm(R * t[n]), atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(ref.transition(d, "to_end", n).numpy(), expm(R * (t[-1] - t[n])), atol=ATOL, rtol=RTOL)
        for i in sorted({0, N // 2, N - 1}):
            np.testing.assert_allclose(ref.transition(d, "step", i).numpy(), expm(R * (t[i + 1] - t[i])), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("S", [20, 50])
def test_rows_sum_to_one_and_composition_holds_across_old_k29_30_boundary(S):
    # The old ordered kernel switched from true powers to separately normalised
    # Gaussian matrices at step 30: P[29] @ Q1 differed from P[30] by up to 0.38.
    ref = CategoricalReference([S], torch.tensor([True]), TimeGrid.uniform(100), mixing_rate=3.0, ordered_bandwidth=0.2)
    for n in range(100):
        A, step, B = (ref.transition(0, "from_start", n).numpy(), ref.transition(0, "step", n).numpy(),
                      ref.transition(0, "from_start", n + 1).numpy())
        np.testing.assert_allclose(A.sum(1), 1.0, atol=ATOL)
        assert (A >= 0).all()
        np.testing.assert_allclose(A @ step, B, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(step @ ref.transition(0, "to_end", n + 1).numpy(),
                                   ref.transition(0, "to_end", n).numpy(), atol=ATOL, rtol=RTOL)
    for n in (28, 29, 30, 31):      # the historical boundary, explicitly
        np.testing.assert_allclose(ref.transition(0, "from_start", n).numpy() @ ref.transition(0, "step", n).numpy(),
                                   ref.transition(0, "from_start", n + 1).numpy(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("S", [20, 50])
def test_bridges_are_normalised_with_exact_endpoints_for_rare_pairs(S, ordered):
    N = 100
    ref = CategoricalReference([S], torch.tensor([ordered]), TimeGrid.uniform(N), mixing_rate=3.0, ordered_bandwidth=0.2)
    # rarest pair under an ordered kernel: opposite ends of the support
    x0 = torch.zeros((1, 1), dtype=torch.long)
    xN = torch.full((1, 1), S - 1)
    QT = ref.transition(0, "from_start", N).numpy()
    for n in (0, 1, 10, 29, 30, 50, 90, 99, 100):
        p = ref.bridge_at_time(x0, xN, n)[0, 0].numpy()
        assert p.sum() == pytest.approx(1.0, abs=ATOL)
        # independent oracle: Q_{0,n}[x0, s] Q_{n,N}[s, xN] / Q_{0,N}[x0, xN]
        oracle = (ref.transition(0, "from_start", n).numpy()[0, :] * ref.transition(0, "to_end", n).numpy()[:, S - 1]) / QT[0, S - 1]
        np.testing.assert_allclose(p[:S], oracle, atol=ATOL, rtol=1e-8)
    e0, eN = ref.bridge_at_time(x0, xN, 0)[0, 0].numpy(), ref.bridge_at_time(x0, xN, N)[0, 0].numpy()
    assert e0[0] == 1.0 and e0.sum() == 1.0              # exact endpoints
    assert eN[S - 1] == 1.0 and eN.sum() == 1.0


def test_one_step_bridges_match_brute_force_conditionals():
    S, N = 6, 5
    grid = TimeGrid.from_points([0, 0.1, 0.25, 0.6, 0.8, 1.0])
    ref = CategoricalReference([S], torch.tensor([True]), grid, mixing_rate=2.0, ordered_bandwidth=0.5)
    step = [ref.transition(0, "step", i).numpy() for i in range(N)]
    # enumerate the joint law of (x_n, x_{n+1}, x_N) given x_0 by brute-force products
    def prod(a, b):
        M = np.eye(S)
        for i in range(a, b):
            M = M @ step[i]
        return M
    for n in range(N):
        for xn in range(S):
            for xN in range(S):
                w = step[n][xn, :] * prod(n + 1, N)[:, xN]
                got = ref.bridge_next_given_prev(torch.tensor([[xn]]), torch.tensor([[xN]]), n)[0, 0].numpy()
                np.testing.assert_allclose(got, w / w.sum(), atol=ATOL, rtol=1e-8)
    for n in range(1, N + 1):
        for x0 in range(S):
            for xn in range(S):
                w = prod(0, n - 1)[x0, :] * step[n - 1][:, xn]
                got = ref.bridge_prev_given_next(torch.tensor([[x0]]), torch.tensor([[xn]]), n)[0, 0].numpy()
                np.testing.assert_allclose(got, w / w.sum(), atol=ATOL, rtol=1e-8)


def test_reference_law_does_not_depend_on_the_number_of_steps():
    # Old per-step kernel: P(stay over the horizon) was 0.633 / 0.446 / 0.301 for K = 50 / 100 / 200.
    stay = [float(CategoricalReference([4], torch.tensor([False]), TimeGrid.uniform(N), mixing_rate=1.0)
                  .transition(0, "from_start", N)[0, 0]) for N in (50, 100, 200)]
    assert max(stay) - min(stay) < 1e-12
    assert stay[0] == pytest.approx(np.exp(-1.0) + (1 - np.exp(-1.0)) / 4, abs=1e-12)


def test_changing_the_kernel_rebuilds_every_cached_transition():
    grid = TimeGrid.uniform(10)
    ref = CategoricalReference([5, 5], torch.tensor([True, False]), grid, mixing_rate=1.0, ordered_bandwidth=0.3)
    before = [[ref.transition(d, k, 3).clone() for k in ("from_start", "to_end", "step")] for d in (0, 1)]
    ref.set_kernel(mixing_rate=2.5)
    fresh = CategoricalReference([5, 5], torch.tensor([True, False]), grid, mixing_rate=2.5, ordered_bandwidth=0.3)
    for d in (0, 1):
        for j, k in enumerate(("from_start", "to_end", "step")):
            assert not torch.allclose(ref.transition(d, k, 3), before[d][j])          # nothing stale
            assert torch.allclose(ref.transition(d, k, 3), fresh.transition(d, k, 3), atol=1e-14)


def test_unreachable_bridge_is_reported_not_smoothed():
    ref = CategoricalReference([50], torch.tensor([True]), TimeGrid.uniform(10), mixing_rate=1e-3, ordered_bandwidth=0.01)
    with pytest.raises(IncompatibleBridgeError):
        ref.bridge_at_time(torch.zeros((1, 1), dtype=torch.long), torch.full((1, 1), 49), 5)


def test_sampling_never_selects_a_padded_category_and_rejects_malformed_probabilities():
    ref = CategoricalReference([2, 7], torch.tensor([False, True]), TimeGrid.uniform(4))
    g = torch.Generator().manual_seed(0)
    x0, xN = ref.sample_prior(4000, generator=g), ref.sample_prior(4000, generator=g)
    for n in range(5):
        xt = ref.sample_x_t(x0, xN, n, generator=g)
        assert int(xt[:, 0].max()) <= 1 and int(xt[:, 1].max()) <= 6 and int(xt.min()) >= 0
    bad = torch.zeros((1, 2, 7), dtype=torch.float64)
    with pytest.raises(ValueError):
        ref.sample_from_probs(bad)                       # all-zero rows used to fall back to uniform sampling
    bad[0, 0, 5] = 1.0
    bad[0, 1, 0] = 1.0
    with pytest.raises(ValueError):
        ref.sample_from_probs(bad)                       # mass on a padded category of column 0


def test_model_induced_steps_are_normalised_and_reduce_to_the_bridge_for_a_point_mass_head():
    S, N = 5, 6
    ref = CategoricalReference([S, 3], torch.tensor([True, False]), TimeGrid.uniform(N), mixing_rate=2.0, ordered_bandwidth=0.4)
    g = torch.Generator().manual_seed(1)
    x = ref.sample_prior(64, generator=g)
    target = ref.sample_prior(64, generator=g)
    logits = torch.full((64, 2, S), -40.0)
    logits.scatter_(2, target.unsqueeze(-1), 40.0)       # head puts (numerically) all mass on `target`
    for n in range(N):
        p = ref.model_induced_next_step(logits, x, n)
        assert torch.allclose(p.sum(-1), torch.ones(64, 2, dtype=torch.float64), atol=1e-12)
        assert float(p[:, 1, 3:].abs().max()) == 0.0     # padded categories of the 3-state column
        assert torch.allclose(p, ref.bridge_next_given_prev(x, target, n), atol=1e-9)
    for n in range(1, N + 1):
        p = ref.model_induced_prev_step(logits, x, n)
        assert torch.allclose(p.sum(-1), torch.ones(64, 2, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(p, ref.bridge_prev_given_next(target, x, n), atol=1e-9)


def test_index_bounds_distinguish_forward_and_backward_endpoints():
    ref = CategoricalReference([3], torch.tensor([False]), TimeGrid.uniform(4))
    x = torch.zeros((2, 1), dtype=torch.long)
    logits = torch.zeros((2, 1, 3))
    with pytest.raises(IndexError):
        ref.model_induced_next_step(logits, x, 4)        # forward steps start from n in [0, N-1]
    with pytest.raises(IndexError):
        ref.model_induced_prev_step(logits, x, 0)        # backward steps start from n in [1, N]
