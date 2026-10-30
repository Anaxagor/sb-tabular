"""Loss normalisation, absent blocks, and one-clock sampler tests."""
import math

import numpy as np
import pytest
import torch

from sbtab.bridge.losses import CSBMLoss, MixedSBMLoss, RegressionLoss
from sbtab.bridge.pathsampler import DiscretePathSampler, MixedPathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.timegrid import TimeGrid


def make_ref(cards, ordered, N=6):
    return CategoricalReference(cards, torch.tensor(ordered), TimeGrid.uniform(N), mixing_rate=2.0, ordered_bandwidth=0.4)


def scalar_loop_forward_loss(ref, logits, x1, xt, n, lmbda):
    """Independent oracle: python loops over rows/columns/categories, 0 log 0 = 0."""
    B, D, _ = logits.shape
    total = 0.0
    for b in range(B):
        for d in range(D):
            S = ref.cardinalities[d]
            z = logits[b, d, :S].double()
            p = torch.softmax(z, 0).numpy()
            nb = int(n[b])
            step = ref.transition(d, "step", nb).numpy()
            to_end_next = ref.transition(d, "to_end", nb + 1).numpy()
            to_end_now = ref.transition(d, "to_end", nb).numpy()
            i, j = int(xt[b, d]), int(x1[b, d])
            target = step[i, :] * to_end_next[:, j] / to_end_now[i, j]
            model = np.zeros(S)
            for s in range(S):
                model[s] = step[i, s] * sum(p[e] * to_end_next[s, e] / to_end_now[i, e] for e in range(S))
            kl = sum(target[s] * (math.log(target[s]) - math.log(model[s])) for s in range(S) if target[s] > 0)
            ce = -math.log(p[j])
            total += kl + lmbda * ce
    return total / (B * D)


def test_batched_categorical_loss_equals_scalar_loop_and_has_finite_gradients():
    torch.manual_seed(0)
    ref = make_ref([3, 5, 4], [False, True, False])
    B, g = 7, torch.Generator().manual_seed(3)
    x0, x1 = ref.sample_prior(B, generator=g), ref.sample_prior(B, generator=g)
    n = torch.randint(0, 6, (B,), generator=g)
    xt = ref.sample_x_t(x0, x1, n, generator=g)
    logits = torch.randn(B, 3, 5, requires_grad=True)
    loss = CSBMLoss(ref, lmbda=0.3).forward_loss(logits, x1, xt, n)
    assert float(loss) == pytest.approx(scalar_loop_forward_loss(ref, logits.detach(), x1, xt, n, 0.3), rel=1e-9)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # padded categories receive exactly zero gradient: they carry no probability mass
    assert float(logits.grad[:, 0, 3:].abs().max()) == 0.0 and float(logits.grad[:, 2, 4:].abs().max()) == 0.0


def test_duplicating_a_categorical_column_does_not_change_the_loss():
    # Reproducer for the double normalisation: the old loss divided the (already
    # per-column-averaged) categorical term by the column count a second time, so
    # duplicating an identical column halved it.
    torch.manual_seed(1)
    g = torch.Generator().manual_seed(5)
    ref1, ref2 = make_ref([4], [False]), make_ref([4, 4], [False, False])
    B = 16
    x0, x1 = ref1.sample_prior(B, generator=g), ref1.sample_prior(B, generator=g)
    n = torch.randint(0, 6, (B,), generator=g)
    xt = ref1.sample_x_t(x0, x1, n, generator=g)
    logits = torch.randn(B, 1, 4)
    dup = lambda t: torch.cat([t, t], dim=1)
    for direction, tgt, idx in (("forward", x1, n), ("backward", x0, n + 1)):
        xt_d = xt if direction == "forward" else ref1.sample_x_t(x0, x1, idx, generator=torch.Generator().manual_seed(9))
        kw = dict(pred_num=None, target_num=None, direction=direction)
        one = MixedSBMLoss(ref1, lambda_num=0.5, lambda_cat=0.7)(pred_logits_cat=logits, true_cat=tgt, x_t_cat=xt_d, n=idx, **kw)
        two = MixedSBMLoss(ref2, lambda_num=0.5, lambda_cat=0.7)(pred_logits_cat=dup(logits), true_cat=dup(tgt),
                                                                  x_t_cat=dup(xt_d), n=idx, **kw)
        assert float(one) == pytest.approx(float(two), rel=1e-12)


def test_absent_blocks_give_finite_losses_and_match_the_specialised_losses():
    torch.manual_seed(2)
    g = torch.Generator().manual_seed(7)
    ref = make_ref([3, 4], [False, True])
    B = 12
    x0, x1 = ref.sample_prior(B, generator=g), ref.sample_prior(B, generator=g)
    n = torch.randint(0, 6, (B,), generator=g)
    xt = ref.sample_x_t(x0, x1, n, generator=g)
    logits, pred, target = torch.randn(B, 2, 4), torch.randn(B, 3), torch.randn(B, 3)

    # pure categorical: no numerical tensor at all, and an empty (B, 0) one
    cat_only = MixedSBMLoss(ref, lambda_num=0.8, lambda_cat=1.0)
    ref_loss = CSBMLoss(ref, lmbda=0.001).forward_loss(logits, x1, xt, n)
    for pn, tn in ((None, None), (torch.zeros(B, 0), torch.zeros(B, 0))):
        got = cat_only(pred_num=pn, target_num=tn, pred_logits_cat=logits, true_cat=x1, x_t_cat=xt, n=n)
        assert torch.isfinite(got) and float(got) == pytest.approx(float(ref_loss), rel=1e-12)   # update parity with CSBM

    # pure numerical: no reference, no logits
    num_only = MixedSBMLoss(None, lambda_num=1.0, lambda_cat=0.2)
    got = num_only(pred_num=pred, target_num=target, pred_logits_cat=None, true_cat=None, x_t_cat=None, n=n)
    assert torch.isfinite(got)
    assert float(got) == pytest.approx(float(RegressionLoss()(pred, target)), rel=1e-6)          # parity with plain MSE
    assert float(got) == pytest.approx(float(((pred - target) ** 2).mean()), rel=1e-6)           # mean over coordinates

    with pytest.raises(ValueError):
        num_only(pred_num=None, target_num=None, pred_logits_cat=None, true_cat=None, x_t_cat=None, n=n)
    with pytest.raises(ValueError):
        RegressionLoss()(torch.zeros(B, 0), torch.zeros(B, 0))   # the old code returned NaN here


class ClockSpy(torch.nn.Module):
    """Records the time fed to the network at every sampler step."""

    def __init__(self, cards, cont_dim=None):
        super().__init__()
        self.cards, self.cont_dim, self.seen = cards, cont_dim, []

    def forward(self, *args):
        t = args[-1]
        self.seen.append(float(t.reshape(-1)[0]))
        B = args[0].shape[0]
        logits = torch.zeros(B, len(self.cards), max(self.cards))
        if self.cont_dim is None:
            return logits
        return torch.zeros(B, self.cont_dim), logits


@pytest.mark.parametrize("grid", [TimeGrid.uniform(5), TimeGrid.from_points([0, 0.1, 0.15, 0.6, 1.0])])
def test_network_continuous_step_and_categorical_index_share_one_clock(grid):
    N, t = grid.num_steps, grid._grid64().numpy()
    ref = CategoricalReference([3, 2], torch.tensor([False, False]), grid)
    x_cat = ref.sample_prior(8, generator=torch.Generator().manual_seed(0))

    for direction, expected in (("forward", t[:-1]), ("backward", t[:0:-1])):
        spy = ClockSpy([3, 2])
        DiscretePathSampler(grid, ref).simulate(x_cat, spy, direction, seed=1)
        # the network always sees the time of the CURRENT state: forward starts at t[0] = 0, backward at t[N] = T
        np.testing.assert_allclose(spy.seen, expected, atol=1e-6)

    # mixed sampler: same clock, and the continuous block integrates over exactly the grid horizon
    drift = 2.0

    class ConstDrift(ClockSpy):
        def forward(self, x_cont, x_cat, t):
            out = super().forward(x_cont, x_cat, t)
            return torch.full_like(x_cont, drift), out[1]

    for direction, expected in (("forward", t[:-1]), ("backward", t[:0:-1])):
        spy = ConstDrift([3, 2], cont_dim=2)
        x_end, _, _ = MixedPathSampler(grid, ref, EulerMaruyama(noise=True, sigma=0.0)).simulate(
            torch.zeros(8, 2), x_cat, spy, direction, seed=1)
        np.testing.assert_allclose(spy.seen, expected, atol=1e-6)
        # sum of step lengths == horizon (the old sampler integrated over ~0.2179 for N = 100)
        assert torch.allclose(x_end, torch.full((8, 2), drift * grid.T), atol=1e-5)


def test_mixed_sampler_handles_absent_blocks():
    grid = TimeGrid.uniform(4)
    ref = CategoricalReference([3], torch.tensor([False]), grid)
    spy = ClockSpy([3], cont_dim=0)
    xc, xk, _ = MixedPathSampler(grid, ref, EulerMaruyama(sigma=0.1)).simulate(
        torch.zeros(5, 0), ref.sample_prior(5), spy, "backward", seed=0)
    assert xc.shape == (5, 0) and xk.shape == (5, 1)

    class NumOnly(torch.nn.Module):
        def forward(self, x_cont, x_cat, t):
            return -x_cont, None

    xc, xk, _ = MixedPathSampler(grid, None, EulerMaruyama(sigma=0.1)).simulate(
        torch.randn(5, 2), torch.zeros(5, 0, dtype=torch.long), NumOnly(), "forward", seed=0)
    assert xc.shape == (5, 2) and xk.shape == (5, 0) and torch.isfinite(xc).all()
