"""MixedSBM / CSBM acceptance tests (targets, clocks, IMF iteration, pure regimes, reload)."""
import numpy as np
import pytest
import torch

from sbtab.bridge.pathsampler import MixedPathSampler
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.timegrid import TimeGrid
from sbtab.solvers.csbm import AnnealedCSBMConfig, CSBMConfig, CSBMSolver
from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver
from sbtab.solvers.msbm.updater import MixedSBMUpdater

TINY = dict(steps_per_direction=None, min_steps_per_direction=0, num_steps=8, epochs_per_direction=2, hidden_dim=32, n_layers=2, time_dim=16, cat_emb_dim=4, dropout=0.0, seed=1)


def toy(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    num = torch.randn(n, 2, generator=g) * 0.5 + torch.tensor([2.0, -1.0])
    cat = torch.stack([torch.randint(0, 3, (n,), generator=g), torch.randint(0, 7, (n,), generator=g)], 1)
    return num, cat


# --------------------------------------------------------------------------- Brownian target
@pytest.mark.parametrize("direction", ["f", "b"])
def test_noise_corrected_target_equals_endpoint_conditioned_target(direction):
    """(x1 - X_t)/(1 - t) and (x0 - X_t)/t, rebuilt here from the tuple alone."""
    cfg = MixedSBMConfig(**{**TINY, "sigma": 0.7, "num_steps": 16})
    s = MixedSBMSolver(3, [], torch.tensor([], dtype=torch.bool), cfg)
    g = torch.Generator().manual_seed(11)
    z0, z1 = torch.randn(4096, 3, generator=g), torch.randn(4096, 3, generator=g) * 2 + 1
    empty = torch.zeros(4096, 0, dtype=torch.long)
    x_t, _, t, n, target, _ = s.updater._make_training_tuple(z0, empty, z1, empty, direction, generator=g)

    grid = s.timegrid.grid()
    assert torch.equal(t.view(-1), grid[n])                       # the network clock IS the bridge clock
    if direction == "f":
        assert int(n.min()) == 0 and int(n.max()) == 15           # every index the forward sampler visits, no t = 1
        keep = (t.view(-1) < 1)
        oracle = (z1 - x_t) / (1 - t)
    else:
        assert int(n.min()) == 1 and int(n.max()) == 16           # every index the backward sampler visits, no t = 0
        keep = (t.view(-1) > 0)
        oracle = (z0 - x_t) / t
    assert keep.all()
    # float32 with 1/(1-t) up to 16: relative 1e-4 is the conditioning-derived tolerance
    assert torch.allclose(target, oracle, rtol=1e-4, atol=1e-4)


def test_analytic_ou_process_moments_and_discretisation_bias():
    """
    Independent N(0,1) endpoints with sigma = sqrt(2): the bridge mixture is
    stationary N(0,1) and the forward drift is -x. Integrating the ORACLE drift with
    the sampler gives the Euler chain x <- (1-h) x + sqrt(2h) z whose variance after
    N steps from variance 1 is known in closed form:
        v_N = r^N + (1 - r^N) / (1 - h/2),  r = (1-h)^2,  h = 1/N.
    n = 200_000 samples, seed 0: the standard error of a variance estimate is
    sqrt(2/n) v ~ 0.0033, so 4 standard errors = 0.013.
    """
    class Oracle(torch.nn.Module):
        def forward(self, x_cont, x_cat, t):
            return -x_cont, None

    n = 200_000
    x0 = torch.randn(n, 1, generator=torch.Generator().manual_seed(0))
    bias = {}
    for N in (4, 8, 16):
        grid = TimeGrid.uniform(N)
        x, _, _ = MixedPathSampler(grid, None, EulerMaruyama(noise=True, sigma=2 ** 0.5)).simulate(
            x0, torch.zeros(n, 0, dtype=torch.long), Oracle(), "forward", seed=1)
        h, r = 1.0 / N, (1 - 1.0 / N) ** 2
        v_closed = r ** N + (1 - r ** N) / (1 - h / 2)
        assert float(x.var()) == pytest.approx(v_closed, abs=0.013)
        assert abs(float(x.mean())) < 4 * (v_closed / n) ** 0.5
        bias[N] = v_closed - 1.0
    assert bias[4] > bias[8] > bias[16] > 0                      # halving the step reduces the deterministic bias


# --------------------------------------------------------------------------- IMF iteration
def test_imf_stages_use_the_previous_learned_coupling_with_the_right_anchor(monkeypatch):
    num, cat = toy()
    captured = []
    real = MixedSBMUpdater.train_epochs

    def spy(self, direction, z0_num, z0_cat, z1_num, z1_cat, **kwargs):
        captured.append((direction, z0_num.clone(), z0_cat.clone(), z1_num.clone(), z1_cat.clone()))
        return real(self, direction, z0_num, z0_cat, z1_num, z1_cat, **kwargs)

    monkeypatch.setattr(MixedSBMUpdater, "train_epochs", spy)
    s = MixedSBMSolver(2, [3, 7], torch.tensor([False, True]), MixedSBMConfig(fb_sequence=("b", "f", "b"), **TINY)).fit(num, cat)

    assert [l["coupling_source"] for l in s.stage_log] == ["independent", "backward_model", "forward_model"]
    assert [l["anchored_endpoint"] for l in s.stage_log] == ["both", "prior", "data"]
    (_, a0n, a0c, a1n, a1c), (_, b0n, b0c, b1n, b1c), (_, c0n, c0c, c1n, c1c) = captured
    # stage 0: independent coupling data (x) prior
    assert torch.equal(a0n, num) and torch.equal(a0c, cat)
    # stage 1 ('f' after 'b'): x1 is the SAME real prior sample, x0 was re-simulated by the backward net
    assert torch.equal(b1n, a1n) and torch.equal(b1c, a1c)
    assert not torch.equal(b0n, num)
    # stage 2 ('b' after 'f'): x0 is exactly the real data, x1 was re-simulated by the forward net
    assert torch.equal(c0n, num) and torch.equal(c0c, cat)
    assert not torch.equal(c1n, a1n)
    # generation uses the LAST backward stage; every stage really trained
    assert s.generation_stage() == 2 and all(l["n_updates"] > 0 for l in s.stage_log)
    # feature/tuning trains both directions through the same network and optimizer.
    assert s.updater.model is s.model
    assert s.updater.n_updates == s.n_updates


def test_fb_sequence_must_alternate():
    with pytest.raises(ValueError):
        MixedSBMSolver(2, [3], torch.tensor([False]), MixedSBMConfig(**{**TINY, "fb_sequence": ("b", "b")}))


# --------------------------------------------------------------------------- pure regimes, sizes, reload
@pytest.mark.integration
@pytest.mark.parametrize("cont_dim,cards,ordered", [(2, [3, 7], [False, True]), (2, [], []), (0, [3, 7], [False, True])])
def test_regimes_small_sample_sizes_and_reload(tmp_path, cont_dim, cards, ordered):
    num, cat = toy(n=60)                                          # 60 rows < batch_size 256: old code trained nothing
    s = MixedSBMSolver(cont_dim, cards, torch.tensor(ordered, dtype=torch.bool),
                       MixedSBMConfig(fb_sequence=("b", "f", "b"), **TINY))
    s.fit(num if cont_dim else None, cat if cards else None)
    assert all(l["n_updates"] > 0 and np.isfinite(l["last_epoch_loss"]) for l in s.stage_log)
    for n in (1, 99, 101):                                        # batch-1, batch+1 around sample batch 100
        gn, gc = s.sample(n, seed=3, batch_size=100)
        assert gn.shape == (n, cont_dim) and gc.shape == (n, len(cards)) and torch.isfinite(gn).all()
        for d, c in enumerate(cards):
            assert 0 <= int(gc[:, d].min()) and int(gc[:, d].max()) < c
    gn, _ = s.sample(200, seed=5, batch_size=100)
    if cont_dim:
        assert not torch.equal(gn[:100], gn[100:])                # batches are not reseeded to identical outputs
    s.save_checkpoint(tmp_path / "m.pt")
    r = MixedSBMSolver.load_checkpoint(tmp_path / "m.pt")
    a, b = s.sample(50, seed=11), r.sample(50, seed=11)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])    # CPU reload is exact


# --------------------------------------------------------------------------- CSBM
CS = dict(num_outer_iterations=2, epochs=2, num_steps=6, hidden_dim=32, emb_dim=4, time_dim=16, seed=3)


def cat_table(n=40):
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 3, (n,), generator=g)
    return torch.stack([a, (a * 2) % 5, torch.randint(0, 9, (n,), generator=g)], 1), [3, 5, 9], [False, False, True]


@pytest.mark.integration
def test_csbm_reference_is_fixed_and_public_sampler_is_exact(tmp_path):
    x, cards, ordered = cat_table()
    s = CSBMSolver(cards, ordered, CSBMConfig(**CS)).fit(x)
    assert s.variant == "csbm" and s.reference.mixing_rate == 1.0
    assert [t["mixing_rate"] for t in s.reference_trace] == [1.0]            # never changed during the fit
    assert [(l["direction"], l["coupling_source"], l["anchored_endpoint"]) for l in s.stage_log] == [
        ("forward", "independent", "both"), ("backward", "forward_model", "data"),
        ("forward", "backward_model", "prior"), ("backward", "forward_model", "data")]
    assert all(l["n_updates"] > 0 for l in s.stage_log)                      # 40 rows < batch size
    for n in (1, 263, 265):
        g = s.sample(n, seed=7, batch_size=100)
        assert g.shape == (n, 3) and g.dtype == torch.long
        assert all(0 <= int(g[:, d].min()) and int(g[:, d].max()) < c for d, c in enumerate(cards))   # masking
    s.save_checkpoint(tmp_path / "c.pt")
    assert torch.equal(s.sample(64, seed=1), CSBMSolver.load_checkpoint(tmp_path / "c.pt").sample(64, seed=1))
    with pytest.raises(ValueError):
        s.fit(torch.full((4, 3), 9))                                        # code 9 is outside column 0's support


@pytest.mark.integration
def test_annealing_exists_only_as_the_named_variant(tmp_path):
    x, cards, ordered = cat_table()
    a = CSBMSolver(cards, ordered, AnnealedCSBMConfig(**CS, anneal_every=1, anneal_multiplier=0.5)).fit(x)
    assert a.variant == "csbm_annealed"
    assert [t["mixing_rate"] for t in a.reference_trace] == [1.0, 0.5, 0.25]
    a.save_checkpoint(tmp_path / "a.pt")
    r = CSBMSolver.load_checkpoint(tmp_path / "a.pt")
    assert r.reference.mixing_rate == 0.25                                  # the FINAL reference is restored
    assert torch.equal(a.sample(32, seed=2), r.sample(32, seed=2))


def test_csbm_updater_feeds_the_grid_time_of_the_current_state():
    x, cards, ordered = cat_table()
    s = CSBMSolver(cards, ordered, CSBMConfig(**CS))
    seen = []
    fwd = s.updater.forward_model.forward
    s.updater.forward_model.forward = lambda xt, t: (seen.append(t.clone()), fwd(xt, t))[1]
    n = torch.tensor([0, 3, 5])
    xt = s.reference.sample_x_t(x[:3], x[:3], n)
    s.updater.train_forward_step(xt, x[:3], n)
    assert torch.allclose(seen[0], torch.tensor([0.0, 0.5, 5 / 6]))          # t[n] = n / N, the sampler's clock
