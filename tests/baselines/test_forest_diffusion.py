"""Forest training coverage, mathematical oracles and real-tree round trips."""
import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")
from sbtab.solvers.ForestDiffusion import ForestDiffusionModel
from sbtab.solvers.ForestDiffusion.utils.utils_diffusion import IterForDMatrix, get_xt, euler_solve
from sbtab.solvers.ForestDiffusion.utils.diffusion import VPSDE, get_pc_sampler
from sbtab.baselines.forest_diffusion import ForestDiffusionConfig, ForestDiffusionWrapper


def tree(x, **kw):
    opts = dict(n_t=3, n_estimators=2, duplicate_K=2, n_batch=3,
                n_jobs=1, nthread=1, seed=17)
    opts.update(kw)
    return ForestDiffusionModel(np.asarray(x, dtype=float), **opts)


@pytest.mark.parametrize("kind", ["flow", "vp"])
def test_interpolation_targets_and_iterator_replay(kind):
    data = np.arange(14, dtype=float).reshape(7, 2) / 5
    noise = np.full_like(data, 0.7)
    sde = VPSDE(beta_min=.1, beta_max=8, N=3)
    state, target = get_xt(data, .37, None, diffusion_type=kind, sde=sde, x0=noise)
    if kind == "flow":
        np.testing.assert_allclose(state, .37 * data + .63 * noise)
        np.testing.assert_allclose(target, data - noise)
    else:
        alpha, std = sde.marginal_prob_coef(data, .37)
        np.testing.assert_allclose(state, alpha * data + std * noise)
        np.testing.assert_array_equal(target, noise)
    it = IterForDMatrix(np.array_split(data, 3), None, t=.37, dim=None,
                       n_epochs=2, diffusion_type=kind, sde=sde, seed=5)
    def collect():
        chunks = []
        while it.next(lambda **batch: chunks.append((batch['data'].copy(), batch['label'].copy()))):
            pass
        return tuple(np.concatenate([c[i] for c in chunks]) for i in range(2))
    first = collect()
    it.reset()
    second = collect()
    assert len(first[0]) == 2 * len(data)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", ["flow", "vp"])
def test_missing_target_not_used_as_supervision(kind):
    data = np.array([[1., 4.], [np.nan, 5.], [2., 6.]])
    it = IterForDMatrix([data], None, t=.3, dim=0, n_epochs=1,
                       diffusion_type=kind, sde=VPSDE(N=3))
    chunks = []
    it.next(lambda **batch: chunks.append(batch))
    assert chunks[0]['data'].shape == (2, 2)
    assert np.isfinite(chunks[0]['data']).all()


@pytest.mark.parametrize("n_batch", [1, 3, 100])
def test_class_batches_cover_every_row_and_arbitrary_labels(monkeypatch, n_batch):
    rows = []
    real = xgb.train
    def capture(params, dtrain, **kw):
        rows.append(dtrain.num_row())
        return real(params, dtrain, **kw)
    monkeypatch.setattr(xgb, "train", capture)
    x = np.arange(11, dtype=float)[:, None]
    labels = np.array([2] * 4 + [9] * 7)
    m = tree(x, label_y=labels, n_batch=n_batch)
    assert rows == [8] * 3 + [14] * 3
    output = m.generate(batch_size=17, seed=9, max_batch_size=5)
    assert output.shape == (17, 2) and np.isfinite(output).all()
    assert set(output[:, 1]) <= {2, 9}
    proba = m.predict_proba(x[:3], n_z=1)
    assert proba.shape == (3, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_fitted_categorical_vocabulary_and_binary_support():
    x = np.column_stack([np.arange(12) / 5, [10, 30, 90] * 4, [7] * 12, [0, 1] * 6])
    m = tree(x, cat_indexes=[1, 2], bin_indexes=[3])
    encoded, _, _ = m.dummify(x[[0, 3]])
    assert encoded.shape[1] == m.c
    np.testing.assert_array_equal(m.clean_onehot_data(encoded), x[[0, 3]])
    output = m.generate(batch_size=20, seed=4)
    assert set(output[:, 1]) <= {10, 30, 90} and set(output[:, 2]) == {7}
    assert set(output[:, 3]) <= {0, 1}
    extreme = np.array([[-20., 10., 7., -1.], [20., 90., 7., 2.]])
    m.clip_extremes(extreme)
    np.testing.assert_array_equal(extreme[:, 0], [-20., 20.])
    np.testing.assert_array_equal(extreme[:, 3], [0., 1.])


def test_flow_constant_velocity_and_vp_reverse_formula():
    y0 = np.array([2., 3.])
    np.testing.assert_allclose(euler_solve(y0, lambda t, y: np.full_like(y, 4.), N=7), y0 + 4)
    sde = VPSDE(beta_min=.2, beta_max=5, N=4)
    calls = []
    def score(y, t):
        calls.append(t)
        return -2 * y
    drift, diffusion = sde.reverse(score).sde(y0, .4)
    beta = .2 + .4 * 4.8
    np.testing.assert_allclose(drift, -.5 * beta * y0 + 2 * beta * y0)
    assert len(calls) == 1
    ode_drift, ode_diffusion = sde.reverse(score, probability_flow=True).sde(y0, .4)
    np.testing.assert_allclose(ode_drift, -.5 * beta * y0 + beta * y0)
    assert ode_diffusion == 0 and diffusion == pytest.approx(np.sqrt(beta))


def test_repaint_forward_jump_uses_exact_vp_transition():
    class NoNoise:
        def normal(self, size):
            return np.zeros(size)
    sde = VPSDE(beta_min=.2, beta_max=3, N=3)
    eps = .2
    sample = get_pc_sampler(lambda y, t: -y, sde, denoise=False,
                            eps=eps, repaint=True, rng=NoNoise())(np.ones(1), r=2, j=1)
    expected = 1.
    times = [1., .6, .2]
    for high, low in zip(times, times[1:]):
        # Reverse Euler (score=-x), exact forward conditional mean, repeat reverse.
        reverse = 1 - .5 * (.2 + 2.8 * high) * (high - low)
        integrated_beta = lambda t: .2 * t + 1.4 * t * t
        forward = np.exp(-.5 * (integrated_beta(high) - integrated_beta(low)))
        expected *= reverse**2 * forward
    np.testing.assert_allclose(sample, expected)


@pytest.mark.parametrize("repaint", [False, True])
def test_vp_tweedie_recovers_point_mass_with_exact_score(repaint):
    sde = VPSDE(beta_min=.1, beta_max=8, N=9)
    center = np.array([2., -3.])
    def score(y, t):
        alpha, std = sde.marginal_prob_coef(y, t)
        return -(y - alpha * center) / std**2
    sample = get_pc_sampler(score, sde, eps=.13, repaint=repaint,
                            rng=np.random.default_rng(2))(np.zeros(2), r=2, j=3)
    np.testing.assert_allclose(sample, center, atol=1e-12)


@pytest.mark.parametrize("kind", ["flow", "vp"])
def test_wrapper_seed_checkpoint_supports_and_no_global_rng(tmp_path, kind):
    frame = pd.DataFrame({'x': np.arange(18) / 7, 'k': [2, 5, 9] * 6,
                          'c': ['a', 'b'] * 9, 'id': np.arange(18)})
    cfg = ForestDiffusionConfig(diffusion_type=kind, n_t=4, n_estimators=3,
                               duplicate_K=2, n_batch=5, n_threads=1)
    np.random.seed(414)
    before = np.random.get_state()
    m = ForestDiffusionWrapper(cfg).fit(frame, continuous_cols=['x'],
        discrete_cols=['k'], categorical_cols=['c'], id_col='id')
    sample = m.sample(11, seed=8)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert set(sample.k) <= {2, 5, 9} and set(sample.c) <= {'a', 'b'}
    assert sample.id.is_unique and not set(sample.id) & set(frame.id)
    assert np.isfinite(sample.x).all() and m.model.X1 is None
    path = m.save_checkpoint(tmp_path / 'forest.pkl')
    restored = ForestDiffusionWrapper.load_checkpoint(path)
    pd.testing.assert_frame_equal(m.sample(11, seed=8).drop(columns='id'),
                                  restored.sample(11, seed=8).drop(columns='id'))
    m.fit(frame, continuous_cols=['x'], discrete_cols=['k'], categorical_cols=['c'], id_col='id')
    pd.testing.assert_frame_equal(sample, m.sample(11, seed=8))


def test_imputation_labels_and_covariate_validation():
    x = np.arange(16, dtype=float).reshape(8, 2) / 8
    m = tree(x, label_y=np.array([2, 9] * 4), diffusion_type='vp')
    incomplete = x[:3].copy()
    incomplete[0, 0] = np.nan
    for labels in [None, [2, 2], [2, 9, 100]]:
        with pytest.raises(ValueError, match='known training label'):
            m.impute(X=incomplete, label_y=labels)
    result = m.impute(X=incomplete, label_y=np.array([2, 9, 2]), seed=1)
    np.testing.assert_array_equal(result[:, :2][np.isfinite(incomplete)], incomplete[np.isfinite(incomplete)])
    assert np.isfinite(result).all()
    conditional = tree(x, X_covs=np.ones((8, 1)))
    with pytest.raises(ValueError, match='covariates'):
        conditional.generate(batch_size=3)
    assert conditional.generate(batch_size=3, X_covs=np.ones((3, 1))).shape == (3, 2)
