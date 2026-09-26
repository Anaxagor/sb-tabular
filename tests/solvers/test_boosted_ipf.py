"""Boosted IPF-DSB acceptance tests: edge indices, units, OU reference, caches, structure, reload."""
import numpy as np
import pandas as pd
import pytest

from sbtab.models.boosted.catboost_continuous_joint import CatBoostContinuousJoint, CatBoostContinuousJointConfig
from sbtab.models.boosted.catboost_continuous_scalar import CatBoostContinuousScalarConfig
from sbtab.models.boosted.catboost_discrete_joint import CatBoostDiscreteJoint, CatBoostDiscreteJointConfig
from sbtab.models.boosted.catboost_discrete_scalar import CatBoostDiscreteScalar, CatBoostDiscreteScalarConfig
from sbtab.solvers import structure as structure_mod
from sbtab.solvers.continuous_time.feature_wise.boosting.ipf_dsb.solver import (
    StructuralContinuousBoostedConfig, StructuralContinuousBoostedSolver)
from sbtab.solvers.continuous_time.joint_distribution.boosting.ipf_dsb.solver import (
    JointContinuousBoostedConfig, JointContinuousBoostedSolver)
from sbtab.solvers.discrete_time.feature_wise.boosting.ipf_dsb.solver import (
    StructuralDiscreteBoostedConfig, StructuralDiscreteBoostedSolver)
from sbtab.solvers.discrete_time.joint_distribution.boosting.ipf_dsb.solver import (
    JointDiscreteBoostedConfig, JointDiscreteBoostedSolver)

CB = dict(iterations=15, depth=2, learning_rate=0.2, thread_count=2)
K = 4


def frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": 0.9 * a + np.sqrt(0.19) * rng.normal(size=n)}).astype(np.float32)


# --------------------------------------------------------------------------- indices
def test_every_edge_model_is_fitted_and_called_under_its_own_index(monkeypatch):
    """
    Reproducer for the index defect: the old code fitted B at min(k+1, K-1) and F at
    max(k-1, 0) — B[0] was never trained, B[K-1] was fitted twice — and fit() itself
    crashed with 'Model for time step 0 is not trained'.
    """
    fits, calls = [], []
    real_fit, real_pred = CatBoostDiscreteJoint.fit_step, CatBoostDiscreteJoint.predict_step

    def spy_fit(self, k, x, y, **kw):
        fits.append((id(self), k))
        return real_fit(self, k, x, y, **kw)

    def spy_pred(self, k, x):
        calls.append((id(self), k))
        return real_pred(self, k, x)

    monkeypatch.setattr(CatBoostDiscreteJoint, "fit_step", spy_fit)
    monkeypatch.setattr(CatBoostDiscreteJoint, "predict_step", spy_pred)
    s = JointDiscreteBoostedSolver(2, JointDiscreteBoostedConfig(num_steps=K, ipf_iters=2,
                                                                 catboost=CatBoostDiscreteJointConfig(**CB)))
    s.fit(frame())                                                # completes (it used to raise)
    fb, ff = id(s.field_b), id(s.field_f)
    for it_fits in (fits[: 2 * K], fits[2 * K:]):                 # each IPF iteration: B on 0..K-1, then F on K-1..0
        assert [k for f, k in it_fits if f == fb] == list(range(K))
        assert [k for f, k in it_fits if f == ff] == list(range(K - 1, -1, -1))
    assert all(s.field_b.is_fitted(k) and s.field_f.is_fitted(k) for k in range(K))

    calls.clear()
    s.sample(5, seed=0)
    assert calls == [(fb, k) for k in range(K - 1, -1, -1)]       # reverse sampling calls K-1..0, incl. index 0


def test_continuous_time_labels_are_one_to_one_and_the_second_evaluation_uses_the_same_edge(monkeypatch):
    """Old code labelled edge k with times[k+1] (clamped) and evaluated F at the wrong time."""
    seen = []
    real = CatBoostContinuousJoint.predict

    def spy(self, x, t=0.0):
        seen.append(float(np.asarray(t).reshape(-1)[0]))
        return real(self, x, t)

    monkeypatch.setattr(CatBoostContinuousJoint, "predict", spy)
    s = JointContinuousBoostedSolver(2, JointContinuousBoostedConfig(num_steps=K, ipf_iters=2,
                                                                     catboost=CatBoostContinuousJointConfig(**CB)))
    s.fit(frame())
    times = [float(t) for t in s.times]
    for log in s.stage_log:
        labels = log["edge_labels"]
        assert sorted(labels) == sorted(times) and len(set(labels)) == K      # no duplicated, no missing label
    # predict() is always called in pairs (curr, next) at the SAME edge time
    assert len(seen) % 2 == 0 and all(seen[i] == seen[i + 1] for i in range(0, len(seen), 2))
    seen.clear()
    s.sample(3, seed=0)
    assert seen == times[::-1]


# --------------------------------------------------------------------------- units and reference
def test_predictions_are_next_state_means_not_drifts_scaled_by_gamma():
    s = JointDiscreteBoostedSolver(2, JointDiscreteBoostedConfig(num_steps=K, ipf_iters=1))
    s._fitted = True
    s.field_b.predict_step = lambda k, x: x + 1.0                 # stub: mean map shifts by exactly 1
    s.gammas = np.zeros_like(s.gammas)                            # switch the noise off
    out = s.sample(6, seed=0)
    start = np.random.default_rng(0).normal(size=(6, 2)).astype(np.float32)
    np.testing.assert_allclose(out, start + K, atol=1e-5)         # +1 per step; a drift would give + sum(gamma)


def test_ou_reference_uses_the_step_interval():
    s = JointDiscreteBoostedSolver(2, JointDiscreteBoostedConfig(num_steps=20, alpha_ou=0.7))
    x = np.ones((3, 2), dtype=np.float32)
    for k in (0, 10, 19):
        np.testing.assert_allclose(s._reference_mean(k, x), 1.0 - 0.7 * s.gammas[k], rtol=1e-6)
    # The old joint code contracted by the CUMULATIVE time t_grid[k] instead of the step gamma_k.
    # At k = 19 (gamma = 0.01, elapsed = 0.0461) with alpha = 0.7 that is 0.9677 instead of 0.9930.
    step_based, cumulative = 1.0 - 0.7 * s.gammas[19], 1.0 - 0.7 * s.t_grid[19]
    assert step_based == pytest.approx(0.9930, abs=1e-4) and cumulative == pytest.approx(0.9677, abs=1e-4)
    assert float(s._reference_mean(19, x)[0, 0]) == pytest.approx(step_based, abs=1e-6)
    assert abs(float(s._reference_mean(19, x)[0, 0]) - cumulative) > 0.02


def test_forward_cache_holds_actual_reference_trajectory_states(monkeypatch):
    """IPF iteration 0: the inputs B[k] is trained on are states of the OU chain at step k+1."""
    cached = {}
    real = CatBoostDiscreteJoint.fit_step
    monkeypatch.setattr(CatBoostDiscreteJoint, "fit_step",
                        lambda self, k, x, y, **kw: (cached.setdefault((id(self), k), np.array(x)), real(self, k, x, y, **kw))[1])
    n = 4000
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(n, 2)).astype(np.float32) * 0.2 + 3.0, columns=["a", "b"])
    s = JointDiscreteBoostedSolver(2, JointDiscreteBoostedConfig(num_steps=K, ipf_iters=1, gamma_min=0.05, gamma_max=0.2,
                                                                 catboost=CatBoostDiscreteJointConfig(**CB)))
    s.fit(df)
    mean, var = 3.0, 0.04
    for k in range(K):                                            # exact OU recursion of the first two moments
        a = 1.0 - s.gammas[k]
        mean, var = a * mean, a * a * var + 2.0 * s.gammas[k]
        x = cached[(id(s.field_b), k)]
        # n = 4000: se(mean) = sqrt(var/n) <= 0.013, se(var) = var sqrt(2/n) <= 0.015 -> 5 se
        assert x.mean() == pytest.approx(mean, abs=5 * np.sqrt(var / n))
        assert x.var() == pytest.approx(var, abs=5 * var * np.sqrt(2.0 / n))


def test_residual_parameterisation_tracks_the_identity_plus_small_correction():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 1)).astype(np.float32)
    y = x * (1 - 0.01)                                            # an O(gamma) mean-matching map
    xt = rng.normal(size=(300, 1)).astype(np.float32)
    err = {}
    for residual in (False, True):
        f = CatBoostDiscreteJoint(1, np.array([0.1]), CatBoostDiscreteJointConfig(residual=residual, **CB))
        f.fit_step(0, x, y)
        pred = f.predict_step(0, xt)
        assert pred.shape == (300, 1)                             # dim = 1 used to broadcast to (n, n)
        err[residual] = float(np.sqrt(np.mean((pred - xt * 0.99) ** 2)))
    assert err[True] < 0.01 < err[False]                          # trees cannot represent the identity map


# --------------------------------------------------------------------------- structure
@pytest.mark.integration
@pytest.mark.parametrize("cls,cfg_cls,cb_cls", [
    (StructuralDiscreteBoostedSolver, StructuralDiscreteBoostedConfig, CatBoostDiscreteScalarConfig),
    (StructuralContinuousBoostedSolver, StructuralContinuousBoostedConfig, CatBoostContinuousScalarConfig)])
def test_structural_graph_is_learned_on_fit_rows_only_and_parents_come_from_the_same_row(monkeypatch, tmp_path, cls, cfg_cls, cb_cls):
    full = frame(n=300)
    full.index = np.arange(1000, 1300)
    train, held = full.iloc[:200], full.iloc[200:]
    seen_index = []
    real = structure_mod.learn_dag
    import importlib
    mod = importlib.import_module(cls.__module__)
    monkeypatch.setattr(mod, "learn_dag", lambda df, n_bins=5: (seen_index.append(df.index.to_numpy().copy()), real(df, n_bins))[1])

    s = cls(cfg_cls(num_steps=3, ipf_iters=1, catboost=cb_cls(**CB))).fit(train)
    assert len(seen_index) == 1 and set(seen_index[0]) == set(train.index) and not set(seen_index[0]) & set(held.index)
    st = s.structure
    st.validate()                                                 # acyclic, parents generated first
    # a -> b and b -> a are Markov equivalent, so the DIRECTION is not identifiable from
    # observational data; what must hold is one edge, and the child generated after its parent.
    assert st.fit_row_count == 200 and len(st.edges) == 1
    parent, kid = st.order
    assert st.parents == {parent: [], kid: [parent]} and {parent, kid} == {"a", "b"}

    # parents fed to the child's model are the parent values generated for the SAME rows
    fed = []
    child = (s.fields if hasattr(s, "fields") else s.models)[kid]
    name = "predict_step" if hasattr(child, "predict_step") else "predict"
    real_pred = getattr(child, name)
    setattr(child, name, lambda *a, **kw: (fed.append(np.array(kw["x0"])), real_pred(*a, **kw))[1])
    out = s.sample(25, seed=4)
    assert list(out.columns) == ["a", "b"] and len(out) == 25
    assert len(fed) == 3 and all(np.array_equal(p[:, 0], out[parent].to_numpy(dtype=np.float32)) for p in fed)

    s.save_checkpoint(tmp_path / "s.pkl")
    r = cls.load_checkpoint(tmp_path / "s.pkl")
    assert r.structure.parents == st.parents                      # explicit ordered parent lists, restored verbatim
    pd.testing.assert_frame_equal(r.sample(25, seed=4), s.sample(25, seed=4))


def test_known_two_variable_relation_is_recovered_by_structure_learning():
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    dag = structure_mod.learn_dag(pd.DataFrame({"x": x, "y": 2 * x + 0.1 * rng.normal(size=500), "z": rng.normal(size=500)}))
    assert set(map(frozenset, dag.edges)) == {frozenset(("x", "y"))} and dag.parents["z"] == []
    const = structure_mod.learn_dag(pd.DataFrame({"x": x, "c": np.ones(500)}))     # constant column: no crash, no edge
    assert const.parents == {"x": [], "c": []}


@pytest.mark.integration
@pytest.mark.parametrize("make", [
    lambda: JointDiscreteBoostedSolver(2, JointDiscreteBoostedConfig(num_steps=K, ipf_iters=1, catboost=CatBoostDiscreteJointConfig(**CB))),
    lambda: JointContinuousBoostedSolver(2, JointContinuousBoostedConfig(num_steps=K, ipf_iters=1, catboost=CatBoostContinuousJointConfig(**CB)))])
def test_joint_solvers_sizes_seeds_and_reload(tmp_path, make):
    s = make().fit(frame())
    assert s.n_updates > 0
    for n in (1, 7):
        assert s.sample(n, seed=1).shape == (n, 2)
    assert np.array_equal(s.sample(9, seed=5), s.sample(9, seed=5))
    assert not np.array_equal(s.sample(9), s.sample(9))
    s.save_checkpoint(tmp_path / "j.pkl")
    assert np.array_equal(type(s).load_checkpoint(tmp_path / "j.pkl").sample(9, seed=5), s.sample(9, seed=5))
    with pytest.raises(RuntimeError):
        make().sample(3)                                          # not fitted


def test_ignored_options_are_rejected():
    with pytest.raises(ValueError):
        from sbtab.models.boosted.catboost_continuous_scalar import CatBoostContinuousScalar
        CatBoostContinuousScalar(CatBoostContinuousScalarConfig(feature_mode="x"))     # used to be silently ignored
    with pytest.raises(ValueError):
        CatBoostDiscreteJoint(2, np.array([0.1]), CatBoostDiscreteJointConfig(**CB)).fit_step(
            0, np.zeros((4, 2)), np.zeros((4, 2)), x0=np.zeros((4, 1)))                 # x0 used to be accepted and deleted
