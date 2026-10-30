"""IPF defaults must mix towards the Gaussian prior and preserve saved clocks."""
from dataclasses import replace
from importlib import import_module
import pickle

import numpy as np
import pandas as pd
import pytest
import torch


CASES = [
    ("continuous_time.joint_distribution.mlp", "IPFDSB"),
    ("discrete_time.joint_distribution.mlp", "IPFDSB"),
    ("continuous_time.joint_distribution.boosting", "JointContinuousBoosted"),
    ("discrete_time.joint_distribution.boosting", "JointDiscreteBoosted"),
    ("continuous_time.feature_wise.boosting", "StructuralContinuousBoosted"),
    ("discrete_time.feature_wise.boosting", "StructuralDiscreteBoosted"),
]


def make_solver(path, name, **kwargs):
    module = import_module(f"sbtab.solvers.{path}.ipf_dsb.solver")
    cfg = getattr(module, name + "Config")(**kwargs)
    if "mlp" in path:
        cfg = replace(cfg, hidden_units=8, n_layers=2, time_features=8,
                      steps_per_phase=2, batch_size=16, cache_batches=1)
    else:
        cfg = replace(cfg, catboost=replace(cfg.catboost, iterations=3, depth=2, thread_count=2))
    cls = getattr(module, name + "Solver")
    return cls(cfg) if "feature_wise" in path else cls(2, cfg)


@pytest.mark.parametrize("path,name", CASES)
@pytest.mark.parametrize("num_steps", [20, 40])
def test_default_horizon_reduces_data_memory_independently_of_step_count(path, name, num_steps):
    solver = make_solver(path, name, num_steps=num_steps)
    dt = solver.timegrid.dt().double().numpy()
    assert solver.timegrid.T == pytest.approx(2.0)
    assert np.all(dt > 0) and np.all(dt < 1)
    # For the declared Euler OU chain, this is E[X_T | X_0] / X_0.
    # The legacy default retains >95% at K=20; the new horizon retains <14%.
    assert 0 < np.prod(1 - dt) < 0.14
    # Starting from a point mass, check that the chain develops prior-scale noise.
    variance = 0.0
    for step in dt:
        variance = (1 - step) ** 2 * variance + 2 * step
    assert 0.9 < variance < 1.4


@pytest.mark.parametrize("path,name", CASES)
def test_explicit_horizon_and_legacy_grid_are_available(path, name):
    assert make_solver(path, name, horizon=0.75).timegrid.T == pytest.approx(0.75)
    legacy = make_solver(path, name, num_steps=20, horizon=None)
    np.testing.assert_allclose(legacy.timegrid.dt(), np.geomspace(1e-4, 1e-2, 20), rtol=1e-6)
    assert legacy.timegrid.T == pytest.approx(0.046095, abs=1e-6)


@pytest.mark.parametrize("path,name", CASES)
def test_coarse_grid_still_enforces_ou_stability(path, name):
    with pytest.raises(ValueError, match=r"alpha_ou \* max\(gamma\)"):
        make_solver(path, name, num_steps=3)


@pytest.mark.parametrize("model_id", [
    "dsb_ct_joint_mlp", "dsb_dt_joint_mlp", "dsb_ct_joint_gbt",
    "dsb_dt_joint_gbt", "dsb_ct_structural_gbt", "dsb_dt_structural_gbt",
])
@pytest.mark.parametrize("config,expected_T", [({}, 2.0), ({"horizon": 0.75}, 0.75),
                                               ({"horizon": None}, 0.04609516184868363)])
def test_adapter_passes_default_and_explicit_horizons_to_solver(model_id, config, expected_T):
    from sbtab.solvers.registry import get_adapter_class

    adapter = get_adapter_class(model_id)()
    adapter.config = adapter.resolve_config(config)
    adapter.seed = 0
    solver = adapter._build(2, ["a", "b"])
    assert solver.timegrid.T == pytest.approx(expected_T)


@pytest.mark.parametrize("path,name", CASES)
@pytest.mark.parametrize("legacy", [False, True])
def test_fitted_checkpoint_preserves_grid_and_samples(tmp_path, path, name, legacy):
    # Four steps with the old raw schedule would fail the OU stability bound if
    # a loader accidentally applied the new T=2 default to an old checkpoint.
    kwargs = {"num_steps": 4, "horizon": None} if legacy else {}
    solver = make_solver(path, name, ipf_iters=1, **kwargs)
    data = pd.DataFrame(np.random.default_rng(5).normal(size=(32, 2)), columns=["a", "b"])
    solver.fit(data)
    checkpoint = tmp_path / "model.bin"
    solver.save_checkpoint(checkpoint)
    if legacy:
        if "mlp" in path:
            state = torch.load(checkpoint, weights_only=True)
            state["config"].pop("horizon")
            torch.save(state, checkpoint)
        else:
            with checkpoint.open("rb") as fh:
                state = pickle.load(fh)
            del state["cfg"].__dict__["horizon"]  # field absent in older pickled dataclasses
            with checkpoint.open("wb") as fh:
                pickle.dump(state, fh)
    restored = type(solver).load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored.timegrid.grid(), solver.timegrid.grid())
    np.testing.assert_array_equal(restored.sample(8, seed=17), solver.sample(8, seed=17))
