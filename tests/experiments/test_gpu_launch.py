"""GPU selection, immutable device overrides, and compute-node smoke checks."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.experiments import pipeline
from sbtab.experiments.cluster_environment import check_cuda
from sbtab.experiments.experiment_common import StageError, read_json
from sbtab.experiments.tune import load_search_space
from sbtab.solvers.registry import get_adapter_class, missing_requirements


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_override_reaches_every_trial_without_changing_ranges(tmp_path, monkeypatch, device):
    monkeypatch.setattr(pipeline, "missing_requirements", lambda _: ())
    models = ["dsb_ct_joint_mlp", "dsbm_ct_joint_mlp", "lightsb", "ctgan", "csbm"]
    result = pipeline.create_plan(tmp_path, datasets=["diabetes", "car_evaluation"], models=models, device=device)
    plan = pipeline.load_plan(result["plan"])
    assert plan["n_trials"] == 100 and plan["n_folds"] == 5 and plan["device"] == device
    for model, entry in plan["search_spaces"].items():
        original = load_search_space(entry["source_path"], model, "production")
        effective = load_search_space(entry["path"], model, "production")
        key = "enable_gpu" if model == "ctgan" else "device"
        assert effective["fixed"][key] == (device == "cuda" if model == "ctgan" else device)
        assert original["params"] == effective["params"]
    assert pipeline.create_plan(tmp_path, datasets=["diabetes", "car_evaluation"], models=models, device=device) == result
    entry = plan["search_spaces"]["lightsb"]
    Path(entry["path"]).write_text("{}")
    with pytest.raises(StageError, match="changed"):
        pipeline.load_plan(result["plan"])
    with pytest.raises(StageError, match="frozen search space"):
        pipeline.create_plan(tmp_path, datasets=["diabetes", "car_evaluation"], models=models, device=device)


def test_gpu_worker_rejects_cpu_before_allocating_trials(tmp_path, monkeypatch):
    result = pipeline.create_plan(tmp_path, datasets=["diabetes"], models=["lightsb"], smoke=True, device="cuda")
    pipeline.prepare(result["plan"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    record = pipeline.worker(result["plan"], 0)
    assert record["status"] == "training_failed" and "CUDA is required" in record["error"]
    assert not list(tmp_path.rglob("study.sqlite3"))


GPU_MODELS = ["dsb_ct_joint_mlp", "dsbm_ct_joint_mlp", "lightsb", "csbm", "csbm_annealed",
              "mixedsbm", "tabddpm", "ve_score_sde_simplified", "ctgan", "tabbyflow", "forestdiffusion"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an allocated CUDA GPU; run this file on the cluster")
@pytest.mark.parametrize("model", GPU_MODELS)
def test_actual_cuda_fit_sample_reload(tmp_path, model):
    assert not missing_requirements(model), f"Install the cluster environment before testing {model}"
    check_cuda()
    rng = np.random.default_rng(19)
    discrete = model in ("csbm", "csbm_annealed")
    frame = pd.DataFrame({"x": rng.integers(0, 3, 80) if discrete else rng.normal(size=80),
                          "c": rng.integers(0, 3, 80), "y": np.tile([0, 1], 40)})
    frame.index.name = "row_id"
    schema = DatasetSchema("cuda_smoke", (ColumnSpec("x", "discrete" if discrete else "continuous"),
        ColumnSpec("c", "categorical"), ColumnSpec("y", "categorical", role="target")), task="classification")
    space = load_search_space(f"configs/search_spaces/smoke/{model}.yaml", model, "smoke")
    config = {**space["fixed"], **{key: spec["choices"][0] if spec["type"] == "categorical" else spec["low"]
                                  for key, spec in space["params"].items()}}
    config.update({"enable_gpu": True} if model == "ctgan" else {"device": "cuda"})
    adapter = get_adapter_class(model)().fit(frame, schema, config, seed=5)
    generated = adapter.sample(32, seed=7)
    assert len(generated) == 32 and np.isfinite(generated.to_numpy(dtype=float)).all()
    adapter.save_checkpoint(tmp_path / model)
    restored = type(adapter).load_checkpoint(tmp_path / model)
    np.testing.assert_allclose(restored.sample(32, seed=7), generated, rtol=1e-5, atol=1e-6)
