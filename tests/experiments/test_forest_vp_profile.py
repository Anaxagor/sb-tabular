"""Separate Forest-VP plans and bounded real tune/CV checkpoint coverage."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sbtab.experiments import cross_validate, pipeline, tune
from sbtab.experiments.experiment_common import StageError, read_json
from sbtab.solvers.registry import get_adapter_class
from test_stages import make_dataset


PRODUCTION = Path("configs/search_spaces/forest_vp")
SMOKE = Path("configs/search_spaces/smoke/forest_vp/forestdiffusion.yaml")
MODEL = "forestdiffusion"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_production_vp_profile_is_selected_and_frozen_with_device_override(tmp_path, monkeypatch, device):
    # Planning assertions do not require a trained backend or an allocated GPU.
    monkeypatch.setattr(pipeline, "missing_requirements", lambda _: ())
    arguments = dict(datasets=["diabetes"], models=[MODEL], device=device)
    result = pipeline.create_plan(tmp_path, search_space_dir=PRODUCTION, **arguments)
    plan = pipeline.load_plan(result["plan"])
    entry = plan["search_spaces"][MODEL]
    source = tune.load_search_space(entry["source_path"], MODEL, "production")
    effective = tune.load_search_space(entry["path"], MODEL, "production")
    flow = tune.load_search_space("configs/search_spaces/forestdiffusion.yaml", MODEL, "production")

    assert plan["n_trials"] == 100 and plan["n_folds"] == 5
    assert Path(entry["source_path"]) == (PRODUCTION / "forestdiffusion.yaml").resolve()
    assert source["fixed"] == {**flow["fixed"], "diffusion_type": "vp"}
    assert effective["fixed"] == {**source["fixed"], "device": device}
    assert source["params"] == effective["params"] == flow["params"]
    assert pipeline.create_plan(tmp_path, search_space_dir=PRODUCTION, **arguments) == result

    # A Flow experiment cannot resume into the same output root as VP.
    with pytest.raises(StageError, match="existing pipeline plan differs"):
        pipeline.create_plan(tmp_path, **arguments)
    pipeline.load_plan(result["plan"])

    # Even a mode-only change to the frozen effective profile is detected.
    effective["fixed"]["diffusion_type"] = "flow"
    Path(entry["path"]).write_text(yaml.safe_dump(effective))
    with pytest.raises(StageError, match="configuration changed"):
        pipeline.load_plan(result["plan"])


@pytest.mark.integration
def test_real_vp_smoke_tuning_cv_and_checkpoint_retain_selected_mode(tmp_path, monkeypatch):
    pytest.importorskip("xgboost", reason="real Forest-VP smoke requires the optional XGBoost backend")
    root, splits_path, _, schema = make_dataset(tmp_path, monkeypatch, n=120)
    splits = read_json(splits_path)
    result = tune.run("toy", MODEL, splits_path, SMOKE, resume=False, smoke=True)
    assert result["counts"]["COMPLETE"] == 3
    assert result["selection"] == "final"
    run = Path(result["run_dir"])
    selected_path = run / "tuning/selected_config.json"
    selected = read_json(selected_path)
    assert selected["config"]["diffusion_type"] == "vp"
    assert read_json(run / "tuning/best.json")["reload_verified"]
    for path in sorted((run / "tuning").glob("trial-*/config.json")):
        assert read_json(path)["effective_config"]["diffusion_type"] == "vp"
        assert len(pd.read_parquet(path.parent / "synthetic.parquet")) == splits["n_V"]

    # One real CV fold is enough to exercise fresh fitting, generation and reload;
    # the small smoke search above keeps this check independent of production cost.
    cv = cross_validate.run("toy", MODEL, selected_path, splits_path, root, smoke=True, folds=[0])
    assert cv["n_ok"] == 1
    fold_dir = run / "cv/fold-0"
    fold = read_json(fold_dir / "manifest.json")
    assert fold["effective_config"]["diffusion_type"] == "vp"
    assert fold["reload_verified"]["ok"]
    assert fold["n_generated"] == len(splits["folds"][0]["train_row_ids"])
    adapter = get_adapter_class(MODEL).load_checkpoint(fold_dir / "checkpoint")
    assert adapter.model.cfg.diffusion_type == "vp"
    assert adapter.model.variant_id == "forest_vp_xgboost_joint_xy"
    generated = adapter.sample(8, seed=19)
    assert len(generated) == 8
    assert np.isfinite(generated[list(schema.continuous)].to_numpy(dtype=float)).all()

    again = tune.run("toy", MODEL, splits_path, SMOKE, resume=True, smoke=True)
    assert again["new_trials"] == 0
    with pytest.raises(StageError, match="search_space_hash"):
        tune.run("toy", MODEL, splits_path, "configs/search_spaces/smoke/forestdiffusion.yaml",
                 resume=True, smoke=True, run_id=run.name)
