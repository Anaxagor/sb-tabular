"""Active model policy and per-model resources are enforced before expensive work."""
import json
from pathlib import Path

import pandas as pd
import pytest

from sbtab.experiments import aggregate_results, calculate_metrics, cluster_environment, cross_validate, pipeline, tune
from sbtab.experiments.experiment_common import StageError, canonical_hash, library_versions, read_json, write_json
from sbtab.solvers.registry import get_adapter_class


@pytest.mark.parametrize("smoke", [False, True])
def test_default_matrix_includes_tabbyflow_and_forest_without_tabpfgen(tmp_path, monkeypatch, smoke):
    monkeypatch.setattr(pipeline, "missing_requirements", lambda _: ())
    result = pipeline.create_plan(tmp_path, datasets=["diabetes", "car_evaluation"], smoke=smoke)
    plan = pipeline.load_plan(result["plan"])
    assert {"tabbyflow", "forestdiffusion"} <= set(plan["search_spaces"])
    assert "tabpfgen" not in plan["search_spaces"]
    assert all(task["model"] != "tabpfgen" for task in plan["tasks"])
    assert any(item["model"] == "tabpfgen" and "Excluded from tuning" in item["reason"] for item in plan["excluded"])
    assert "pretrained_models" not in plan
    assert not (Path("configs/search_spaces") / "tabpfgen.yaml").exists()
    assert not (Path("configs/search_spaces/smoke") / "tabpfgen.yaml").exists()


def test_tabpfgen_is_rejected_even_when_explicitly_requested(tmp_path):
    with pytest.raises(StageError, match="tabpfgen.*unavailable"):
        pipeline.create_plan(tmp_path, datasets=["diabetes"], models=["tabpfgen"])
    with pytest.raises(StageError, match="tabpfgen.*unavailable"):
        tune.run("diabetes", "tabpfgen", "missing-splits.json", "missing-space.yaml", resume=False, smoke=False)
    with pytest.raises(StageError, match="tabpfgen.*unavailable"):
        cross_validate.run("diabetes", "tabpfgen", "missing-selection.json", "missing-splits.json")
    with pytest.raises(LookupError, match="tabpfgen.*unavailable"):
        get_adapter_class("tabpfgen")
    manifest = tmp_path / "old-cv.json"
    write_json(manifest, {"model": "tabpfgen"})
    with pytest.raises(StageError, match="tabpfgen.*unavailable"):
        calculate_metrics.run(manifest, "missing-metrics.yaml", dry_run=True)
    assert not list(tmp_path.rglob("study.sqlite3"))


def test_old_plan_cannot_reactivate_tabpfgen(tmp_path):
    result = pipeline.create_plan(tmp_path, datasets=["diabetes"], models=["lightsb"], smoke=True)
    plan = read_json(result["plan"])
    plan["tasks"][0]["model"] = "tabpfgen"
    plan["plan_hash"] = canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    write_json(result["plan"], plan)
    with pytest.raises(StageError, match="tabpfgen.*unavailable"):
        pipeline.load_plan(result["plan"])
    record = pipeline.worker(result["plan"], 0)
    assert record["status"] == "undefined" and "tabpfgen" in record["error"]
    assert not list(tmp_path.rglob("study.sqlite3"))


def test_cluster_preflight_needs_no_pretrained_model_or_historical_package(monkeypatch, capsys):
    assert not hasattr(cluster_environment, "check_tabpfn_cache")
    monkeypatch.setattr(cluster_environment, "check_packages", lambda: {"torch": "2.6.0"})
    assert cluster_environment.main([]) == 0
    assert json.loads(capsys.readouterr().out) == {"packages": {"torch": "2.6.0"}, "status": "ok"}
    assert not {"tabpfgen", "tabpfn"} & set(library_versions())
    requirements = Path("requirements-cluster.txt").read_text().lower()
    assert "tabpfgen" not in requirements and "tabpfn" not in requirements


def _historical_evaluation(root, model):
    directory = root / "toy" / model / "run-old" / "evaluation" / "metrics-v2"
    directory.mkdir(parents=True)
    records = [{"dataset": "toy", "fold": fold, "model": model, "metric": "wd", "value": .2,
                "status": "ok", "direction": "lower_is_better", "unit": "standardised"} for fold in range(5)]
    pd.DataFrame(records).to_csv(directory / "per_fold.csv", index=False)
    write_json(directory / "summary.json", {"dataset": "toy", "model": model, "complete_five_fold": True, "metrics": []})
    write_json(directory / "fold-0" / "metrics.json", {"records": records})
    return directory


@pytest.mark.parametrize("include_active", [False, True])
def test_new_aggregates_exclude_tabpfgen_without_modifying_saved_results(tmp_path, include_active):
    old = _historical_evaluation(tmp_path, "tabpfgen")
    before = {path: path.read_bytes() for path in old.rglob("*") if path.is_file()}
    if include_active:
        _historical_evaluation(tmp_path, "lightsb")
    out = aggregate_results.aggregate_root(tmp_path, rank_metrics=["wd"], models=["tabpfgen", "lightsb"])
    assert out["n_runs"] == int(include_active)
    assert out["excluded_models"] and all(item["model"] == "tabpfgen" for item in out["excluded_models"])
    assert all("Excluded from tuning" in item["reason"] for item in out["excluded_models"])
    combined = pd.read_csv(tmp_path / "aggregate" / "per_fold_all.csv")
    assert "tabpfgen" not in set(combined["model"])
    assert {path: path.read_bytes() for path in before} == before
    with pytest.raises(StageError, match="tabpfgen.*excluded"):
        aggregate_results.aggregate_run(old)
    assert {path: path.read_bytes() for path in before} == before


def test_auto_device_mapping_is_frozen_per_model_and_preserves_search_ranges(tmp_path):
    result = pipeline.create_plan(tmp_path, datasets=["diabetes"],
        models=["lightsb", "tabbyflow", "forestdiffusion", "dsb_ct_joint_gbt"], device="auto")
    plan = pipeline.load_plan(result["plan"])
    assert plan["model_devices"] == {
        "lightsb": "cuda", "tabbyflow": "cuda", "forestdiffusion": "cpu", "dsb_ct_joint_gbt": "cpu"}
    assert {task["model"]: task["device"] for task in plan["tasks"]} == plan["model_devices"]
    for model, item in plan["search_spaces"].items():
        current = tune.load_search_space(item["path"], model, "production")
        original = tune.load_search_space(f"configs/search_spaces/{model}.yaml", model, "production")
        assert current["params"] == original["params"]
        if model == "dsb_ct_joint_gbt":
            assert current == original and "source_path" not in item
        else:
            assert current["fixed"]["device"] == plan["model_devices"][model]


def test_no_device_override_retains_profile_cpu_defaults(tmp_path):
    result = pipeline.create_plan(tmp_path, datasets=["diabetes"], models=["lightsb", "forestdiffusion"])
    plan = pipeline.load_plan(result["plan"])
    assert plan["device"] is None and set(plan["model_devices"].values()) == {"cpu"}
    assert all("source_path" not in item for item in plan["search_spaces"].values())


def _stub_generation(monkeypatch, root, model, calls):
    def fit(*args, **kwargs):
        calls.append("tune")
        write_json(root / "diabetes" / model / pipeline.RUN_ID / "tuning" / "selected_config.json", {})
        return {"selection": "final", "counts": {"allocated": 3}}

    def cv(*args, **kwargs):
        calls.append("validate_cv" if kwargs.get("dry_run") else ("cv", kwargs["folds"][0]))
        return {"n_ok": 5}

    monkeypatch.setattr(tune, "run", fit)
    monkeypatch.setattr(cross_validate, "run", cv)
    monkeypatch.setattr(calculate_metrics, "run", lambda *a, **k: pytest.fail("generation must not run CPU metrics"))


@pytest.mark.parametrize("model", ["forestdiffusion", "lightsb"])
def test_generate_stage_runs_tuning_and_five_folds_with_the_task_device(tmp_path, monkeypatch, model):
    result = pipeline.create_plan(tmp_path, datasets=["diabetes"], models=[model], smoke=True, device="auto")
    plan = pipeline.load_plan(result["plan"])
    write_json(tmp_path / "pipeline" / "preparation.json",
               {"plan_hash": plan["plan_hash"], "datasets": {"diabetes": {"split_status": "ok"}}})
    calls = []
    _stub_generation(monkeypatch, tmp_path, model, calls)
    monkeypatch.setattr(cluster_environment, "check_cuda", lambda: calls.append("cuda") or {"device": "test"})
    record = pipeline.worker(result["plan"], 0, stage="generate")
    assert record["status"] == "ok" and set(record["stages"]) == {"tune", "cv"}
    expected = (["cuda"] if model == "lightsb" else []) + ["tune", "validate_cv"] + [("cv", k) for k in range(5)]
    assert calls == expected
    assert pipeline.aggregate(result["plan"])["complete"] is False
