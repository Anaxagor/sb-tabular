"""Default DSB scope stays small; every registered variant can be explicitly selected."""
import pytest

from sbtab.experiments import cross_validate, pipeline, tune
from sbtab.experiments.experiment_common import StageError, canonical_hash, read_json, write_json
from sbtab.experiments.model_selection import BASIC_DSB_MODELS
from sbtab.solvers.registry import get_adapter_class, missing_requirements, solver_registry


EXCLUDED = sorted(entry.id for entry in solver_registry.values()
                  if entry.family in BASIC_DSB_MODELS and entry.id not in BASIC_DSB_MODELS.values())


@pytest.mark.parametrize("smoke", [False, True])
def test_default_plans_keep_only_basic_dsb_models_and_other_available_families(tmp_path, smoke):
    result = pipeline.create_plan(tmp_path / "experiment", datasets=["diabetes", "car_evaluation"], smoke=smoke)
    plan = pipeline.load_plan(result["plan"])
    selected = {task["model"] for task in plan["tasks"]}
    expected = {entry.id for entry in solver_registry.values()
                if entry.status != "unavailable" and not missing_requirements(entry.id) and entry.id not in EXCLUDED}
    assert selected == expected
    assert selected & {entry.id for entry in solver_registry.values() if entry.family in BASIC_DSB_MODELS} == {
        "dsb_ct_joint_mlp", "dsbm_ct_joint_mlp"}
    assert not set(EXCLUDED) & set(plan["search_spaces"])
    assert set(EXCLUDED) <= {item["model"] for item in plan["excluded"] if "excluded from default experiments" in item["reason"]}
    assert plan["basic_dsb_models"] == BASIC_DSB_MODELS
    assert plan["n_trials"] == (3 if smoke else 100) and plan["n_folds"] == 5


@pytest.mark.parametrize("model", EXCLUDED)
def test_explicit_variants_use_the_same_pipeline_and_stages(tmp_path, monkeypatch, model):
    result = pipeline.create_plan(tmp_path / "experiment", datasets=["diabetes"], models=[model])
    plan = pipeline.load_plan(result["plan"])
    assert [task["model"] for task in plan["tasks"]] == [model]
    assert plan["n_trials"] == 100 and plan["n_folds"] == 5

    def reached_splits(*args):
        raise StageError("undefined", "reached split validation")
    monkeypatch.setattr(tune, "load_split_artifacts", reached_splits)
    monkeypatch.setattr(cross_validate, "load_split_artifacts", reached_splits)
    with pytest.raises(StageError, match="reached split validation"):
        tune.run("diabetes", model, "missing-splits.json", "missing-space.yaml", resume=False, smoke=False)
    with pytest.raises(StageError, match="reached split validation"):
        cross_validate.run("diabetes", model, "missing-selection.json", "missing-splits.json")


def test_existing_plans_record_the_default_model_scope(tmp_path):
    result = pipeline.create_plan(tmp_path / "experiment", datasets=["diabetes"], models=["dsb_ct_joint_mlp"])
    plan = read_json(result["plan"])
    plan.pop("basic_dsb_models")
    plan["plan_hash"] = canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    write_json(result["plan"], plan)
    with pytest.raises(StageError, match="selection changed"):
        pipeline.load_plan(result["plan"])


@pytest.mark.parametrize("model", [model for model in EXCLUDED
                                  if not {"device", "enable_gpu"} & get_adapter_class(model).DEFAULTS.keys()])
def test_explicit_cpu_only_variants_accept_cpu_device_and_reject_cuda(tmp_path, model):
    result = pipeline.create_plan(tmp_path / "cpu", datasets=["diabetes"], models=[model], device="cpu")
    plan = pipeline.load_plan(result["plan"])
    assert plan["device"] == "cpu" and [task["model"] for task in plan["tasks"]] == [model]
    space = plan["search_spaces"][model]
    assert "source_path" not in space     # No ignored execution key is injected into the model configuration.
    with pytest.raises(StageError, match="CPU-only"):
        pipeline.create_plan(tmp_path / "cuda", datasets=["diabetes"], models=[model], device="cuda")


def test_structural_dsbm_missing_graph_package_stops_before_trial_allocation(tmp_path, monkeypatch):
    import importlib

    original_import = importlib.import_module

    def without_pgmpy(name, *args, **kwargs):
        if name == "pgmpy":
            raise ModuleNotFoundError("pgmpy is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_pgmpy)
    model = "dsbm_dt_structural_gbt"
    assert missing_requirements(model) == ("pgmpy",)
    with pytest.raises(StageError, match="missing dependencies.*pgmpy"):
        pipeline.create_plan(tmp_path / "experiment", datasets=["diabetes"], models=[model])
    with pytest.raises(StageError, match="optional packages.*pgmpy"):
        tune.run("diabetes", model, "missing-splits.json", "missing-space.yaml", resume=False, smoke=False)
    with pytest.raises(StageError, match="optional packages.*pgmpy"):
        cross_validate.run("diabetes", model, "missing-selection.json", "missing-splits.json")
    assert not (tmp_path / "experiment").exists()
