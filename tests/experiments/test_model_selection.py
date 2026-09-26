"""The experiment includes exactly one basic joint MLP for each DSB family."""
import pytest

from sbtab.experiments import cross_validate, pipeline, tune
from sbtab.experiments.experiment_common import StageError, canonical_hash, read_json, write_json
from sbtab.experiments.model_selection import BASIC_DSB_MODELS
from sbtab.solvers.registry import missing_requirements, solver_registry


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
    assert set(EXCLUDED) <= {item["model"] for item in plan["excluded"] if "excluded from experiments" in item["reason"]}
    assert plan["basic_dsb_models"] == BASIC_DSB_MODELS
    assert plan["n_trials"] == (3 if smoke else 100) and plan["n_folds"] == 5


@pytest.mark.parametrize("model", EXCLUDED)
def test_excluded_variants_cannot_be_requested_or_trained_directly(tmp_path, model):
    with pytest.raises(StageError, match="excluded from experiments"):
        pipeline.create_plan(tmp_path / "experiment", datasets=["diabetes"], models=[model])
    # Reject before reading splits/configs or initializing any model.
    with pytest.raises(StageError, match="excluded from experiments"):
        tune.run("diabetes", model, "missing-splits.json", "missing-space.yaml", resume=False, smoke=False)
    with pytest.raises(StageError, match="excluded from experiments"):
        cross_validate.run("diabetes", model, "missing-selection.json", "missing-splits.json")
    assert not (tmp_path / "experiment").exists()


@pytest.mark.parametrize("change", ["old_selection", "excluded_task"])
def test_existing_plans_cannot_bypass_current_model_selection(tmp_path, change):
    result = pipeline.create_plan(tmp_path / "experiment", datasets=["diabetes"], models=["dsb_ct_joint_mlp"])
    plan = read_json(result["plan"])
    if change == "old_selection":
        plan.pop("basic_dsb_models")
    else:
        plan["tasks"][0]["model"] = "dsb_dt_joint_mlp"
    plan["plan_hash"] = canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    write_json(result["plan"], plan)
    with pytest.raises(StageError, match="selection changed|excluded from experiments"):
        pipeline.load_plan(result["plan"])
