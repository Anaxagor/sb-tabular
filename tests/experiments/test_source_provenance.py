"""Identical runtime files must validate on login and compute nodes without Git."""
import shutil

import pytest

from sbtab.experiments import experiment_common as common, pipeline


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "sbtab").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "sbtab/model.py").write_text("steps = 10\n")
    (root / "configs/model.yaml").write_text("lr: 0.01\n")
    monkeypatch.setattr(common, "REPO_ROOT", root)
    return root


def test_content_identity_ignores_git_availability_and_metadata(source_tree, monkeypatch):
    def git_metadata(*args):
        return {"rev-parse": "commit-a\n", "diff": "old diff\n", "ls-files": ""}[args[0]]
    monkeypatch.setattr(common, "_git", git_metadata)
    login = common.source_provenance()
    monkeypatch.setattr(common, "_git", lambda *args: None)
    compute = common.source_provenance()
    assert login["commit"] != compute["commit"]
    assert common.implementation_hash(login) == common.implementation_hash(compute)
    assert common.implementation_hash() == common.implementation_hash(login)


@pytest.mark.parametrize("change", ["edit_python", "edit_config", "add_python", "delete_python"])
def test_content_identity_detects_runtime_changes(source_tree, change):
    before = common.implementation_hash()
    if change == "edit_python":
        (source_tree / "sbtab/model.py").write_text("steps = 11\n")
    elif change == "edit_config":
        (source_tree / "configs/model.yaml").write_text("lr: 0.02\n")
    elif change == "add_python":
        (source_tree / "sbtab/helper.py").write_text("value = 1\n")
    else:
        (source_tree / "sbtab/model.py").unlink()
    assert common.implementation_hash() != before


def test_content_identity_ignores_generated_files_and_checkout_location(source_tree, monkeypatch):
    before = common.implementation_hash()
    for name in ("README.md", "slurm_logs/prepare-1.out", "sbtab/__pycache__/model.pyc",
                 "artifacts/pipeline/plan.json"):
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated output")
    assert common.implementation_hash() == before
    other = source_tree.parent / "compute-checkout"
    shutil.copytree(source_tree, other)
    monkeypatch.setattr(common, "REPO_ROOT", other)
    assert common.implementation_hash() == before


def test_plan_created_with_git_loads_without_git(tmp_path, monkeypatch):
    plan = pipeline.create_plan(tmp_path, datasets=["diabetes"], models=["lightsb"], smoke=True)
    expected = pipeline.load_plan(plan["plan"])
    monkeypatch.setattr(common, "_git", lambda *args: None)
    assert pipeline.load_plan(plan["plan"]) == expected
    assert common.implementation_hash(common.source_provenance()) == expected["implementation_hash"]


@pytest.mark.parametrize("legacy", [False, True])
def test_plan_reports_protocol_change_separately_from_legacy_hash(tmp_path, legacy):
    result = pipeline.create_plan(tmp_path, datasets=["diabetes"], models=["lightsb"], smoke=True)
    plan = common.read_json(result["plan"])
    if legacy:
        plan.pop("implementation_hash_version")
        message = "legacy Git-based"
    else:
        plan["protocol_hash"] = "old-protocol-hash"
        message = "protocol changed after planning: expected old-protocol-hash"
    plan["plan_hash"] = common.canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    common.write_json(result["plan"], plan)
    with pytest.raises(common.StageError, match=message):
        pipeline.load_plan(result["plan"])
