"""Cluster orchestration: protocol, recovery, isolation, and real process races."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

import pytest

from sbtab.experiments import pipeline
from sbtab.experiments.experiment_common import StageError, file_lock, read_json, write_json


def plan_for(tmp_path, smoke=True, datasets=None, models=None):
    result = pipeline.create_plan(tmp_path / "experiment", datasets=datasets or ["diabetes"],
                                  models=models or ["lightsb"], smoke=smoke)
    return Path(result["plan"])


def test_production_plan_keeps_full_protocol_and_compatible_tasks(tmp_path):
    path = plan_for(tmp_path, smoke=False, datasets=["diabetes", "car_evaluation"], models=["lightsb", "csbm"])
    plan = pipeline.load_plan(path)
    assert plan["n_trials"] == 100 and plan["n_folds"] == 5
    assert [(t["dataset"], t["model"]) for t in plan["tasks"]] == [
        ("car_evaluation", "csbm"), ("car_evaluation", "lightsb"), ("diabetes", "lightsb")]
    assert plan["excluded"] == [{"dataset": "diabetes", "model": "csbm", "reason": "incompatible data regime"}]
    assert plan_for(tmp_path, smoke=False, datasets=["diabetes", "car_evaluation"], models=["lightsb", "csbm"]) == path
    assert not list(path.parent.parent.glob("*/data.parquet"))  # no data preparation/training on the login node


def test_plan_refuses_changes_and_missing_explicit_dependencies(tmp_path, monkeypatch):
    path = plan_for(tmp_path)
    with pytest.raises(StageError, match="differs"):
        plan_for(tmp_path, models=["mixedsbm"])
    monkeypatch.setattr(pipeline, "missing_requirements", lambda _: ("missing_package",))
    with pytest.raises(StageError, match="missing dependencies"):
        plan_for(tmp_path / "missing")
    plan = read_json(path)
    plan["n_trials"] = 1
    write_json(path, plan)
    with pytest.raises(StageError, match="hash mismatch"):
        pipeline.load_plan(path)


def test_changed_implementation_is_visible_in_task_status(tmp_path, monkeypatch):
    path = plan_for(tmp_path)
    monkeypatch.setattr(pipeline, "implementation_hash", lambda: "changed")
    record = pipeline.worker(path, 0)
    assert record["status"] == "undefined"
    assert "implementation changed" in record["error"]
    assert read_json(path.parent / "tasks/00000.json")["status"] == "undefined"


def test_blocked_preflight_never_trains_and_is_in_final_summary(tmp_path, monkeypatch):
    from sbtab.experiments import prepare_splits, tune
    path = plan_for(tmp_path)
    monkeypatch.setattr(prepare_splits, "run", lambda *a, **k: {"split_status": "blocked_support"})
    monkeypatch.setattr(tune, "run", lambda *a, **k: pytest.fail("blocked dataset was tuned"))
    assert pipeline.prepare(path)["counts"] == {"blocked_support": 1}
    assert pipeline.worker(path, 0)["status"] == "blocked_support"
    assert pipeline.aggregate(path)["complete"] is False
    assert read_json(path.parent / "summary.json")["counts"] == {"blocked_support": 1}


def test_missing_and_killed_tasks_are_not_reported_complete(tmp_path):
    path = plan_for(tmp_path, models=["lightsb", "mixedsbm"])
    write_json(path.parent / "tasks/00000.json", {"task_id": 0, "status": "running"})
    summary = pipeline.aggregate(path)
    assert summary["counts"] == {"unfinished": 1, "not_run": 1}
    assert not summary["complete"]


def _contend_for_lock(path, queue):
    try:
        with file_lock(path, blocking=False):
            queue.put("acquired")
    except StageError:
        queue.put("blocked")


def test_duplicate_task_lock_is_released_after_owner_exits(tmp_path):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    path = str(tmp_path / "task.lock")
    with file_lock(path):
        process = ctx.Process(target=_contend_for_lock, args=(path, queue))
        process.start()
        assert queue.get(timeout=30) == "blocked"
        process.join(30)
        assert process.exitcode == 0
    with file_lock(path, blocking=False):
        pass


def _reference_race(dataset_dir, barrier, queue):
    from sbtab.experiments.calculate_metrics import real_reference
    root = Path(dataset_dir)
    def compute():
        with (root / "calls.txt").open("a") as handle:
            handle.write("fit\n")
        time.sleep(0.1)
        return {"status": "ok", "scores": {"r2": 0.7}, "predictions": {"pred": [1.0, 2.0]}}
    barrier.wait(timeout=30)
    result = real_reference(root, "abc", 0, compute)
    queue.put(result)


def test_parallel_models_share_one_real_reference_fit(tmp_path):
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_reference_race, args=(str(tmp_path), barrier, queue)) for _ in range(2)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=45) for _ in processes]
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    assert (tmp_path / "calls.txt").read_text() == "fit\n"
    assert sorted(result.pop("cache") for result in results) == ["hit", "miss"]
    assert results[0] == results[1]
    import pandas as pd
    assert len(pd.read_parquet(tmp_path / "utility_reference/abc/fold-0_predictions.parquet")) == 2


@pytest.mark.integration
def test_real_pipeline_resumes_partial_study_and_keeps_generator_artifacts(tmp_path, monkeypatch):
    from sbtab.experiments import tune, cross_validate
    from sbtab.experiments.experiment_common import file_hash
    path = plan_for(tmp_path)
    plan = pipeline.load_plan(path)
    assert pipeline.prepare(path)["counts"] == {"ok": 1}
    root = path.parent.parent
    partial = tune.run("diabetes", "lightsb", root / "diabetes/splits.json",
                       plan["search_spaces"]["lightsb"]["path"], resume=True, smoke=True,
                       protocol_path=plan["protocol_path"], run_id=plan["run_id"], max_new_trials=1)
    assert partial["counts"]["allocated"] == 1 and partial["selection"] == "provisional"
    original_cv = cross_validate.run
    def interrupted_cv(*args, **kwargs):
        if kwargs.get("folds") == [2]:
            raise KeyboardInterrupt("simulated preemption")
        return original_cv(*args, **kwargs)
    monkeypatch.setattr(cross_validate, "run", interrupted_cv)
    interrupted = pipeline.worker(path, 0)
    assert interrupted["status"] == "interrupted"
    assert interrupted["stages"]["tune"]["new_trials"] == 2
    assert interrupted["stages"]["cv"]["n_ok"] == 2
    run_dir = Path(interrupted["run_dir"])
    partial_hashes = {str(p): file_hash(p) for k in (0, 1)
                      for p in (run_dir / "cv" / f"fold-{k}").rglob("*") if p.is_file()}
    monkeypatch.setattr(cross_validate, "run", original_cv)
    result = pipeline.worker(path, 0)
    assert result["status"] == "ok", result.get("error")
    assert result["stages"]["tune"]["new_trials"] == 0
    assert result["stages"]["cv"]["n_ok"] == 5
    assert partial_hashes == {p: file_hash(p) for p in partial_hashes}
    run_dir = Path(result["run_dir"])
    before = {str(p): file_hash(p) for p in (run_dir / "cv").rglob("*") if p.is_file()}
    again = pipeline.worker(path, 0, stage="metrics")
    assert again["status"] == "ok", again.get("error")
    assert before == {p: file_hash(p) for p in before}
    assert pipeline.aggregate(path)["complete"]
    best = read_json(run_dir / "tuning/best.json")
    assert best["counts"]["allocated"] == 3
    for k in range(5):
        fold = read_json(run_dir / "cv" / f"fold-{k}/manifest.json")
        assert fold["n_requested"] == fold["n_train_rows"]
        assert fold["reload_verified"]


@pytest.mark.integration
def test_failed_fold_is_evaluated_and_only_retried_explicitly(tmp_path, monkeypatch):
    from sbtab.adapters.neural import LightSBAdapter
    from sbtab.experiments.experiment_common import file_hash
    path = plan_for(tmp_path)
    pipeline.prepare(path)
    assert pipeline.worker(path, 0, stage="tune")["status"] == "ok"
    original_fit = LightSBAdapter._fit_model
    calls = []
    def fail_once(self):
        calls.append(self.seed)
        if len(calls) == 3:
            raise RuntimeError("simulated failed fold")
        return original_fit(self)
    monkeypatch.setattr(LightSBAdapter, "_fit_model", fail_once)
    cv = pipeline.worker(path, 0, stage="cv")
    assert cv["stages"]["cv"]["n_ok"] == 4
    metrics = pipeline.worker(path, 0, stage="metrics")
    assert metrics["status"] == "evaluation_failed"
    assert metrics["fold_problems"] == {"2": ["training_failed"]}
    assert not pipeline.aggregate(path)["complete"]
    run_dir = Path(cv["run_dir"])
    before = {str(p): file_hash(p) for k in (0, 1, 3, 4)
              for p in (run_dir / "cv" / f"fold-{k}").rglob("*") if p.is_file()}
    assert pipeline.worker(path, 0, stage="cv")["stages"]["cv"]["n_ok"] == 4
    assert len(calls) == 5
    assert pipeline.worker(path, 0, stage="cv", retry_failed_folds=True)["stages"]["cv"]["n_ok"] == 5
    assert len(calls) == 6
    assert before == {p: file_hash(p) for p in before}
    assert pipeline.worker(path, 0, stage="metrics")["status"] == "ok"
    assert pipeline.aggregate(path)["complete"]


def _params_race(dataset_dir, key, barrier, queue):
    from sbtab import evaluation
    from sbtab.experiments.calculate_metrics import utility_params
    def resolve(*args, **kwargs):
        time.sleep(0.1)
        return {"test_key": key}
    evaluation.resolve_utility_params = resolve
    barrier.wait(timeout=30)
    queue.put(utility_params(Path(dataset_dir), key, "regression", None, None, [], None))


def test_parallel_utility_configs_do_not_lose_other_cache_keys(tmp_path):
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_params_race, args=(str(tmp_path), key, barrier, queue)) for key in ("a", "b")]
    for process in processes:
        process.start()
    assert {queue.get(timeout=45)["test_key"] for _ in processes} == {"a", "b"}
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    assert set(read_json(tmp_path / "utility_config.json")) == {"a", "b"}


def test_batch_scripts_quote_paths_and_submit_correct_dependencies(tmp_path):
    # Simulated scheduler verifies the submission interface without requiring Slurm.
    scripts = Path(__file__).resolve().parents[2] / "scripts/slurm"
    for path in list(scripts.glob("*.sh")) + list(scripts.glob("*.sbatch")):
        subprocess.run(["bash", "-n", str(path)], check=True)
    fakebin = tmp_path / "fake bin"
    fakebin.mkdir()
    scheduler = fakebin / "sbatch"
    scheduler.write_text("#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\np=Path(os.environ['SBATCH_CALLS'])\nrows=p.read_text().splitlines() if p.exists() else []\nwith p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nprint(str(100+len(rows))+';cluster')\n")
    scheduler.chmod(0o755)
    config = tmp_path / "cluster settings.sh"
    import shlex
    config.write_text(f"SBTAB_PYTHON={shlex.quote(os.sys.executable)}\nSBATCH_SITE_ARGS=(--partition=rocky --account=proj_1752)\n")
    env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
           "SBATCH_CALLS": str(tmp_path / "calls.jsonl")}
    args = ["bash", str(scripts / "submit.sh"), str(config), "--output-root", str(tmp_path / "output with spaces"),
            "--datasets", "diabetes", "--models", "lightsb", "--smoke"]
    submitted = subprocess.run(args, env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in Path(env["SBATCH_CALLS"]).read_text().splitlines()]
    assert len(calls) == 3
    assert f"Cluster configuration: {config}" in submitted.stderr
    logged_calls = [shlex.split(line)[1:] for line in submitted.stderr.splitlines()
                    if line.startswith("sbatch ")]
    assert logged_calls == calls
    assert "--dependency=afterok:100" in calls[1]
    assert "--kill-on-invalid-dep=yes" in calls[1]
    assert "--dependency=afterany:101" in calls[2]
    assert "--array=0-0%8" in calls[1]
    assert "--gpus=1" in calls[1] and "--cpus-per-task=8" in calls[1]
    assert any(arg.endswith("experiment-%A_%a.err") for arg in calls[1])
    plan = read_json(calls[0][-1])
    assert plan["device"] == "cuda"
    from sbtab.experiments.experiment_common import load_yaml
    assert load_yaml(plan["search_spaces"]["lightsb"]["path"])["fixed"]["device"] == "cuda"
    assert all("--account=proj_1752" in call and "--partition=rocky" in call for call in calls)
    assert calls[0][-2] == str(config)
    assert calls[0][-1].endswith("output with spaces/pipeline/plan.json")
    preview = subprocess.run([args[0], args[1], "--dry-run", *args[2:]], env=env, check=True,
                             capture_output=True, text=True)
    assert f"Cluster configuration: {config}" in preview.stderr
    preview_calls = [shlex.split(line)[1:] for line in preview.stderr.splitlines()
                     if line.startswith("sbatch ")]
    assert len(preview_calls) == 3
    assert all("--account=proj_1752" in call for call in preview_calls)
    assert len(Path(env["SBATCH_CALLS"]).read_text().splitlines()) == 3
