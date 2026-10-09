"""Plan and execute the complete experiment, locally or as SLURM array tasks.

One task owns one dataset/model pair: 100 sequential Optuna trials, five fresh
CV fits, then all held-out metrics. Splits are prepared once, before workers.
This module orchestrates the canonical stages; it does not reimplement them.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import yaml

from sbtab.data.registry import available_datasets, load_dataset_config, schema_from_config
from sbtab.experiments.experiment_common import (
    REPO_ROOT, SOURCE_HASH_VERSION, StageError, atomic_write_text, canonical_hash, claim_output_root, file_hash, file_lock,
    implementation_hash, library_versions, load_metric_config, load_protocol, read_json, write_json,
)
from sbtab.solvers.registry import get_entry, missing_requirements, solver_registry
from sbtab.experiments.model_selection import BASIC_DSB_MODELS, exclusion_reason, require_experiment_model

PLAN_VERSION = "sbtab.pipeline/1"
RUN_ID = "run-pipeline"


def create_plan(output_root, datasets=None, models=None, protocol_path=None, smoke=False,
                search_space_dir=None, dataset_config_dir="configs/datasets", include_heuristic=True, device=None):
    from sbtab.evaluation import MetricConfig
    from sbtab.experiments.tune import load_search_space

    root = Path(output_root).resolve()
    if device not in (None, "cpu", "cuda"):
        raise StageError("undefined", "device must be cpu or cuda")
    protocol = load_protocol(protocol_path, smoke=smoke)
    metrics = load_metric_config(protocol)
    MetricConfig.from_document(metrics)
    metric_path = (REPO_ROOT / protocol["metrics_config"]).resolve()
    config_dir = Path(dataset_config_dir).resolve()
    names = sorted(set(datasets if datasets is not None else available_datasets(config_dir)))
    if not names:
        raise StageError("undefined", "no datasets selected")
    schemas = {name: schema_from_config(load_dataset_config(name, config_dir)) for name in names}
    chosen = sorted(set(models if models is not None else solver_registry))
    spaces_dir = Path(search_space_dir or ("configs/search_spaces/smoke" if smoke else "configs/search_spaces")).resolve()
    entries, spaces, excluded, generated_spaces = {}, {}, [], {}
    for model in chosen:
        entry = get_entry(model)
        reason = exclusion_reason(model)
        if reason is None and entry.status == "unavailable":
            reason = f"unavailable: {entry.notes}"
        elif reason is None and entry.status == "heuristic" and not include_heuristic and models is None:
            reason = "heuristic excluded by --exclude-heuristic"
        elif reason is None and missing_requirements(model):
            reason = f"missing dependencies: {list(missing_requirements(model))}"
        if reason:
            if models is not None:
                raise StageError("undefined", f"requested model {model}: {reason}")
            excluded.append({"model": model, "reason": reason})
            continue
        path = spaces_dir / f"{model}.yaml"
        space = load_search_space(path, model, protocol.kind)
        spaces[model] = {"path": str(path), "hash": file_hash(path)}
        if device is not None:
            from sbtab.solvers.registry import get_adapter_class
            keys = get_adapter_class(model).DEFAULTS
            key = "device" if "device" in keys else "enable_gpu" if "enable_gpu" in keys else None
            if key is None or key in space["params"]:
                raise StageError("undefined", f"{model}: cannot fix execution device in this search space")
            space["fixed"][key] = device if key == "device" else device == "cuda"
            # Use YAML's float spelling (1.0e-05), since YAML 1.1 can read JSON's
            # 1e-05 as a string. Keep every hyperparameter range unchanged.
            contents = yaml.safe_dump(space, sort_keys=False)
            effective = root / "pipeline" / "search_spaces" / f"{model}.yaml"
            generated_spaces[effective] = contents
            spaces[model] = {"path": str(effective), "hash": hashlib.sha256(contents.encode()).hexdigest(),
                             "source_path": str(path), "source_hash": file_hash(path)}
        entries[model] = entry
    tasks = []
    for dataset in names:
        for model, entry in entries.items():
            if schemas[dataset].regime not in entry.regimes:
                excluded.append({"dataset": dataset, "model": model, "reason": "incompatible data regime"})
                continue
            tasks.append({"task_id": len(tasks), "dataset": dataset, "model": model,
                          "regime": schemas[dataset].regime, "model_status": entry.status})
    if not tasks:
        raise StageError("not_applicable", "no compatible dataset/model tasks remain")
    pretrained = {}
    if "tabpfgen" in entries:
        from sbtab.experiments.cluster_environment import check_tabpfn_cache
        pretrained["tabpfgen"] = check_tabpfn_cache()
    plan = {
        "version": PLAN_VERSION, "repo_root": str(REPO_ROOT), "output_root": str(root), "run_id": RUN_ID,
        "protocol_path": str(Path(protocol.path).resolve()), "protocol_hash": protocol.hash(),
        "protocol_id": protocol.id, "smoke": smoke, "n_trials": protocol["tuning"]["n_trials"],
        "n_folds": protocol["cv"]["n_splits"], "metrics_path": str(metric_path),
        "metrics_hash": file_hash(metric_path), "dataset_config_dir": str(config_dir),
        "datasets": {name: {"config_hash": file_hash(config_dir / f"{name}.yaml"), "regime": schema.regime}
                     for name, schema in schemas.items()},
        "search_spaces": spaces, "tasks": tasks, "excluded": excluded,
        "basic_dsb_models": dict(BASIC_DSB_MODELS),
        "device": device,
        "pretrained_models": pretrained,
        "implementation_hash_version": SOURCE_HASH_VERSION,
        "implementation_hash": implementation_hash(), "libraries": library_versions(),
        "statistical_note": protocol.data.get("statistical_note", ""),
    }
    plan["plan_hash"] = canonical_hash(plan)
    path = root / "pipeline" / "plan.json"
    with file_lock(root / "pipeline" / ".plan.lock", blocking=False):
        claim_output_root(root, protocol)
        if path.exists():
            if read_json(path) != plan:
                raise StageError("undefined", "existing pipeline plan differs; use a new output root")
            for item in spaces.values():
                if not Path(item["path"]).is_file() or file_hash(item["path"]) != item["hash"]:
                    raise StageError("undefined", "frozen search space changed or is missing")
        else:
            # Mixing unrelated runs in this root would make rank selection ambiguous.
            if any(root.glob("*/*/*/run_manifest.json")) or any(root.glob("*/*/*/cv/cv_run_manifest.json")):
                raise StageError("undefined", "choose a fresh output root for the pipeline plan")
            for filename, contents in generated_spaces.items():
                atomic_write_text(filename, contents)
            write_json(path, plan)
    return {"plan": str(path), "n_tasks": len(tasks), "n_datasets": len(names),
            "n_models": len(entries), "n_trials": plan["n_trials"], "n_folds": plan["n_folds"],
            "n_excluded": len(excluded), "smoke": smoke}


def load_plan(path, verify=True):
    plan = read_json(path)
    if plan.get("version") != PLAN_VERSION:
        raise StageError("undefined", "unsupported pipeline plan version")
    if canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"}) != plan["plan_hash"]:
        raise StageError("undefined", "pipeline plan hash mismatch")
    if not verify:
        return plan
    if plan.get("basic_dsb_models") != BASIC_DSB_MODELS:
        raise StageError("undefined", "experiment model selection changed; create a new plan in a new output root")
    for task in plan["tasks"]:
        require_experiment_model(task["model"])
    if str(REPO_ROOT) != plan["repo_root"]:
        raise StageError("undefined", "plan belongs to another checkout; create it on the cluster at its final path")
    protocol = load_protocol(plan["protocol_path"], smoke=plan["smoke"])
    current_protocol_hash = protocol.hash()
    if current_protocol_hash != plan["protocol_hash"]:
        raise StageError("undefined", f"protocol changed after planning: expected {plan['protocol_hash']}, "
                         f"current {current_protocol_hash}; use a new output root")
    if plan.get("implementation_hash_version") != SOURCE_HASH_VERSION:
        raise StageError("undefined", "plan uses legacy Git-based implementation verification; "
                         "create a new plan in a new output root with the updated code")
    current_implementation_hash = implementation_hash()
    if current_implementation_hash != plan["implementation_hash"]:
        raise StageError("undefined", f"implementation changed after planning: expected {plan['implementation_hash']}, "
                         f"current {current_implementation_hash}; source/config files changed, use a new output root")
    paths = {plan["metrics_path"]: plan["metrics_hash"]}
    paths.update({s["path"]: s["hash"] for s in plan["search_spaces"].values()})
    paths.update({s["source_path"]: s["source_hash"] for s in plan["search_spaces"].values() if "source_path" in s})
    paths.update({str(Path(plan["dataset_config_dir"]) / f"{name}.yaml"): info["config_hash"]
                  for name, info in plan["datasets"].items()})
    for filename, expected in paths.items():
        if not Path(filename).is_file() or file_hash(filename) != expected:
            raise StageError("undefined", f"configuration changed after planning: {filename}")
    if library_versions() != plan["libraries"]:
        raise StageError("undefined", "Python/dependency versions differ from the pipeline plan")
    if "tabpfgen" in plan.get("pretrained_models", {}):
        from sbtab.experiments.cluster_environment import check_tabpfn_cache
        if check_tabpfn_cache() != plan["pretrained_models"]["tabpfgen"]:
            raise StageError("undefined", "TabPFN pretrained weights or cache location changed after planning")
    return plan


def prepare(plan_path):
    from sbtab.experiments.prepare_splits import run
    from sbtab.experiments import tune
    plan = load_plan(plan_path)
    root = Path(plan["output_root"])
    protocol = load_protocol(plan["protocol_path"], smoke=plan["smoke"])
    result = {"plan_hash": plan["plan_hash"], "datasets": {}}
    with file_lock(root / "pipeline" / ".prepare.lock", blocking=False):
        for name in plan["datasets"]:
            try:
                value = run(name, protocol, root, config_dir=plan["dataset_config_dir"])
                if value["split_status"] == "ok" and "tabpfgen" in plan["search_spaces"]:
                    try:
                        preview = tune.run(name, "tabpfgen", root / name / "splits.json", plan["search_spaces"]["tabpfgen"]["path"],
                                           resume=True, smoke=plan["smoke"], protocol_path=plan["protocol_path"], dry_run=True)
                        support = {"status": "ok", "context": preview["tabpfgen_context"]}
                    except Exception as error:
                        support = {"status": getattr(error, "status", "undefined"), "error": str(error)}
                    value["models"] = {"tabpfgen": support}
            except Exception as e:
                value = {"dataset": name, "split_status": getattr(e, "status", "undefined"),
                         "error": str(e), "trace": traceback.format_exc()}
            result["datasets"][name] = value
            write_json(root / "pipeline" / "preparation.json", result)
    return {"datasets": result["datasets"], "counts": dict(Counter(v["split_status"] for v in result["datasets"].values()))}


def _task(plan, task_id):
    if task_id < 0 or task_id >= len(plan["tasks"]):
        raise StageError("undefined", f"task index must be between 0 and {len(plan['tasks']) - 1}")
    return plan["tasks"][task_id]


def worker(plan_path, task_id, stage="all", retry_failed_folds=False):
    # Validate inside the status handler so even a provenance failure is visible to aggregation.
    plan = load_plan(plan_path, verify=False)
    task = _task(plan, task_id)
    root = Path(plan["output_root"])
    run_dir = root / task["dataset"] / task["model"] / plan["run_id"]
    control = root / "pipeline"
    status_path = control / "tasks" / f"{task_id:05d}.json"
    record = {**task, "plan_hash": plan["plan_hash"], "stage": stage, "status": "running",
              "run_dir": str(run_dir), "started_at": time.time(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "stages": {}}
    with file_lock(control / "locks" / f"task-{task_id}.lock", blocking=False):
        if status_path.exists():
            previous = read_json(status_path)
            record["stages"] = previous.get("stages", {})
        write_json(status_path, record)
        try:
            load_plan(plan_path)
            preparation = read_json(control / "preparation.json")
            if preparation["plan_hash"] != plan["plan_hash"]:
                raise StageError("undefined", "preparation belongs to a different plan")
            dataset_status = preparation["datasets"][task["dataset"]]
            if dataset_status["split_status"] != "ok":
                raise StageError(dataset_status["split_status"], f"dataset preflight failed: {dataset_status}")
            support = dataset_status.get("models", {}).get(task["model"], {"status": "ok"})
            if support["status"] != "ok":
                raise StageError(support["status"], support.get("error", "model preflight failed"))
            if plan.get("device") == "cuda" and stage != "metrics":
                from sbtab.experiments.cluster_environment import check_cuda
                record["cuda"] = check_cuda()
            from sbtab.experiments import tune, cross_validate, calculate_metrics
            splits = root / task["dataset"] / "splits.json"
            selected = run_dir / "tuning" / "selected_config.json"

            def save_stage(name, output):
                record["stages"][name] = output
                write_json(status_path, record)

            if stage in ("all", "tune"):
                record["stages"].pop("cv", None)
                record["stages"].pop("metrics", None)
                record["stage"] = "tune"
                write_json(status_path, record)
                tuned = tune.run(task["dataset"], task["model"], splits,
                                 plan["search_spaces"][task["model"]]["path"], resume=True,
                                 smoke=plan["smoke"], protocol_path=plan["protocol_path"], run_id=plan["run_id"])
                save_stage("tune", tuned)
                if tuned["selection"] != "final" or tuned["counts"]["allocated"] != plan["n_trials"]:
                    raise StageError("training_failed", "tuning did not produce a final selected configuration")
            if stage in ("all", "cv"):
                record["stages"].pop("metrics", None)
                record["stage"] = "cv"
                write_json(status_path, record)
                if not selected.is_file():
                    raise StageError("training_failed", "no selected configuration; complete tuning first")
                # Check compatibility even if every previously attempted fold failed.
                cross_validate.run(task["dataset"], task["model"], selected, splits, root,
                                   smoke=plan["smoke"], protocol_path=plan["protocol_path"], dry_run=True)
                for k in range(plan["n_folds"]):
                    manifest = run_dir / "cv" / f"fold-{k}" / "manifest.json"
                    failed = manifest.exists() and read_json(manifest)["status"] != "ok"
                    if failed and not retry_failed_folds:
                        continue
                    result = cross_validate.run(task["dataset"], task["model"], selected, splits, root,
                                                smoke=plan["smoke"], protocol_path=plan["protocol_path"],
                                                folds=[k], force=failed and retry_failed_folds)
                    save_stage("cv", result)
            if stage in ("all", "metrics"):
                record["stage"] = "metrics"
                write_json(status_path, record)
                cv_manifest = run_dir / "cv" / "cv_run_manifest.json"
                if not cv_manifest.is_file():
                    raise StageError("training_failed", "no CV manifest; complete CV first")
                result = calculate_metrics.run(cv_manifest, plan["metrics_path"])
                save_stage("metrics", result)
                evaluation = Path(result["evaluation_dir"])
                problems = {}
                for k in range(plan["n_folds"]):
                    fold = read_json(evaluation / f"fold-{k}" / "metrics.json")
                    statuses = {fold["status"], fold.get("utility", {}).get("status", "ok")}
                    bad = statuses - {"ok", "not_applicable", "incomplete_conditional_coverage"}
                    if bad:
                        problems[str(k)] = sorted(bad)
                if problems:
                    record["fold_problems"] = problems
                    raise StageError("evaluation_failed", "metrics saved, but one or more folds failed; see fold_problems")
            record["status"] = "ok"
        except (Exception, KeyboardInterrupt) as e:
            record["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else getattr(e, "status", "failed")
            record["error"] = f"{type(e).__name__}: {e}"
            record["trace"] = traceback.format_exc()
        finally:
            record["finished_at"] = time.time()
            write_json(status_path, record)
    return record


def aggregate(plan_path):
    from sbtab.experiments.aggregate_results import aggregate_root
    # Collect finished evidence even if code/dependencies changed after a failed job.
    plan = load_plan(plan_path, verify=False)
    root = Path(plan["output_root"])
    with file_lock(root / "pipeline" / ".aggregate.lock", blocking=False):
        tasks = []
        for task in plan["tasks"]:
            path = root / "pipeline" / "tasks" / f"{task['task_id']:05d}.json"
            saved = read_json(path) if path.exists() else {**task, "status": "not_run"}
            if saved["status"] == "running":
                saved = {**saved, "status": "unfinished", "note": "worker did not record completion; may have been killed"}
            if saved.get("plan_hash", plan["plan_hash"]) != plan["plan_hash"]:
                raise StageError("undefined", "task status belongs to another plan")
            tasks.append(saved)
        result = {"plan_hash": plan["plan_hash"], "n_tasks": len(tasks), "tasks": tasks,
                  "counts": dict(Counter(t["status"] for t in tasks)), "excluded": plan["excluded"],
                  "complete": all(t["status"] == "ok" and "metrics" in t.get("stages", {}) for t in tasks)}
        try:
            result["metrics"] = aggregate_root(root)
        except Exception as e:
            result.update(complete=False, aggregation_error=f"{type(e).__name__}: {e}")
        write_json(root / "pipeline" / "summary.json", result)
    return {k: v for k, v in result.items() if k not in ("tasks", "excluded", "metrics")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="create an immutable task manifest; no model fitting")
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--datasets", nargs="+", default=None)
    plan.add_argument("--models", nargs="+", default=None)
    plan.add_argument("--protocol", dest="protocol_path", default=None)
    plan.add_argument("--smoke", action="store_true")
    plan.add_argument("--search-space-dir", default=None)
    plan.add_argument("--device", choices=("cpu", "cuda"), default=None,
                      help="freeze a device override for all selected generators")
    plan.add_argument("--dataset-config-dir", default="configs/datasets")
    plan.add_argument("--exclude-heuristic", dest="include_heuristic", action="store_false")
    for name in ("prepare", "worker", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        if name == "worker":
            command.add_argument("--task-id", type=int, default=None)
            command.add_argument("--stage", choices=("all", "tune", "cv", "metrics"), default="all")
            command.add_argument("--retry-failed-folds", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "plan":
            result = create_plan(**args)
        elif command == "prepare":
            result = prepare(args["plan"])
        elif command == "aggregate":
            result = aggregate(args["plan"])
        else:
            task_id = args["task_id"]
            if task_id is None:
                if "SLURM_ARRAY_TASK_ID" not in os.environ:
                    parser.error("worker requires --task-id or SLURM_ARRAY_TASK_ID")
                task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"received signal {signum}")
            signal.signal(signal.SIGTERM, interrupted)
            result = worker(args["plan"], task_id, args["stage"], args["retry_failed_folds"])
        print(json.dumps(result, indent=2))
        if command == "worker":
            return 0 if result["status"] == "ok" else 1
        if command == "aggregate":
            return 0 if result["complete"] else 1
        return 0
    except Exception as e:
        print(json.dumps({"status": getattr(e, "status", "failed"), "error": f"{type(e).__name__}: {e}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
