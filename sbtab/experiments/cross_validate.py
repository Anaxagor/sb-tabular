"""
Stage 2 — fixed-hyperparameter cross-validation on the 85 % pool T. No Optuna.

    python -m sbtab.experiments.cross_validate --dataset adult --model mixedsbm \
        --selected-config <selected-config.json> --splits <splits.json> --output-root <artifact-root>

For every fold k a FRESH preprocessing and a FRESH model are fitted on T_k only,
using the selected HYPERPARAMETERS. Tuned weights, optimiser moments, learned
graphs, transformation state and generated caches are never loaded. Exactly
len(T_k) rows are generated, target included, and the complete table is saved
together with the fold checkpoint, timing and provenance — before and
independently of any metric. A failed fold keeps its record.

This is fixed-hyperparameter CV after dataset-level tuning, NOT nested CV: the
tuning candidates were trained on all of T, which contains every E_k.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
from pathlib import Path
from typing import List, Optional

from sbtab.adapters.base import ADAPTER_FORMAT
from sbtab.experiments.experiment_common import (
    SeedLedger, StageError, canonical_hash, file_hash, file_lock, hardware_info, implementation_hash, library_versions, load_metric_config, load_protocol,
    read_json, source_provenance, write_json,
)
from sbtab.experiments.prepare_splits import load_split_artifacts, require_ok
from sbtab.experiments.runner import fit_generate
from sbtab.experiments.model_selection import require_experiment_model
from sbtab.solvers.registry import get_adapter_class, missing_requirements

CV_VERSION = "sbtab.cv/1"


def resolve_run_dir(selected_path: Path, dataset_dir: Path, model_id: str, selected: dict) -> Path:
    p = selected_path.resolve()
    if p.parent.name == "tuning" and p.parent.parent.parent == (dataset_dir / model_id).resolve():
        return p.parent.parent                      # CV sits next to the tuning it came from
    return dataset_dir / model_id / f"run-{canonical_hash(selected)[:10]}"


def load_tuned_selection(path, model_id: str, budget: int) -> dict:
    """Require the completed tuning artifacts, without loading any fitted model."""
    path = Path(path)
    selected = read_json(path)
    if selected.get("model") != model_id:
        raise StageError("undefined", f"selected config is for model {selected.get('model')!r}, not {model_id!r}")
    if selected.get("source") != "tuning" or not isinstance(selected.get("compatibility"), dict):
        raise StageError("undefined", "cross-validation requires a final tuning selection with provenance; complete tuning first")
    best_path = path.parent / "best.json"
    if not best_path.is_file():
        raise StageError("undefined", "selected config must accompany its final tuning best.json")
    best = read_json(best_path)
    counts = best.get("counts", {})
    if (best.get("selection") != "final" or best.get("reload_verified") is not True
            or best.get("budget") != budget or counts.get("allocated") != budget
            or counts.get("RUNNING") != 0 or counts.get("WAITING") != 0
            or sum(counts.get(state, 0) for state in ("COMPLETE", "FAIL", "PRUNED")) != budget):
        raise StageError("undefined", "selected config requires completed tuning with the exact allocated budget and verified reload")
    for key in ("version", "model", "trial", "objective", "compatibility"):
        if key not in selected or selected[key] != best.get(key):
            raise StageError("undefined", f"selected config differs from the final tuning selection in {key}")
    trial = selected["trial"]
    config_path = path.parent / f"trial-{trial:03d}" / "config.json"
    if not config_path.is_file() or selected.get("config") != read_json(config_path).get("effective_config"):
        raise StageError("undefined", "selected hyperparameters differ from the winning trial's effective configuration")

    # best.json is a convenient pointer, not an independent source of tuning evidence.
    # Read the actual study so edited/stale pointers cannot bypass the minimum rule.
    import optuna
    from sbtab.experiments.tune import counts as trial_counts, select_best
    database = path.parent / "study.sqlite3"
    if not database.is_file():
        raise StageError("undefined", "selected config requires its completed Optuna study.sqlite3")
    study = optuna.load_study(study_name="study", storage=f"sqlite:///{database}")
    if (study.direction.name != "MINIMIZE" or trial_counts(study) != counts
            or study.user_attrs.get("compatibility") != selected["compatibility"]
            or study.user_attrs.get("compatibility_hash") != canonical_hash(selected["compatibility"])):
        raise StageError("undefined", "selected config provenance or trial budget differs from its Optuna study")
    winner = select_best(study)
    if winner is None or winner.number != trial or winner.value != selected["objective"]:
        raise StageError("undefined", "selected config is not the minimum finite COMPLETE trial in its Optuna study")
    if winner.user_attrs.get("effective_config") != selected["config"]:
        raise StageError("undefined", "selected hyperparameters differ from the effective configuration recorded in Optuna")
    return selected


def run(dataset: str, model_id: str, selected_config_path, splits_path, output_root=None, smoke: bool = False,
        protocol_path=None, folds: Optional[List[int]] = None, dry_run: bool = False, force: bool = False) -> dict:
    entry = require_experiment_model(model_id)
    protocol = load_protocol(protocol_path, smoke=smoke)
    if entry.status == "unavailable":
        raise StageError("undefined", f"model {model_id!r} is unavailable: {entry.notes}")
    missing = missing_requirements(model_id)
    if missing:
        raise StageError("undefined", f"model {model_id!r} needs the optional packages {list(missing)}, which are not installed")

    frame, schema, splits = load_split_artifacts(splits_path)
    if splits["dataset"] != dataset:
        raise StageError("undefined", f"splits.json is for dataset {splits['dataset']!r}, not {dataset!r}")
    if splits["protocol_hash"] != protocol.hash():
        raise StageError("undefined", "splits.json was produced under a different protocol (hash mismatch)")
    require_ok(splits)
    if schema.regime not in entry.regimes:
        raise StageError("not_applicable", f"model {model_id!r} does not support the {schema.regime!r} regime")
    dataset_dir = Path(splits_path).resolve().parent
    if output_root is not None and Path(output_root).resolve() != dataset_dir.parent:
        raise StageError("undefined", f"--output-root {output_root} is not the root that holds {splits_path}")

    selected = load_tuned_selection(selected_config_path, model_id, int(protocol["tuning"]["n_trials"]))
    config = get_adapter_class(model_id).resolve_config(selected["config"])    # strict: no ignored hyperparameter
    comp = selected["compatibility"]
    prov = source_provenance()
    versions = library_versions()
    metric_config = load_metric_config(protocol)
    for key, have in (("membership_hash", splits["membership_hash"]), ("schema_hash", splits["schema_hash"]),
                      ("protocol_hash", protocol.hash()), ("dataset_fingerprint", splits["dataset_fingerprint"]),
                      ("implementation_hash", implementation_hash(prov)), ("checkpoint_format", ADAPTER_FORMAT),
                      ("adapter", entry.adapter), ("libraries", versions), ("model", model_id),
                      ("metric_config_hash", canonical_hash(metric_config)), ("metric_version", metric_config["metric_version"])):
        if comp.get(key) != have:
            raise StageError("undefined", f"selected config was tuned under a different {key}")

    n_splits = splits["cv"]["n_splits"]
    fold_ids = list(range(n_splits)) if folds is None else sorted(set(folds))
    if any(k < 0 or k >= n_splits for k in fold_ids):
        raise StageError("undefined", f"fold indices must lie in [0, {n_splits})")
    run_dir = resolve_run_dir(Path(selected_config_path), dataset_dir, model_id, selected)
    cv_dir = run_dir / "cv"
    compatibility = {"version": CV_VERSION, "model": model_id, "adapter": entry.adapter,
                     "checkpoint_format": ADAPTER_FORMAT, "protocol_hash": protocol.hash(),
                     "dataset_fingerprint": splits["dataset_fingerprint"], "schema_hash": splits["schema_hash"],
                     "membership_hash": splits["membership_hash"], "config": config,
                     "libraries": versions,
                     "implementation_hash": implementation_hash(prov)}
    compatibility_hash = canonical_hash(compatibility)
    # Standalone --folds calls share both fold artifacts and the run manifest.
    # Hold one lock through validation, generation and manifest merging.
    with nullcontext() if dry_run else file_lock(cv_dir / ".cv.lock", blocking=False):
        existing = cv_dir / "cv_run_manifest.json"
        if existing.exists() and read_json(existing).get("compatibility_hash") != compatibility_hash:
            raise StageError("undefined", "existing CV run has incompatible data, configuration or implementation; "
                             "use a new run directory to avoid mixing fold provenance")
        plan = {"dataset": dataset, "model": model_id, "protocol_id": protocol.id, "kind": protocol.kind,
                "run_dir": str(run_dir), "folds": fold_ids, "n_splits": n_splits, "cv": splits["cv"],
                "n_generated": "len(T_k)", "config_source": selected["source"], "config": config}
        if dry_run:
            return {**plan, "dry_run": True}

        V = set(splits["V_row_ids"])
        seeds = SeedLedger(int(protocol["seeds"]["base"]))
        fold_records = []
        for k in fold_ids:
            f = splits["folds"][k]
            assert not (set(f["train_row_ids"]) | set(f["test_row_ids"])) & V, "a row of V entered a CV fold"
            d = cv_dir / f"fold-{k}"
            manifest_path = d / "manifest.json"
            if manifest_path.exists() and not force:
                old = read_json(manifest_path)
                if old.get("status") == "ok" and old.get("train_row_hash") == f["train_hash"] \
                        and old.get("effective_config") == config and old.get("compatibility_hash") == compatibility_hash:
                    hashes = old.get("artifact_hashes", {})
                    if not hashes or any(not (d / p).is_file() or file_hash(d / p) != h for p, h in hashes.items()):
                        raise StageError("undefined", f"{d}: missing or changed generator artifacts; pass --force to regenerate")
                    fold_records.append({**old, "skipped_existing": True})
                    continue
                raise StageError("undefined", f"{d} already holds a different or failed fold run; generator artifacts are "
                                              "never overwritten silently — pass --force or use a new run id")
            T_k = frame.loc[f["train_row_ids"]]
            rec = fit_generate(model_id, T_k, schema, config, seeds, ("fold", k), len(T_k), d)
            manifest = {
                "version": CV_VERSION, "stage": "cv", "dataset": dataset, "model": model_id, "fold": k,
                "compatibility_hash": compatibility_hash,
                "protocol_id": protocol.id, "protocol_hash": protocol.hash(), "status": rec["status"], "failure": rec["failure"],
                "train_row_hash": f["train_hash"], "test_row_hash": f["test_hash"],
                "n_train_rows": len(f["train_row_ids"]), "n_test_rows": len(f["test_row_ids"]),
                "n_requested": rec["n_requested"], "n_generated": rec["n_generated"],
                "requested_config": rec["requested_config"], "effective_config": rec["effective_config"],
                "seeds": rec["seeds"], "n_updates": rec["n_updates"], "describe": rec["describe"],
                "decoding_report": rec["decoding_report"], "preprocessor": rec.get("preprocessor"),
                "checkpoint": rec["checkpoint"], "reload_verified": rec["reload_verified"],
                "synthetic": rec["synthetic"], "synthetic_format": rec["synthetic_format"], "timing": rec["timing"],
                "fresh_state": "new preprocessing, model, optimiser, caches and graph; hyperparameters only from tuning",
                "provenance": prov,
            }
            manifest["artifact_hashes"] = {
                str(p.relative_to(d)): file_hash(p)
                for p in sorted(d.rglob("*")) if p.is_file() and
                (p.name in ("synthetic.parquet", "timing.json", "preprocessor.json") or "checkpoint" in p.relative_to(d).parts)
            }
            if rec["status"] == "ok" and rec["train_row_hash"] != f["train_hash"]:
                raise RuntimeError("fold was fitted on rows other than T_k")
            write_json(manifest_path, manifest)
            fold_records.append(manifest)

        run_manifest = {
            "version": CV_VERSION, "stage": "cv", **plan, "selected_config": str(selected_config_path),
            "compatibility": compatibility, "compatibility_hash": compatibility_hash,
            "selected_config_hash": canonical_hash(selected), "splits": str(Path(splits_path).resolve()),
            "membership_hash": splits["membership_hash"], "schema_hash": splits["schema_hash"],
            "dataset_fingerprint": splits["dataset_fingerprint"], "T_hash": splits["T_hash"], "V_hash": splits["V_hash"],
            "metric_config": load_metric_config(protocol), "provenance": prov, "libraries": versions,
            "hardware": hardware_info(), "seed_ledger": seeds.to_dict(),
            "target_handling": {"target": schema.target, "task": schema.task, "generated_as": "part of the row (X, y)"},
            "folds_status": {str(r["fold"]): r["status"] for r in fold_records},
            "n_expected": n_splits, "n_ok": sum(r["status"] == "ok" for r in fold_records),
            "statistical_note": splits.get("statistical_note", ""),
        }
        if existing.exists():       # folds may be run independently: merge their status
            prev = read_json(existing)
            merged = {**prev.get("folds_status", {}), **run_manifest["folds_status"]}
            run_manifest["folds_status"] = merged
            run_manifest["n_ok"] = sum(v == "ok" for v in merged.values())
            run_manifest["seed_ledger"]["derived"] = {
                **prev.get("seed_ledger", {}).get("derived", {}), **seeds.to_dict()["derived"]}
        run_manifest["folds"] = sorted(int(k) for k in run_manifest["folds_status"])
        write_json(existing, run_manifest)
        return {"run_dir": str(run_dir), "cv_run_manifest": str(existing), "folds_status": run_manifest["folds_status"],
                "n_ok": run_manifest["n_ok"], "n_expected": n_splits}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Five fresh fold fits, fold checkpoints, complete synthetic datasets, timings; no Optuna.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--selected-config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="run only these folds (each fold is independent)")
    ap.add_argument("--force", action="store_true", help="re-run a fold that already has artifacts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    try:
        out = run(args.dataset, args.model, args.selected_config, args.splits, args.output_root, args.smoke,
                  args.protocol, args.folds, args.dry_run, args.force)
    except StageError as e:
        print({"status": e.status, "error": str(e)})
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
