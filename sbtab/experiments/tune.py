"""
Stage 1 — dataset-level hyperparameter tuning.

    python -m sbtab.experiments.tune --dataset adult --model mixedsbm \
        --splits artifacts/sbtab_8515_hpo100_cv5_v2/adult/splits.json \
        --search-space configs/search_spaces/mixedsbm.yaml --resume

Every trial builds fresh preprocessing and a fresh model, fits T only, generates
exactly len(V) rows and is scored against V with the regime objective of
``sbtab.evaluation.tuning_objective``. Additional joint/conditional diagnostics
never influence selection.

Budget: ``n_trials`` counts ALLOCATED trial records — COMPLETE, FAIL and PRUNED
alike. A failed trial is kept and counted; it is never replaced to obtain more
successes. Resuming allocates only the remaining budget; stale RUNNING trials of a
dead process are reconciled to FAIL explicitly. The best trial is the minimum
finite objective among COMPLETE trials (ties: lowest trial number).
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import traceback
from pathlib import Path
from typing import Optional

import optuna
import pandas as pd
from optuna.trial import TrialState

from sbtab.adapters.base import ADAPTER_FORMAT
from sbtab.data.preprocessing import CommonPreprocessor
from sbtab.experiments.experiment_common import (
    SeedLedger, StageError, canonical_hash, hardware_info, implementation_hash, library_versions, load_metric_config,
    load_protocol, load_yaml, read_json, source_provenance, write_json,
)
from sbtab.experiments.prepare_splits import load_split_artifacts, require_ok
from sbtab.experiments.runner import fit_generate, read_synthetic
from sbtab.experiments.model_selection import require_experiment_model
from sbtab.solvers.registry import get_adapter_class, missing_requirements

TUNING_VERSION = "sbtab.tuning/1"


# --------------------------------------------------------------------------- search space
def load_search_space(path, model_id: str, protocol_kind: str) -> dict:
    space = load_yaml(path)
    if space.get("model") != model_id:
        raise StageError("undefined", f"search space {path} is for model {space.get('model')!r}, not {model_id!r}")
    kind = space.get("kind", "production")
    if kind != protocol_kind:
        raise StageError("undefined", f"search space kind {kind!r} cannot be used with a {protocol_kind!r} protocol; "
                                      "reduced budgets belong to the separate smoke protocol")
    adapter = get_adapter_class(model_id)
    fixed, params = dict(space.get("fixed") or {}), dict(space.get("params") or {})
    overlap = sorted(set(fixed) & set(params))
    if overlap:
        raise StageError("undefined", f"search space keys {overlap} are both fixed and searched")
    adapter.resolve_config({**fixed, **{k: None for k in params}})     # every key must be consumed by the adapter
    for name, p in params.items():
        t = p.get("type")
        if t == "categorical":
            if not p.get("choices"):
                raise StageError("undefined", f"search space param {name!r}: empty choices")
        elif t in ("float", "int"):
            if not p["low"] <= p["high"]:
                raise StageError("undefined", f"search space param {name!r}: low > high")
            if p.get("log") and p["low"] <= 0:
                raise StageError("undefined", f"search space param {name!r}: log scale needs low > 0")
        else:
            raise StageError("undefined", f"search space param {name!r}: unknown type {t!r}")
    if "noise" in params or "noise" in fixed:
        raise StageError("undefined", "`noise` is not part of a canonical SB search space: dropping the dynamics "
                                      "noise of a stochastic-drift model is not a marginal-preserving sampler")
    return {"model": model_id, "kind": kind, "version": space.get("version", 1), "fixed": fixed, "params": params}


def suggest(trial: optuna.Trial, space: dict) -> dict:
    cfg = dict(space["fixed"])
    for name, p in space["params"].items():
        if p["type"] == "categorical":
            cfg[name] = trial.suggest_categorical(name, list(p["choices"]))
        elif p["type"] == "float":
            cfg[name] = trial.suggest_float(name, float(p["low"]), float(p["high"]), log=bool(p.get("log", False)))
        else:
            cfg[name] = trial.suggest_int(name, int(p["low"]), int(p["high"]), log=bool(p.get("log", False)),
                                          step=int(p.get("step", 1)))
    return cfg


# --------------------------------------------------------------------------- study helpers
def trial_dir(tuning_dir: Path, number: int) -> Path:
    return tuning_dir / f"trial-{number:03d}"


def counts(study: optuna.Study) -> dict:
    states = [t.state for t in study.get_trials(deepcopy=False)]
    return {"allocated": len(states), "COMPLETE": states.count(TrialState.COMPLETE), "FAIL": states.count(TrialState.FAIL),
            "PRUNED": states.count(TrialState.PRUNED), "RUNNING": states.count(TrialState.RUNNING),
            "WAITING": states.count(TrialState.WAITING)}


def save_sampler(study: optuna.Study, path: Path) -> None:
    """Save after suggestions as well as completion, so an interrupted fit cannot rewind the RNG."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(study.sampler, fh)
    tmp.replace(path)


def reconcile_stale_running(study: optuna.Study, tuning_dir: Path) -> list:
    """A RUNNING trial found at start-up belongs to a dead process (n_jobs = 1): mark it FAIL, keep its record."""
    stale = []
    for t in study.get_trials(deepcopy=False):
        if t.state == TrialState.RUNNING:
            study._storage.set_trial_user_attr(t._trial_id, "failure", "stale_running_reconciled_on_resume")
            study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
            d = trial_dir(tuning_dir, t.number)
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / "status.json", {"trial": t.number, "state": "FAIL", "status": "training_failed",
                                           "failure": {"type": "Interrupted", "message": "stale RUNNING trial reconciled on resume"}})
            stale.append(t.number)
    return stale


def select_best(study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
    done = [t for t in study.get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)]
    return min(done, key=lambda t: (t.value, t.number)) if done else None


def write_trials_csv(study: optuna.Study, tuning_dir: Path) -> None:
    rows = []
    for t in study.get_trials(deepcopy=False):
        rows.append({"trial": t.number, "state": t.state.name, "objective": t.value,
                     "status": t.user_attrs.get("status"), "mean_wd": t.user_attrs.get("mean_wd"),
                     "mean_js": t.user_attrs.get("mean_js"), "n_updates": t.user_attrs.get("n_updates"),
                     "generator_fit_seconds": t.user_attrs.get("generator_fit_seconds"),
                     "generation_seconds": t.user_attrs.get("generation_seconds"),
                     "failure": t.user_attrs.get("failure"), **{f"param_{k}": v for k, v in t.params.items()}})
    tmp = tuning_dir / "trials.csv.tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(tuning_dir / "trials.csv")


def write_best(study: optuna.Study, tuning_dir: Path, budget: int, final: bool, model_id: str, compat: dict) -> dict:
    best, c = select_best(study), counts(study)
    out = {"version": TUNING_VERSION, "model": model_id, "budget": budget, "counts": c, "compatibility": compat,
           "selection_rule": "minimum finite objective among COMPLETE trials; ties -> lowest trial number"}
    if best is None:
        out.update(selection="failed" if final else "provisional", trial=None,
                   reason="no COMPLETE trial with a finite objective")
    else:
        d = trial_dir(tuning_dir, best.number)
        out.update(selection="provisional", trial=best.number, objective=best.value,
                   checkpoint=str(Path(d.name) / "checkpoint"), config=str(Path(d.name) / "config.json"),
                   validation_table=str(Path(d.name) / "synthetic.parquet"), reload_verified=None)
        if final:
            # final only after the budget is spent AND the exact selected checkpoint reloads
            adapter_cls = get_adapter_class(model_id)
            try:
                reloaded = adapter_cls.load_checkpoint(d / "checkpoint")
                reloaded.sample(2, seed=0)
                out.update(selection="final", reload_verified=True)
            except Exception as e:
                out.update(selection="failed", reload_verified=False, reason=f"selected checkpoint does not reload: {e}")
    write_json(tuning_dir / "best.json", out)
    if out["selection"] == "final":
        cfg = read_json(trial_dir(tuning_dir, best.number) / "config.json")
        write_json(tuning_dir / "selected_config.json", {
            "version": TUNING_VERSION, "model": model_id, "source": "tuning", "trial": best.number,
            "objective": best.value, "config": cfg["effective_config"], "compatibility": compat,
            "note": "hyperparameters only — cross-validation never loads tuned weights, optimiser state, "
                    "graphs, transforms or caches"})
    elif (tuning_dir / "selected_config.json").exists():
        # A failed reload must revoke a previously final selection.
        (tuning_dir / "selected_config.json").unlink()
    return out


# --------------------------------------------------------------------------- stage
def run(dataset: str, model_id: str, splits_path, search_space_path, resume: bool, smoke: bool,
        protocol_path=None, run_id: Optional[str] = None, dry_run: bool = False, max_new_trials: Optional[int] = None) -> dict:
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

    space = load_search_space(search_space_path, model_id, protocol.kind)
    context_plan = None
    if model_id == "tabpfgen":
        from sbtab.baselines.tabpfn.model import plan_tabpfn_context
        train = frame.loc[splits["T_row_ids"]]
        n_features = sum(train[c].nunique(dropna=False) if c in schema.categorical else 1
                         for c in schema.column_order if c != schema.target)
        n_classes = train[schema.target].nunique() if schema.task == "classification" else 0
        try:
            context_plan = plan_tabpfn_context(len(train), n_features, n_classes, space["fixed"].get("device", "cpu"))
        except ValueError as error:
            raise StageError("not_applicable", str(error)) from error
    metric_cfg = load_metric_config(protocol)
    from sbtab.evaluation import MetricConfig
    metric_config = MetricConfig.from_document(metric_cfg)
    prov = source_provenance()
    versions = library_versions()
    compat = {
        "tuning_version": TUNING_VERSION, "protocol_id": protocol.id, "protocol_hash": protocol.hash(),
        "dataset_fingerprint": splits["dataset_fingerprint"], "schema_hash": splits["schema_hash"],
        "membership_hash": splits["membership_hash"], "search_space_hash": canonical_hash(space),
        "metric_config_hash": canonical_hash(metric_cfg), "metric_version": metric_cfg["metric_version"],
        "checkpoint_format": ADAPTER_FORMAT, "adapter": entry.adapter, "model": model_id,
        "implementation_hash": implementation_hash(prov),
        "libraries": versions,
    }
    compat_hash = canonical_hash(compat)
    run_id = run_id or f"run-{compat_hash[:10]}"
    budget = int(protocol["tuning"]["n_trials"])
    run_dir = Path(splits_path).resolve().parent / model_id / run_id
    tuning_dir = run_dir / "tuning"

    plan = {"dataset": dataset, "model": model_id, "protocol_id": protocol.id, "kind": protocol.kind, "run_id": run_id,
            "run_dir": str(run_dir), "budget": budget, "n_T": splits["n_T"], "n_V": splits["n_V"],
            "split": {"test_size": splits["split"]["test_size"], "random_state": splits["split"]["random_state"]},
            "cv": splits["cv"], "sampler_seed": protocol["tuning"]["sampler_seed"], "regime": schema.regime,
            "searched": sorted(space["params"]), "fixed": space["fixed"]}
    if context_plan is not None:
        plan["tabpfgen_context"] = context_plan
    if dry_run:
        return {**plan, "dry_run": True}

    storage = f"sqlite:///{tuning_dir / 'study.sqlite3'}"
    exists = (tuning_dir / "study.sqlite3").exists()
    if exists and not resume:
        raise StageError("undefined", f"{tuning_dir} already holds a study; pass --resume to continue it "
                                      "(earlier runs are never removed or overwritten)")
    tuning_dir.mkdir(parents=True, exist_ok=True)
    sampler_path = tuning_dir / "sampler.pkl"
    sampler = optuna.samplers.TPESampler(seed=int(protocol["tuning"]["sampler_seed"]))
    sampler_state = "fresh"
    if exists and sampler_path.exists():
        with open(sampler_path, "rb") as fh:      # the study DB alone does not hold the sampler RNG state
            sampler = pickle.load(fh)
        sampler_state = "restored"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="study", storage=storage, direction=protocol["tuning"]["direction"],
                                sampler=sampler, pruner=optuna.pruners.NopPruner(), load_if_exists=True)
    if exists and counts(study)["allocated"] and not sampler_path.exists():
        raise StageError("undefined", "cannot resume: sampler.pkl is missing; restarting the sampler would rewind its RNG")
    stored = study.user_attrs.get("compatibility_hash")
    if stored is None:
        study.set_user_attr("compatibility_hash", compat_hash)
        study.set_user_attr("compatibility", compat)
    elif stored != compat_hash:
        old = study.user_attrs.get("compatibility", {})
        changed = sorted(k for k in compat if old.get(k) != compat[k])
        raise StageError("undefined", f"refusing to resume: the study is incompatible in {changed} "
                                      "(data, split, search space, metric, checkpoint/adapter, protocol or implementation changed)")

    stale = reconcile_stale_running(study, tuning_dir)
    save_sampler(study, sampler_path)
    seeds = SeedLedger(int(protocol["seeds"]["base"]))
    ledger_path = run_dir / "seed_ledger_tuning.json"
    if ledger_path.exists():
        seeds.records.update(read_json(ledger_path)["derived"])
    T_raw, V_raw = frame.loc[splits["T_row_ids"]], frame.loc[splits["V_row_ids"]]
    from sbtab.evaluation import check_validity, tuning_objective, MetricConfig, MetricContext

    write_json(run_dir / "run_manifest.json", {
        "stage": "tuning", **plan, "compatibility": compat, "provenance": prov, "libraries": versions,
        "hardware": hardware_info(), "search_space": space, "sampler": {"kind": "TPESampler", "state": sampler_state,
        "seed": protocol["tuning"]["sampler_seed"], "n_jobs": 1, "pruner": "none"},
        "target_handling": {"target": schema.target, "task": schema.task, "generated_as": "part of the row (X, y)"},
        "T_hash": splits["T_hash"], "V_hash": splits["V_hash"], "stale_running_reconciled": stale,
        "statistical_note": splits.get("statistical_note", "")})

    new = 0
    while counts(study)["allocated"] < budget and (max_new_trials is None or new < max_new_trials):
        trial = study.ask()
        new += 1
        d = trial_dir(tuning_dir, trial.number)
        state, value = TrialState.FAIL, None
        try:
            cfg = suggest(trial, space)
            save_sampler(study, sampler_path)
            rec = fit_generate(model_id, T_raw, schema, cfg, seeds, ("trial", trial.number), len(V_raw), d)
            metrics = {"status": rec["status"], "objective": None}
            if rec["status"] == "ok":
                pre = CommonPreprocessor.load(d / "preprocessor", schema)
                V = pre.transform(V_raw)
                synth = read_synthetic(d / "synthetic.parquet", schema)
                ctx = MetricContext.fit(pre.transform(T_raw), schema, metric_config,
                    train_row_ids=list(T_raw.index))
                validity = check_validity(synth, schema, ctx)
                obj = tuning_objective(V, synth, schema) if validity["status"] == "ok" else \
                    {"objective": None, "status": validity["status"], "regime": schema.regime, "mean_wd": None, "mean_js": None}
                metrics = {**obj, "validity": validity, "n_validation_rows": int(len(V)), "n_generated": rec["n_generated"],
                           "metric_version": metric_cfg["metric_version"], "objective_space": "common (standardised) space"}
                if obj["status"] == "ok" and obj["objective"] is not None and math.isfinite(obj["objective"]):
                    state, value = TrialState.COMPLETE, float(obj["objective"])
            write_json(d / "config.json", {"trial": trial.number, "sampled_params": trial.params,
                                           "requested_config": rec["requested_config"],
                                           "effective_config": rec["effective_config"], "seeds": rec["seeds"]})
            write_json(d / "metrics.json", metrics)
            write_json(d / "status.json", {"trial": trial.number, "state": state.name, "status": metrics["status"],
                                           "failure": rec["failure"], "n_updates": rec["n_updates"],
                                           "reload_verified": rec["reload_verified"], "describe": rec["describe"]})
            for k in ("mean_wd", "mean_js"):
                trial.set_user_attr(k, metrics.get(k))
            trial.set_user_attr("status", metrics["status"])
            trial.set_user_attr("n_updates", rec["n_updates"])
            trial.set_user_attr("generator_fit_seconds", rec["timing"].get("generator_fit_seconds"))
            trial.set_user_attr("generation_seconds", rec["timing"].get("generation_seconds"))
            if rec["failure"]:
                trial.set_user_attr("failure", f"{rec['failure']['type']}: {rec['failure']['message']}"[:500])
        except Exception as e:      # e.g. an invalid configuration discovered after allocation: keep the record
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / "status.json", {"trial": trial.number, "state": "FAIL", "status": "training_failed",
                                           "failure": {"type": type(e).__name__, "message": str(e), "trace": traceback.format_exc()}})
            trial.set_user_attr("status", "training_failed")
            trial.set_user_attr("failure", f"{type(e).__name__}: {e}"[:500])
        study.tell(trial, value, state=state)
        save_sampler(study, sampler_path)
        write_json(ledger_path, seeds.to_dict())
        write_trials_csv(study, tuning_dir)
        write_best(study, tuning_dir, budget, final=False, model_id=model_id, compat=compat)

    c = counts(study)
    final = c["allocated"] >= budget
    best = write_best(study, tuning_dir, budget, final=final, model_id=model_id, compat=compat)
    write_trials_csv(study, tuning_dir)
    write_json(run_dir / "seed_ledger_tuning.json", seeds.to_dict())
    return {**plan, "counts": c, "new_trials": new, "selection": best["selection"], "best_trial": best.get("trial"),
            "objective": best.get("objective"), "stale_running_reconciled": stale}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Persistent Optuna study: trial metrics/timings/configs, checkpoints, selected config.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="solver_registry id")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--search-space", required=True)
    ap.add_argument("--protocol", default=None, help="must be the protocol the splits were made under")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true", help="continue an existing study; allocates only the remaining budget")
    ap.add_argument("--dry-run", action="store_true", help="validate everything and print the resolved plan")
    ap.add_argument("--smoke", action="store_true", help="separate smoke protocol + smoke search space")
    args = ap.parse_args(argv)     # NOTE: there is deliberately no --n-trials / --seed / --test-size override
    try:
        out = run(args.dataset, args.model, args.splits, args.search_space, args.resume, args.smoke,
                  args.protocol, args.run_id, args.dry_run)
    except StageError as e:
        print({"status": e.status, "error": str(e)})
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
