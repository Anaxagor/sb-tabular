"""
Stage 3 — metrics from SAVED artifacts. No generator is fitted, no checkpoint is loaded.

    python -m sbtab.experiments.calculate_metrics --cv-run <cv-run-manifest.json> --metrics-config <metrics.yaml>

Inputs per fold: the immutable split (real row ids), the saved fold preprocessor and
the saved synthetic table. Everything a metric learns (histogram edges, supports,
conditioning definitions, MMD bandwidth/scales) comes from the fold's TRAINING rows.
Generated rows are compared with the held-out E_k.

Results go to ``evaluation/<metric-version>-<config-hash>/``: a changed metric
definition gets a new namespace instead of overwriting earlier results. Generator
artifacts under ``cv/`` (checkpoints, synthetic tables, fit timing) are only read.

The real-data TSTR reference and the frozen CatBoost parameters are cached per
dataset / split / schema / evaluator configuration — NOT per generator — so they
are numerically identical in every model's result table.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from sbtab.data.dataset_schema import CATEGORICAL, CONTINUOUS, DatasetSchema
from sbtab.data.preprocessing import CommonPreprocessor
from sbtab.experiments.experiment_common import (
    StageError, Timer, canonical_hash, file_hash, file_lock, hardware_info, library_versions, load_yaml, read_json, source_provenance, write_json,
)
from sbtab.experiments.prepare_splits import load_split_artifacts
from sbtab.experiments.runner import read_synthetic
from sbtab.experiments.model_selection import require_experiment_model
from sbtab.experiments.status_policy import fold_completion
from sbtab.evaluation.validity import numerical_diagnostics

EVAL_VERSION = "sbtab.evalstage/3"

_SKIP_KEYS = {"metric_version", "context_spec_hash", "validity", "seeds", "subsets", "floor_subsets", "reasons",
              "missing_columns", "extra_columns", "per_column", "predictions", "params", "label_universe",
              "excluded_conditioners", "incomplete_conditioners", "constant_training_columns", "classes_missing_in_synth"}


def metric_meta(name: str) -> Tuple[str, str]:
    """(direction, unit) from the metric name."""
    n = name.lower()
    leaf = n.rsplit(".", 1)[-1]
    if ".gaps." in n:
        score = n.split(".gaps.", 1)[1].split(".", 1)[0]
        if leaf == "delta_pct":
            return "lower_is_better", "percent absolute difference from real reference"
        if leaf == "abs_gap":
            return "lower_is_better", metric_meta(f"utility.scores.{score}")[1]
        if leaf in ("real", "synth"):
            return metric_meta(f"utility.scores.{score}")
    if leaf.startswith("n_") or leaf.endswith(("_count", "_columns", "_levels", "_pairs", "_rows")) or leaf in ("n", "subsample_size", "floor_size"):
        return "none", "count"
    if leaf.endswith("_seconds"):
        return "lower_is_better", "seconds"
    if "mass" in leaf or leaf.endswith("_rate") or "coverage" in leaf:
        return "none", "fraction"
    if leaf.endswith(("_pct", "_percent")) or "gap_pct" in leaf or "relative_gap" in leaf:
        return "lower_is_better", "percent absolute difference from real reference"
    if any(k in leaf for k in ("f1", "r2")):
        return "higher_is_better", "score"
    if "mape" in leaf:
        return "lower_is_better", "percent"
    if any(k in leaf for k in ("mae", "rmse")) and "utility" in n:
        return "lower_is_better", "raw target units"
    if "wd" in leaf.split("_"):
        return "lower_is_better", "standardised units"
    if any(k in leaf.split("_") for k in ("kl", "js")):
        return "lower_is_better", "nats"
    if "mmd" in n:
        return "closer_to_zero_is_better", "kernel units (signed unbiased estimate)"
    if any(k in leaf for k in ("frobenius", "rmse", "abs_error", "error")):
        return "lower_is_better", "association units"
    return "none", "value"


def flatten(summary: Any, prefix: str, status: str = "ok") -> Iterator[Tuple[str, Optional[float], str]]:
    """Numeric leaves of a summary -> (name, value-or-None, status of the nearest enclosing block)."""
    if isinstance(summary, dict):
        status = summary.get("status", status) if isinstance(summary.get("status", status), str) else status
        for k, v in summary.items():
            if k in _SKIP_KEYS or k == "status":
                continue
            score_status_key = {"scores": "score_status", "scores_real": "score_status_real",
                                "scores_synth": "score_status_synth"}.get(k)
            if score_status_key and isinstance(v, dict):
                for score, value in v.items():
                    yield from flatten(value, f"{prefix}.{k}.{score}", summary.get(score_status_key, {}).get(score, status))
            else:
                yield from flatten(v, f"{prefix}.{k}" if prefix else str(k), status)
    elif isinstance(summary, bool) or isinstance(summary, str) or isinstance(summary, (list, tuple)):
        return
    elif summary is None:
        yield prefix, None, (status if status != "ok" else "not_applicable")
    elif isinstance(summary, (int, float, np.integer, np.floating)):
        f = float(summary)
        yield prefix, (f if np.isfinite(f) else None), (status if np.isfinite(f) else "undefined")


def target_raw(pre: CommonPreprocessor, schema: DatasetSchema, column: pd.Series) -> np.ndarray:
    """Target in RAW units / labels (utility is evaluated after inverse transformation)."""
    spec = schema.spec(schema.target)
    if spec.type == CONTINUOUS:
        return column.to_numpy(dtype=np.float64) * pre.scales[spec.name] + pre.means[spec.name]
    if spec.type == CATEGORICAL:
        vocab = np.asarray(pre.vocab[spec.name], dtype=object)
        return vocab[column.to_numpy(dtype=np.int64)]
    return column.to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- utility caches (per dataset, not per generator)
def utility_params(dataset_dir: Path, key: str, task: str, X0: pd.DataFrame, y0, cat_features: List[str], cfg) -> dict:
    from sbtab.evaluation import resolve_utility_params
    from sbtab.evaluation.utility import UTILITY_PARAMETER_POLICY
    path = dataset_dir / "utility_config.json"
    with file_lock(dataset_dir / ".utility_config.lock"):
        store = read_json(path) if path.exists() else {}
        if key not in store:
            store[key] = {"params": resolve_utility_params(task, X0, y0, cat_features, cfg), "task": task,
                          "parameter_policy": UTILITY_PARAMETER_POLICY,
                          "convention": "fixed task-appropriate CatBoost CPU defaults-based preset; "
                                        "no probe fit, automatic learning rate or row-count-dependent subsampling; "
                                        "identical parameters for real and synthetic fits in every fold and generator; "
                                        "no validation set or early stopping, configured seed and thread count",
                          "libraries": {"catboost": library_versions().get("catboost")}}
            write_json(path, store)
        return store[key]["params"]


def real_reference(dataset_dir: Path, key: str, fold: int, compute) -> dict:
    d = dataset_dir / "utility_reference" / key[:16]
    path = d / f"fold-{fold}.json"
    with file_lock(d / f".fold-{fold}.lock"):
        if path.exists():
            ref = read_json(path)
            ref["cache"] = "hit"
            return ref
        ref = compute()
        preds = ref.pop("predictions", None)
        if preds is not None:
            pd.DataFrame(preds).to_parquet(d / f"fold-{fold}_predictions.parquet", index=False)
        write_json(path, ref)
        ref = read_json(path)       # the same JSON round-trip is returned to every generator
        ref["cache"] = "miss"
        return ref


# --------------------------------------------------------------------------- one fold
def evaluate_fold(k: int, frame, schema, splits, run_dir: Path, out_dir: Path, mcfg, metric_cfg_dict: dict,
                  base_record: dict, dataset_dir: Path) -> List[dict]:
    from sbtab import evaluation as ev
    fold_dir = run_dir / "cv" / f"fold-{k}"
    d = out_dir / f"fold-{k}"
    d.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    t_wall = time.perf_counter()
    records: List[dict] = []

    def rec(name, value, status, **extra):
        direction, unit = metric_meta(name)
        records.append({**base_record, "fold": k, "metric": name, "value": value, "status": status,
                        "direction": direction, "unit": unit, **extra})

    manifest = read_json(fold_dir / "manifest.json") if (fold_dir / "manifest.json").exists() else None
    if manifest is None or manifest["status"] != "ok":
        status = "training_failed" if manifest is None else manifest["status"]
        rec("fold_status", None, status)
        document = {"fold": k, "status": status, "reason": None if manifest is None else manifest.get("failure"),
                    "records": records}
        if manifest is not None:
            for key in ("validity", "numerical_diagnostics", "checkpoint_loaded", "sampling_probe", "failure_kind",
                        "serialization_failure"):
                if manifest.get(key) is not None:
                    document[key] = manifest[key]
        document["completion"] = fold_completion(document)
        write_json(d / "metrics.json", document)
        return records

    f = splits["folds"][k]
    if manifest["train_row_hash"] != f["train_hash"]:
        raise StageError("undefined", f"fold {k}: generator was fitted on rows other than T_k")
    if manifest["test_row_hash"] != f["test_hash"]:
        raise StageError("undefined", f"fold {k}: test membership differs from the generator manifest")
    if base_record.get("cv_compatibility_hash") != manifest.get("compatibility_hash"):
        raise StageError("undefined", f"fold {k}: incompatible generator provenance")
    for rel in ("synthetic.parquet", "preprocessor/preprocessor.json", "timing.json"):
        expected = manifest.get("artifact_hashes", {}).get(rel)
        if expected is not None and (not (fold_dir / rel).is_file() or file_hash(fold_dir / rel) != expected):
            raise StageError("undefined", f"fold {k}: missing or changed artifact {rel}")
    pre = CommonPreprocessor.load(fold_dir / "preprocessor", schema)
    if pre.fit_row_hash != f["train_hash"]:
        raise StageError("undefined", f"fold {k}: the saved fold transform was not fitted on T_k")
    T_k, E_k = pre.transform(frame.loc[f["train_row_ids"]]), pre.transform(frame.loc[f["test_row_ids"]])
    synth = read_synthetic(fold_dir / "synthetic.parquet", schema)
    if len(synth) != len(T_k) or len(synth) != manifest["n_generated"]:
        raise StageError("undefined", f"fold {k}: saved synthetic row count does not equal len(T_k)")
    ctx = ev.MetricContext.fit(T_k, schema, mcfg, train_row_ids=f["train_row_ids"])
    write_json(d / "metric_context.json", ctx.to_dict())
    counts = {"n_real": int(len(E_k)), "n_synth": int(len(synth)), "n_train": int(len(T_k)),
              "metric_spec_hash": ctx.spec_hash(), "transform_hash": canonical_hash(pre.to_dict()),
              "source": {"synthetic": str(fold_dir / "synthetic.parquet"), "preprocessor": str(fold_dir / "preprocessor"),
                         "splits": base_record["splits"]}, "seeds": manifest["seeds"]}

    summary: Dict[str, Any] = {"fold": k, "status": "ok"}
    with timer.measure("fidelity_metrics_seconds"):
        validity = ev.check_validity(synth, schema, ctx)
        summary["validity"] = validity
        # Compute from the fold's own training rows, including for legacy saved
        # generations. These flags never change validity or completion.
        summary["numerical_diagnostics"] = numerical_diagnostics(synth, schema, T_k)
        for name, value, status in flatten({k2: v for k2, v in validity.items() if k2.endswith("_rate") or k2 == "n_invalid_rows"}, "validity", validity["status"]):
            rec(name, value, "ok", **counts)
        if validity["status"] != "ok":
            summary["status"] = "invalid_generated_data"
            rec("fidelity", None, "invalid_generated_data", **counts)
        else:
            marg, per_feature = ev.marginal_metrics(ctx, E_k, synth)
            assoc, arrays = ev.association_metrics(ctx, E_k, synth)
            cond, per_condition = ev.conditional_metrics(ctx, E_k, synth)
            mmd = ev.mmd_metrics(ctx, E_k, synth, real_row_ids=list(E_k.index), synth_row_ids=list(synth.index))
            summary.update(marginal=marg, association=assoc, conditional=cond, mmd=mmd)
            per_feature.to_parquet(d / "per_feature.parquet", index=False)
            per_condition.astype({"level": str}).to_parquet(d / "per_condition.parquet", index=False)
            np.savez_compressed(d / "associations.npz", **{a: np.asarray(v) for a, v in arrays.items()})
            for block, body in (("marginal", marg), ("association", assoc), ("conditional", cond), ("mmd", mmd)):
                for name, value, status in flatten(body, block):
                    rec(name, value, status, **counts)

    # ---------------------------------------------------------------- TSTR utility
    if schema.target is not None and summary["status"] == "ok":
        target, task = schema.target, schema.task
        feats = [c for c in schema.column_order if c != target]           # y is stripped from the predictors
        cats = [c for c in schema.categorical if c != target]             # every nominal predictor -> cat_features
        universe = list(pre.vocab[target]) if task == "classification" else None
        key = canonical_hash({"dataset": splits["dataset_fingerprint"], "membership": splits["membership_hash"],
                              "schema": splits["schema_hash"], "evaluator": metric_cfg_dict,
                              "catboost": library_versions().get("catboost")})
        try:
            with timer.measure("utility_defaults_seconds"):
                params = utility_params(dataset_dir, key, task, T_k[feats], target_raw(pre, schema, T_k[target]), cats, mcfg)
            y_tr, y_te = target_raw(pre, schema, T_k[target]), target_raw(pre, schema, E_k[target])
            t0 = time.perf_counter()
            ref = real_reference(dataset_dir, key, k, lambda: ev.utility_reference(
                task, T_k[feats], y_tr, E_k[feats], y_te, cats, params, label_universe=universe))
            if ref["cache"] == "miss":
                timer.add("utility_real_fit_and_predict_seconds", time.perf_counter() - t0)
            try:
                y_sy = target_raw(pre, schema, synth[target])
                with timer.measure("utility_synth_fit_and_predict_seconds"):
                    tstr = ev.utility_tstr(task, synth[feats], y_sy, E_k[feats], y_te, cats, params, reference=ref,
                                           label_universe=universe)
            except Exception as e:
                tstr = {"status": "utility_fit_failed", "failure": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            preds = tstr.pop("predictions", None)
            if preds is not None:
                pd.DataFrame(preds).to_parquet(d / "utility_predictions.parquet", index=False)
            for sub in ("fit_seconds", "predict_seconds"):
                if isinstance(tstr.get(sub), (int, float)):
                    timer.add(f"utility_synth_{sub}", tstr[sub])
            summary["utility"] = {"status": tstr["status"] if ref["status"] == "ok" else ref["status"],
                                  "task": task, "reference_cache": ref["cache"], "reference_key": key[:16], "reference": ref,
                                  "tstr": tstr, "n_synth_train_rows": int(len(synth)), "cat_features": cats}
            for name, value, status in flatten(ref, "utility.real"):
                rec(name, value, status, **counts)
            for name, value, status in flatten(tstr, "utility.synth"):
                rec(name, value, status, **counts)
        except Exception as e:
            # Utility setup/reference fitting is independent of generator fidelity.
            # Preserve the other metrics even if defaults cannot be resolved.
            summary["utility"] = {"status": "utility_fit_failed", "task": task,
                                  "failure": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            rec("utility", None, "utility_fit_failed", **counts)
    elif schema.target is None:
        summary["utility"] = {"status": "not_applicable"}
        rec("utility", None, "not_applicable", **counts)

    # generator timings are first-class metrics; they are READ from the CV artifacts, never rewritten
    gen_timing = read_json(fold_dir / "timing.json")
    for name, value in gen_timing["seconds"].items():
        rec(f"generator.{name}", float(value), "ok", **counts)
    rec("generator.n_updates", None if manifest["n_updates"] is None else float(manifest["n_updates"]), "ok", **counts)

    timer.seconds["stage_wall_seconds"] = time.perf_counter() - t_wall
    write_json(d / "timing.json", {"seconds": timer.to_dict(), "hardware": hardware_info(),
                                   "generator_timing_source": str(fold_dir / "timing.json")})
    for name, value in timer.to_dict().items():
        rec(f"evaluation.{name}", float(value), "ok", **counts)
    document = {**summary, "records": records, "counts": counts}
    document["completion"] = fold_completion(document)
    write_json(d / "metrics.json", document)
    return records


# --------------------------------------------------------------------------- stage
def run(cv_run_manifest, metrics_config_path, folds: Optional[List[int]] = None, dry_run: bool = False) -> dict:
    from sbtab import evaluation as ev
    cv_manifest = read_json(cv_run_manifest)
    require_experiment_model(cv_manifest["model"])
    run_dir = Path(cv_manifest["run_dir"])
    metric_cfg_dict = load_yaml(metrics_config_path)
    if metric_cfg_dict.get("metric_version") != ev.METRIC_VERSION:
        raise StageError("undefined", f"metrics config declares {metric_cfg_dict.get('metric_version')!r} but the "
                                      f"installed metric implementation is {ev.METRIC_VERSION!r}")
    mcfg = ev.MetricConfig.from_document(metric_cfg_dict)
    frame, schema, splits = load_split_artifacts(cv_manifest["splits"])
    for key in ("membership_hash", "schema_hash", "dataset_fingerprint"):
        if splits[key] != cv_manifest[key]:
            raise StageError("undefined", f"the CV run was produced under a different {key}")
    namespace = f"{ev.METRIC_VERSION.replace('/', '_')}-{canonical_hash(metric_cfg_dict)[:8]}-eval3"
    out_dir = run_dir / "evaluation" / namespace
    n_splits = splits["cv"]["n_splits"]
    fold_ids = list(range(n_splits)) if folds is None else sorted(set(folds))
    if any(k < 0 or k >= n_splits for k in fold_ids):
        raise StageError("undefined", f"fold indices must lie in [0, {n_splits})")
    if dry_run:
        return {"run_dir": str(run_dir), "evaluation_dir": str(out_dir), "folds": fold_ids, "dry_run": True}

    prov = source_provenance()
    base = {"protocol_id": cv_manifest["protocol_id"], "metric_version": ev.METRIC_VERSION,
            "evaluation_version": EVAL_VERSION,
            "metric_config_hash": canonical_hash(metric_cfg_dict), "dataset": cv_manifest["dataset"],
            "membership_hash": splits["membership_hash"], "schema_hash": splits["schema_hash"],
            "cv_compatibility_hash": cv_manifest.get("compatibility_hash"),
            "dataset_fingerprint": splits["dataset_fingerprint"], "model": cv_manifest["model"],
            "run_id": run_dir.name, "commit": prov["commit"], "dirty_diff_hash": prov["dirty_diff_hash"],
            "stage": "cv_evaluation", "splits": str(cv_manifest["splits"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[dict] = []
    for k in fold_ids:
        all_records += evaluate_fold(k, frame, schema, splits, run_dir, out_dir, mcfg, metric_cfg_dict, base,
                                     Path(cv_manifest["splits"]).resolve().parent)

    from sbtab.experiments.aggregate_results import aggregate_run
    summary = aggregate_run(out_dir, n_expected=n_splits)
    write_json(out_dir / "evaluation_manifest.json", {
        "version": EVAL_VERSION, **base, "namespace": namespace, "metric_config": metric_cfg_dict,
        "folds": fold_ids, "provenance": prov, "libraries": library_versions(),
        "generator_artifacts": "read-only: cv/ checkpoints, synthetic tables and fit timings are never modified"})
    return {"evaluation_dir": str(out_dir), "n_records": len(all_records), "n_metrics": len(summary["metrics"]),
            "folds": fold_ids}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-fold metrics, utility predictions, diagnostics and aggregates; no generator retraining.")
    ap.add_argument("--cv-run", required=True, help="cv/cv_run_manifest.json written by cross_validate")
    ap.add_argument("--metrics-config", default="configs/metrics/metrics_v2.yaml")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="accepted for symmetry; metrics are identical in smoke runs")
    args = ap.parse_args(argv)
    try:
        print(run(args.cv_run, args.metrics_config, args.folds, args.dry_run))
    except StageError as e:
        print({"status": e.status, "error": str(e)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
