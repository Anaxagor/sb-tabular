"""
Stage 4 — aggregation of validated per-fold metric records.

    python -m sbtab.experiments.aggregate_results --output-root <artifact-root> [--ranks-on marginal.groups.continuous.mean_wd ...]

Fold results are saved BEFORE aggregation and are never recomputed here.

Per run: arithmetic mean and SAMPLE standard deviation (ddof = 1) over the eligible
fold values, with n_expected, n_valid, n_failed and the excluded folds listed. A
model with fewer valid folds is flagged ``complete = False`` and never appears as a
complete five-fold comparison. Missing results stay null; they are not zeros.

Across runs: tidy CSV / Parquet / JSON. Average ranks are recomputed ONLY on
identical applicable (dataset, fold) sets, and the number of compared models is
kept with every rank: per dataset among the models that have every evaluated fold
(a model missing a fold is listed as excluded, never ranked on a different fold
set), and overall only over cells shared by ALL compared models (``--models``).
Models that were run on different datasets are never ranked against each other.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from sbtab.experiments.experiment_common import StageError, read_json, write_json
from sbtab.experiments.model_selection import unavailable_experiment_reason

ELIGIBLE = ("ok", "incomplete_conditional_coverage")     # a value exists; incompleteness is carried along, not averaged away
RECORD_COLUMNS = ["protocol_id", "metric_version", "metric_config_hash", "dataset", "dataset_fingerprint", "model", "run_id",
                  "commit", "dirty_diff_hash", "stage", "fold", "metric", "value", "status", "direction", "unit",
                  "n_real", "n_synth", "n_train", "metric_spec_hash", "transform_hash", "membership_hash", "schema_hash",
                  "evaluation_version"]


def load_fold_records(evaluation_dir: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(evaluation_dir.glob("fold-*")):
        m = fold_dir / "metrics.json"
        if m.exists():
            rows += read_json(m).get("records", [])
    df = pd.DataFrame(rows)
    for c in RECORD_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df


def summarise(df: pd.DataFrame, n_expected: int) -> List[dict]:
    out = []
    failed_folds = sorted(df.loc[df["metric"] == "fold_status", "fold"].unique().tolist()) if len(df) else []
    for metric, g in df[df["metric"] != "fold_status"].groupby("metric", sort=True):
        ok = g[g["status"].isin(ELIGIBLE) & g["value"].notna()]
        vals = ok["value"].to_numpy(dtype=np.float64)
        excluded = [{"fold": int(r.fold), "status": r.status} for r in g.itertuples() if r.Index not in ok.index]
        excluded += [{"fold": int(k), "status": "fold_failed"} for k in failed_folds]
        missing = sorted(set(range(n_expected)) - set(int(x) for x in g["fold"]) - set(failed_folds))
        excluded += [{"fold": k, "status": "not_evaluated"} for k in missing]
        statuses = sorted(set(g["status"]))
        out.append({
            "metric": metric, "direction": g["direction"].iloc[0], "unit": g["unit"].iloc[0],
            "mean": float(vals.mean()) if len(vals) else None,
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else None, "std_ddof": 1,
            "n_expected": int(n_expected), "n_valid": int(len(vals)), "n_failed": int(n_expected - len(vals)),
            "complete": bool(len(vals) == n_expected), "excluded": excluded, "statuses": statuses,
            "n_incomplete_coverage": int((ok["status"] == "incomplete_conditional_coverage").sum()),
            "fold_values": {str(int(r.fold)): (None if pd.isna(r.value) else float(r.value)) for r in g.itertuples()},
        })
    return out


def aggregate_run(evaluation_dir, n_expected: int = 5) -> dict:
    evaluation_dir = Path(evaluation_dir)
    df = load_fold_records(evaluation_dir)
    for model in df["model"].dropna().unique():
        reason = unavailable_experiment_reason(model)
        if reason:
            raise StageError("undefined", f"model {model!r} is excluded from new experiment aggregates: {reason}")
    tmp = evaluation_dir / "per_fold.csv.tmp"
    df[RECORD_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(evaluation_dir / "per_fold.csv")
    metrics = summarise(df, n_expected)
    head = {c: (df[c].iloc[0] if len(df) else None) for c in ("protocol_id", "metric_version", "metric_config_hash",
                                                              "dataset", "dataset_fingerprint", "model", "run_id",
                                                              "commit", "dirty_diff_hash")}
    observed = set(df.loc[df["metric"] != "fold_status", "fold"])
    failed = set(df.loc[df["metric"] == "fold_status", "fold"])
    n_ok_folds = len(observed - failed)
    summary = {**head, "n_expected": n_expected, "n_folds_with_generator_output": n_ok_folds,
               "complete_five_fold": bool(n_ok_folds == n_expected), "aggregation": "arithmetic mean, sample std (ddof=1)",
               "metrics": metrics}
    write_json(evaluation_dir / "summary.json", summary)
    flat = pd.DataFrame([{**head, **{k: v for k, v in m.items() if k not in ("excluded", "fold_values", "statuses")},
                          "excluded_folds": ";".join(f"{e['fold']}:{e['status']}" for e in m["excluded"]),
                          "statuses": ";".join(m["statuses"])} for m in metrics])
    tmp = evaluation_dir / "summary.csv.tmp"
    flat.to_csv(tmp, index=False)
    tmp.replace(evaluation_dir / "summary.csv")
    return summary


def _rank_block(wide: pd.DataFrame, direction: str) -> pd.DataFrame:
    score = wide.abs() if direction == "closer_to_zero_is_better" else wide
    return score.rank(axis=1, ascending=direction != "higher_is_better", method="average")


def average_ranks(per_fold: pd.DataFrame, metric: str, models: Optional[List[str]] = None) -> dict:
    """
    Average ranks on IDENTICAL applicable (dataset, fold) sets only, always with the number of compared models.

    ``per_dataset``  for each dataset, the models that have a value on EVERY evaluated fold of that dataset are
                     ranked against each other on exactly those folds. A model missing a fold is listed under
                     ``excluded_models`` instead of being ranked on a different fold set.
    ``overall``      one rank per model over the (dataset, fold) cells shared by ALL compared models. The compared
                     set is ``models`` if given, else every model that has the metric. If those models were not all
                     run on a common dataset the overall rank is ``not_applicable`` — models evaluated on different
                     datasets are never ranked against each other.
    """
    all_rows = per_fold[per_fold["metric"].isin([metric, "fold_status"])]
    if models is not None:
        all_rows = all_rows[all_rows["model"].isin(models)]
    d = all_rows[(all_rows["metric"] == metric) & all_rows["status"].isin(ELIGIBLE) & all_rows["value"].notna()]
    if d.empty:
        return {"metric": metric, "status": "not_applicable", "reason": "no eligible value", "n_models": 0,
                "overall": {"status": "not_applicable", "ranks": {}}, "per_dataset": {}}
    direction = d["direction"].iloc[0]
    if direction == "none":
        return {"metric": metric, "status": "not_applicable", "reason": "metric has no direction", "n_models": 0,
                "overall": {"status": "not_applicable", "ranks": {}}, "per_dataset": {}}
    if all_rows.duplicated(["dataset", "fold", "model"]).any():
        raise StageError("undefined", "ambiguous ranks: multiple runs supply the same dataset/fold/model; "
                         "select one run per model in a separate comparison root")
    for key in ("protocol_id", "metric_version", "metric_config_hash", "direction", "unit"):
        if key in d and d[key].dropna().nunique() > 1:
            raise StageError("undefined", f"incompatible {key} in rank comparison")
    for key in ("dataset_fingerprint", "membership_hash", "schema_hash", "transform_hash", "metric_spec_hash"):
        if key in d and (d.groupby(["dataset", "fold"])[key].nunique() > 1).any():
            raise StageError("undefined", f"incompatible {key} in rank comparison")
    compared = sorted(set(models) if models is not None else set(all_rows["model"]))
    cells = pd.MultiIndex.from_frame(all_rows[["dataset", "fold"]].drop_duplicates())
    wide = d.pivot(index=["dataset", "fold"], columns="model", values="value").reindex(index=cells, columns=compared)

    per_dataset = {}
    for dataset, block in wide.groupby(level="dataset"):
        dataset_models = sorted(set(all_rows.loc[all_rows["dataset"] == dataset, "model"]))
        block = block.reindex(columns=dataset_models)
        complete = [m for m in block.columns if block[m].notna().all()]
        excluded = [m for m in block.columns if m not in complete]
        if len(complete) < 2:
            per_dataset[dataset] = {"status": "not_applicable", "reason": "fewer than two models with all folds",
                                    "n_models": len(complete), "models": complete, "excluded_models": excluded, "ranks": {}}
            continue
        r = _rank_block(block[complete], direction)
        per_dataset[dataset] = {"status": "ok", "n_models": len(complete), "models": complete, "excluded_models": excluded,
                                "n_cells": int(len(block)), "folds": sorted(int(i[1]) for i in block.index),
                                "ranks": {m: float(r[m].mean()) for m in complete}}

    common = wide[compared].dropna(how="any")                 # cells EVERY compared model has
    if common.empty or len(compared) < 2:
        overall = {"status": "not_applicable", "n_models": len(compared), "models": compared, "ranks": {},
                   "reason": "no (dataset, fold) cell is shared by all compared models; see per_dataset, or pass an "
                             "explicit model set that was run on common datasets"}
    else:
        r = _rank_block(common, direction)
        overall = {"status": "ok", "n_models": len(compared), "models": compared, "n_cells": int(len(common)),
                   "n_cells_dropped": int(len(wide) - len(common)), "datasets": sorted({i[0] for i in common.index}),
                   "ranks": {m: float(r[m].mean()) for m in compared}}
    return {"metric": metric, "status": "ok", "direction": direction, "n_models": len(compared),
            "overall": overall, "per_dataset": per_dataset}


def aggregate_root(output_root, rank_metrics: Optional[List[str]] = None, models: Optional[List[str]] = None) -> dict:
    root = Path(output_root)
    frames, summaries, excluded = [], [], []
    if models is not None:
        for model in models:
            reason = unavailable_experiment_reason(model)
            if reason:
                excluded.append({"model": model, "reason": reason, "source": "requested rank comparison"})
        models = [model for model in models if unavailable_experiment_reason(model) is None]
    for per_fold in sorted(root.glob("*/*/*/evaluation/*/per_fold.csv")):
        df = pd.read_csv(per_fold)
        blocked = {model: reason for model in df["model"].dropna().unique()
                   if (reason := unavailable_experiment_reason(model)) is not None}
        for model, reason in blocked.items():
            excluded.append({"model": model, "reason": reason, "evaluation_dir": str(per_fold.parent),
                             "n_records": int((df["model"] == model).sum())})
        df = df.loc[~df["model"].isin(blocked)].copy()
        if df.empty:
            continue
        df["namespace"] = per_fold.parent.name
        frames.append(df)
        s = read_json(per_fold.parent / "summary.json")
        summaries.append({k: v for k, v in s.items() if k != "metrics"} | {"namespace": per_fold.parent.name,
                                                                          "evaluation_dir": str(per_fold.parent)})
    if not frames and not excluded:
        return {"n_runs": 0}
    all_folds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RECORD_COLUMNS + ["namespace"])
    out = root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    all_folds.to_csv(out / "per_fold_all.csv", index=False)
    all_folds.to_parquet(out / "per_fold_all.parquet", index=False)
    result = {"n_runs": len(frames), "runs": summaries, "namespaces": sorted(all_folds["namespace"].unique()),
              "incomplete_runs": [s for s in summaries if not s.get("complete_five_fold")], "ranks": [],
              "excluded_models": excluded}
    for ns, g in all_folds.groupby("namespace"):              # never rank across metric definitions
        for metric in rank_metrics or []:
            result["ranks"].append({"namespace": ns, **average_ranks(g, metric, models)})
    write_json(out / "summary_all.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tidy CSV/Parquet and JSON summaries; retains failed/inapplicable counts.")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--ranks-on", nargs="*", default=None, help="metric names to compute average ranks for")
    ap.add_argument("--models", nargs="*", default=None, help="compared model set for the overall rank (default: all)")
    args = ap.parse_args(argv)
    r = aggregate_root(args.output_root, args.ranks_on, args.models)
    print({k: v for k, v in r.items() if k not in ("runs",)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
