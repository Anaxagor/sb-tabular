"""
Stage 0 — immutable split manifests and the category-support report. No model fitting.

    python -m sbtab.experiments.prepare_splits --dataset adult \
        --protocol configs/protocols/sbtab_8515_hpo100_cv5_v2.yaml \
        --output-root artifacts/sbtab_8515_hpo100_cv5_v2

Names: D complete eligible dataset (the source table after the protocol's dataset-eligibility rule — none
under v1; under v2 rows whose value in a finite-support column occurs < 3 times are removed, see
``eligibility_report.json``), T the 85 % tuning-training pool, V the 15 %
tuning-validation set, (T_k, E_k) the train/test pair of CV fold k inside T.

Target stratification does not guarantee support coverage of every feature, so
support is validated explicitly. If the fixed split fails, the dataset is stopped
with ``split_status = blocked_support`` and a structured report. No alternative
seed is searched, no rare row is pinned, no category is merged, K is not changed,
no row is discarded BECAUSE OF THE SPLIT and no global encoder is fitted.

The v2 eligibility rule is a different thing: it removes rows by value counts of the
whole source table, before and independently of any split. It reduces blocking but
cannot guarantee coverage (3 rows of a value can still fall 1 into V and 2 into the
same CV test fold), so this validation runs after it and can still stop a dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from sbtab.data.dataset_schema import DatasetSchema
from sbtab.data.eligibility import apply_eligibility, finite_levels
from sbtab.data.loading import frame_fingerprint
from sbtab.data.preprocessing import row_id_hash
from sbtab.data.registry import DEFAULT_CONFIG_DIR, load_dataset
from sbtab.experiments.experiment_common import (
    ArtifactPaths, Protocol, StageError, canonical_hash, claim_output_root, library_versions,
    load_protocol, read_json, source_provenance, write_json,
)

SPLITS_VERSION = "sbtab.splits/1"


# --------------------------------------------------------------------------- stratification
def regression_strata(y: np.ndarray, start_bins: int, test_size: float) -> Tuple[np.ndarray, dict]:
    """
    Deterministic target-quantile strata: start with ``start_bins`` quantile bins,
    drop repeated edges, and reduce the bin count until a stratified split is
    feasible (every stratum has >= 2 rows and both sides can hold every stratum).
    These are splitting metadata, not a generator transform.
    """
    n = len(y)
    n_test = int(np.ceil(test_size * n))
    for bins in range(int(start_bins), 1, -1):
        edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 3:
            continue
        strata = np.digitize(y, edges[1:-1], right=True)
        counts = np.bincount(strata)
        counts = counts[counts > 0]
        if counts.min() >= 2 and len(counts) <= n_test and len(counts) <= n - n_test:
            return strata, {"kind": "target_quantiles", "requested_bins": int(start_bins), "chosen_bins": int(bins),
                            "n_strata": int(len(counts)), "edges": [float(e) for e in edges],
                            "interior_edges": [float(e) for e in edges[1:-1]], "rule": "digitize(right=True)"}
    raise StageError("undefined", "no feasible target-quantile stratification; define a stratification "
                                  "variable in the dataset configuration")


def stratification_vector(frame: pd.DataFrame, schema: DatasetSchema, protocol: Protocol):
    if not protocol["split"]["stratify"]:
        raise StageError("undefined", "the canonical protocol requires a stratified split")
    if schema.target is None:
        raise StageError("undefined", "no stratification variable: the dataset declares no target; define one "
                                      "in the dataset configuration instead of splitting unstratified")
    y = frame[schema.target]
    if y.isna().any():
        raise StageError("undefined", f"target {schema.target!r} has {int(y.isna().sum())} missing values; rows are "
                                      "never filtered by outcome, so this dataset needs an explicit decision")
    if schema.task == "classification":
        labels = y.astype(str).to_numpy()
        values, counts = np.unique(labels, return_counts=True)
        return labels, {"kind": "task_label", "class_counts": {str(v): int(c) for v, c in zip(values, counts)}}
    return regression_strata(y.to_numpy(dtype=np.float64), protocol["split"]["regression_strata_bins"],
                             protocol["split"]["test_size"])


# --------------------------------------------------------------------------- support validation
def support_violations(frame: pd.DataFrame, schema: DatasetSchema, train_ids, held_ids, where: str) -> List[dict]:
    out = []
    train, held = frame.loc[train_ids], frame.loc[held_ids]
    for col in schema.finite_support:
        lt, lh = finite_levels(train, col, schema), finite_levels(held, col, schema)
        missing = sorted(set(lh.dropna().unique()) - set(lt.dropna().unique()))
        for value in missing:
            rows = lh.index[lh == value].tolist()
            out.append({"where": where, "column": col, "column_type": schema.type_of(col),
                        "is_target": col == schema.target, "value": value,
                        "count_in_held_out": int(len(rows)), "count_in_training": 0,
                        "count_in_dataset": int((finite_levels(frame, col, schema) == value).sum()),
                        "affected_row_ids": [int(r) for r in rows]})
    # imputation needs at least one observed training value per numeric column
    if schema.missing_policy == "impute":
        for col in schema.continuous + schema.discrete:
            if train[col].notna().sum() == 0:
                out.append({"where": where, "column": col, "column_type": schema.type_of(col),
                            "is_target": col == schema.target, "value": "<no observed training value>",
                            "count_in_held_out": int(held[col].isna().sum()), "count_in_training": 0,
                            "count_in_dataset": int(frame[col].notna().sum()), "affected_row_ids": []})
    return out


# --------------------------------------------------------------------------- stage
def build_splits(frame: pd.DataFrame, schema: DatasetSchema, manifest: dict, protocol: Protocol) -> Tuple[dict, dict]:
    sp, cv = protocol["split"], protocol["cv"]
    if schema.missing_policy == "reject" and manifest["missing_counts"]:
        raise StageError("undefined", f"missing values {manifest['missing_counts']} with missing_policy 'reject': "
                                      "the preflight rejects unspecified missing-value handling")
    strata, strata_meta = stratification_vector(frame, schema, protocol)

    row_ids = frame.index.to_numpy(dtype=np.int64)
    try:
        # integer rounding of the 15 % side is scikit-learn's own
        t_ids, v_ids = train_test_split(row_ids, test_size=sp["test_size"], random_state=sp["random_state"],
                                        stratify=strata)
    except ValueError as e:
        raise StageError("undefined", f"stratified {1 - sp['test_size']:.0%}/{sp['test_size']:.0%} split infeasible: {e}",
                         {"stratification": strata_meta}) from e
    T = np.sort(t_ids)          # pool_order: sorted_row_id — KFold positions refer to this order
    V = np.sort(v_ids)

    kf = KFold(n_splits=cv["n_splits"], shuffle=cv["shuffle"], random_state=cv["random_state"])
    folds = []
    for k, (tr_pos, te_pos) in enumerate(kf.split(T)):
        folds.append({"fold": k, "train_row_ids": [int(i) for i in T[tr_pos]], "test_row_ids": [int(i) for i in T[te_pos]],
                      "train_hash": row_id_hash(T[tr_pos]), "test_hash": row_id_hash(T[te_pos])})

    violations = support_violations(frame, schema, T, V, "V_vs_T")
    for f in folds:
        violations += support_violations(frame, schema, f["train_row_ids"], f["test_row_ids"], f"fold_{f['fold']}")
    status = "blocked_support" if violations else "ok"

    value_counts = {}
    for col in sorted({v["column"] for v in violations}):
        vc = finite_levels(frame, col, schema).value_counts()
        value_counts[col] = {str(k): int(n) for k, n in vc.items()}

    support_report = {
        "version": SPLITS_VERSION, "dataset": schema.name, "protocol_id": protocol.id, "split_status": status,
        "checked_columns": schema.finite_support,
        "conditions": ["support(V) ⊆ support(T)", "support(E_k) ⊆ support(T_k) for every fold k",
                       "categorical missingness is the level '__missing__' and must be covered like any other"],
        "n_violations": len(violations),
        "blocking_columns": sorted({v["column"] for v in violations}),
        "affected_folds": sorted({v["where"] for v in violations}),
        "violations": violations,
        "value_counts_of_blocking_columns": value_counts,
        "policy": "No alternative seed, pinned row, merged category, changed K, discarded row or global encoder "
                  "is used to bypass a failure. A support-constrained split would be a separately versioned protocol.",
    }
    splits = {
        "version": SPLITS_VERSION, "dataset": schema.name,
        "protocol_id": protocol.id, "protocol_kind": protocol.kind, "protocol_hash": protocol.hash(),
        "split_status": status, "eligibility_rule": protocol.eligibility,
        "dataset_fingerprint": manifest["fingerprint"], "schema_hash": schema.hash(),
        "n_rows": int(len(frame)), "n_T": int(len(T)), "n_V": int(len(V)),
        "split": {"test_size": sp["test_size"], "random_state": sp["random_state"], "splitter": "sklearn.train_test_split",
                  "stratification": strata_meta, "pool_order": sp["pool_order"]},
        "cv": {"splitter": "sklearn.KFold", "n_splits": cv["n_splits"], "shuffle": cv["shuffle"],
               "random_state": cv["random_state"], "population": "T"},
        "T_row_ids": [int(i) for i in T], "V_row_ids": [int(i) for i in V],
        "T_hash": row_id_hash(T), "V_hash": row_id_hash(V),
        "folds": folds,
        "statistical_note": protocol.data.get("statistical_note", ""),
    }
    splits["membership_hash"] = canonical_hash({"T": splits["T_hash"], "V": splits["V_hash"],
                                                "folds": [(f["train_hash"], f["test_hash"]) for f in folds]})
    return splits, support_report


def load_eligible_dataset(dataset: str, protocol: Protocol, config_dir=DEFAULT_CONFIG_DIR):
    """
    D, the complete ELIGIBLE dataset of this protocol: the source table after the protocol's
    dataset-eligibility rule (none under v1). Applied before any split; original row ids are kept.
    """
    source, schema, manifest = load_dataset(dataset, config_dir)
    try:
        frame, report = apply_eligibility(source, schema, protocol.eligibility)
    except ValueError as e:
        raise StageError("undefined", f"dataset {dataset!r}: {e} under eligibility rule {protocol.eligibility}") from e
    manifest = {**manifest, "source_fingerprint": manifest["fingerprint"], "n_source_rows": int(len(source)),
                "fingerprint": frame_fingerprint(frame), "n_rows": int(len(frame)),
                "missing_counts": {c: int(n) for c, n in frame.isna().sum().items() if n},
                "row_filtering": "none" if not report["n_removed_rows"] else
                f"eligibility rule {report['rule']} removed {report['n_removed_rows']} of {len(source)} rows before splitting",
                "eligibility": {k: report[k] for k in ("rule", "applied", "n_removed_rows", "removed_fraction",
                                                       "passes", "target_values_removed", "task_changed")}}
    return frame, schema, manifest, report


def run(dataset: str, protocol: Protocol, output_root, config_dir=DEFAULT_CONFIG_DIR, dry_run: bool = False) -> dict:
    frame, schema, manifest, eligibility_report = load_eligible_dataset(dataset, protocol, config_dir)
    splits, support_report = build_splits(frame, schema, manifest, protocol)
    summary = {"dataset": dataset, "protocol_id": protocol.id, "split_status": splits["split_status"],
               "n_source_rows": manifest["n_source_rows"], "n_removed_by_eligibility": eligibility_report["n_removed_rows"],
               "task_changed": eligibility_report["task_changed"],
               "n_rows": splits["n_rows"], "n_T": splits["n_T"], "n_V": splits["n_V"],
               "regime": schema.regime, "blocking_columns": support_report["blocking_columns"],
               "membership_hash": splits["membership_hash"]}
    if dry_run:
        return summary

    root = claim_output_root(output_root, protocol)
    paths = ArtifactPaths(root=root, dataset=dataset)
    if paths.splits.exists():
        old = read_json(paths.splits)
        if any(old.get(key) != splits[key] for key in
               ("membership_hash", "schema_hash", "dataset_fingerprint", "protocol_hash")):
            raise StageError("undefined", f"{paths.splits} already exists with different data/membership/schema/protocol; split "
                                          "manifests are immutable — use a new output root or protocol id")
        load_split_artifacts(paths.splits)
        return {**summary, "note": "existing identical split manifest kept"}

    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    # Lossless local copy: numeric columns as float64, labels as strings; the schema sits alongside.
    frame.reset_index().to_parquet(paths.data, index=False)
    write_json(paths.schema, schema.to_dict())
    write_json(paths.dataset_dir / "dataset_manifest.json",
               {**manifest, "data_file": paths.data.name, "provenance": source_provenance(),
                "libraries": library_versions()})
    write_json(paths.dataset_dir / "eligibility_report.json", eligibility_report)
    write_json(paths.dataset_dir / "support_report.json", support_report)
    write_json(paths.splits, splits)
    return summary


def load_split_artifacts(splits_path) -> Tuple[pd.DataFrame, DatasetSchema, dict]:
    """Read the immutable manifests written by this stage (used by every later stage)."""
    splits_path = Path(splits_path)
    splits = read_json(splits_path)
    d = splits_path.parent
    schema = DatasetSchema.from_dict(read_json(d / "schema.json"))
    if schema.hash() != splits["schema_hash"]:
        raise StageError("undefined", "schema.json does not match the schema hash recorded in splits.json")
    frame = pd.read_parquet(d / "data.parquet").set_index("row_id")
    if list(frame.columns) != schema.column_order:
        raise StageError("undefined", "data.parquet columns do not match schema.json")
    for c in schema.categorical:
        frame[c] = frame[c].astype(object).where(~frame[c].isna(), None)
    validate_split_artifacts(frame, splits)
    return frame, schema, splits


def validate_split_artifacts(frame: pd.DataFrame, splits: dict) -> None:
    """Verify persisted values and actual memberships, rather than trusting stored hashes."""
    def require(ok, message):
        if not ok:
            raise StageError("undefined", f"invalid split artifacts: {message}")

    def ids(values, name):
        require(all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values),
                f"{name} row IDs must be integers")
        require(len(values) == len(set(values)), f"duplicate row IDs in {name}")
        return set(values)

    rows = ids(frame.index.tolist(), "data.parquet")
    require(frame.index.tolist() == sorted(rows), "saved dataset row IDs must retain source order")
    require(frame_fingerprint(frame) == splits["dataset_fingerprint"], "dataset fingerprint mismatch")
    require(len(frame) == splits["n_rows"], "dataset row count mismatch")
    T, V = ids(splits["T_row_ids"], "T"), ids(splits["V_row_ids"], "V")
    require(not T & V and T | V == rows, "T and V must partition the saved dataset")
    require(len(T) == splits["n_T"] and len(V) == splits["n_V"], "T/V row counts mismatch")
    require(splits["T_row_ids"] == sorted(T), "T must be in sorted row-ID order")
    for name in ("T", "V"):
        require(row_id_hash(splits[f"{name}_row_ids"]) == splits[f"{name}_hash"], f"{name} membership hash mismatch")
    cv = splits["cv"]
    require(len(splits["folds"]) == cv["n_splits"], "fold count mismatch")
    kf = KFold(n_splits=cv["n_splits"], shuffle=cv["shuffle"], random_state=cv["random_state"])
    pool = np.asarray(splits["T_row_ids"])
    for k, ((tr, te), f) in enumerate(zip(kf.split(pool), splits["folds"])):
        require(f["fold"] == k, "fold IDs/order mismatch")
        train, test = ids(f["train_row_ids"], f"fold {k} train"), ids(f["test_row_ids"], f"fold {k} test")
        require(not train & test and train | test == T, f"fold {k} must partition T")
        require(f["train_row_ids"] == pool[tr].tolist() and f["test_row_ids"] == pool[te].tolist(),
                f"fold {k} does not match the declared sklearn.KFold membership")
        for name in ("train", "test"):
            require(row_id_hash(f[f"{name}_row_ids"]) == f[f"{name}_hash"], f"fold {k} {name} hash mismatch")
    membership = canonical_hash({"T": splits["T_hash"], "V": splits["V_hash"],
                                 "folds": [(f["train_hash"], f["test_hash"]) for f in splits["folds"]]})
    require(membership == splits["membership_hash"], "overall membership hash mismatch")


def require_ok(splits: dict) -> None:
    if splits["split_status"] != "ok":
        raise StageError(splits["split_status"], f"dataset {splits['dataset']!r} is stopped before tuning: "
                         f"split_status={splits['split_status']} (see support_report.json)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validated split manifests, support report and hashes; no model fitting.")
    ap.add_argument("--dataset", required=True, help="dataset name (configs/datasets/<name>.yaml) or 'all'")
    ap.add_argument("--protocol", default=None, help="versioned protocol YAML (default: the canonical production protocol)")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--dataset-configs", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--dry-run", action="store_true", help="validate and report without writing anything")
    ap.add_argument("--smoke", action="store_true", help="use the separate smoke protocol")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.protocol, smoke=args.smoke)
    from sbtab.data.registry import available_datasets
    names = available_datasets(args.dataset_configs) if args.dataset == "all" else [args.dataset]
    rc = 0
    for name in names:
        try:
            s = run(name, protocol, args.output_root, args.dataset_configs, dry_run=args.dry_run)
        except StageError as e:
            s = {"dataset": name, "protocol_id": protocol.id, "split_status": e.status, "error": str(e)}
            rc = 1
        print(s)
    return rc


if __name__ == "__main__":
    sys.exit(main())
