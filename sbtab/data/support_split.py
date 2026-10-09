"""Deterministic same-stratum swaps to cover finite values in fixed KFold training sets.

Only split membership is changed. Rows, values, stratum sizes, seeds and KFold's
definition are preserved. This bounded greedy search can fail even when a solution
exists; callers must retain their full support checks and block unresolved datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from sbtab.data.eligibility import finite_levels


METHOD = "stratum_swap/1"


def validate_repair_rule(rule):
    if not isinstance(rule, dict) or set(rule) != {
        "method", "validation_coverage", "max_swaps", "max_candidate_evaluations"
    }:
        raise ValueError("support_repair requires method, validation_coverage, max_swaps and max_candidate_evaluations")
    if rule["method"] != METHOD:
        raise ValueError(f"unsupported support repair method: {rule['method']!r}")
    if rule["validation_coverage"] != "blocking_columns":
        raise ValueError("support_repair.validation_coverage must be blocking_columns")
    for key in ("max_swaps", "max_candidate_evaluations"):
        if isinstance(rule[key], bool) or not isinstance(rule[key], int) or rule[key] < 1:
            raise ValueError(f"support_repair.{key} must be a positive integer")


def repair_support_split(frame, schema, train_ids, validation_ids, strata, split_seed, cv, rule):
    """Return sorted T/V row IDs and an audit trail; never use model scores or fitted transforms."""
    validate_repair_rule(rule)
    rows = frame.index.to_numpy(dtype=np.int64)
    positions = {int(r): i for i, r in enumerate(rows)}
    train = np.sort([positions[int(r)] for r in train_ids])
    validation = np.sort([positions[int(r)] for r in validation_ids])
    strata = np.asarray(strata)
    tokens, token_columns, offset = [], [], 0
    for col in schema.finite_support:
        codes, levels = pd.factorize(finite_levels(frame, col, schema), sort=True)
        tokens.append(np.where(codes < 0, -1, codes + offset))
        token_columns.extend([col] * len(levels))
        offset += len(levels)
    # Numeric imputation also requires an observed value in every training set.
    if schema.missing_policy == "impute":
        for col in schema.continuous + schema.discrete:
            tokens.append(np.where(frame[col].notna(), offset, -1))
            token_columns.append(None)
            offset += 1
    codes = np.column_stack(tokens) if tokens else np.empty((len(frame), 0), dtype=int)
    fold_positions = [te for _, te in KFold(n_splits=cv["n_splits"], shuffle=cv["shuffle"],
                                            random_state=cv["random_state"]).split(train)]

    def counts(ids):
        values = codes[ids].ravel()
        return np.bincount(values[values >= 0], minlength=offset)

    total = counts(np.arange(len(frame)))
    validation_required = np.zeros(offset, dtype=bool)

    def score(ids):
        n_train = counts(ids)
        largest_fold = np.max([counts(ids[te]) for te in fold_positions], axis=0)
        # Every level needs >=2 T rows, spread over >=2 test folds, to occur in
        # every fold's training set. Count shortages so 0 -> 1 -> 2 can improve.
        deficit = np.maximum(2 - n_train, 0)
        deficit += (n_train >= 2) & (largest_fold == n_train)
        deficit += validation_required & (n_train == total)
        return int(deficit.sum()), deficit, n_train

    initial, deficits, n_train = score(train)
    validation_columns = sorted({token_columns[i] for i in np.flatnonzero(deficits)
                                 if token_columns[i] is not None})
    validation_required = np.isin(np.array(token_columns, dtype=object), validation_columns)
    initial, deficits, n_train = score(train)
    current = initial
    audit = {"method": METHOD, "seed": int(split_seed), "initial_deficit": initial,
             "final_deficit": initial, "candidate_evaluations": 0, "swaps": [],
             "status": "unchanged" if initial == 0 else "unresolved",
             "validation_columns": validation_columns}
    # Singleton levels cannot occur in the training part of every ordinary CV fold.
    infeasible = np.any(total < 2 + validation_required.astype(int))
    if initial == 0 or infeasible:
        if infeasible:
            audit["reason"] = "insufficient rows for training-fold and validation coverage"
        return rows[train], rows[validation], audit

    rng = np.random.default_rng(int(split_seed))
    for _ in range(rule["max_swaps"]):
        changed = False
        incoming = rng.permutation(validation)
        bad_codes = np.flatnonzero(deficits)
        priority = np.isin(codes[incoming], bad_codes).any(axis=1)
        incoming = np.concatenate([incoming[priority], incoming[~priority]])
        missing_validation = validation_required & (n_train == total)
        only_validation_missing = not np.any(deficits - missing_validation.astype(int))
        for add in incoming:
            outgoing = rng.permutation(train[strata[train] == strata[add]])
            # Prefer donors that do not remove scarce training support.
            fragile = np.isin(codes[outgoing], np.flatnonzero(n_train <= 2)).any(axis=1)
            outgoing = np.concatenate([outgoing[~fragile], outgoing[fragile]])
            fills_validation = np.isin(codes[outgoing], np.flatnonzero(missing_validation)).any(axis=1)
            # If training coverage is already satisfied, only a donor carrying a
            # missing validation level can reduce the deficit; skip irrelevant pairs.
            outgoing = (outgoing[fills_validation] if only_validation_missing else
                        np.concatenate([outgoing[fills_validation], outgoing[~fills_validation]]))
            for remove in outgoing:
                if audit["candidate_evaluations"] >= rule["max_candidate_evaluations"]:
                    break
                candidate = np.sort(np.append(train[train != remove], add))
                value, candidate_deficits, candidate_counts = score(candidate)
                audit["candidate_evaluations"] += 1
                if value < current:
                    validation = np.sort(np.append(validation[validation != add], remove))
                    train, current = candidate, value
                    deficits, n_train = candidate_deficits, candidate_counts
                    audit["swaps"].append({"to_training": int(rows[add]), "to_validation": int(rows[remove]),
                                            "deficit_after": value})
                    changed = True
                    break
            if changed or audit["candidate_evaluations"] >= rule["max_candidate_evaluations"]:
                break
        if current == 0 or not changed:
            break
    audit.update(final_deficit=current, status="repaired" if current == 0 else "unresolved")
    return rows[train], rows[validation], audit
