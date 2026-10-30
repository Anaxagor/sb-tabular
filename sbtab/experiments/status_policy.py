"""One completion policy for workers and saved-result summaries.

Completion is distinct from model quality. Undefined individual metrics (for
example MAPE at zero targets) and inapplicable metric blocks are not failures.
Record-only historical results cannot establish utility completion when no
utility evidence was saved; a stale summary flag is never used as evidence.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

COMPLETION_VERSION = "sbtab.fold-completion/1"
VALUE_STATUSES = frozenset({"ok", "incomplete_conditional_coverage"})
OPTIONAL_METRIC_STATUSES = frozenset({"not_applicable", "undefined", "insufficient_data"})


def finite_metric_value(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _record_problems(records):
    problems = set()
    for record in records:
        status = record.get("status", "undefined")
        if status not in VALUE_STATUSES | OPTIONAL_METRIC_STATUSES:
            problems.add(status)
        if status in VALUE_STATUSES and record.get("value") is not None and not finite_metric_value(record["value"]):
            problems.add("nonfinite_metric")
    return problems


def fold_completion(document: Mapping | None) -> dict:
    """Recompute completion from a fold document, ignoring saved completion flags."""
    result = {"version": COMPLETION_VERSION, "generator_complete": False,
              "fidelity_complete": False, "utility_complete": False,
              "utility_applicable": None, "evaluation_complete": False, "problems": []}
    if not document:
        return {**result, "problems": ["not_evaluated"]}
    records = document.get("records") or []
    status = document.get("status")
    failures = {r.get("status", "undefined") for r in records
                if r.get("metric") == "fold_status" and r.get("status") not in VALUE_STATUSES}
    if status is not None and status not in VALUE_STATUSES:
        failures.add(status)
    validity = (document.get("validity") or {}).get("status")
    if validity is not None and validity != "ok":
        failures.add(validity)
    if failures:
        return {**result, "problems": sorted(failures)}
    if status is None and not records:
        return {**result, "problems": ["not_evaluated"]}
    result["generator_complete"] = True

    fidelity = [r for r in records if not str(r.get("metric", "")).startswith(
        ("utility", "validity", "generator.", "evaluation.")) and r.get("metric") != "fold_status"]
    fidelity_problems = _record_problems(fidelity)
    fidelity_values = any(r.get("status") in VALUE_STATUSES and finite_metric_value(r.get("value")) for r in fidelity)
    if not fidelity_values:
        fidelity_problems.add("no_valid_fidelity_metric")
    result["fidelity_complete"] = not fidelity_problems

    utility_records = [r for r in records if str(r.get("metric", "")).startswith("utility")]
    utility = document.get("utility") or {}
    utility_status = utility.get("status")
    utility_problems = _record_problems(utility_records)
    if utility_status == "not_applicable" or (not utility and utility_records and
            all(r.get("status") == "not_applicable" for r in utility_records)):
        result["utility_applicable"] = False
    elif utility or utility_records:
        result["utility_applicable"] = True
        if utility_status is not None and utility_status != "ok":
            utility_problems.add(utility_status)
        for component in ("reference", "tstr"):
            component_status = (utility.get(component) or {}).get("status")
            if component_status is not None and component_status != "ok":
                utility_problems.add(component_status)
        # Status-only legacy metadata can establish a completed utility fit.
        # With records, both evaluated score sets need an available score; optional undefined
        # MAPE/gaps do not erase valid MAE/RMSE/R2 or classification scores.
        scores = [r for r in utility_records if ".scores" in str(r.get("metric", "")) or r.get("metric") == "utility"]
        score_groups = [[r for r in scores if str(r.get("metric", "")).startswith(prefix)]
                        for prefix in ("utility.real.scores.", "utility.synth.scores_synth.")]
        if utility_status != "ok" or scores:
            # Once canonical score evidence is present, a missing counterpart
            # is truncated evaluation evidence, not an inapplicable metric.
            required_groups = score_groups if any(score_groups) else [scores]
            if any(not any(r.get("status") in VALUE_STATUSES and finite_metric_value(r.get("value"))
                           for r in group) for group in required_groups):
                if not utility_problems:
                    utility_problems.add("no_valid_utility_metric")
            if utility_status is None and not score_groups[1]:
                utility_problems.add("utility_not_recorded")
    else:
        utility_problems.add("utility_not_recorded")
    result["utility_complete"] = not utility_problems
    result["evaluation_complete"] = result["fidelity_complete"] and result["utility_complete"]
    result["problems"] = sorted(fidelity_problems | utility_problems)
    return result


def completion_summary(documents: Mapping[int, Mapping], n_expected: int = 5) -> dict:
    """Summarize all expected folds, including ones absent from every input."""
    if isinstance(n_expected, bool) or not isinstance(n_expected, int) or n_expected < 1:
        raise ValueError("n_expected must be a positive integer")
    unexpected = set(documents) - set(range(n_expected))
    if unexpected:
        raise ValueError(f"unexpected fold indices: {sorted(unexpected)}")
    folds = {str(k): fold_completion(documents.get(k)) for k in range(n_expected)}
    summary = {"completion_version": COMPLETION_VERSION, "n_expected": n_expected, "fold_completion": folds}
    for stage in ("generator", "fidelity", "utility", "evaluation"):
        count = sum(f[f"{stage}_complete"] for f in folds.values())
        summary[f"n_{stage}_complete"] = count
        summary[f"{stage}_complete"] = count == n_expected
    summary["n_folds_with_generator_output"] = summary["n_generator_complete"]
    # Compatibility name now means successful evaluation, not mere output presence.
    summary["complete_five_fold"] = summary["evaluation_complete"]
    return summary
