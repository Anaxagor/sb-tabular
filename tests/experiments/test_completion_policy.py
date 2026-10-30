"""Completion must describe successful evaluation, not just existing files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbtab.experiments import aggregate_results as agg
from sbtab.experiments.experiment_common import read_json, write_json
from sbtab.experiments.status_policy import completion_summary, fold_completion


def record(fold, metric, value, status="ok", model="lightsb", dataset="toy"):
    return {"dataset": dataset, "model": model, "fold": fold, "metric": metric,
            "value": value, "status": status, "direction": "lower_is_better", "unit": "score"}


def evaluated(fold, *, utility_status="ok"):
    records = [record(fold, "marginal.groups.continuous.mean_wd", .2),
               record(fold, "utility.real.scores.r2", .8),
               record(fold, "utility.synth.scores_synth.r2", .7 if utility_status == "ok" else None,
                      utility_status)]
    return {"fold": fold, "status": "ok", "validity": {"status": "ok"},
            "utility": {"status": utility_status, "reference": {"status": "ok"},
                        "tstr": {"status": utility_status}}, "records": records}


def save_folds(directory, documents):
    for doc in documents:
        write_json(directory / f"fold-{doc['fold']}" / "metrics.json", doc)


@pytest.mark.parametrize("failure", ["utility_fit_failed", "invalid_generated_data", "sampling_failed"])
def test_summary_distinguishes_generation_fidelity_and_utility(tmp_path, failure):
    docs = [evaluated(k) for k in range(5)]
    if failure == "utility_fit_failed":
        docs[2] = evaluated(2, utility_status=failure)
    elif failure == "invalid_generated_data":
        docs[2] = {"fold": 2, "status": failure, "validity": {"status": failure},
                   "records": [record(2, "validity.n_invalid_rows", 1), record(2, "fidelity", None, failure)]}
    else:
        docs[2] = {"fold": 2, "status": failure, "records": [record(2, "fold_status", None, failure)]}
    save_folds(tmp_path, docs)
    summary = agg.aggregate_run(tmp_path)
    assert summary["generator_complete"] == (failure == "utility_fit_failed")
    assert summary["fidelity_complete"] == (failure == "utility_fit_failed")
    assert not summary["utility_complete"]
    assert not summary["evaluation_complete"]
    assert not summary["complete_five_fold"]
    assert summary["n_evaluation_complete"] == 4
    assert summary["fold_completion"]["2"]["problems"] == [failure]


def test_optional_undefined_metrics_and_finite_quality_warnings_do_not_fail_fold():
    doc = evaluated(0)
    doc["records"] += [record(0, "utility.synth.scores_synth.mape", None, "undefined"),
                       record(0, "conditional.js", .3, "incomplete_conditional_coverage"),
                       record(0, "association.pearson", None, "not_applicable")]
    doc["validity"]["numerical_diagnostics"] = {"warnings": ["extreme_finite_values"]}
    assert fold_completion(doc)["evaluation_complete"]
    # An entirely unavailable score set is distinct from an unavailable MAPE.
    doc["records"][2].update(value=None, status="undefined")
    assert not fold_completion(doc)["utility_complete"]


def test_not_applicable_utility_is_complete_but_missing_utility_is_unknown():
    doc = {"fold": 0, "status": "ok", "utility": {"status": "not_applicable"},
           "records": [record(0, "marginal.wd", .1), record(0, "utility", None, "not_applicable")]}
    assert fold_completion(doc)["evaluation_complete"]
    assert fold_completion(doc)["utility_applicable"] is False
    assert fold_completion({"records": [record(0, "wd", .1)]})["problems"] == ["utility_not_recorded"]


def test_nullable_legacy_metadata_uses_record_evidence():
    doc = evaluated(0)
    doc.update(validity=None, utility=None)
    assert fold_completion(doc)["evaluation_complete"]
    doc["utility"] = {"status": "ok", "reference": None, "tstr": None}
    assert fold_completion(doc)["evaluation_complete"]


@pytest.mark.parametrize("missing", ["utility.real.scores.", "utility.synth.scores_synth."])
@pytest.mark.parametrize("metadata", [True, False])
def test_missing_canonical_score_group_cannot_claim_utility_completion(missing, metadata):
    doc = evaluated(0)
    doc["records"] = [r for r in doc["records"] if not r["metric"].startswith(missing)]
    if not metadata:
        doc.pop("utility")
    result = fold_completion(doc)
    assert result["generator_complete"] and result["fidelity_complete"]
    assert not result["utility_complete"]
    assert not result["evaluation_complete"]
    assert "no_valid_utility_metric" in result["problems"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_success_records_are_not_complete_or_aggregated(tmp_path, value):
    docs = [evaluated(k) for k in range(5)]
    docs[2]["records"][0]["value"] = value
    # Model malformed legacy JSON explicitly: write_json correctly removes NaN/Inf.
    for doc in docs:
        path = tmp_path / f"fold-{doc['fold']}" / "metrics.json"
        path.parent.mkdir()
        path.write_text(json.dumps(doc))
    assert "nonfinite_metric" in fold_completion(docs[2])["problems"]
    summary = agg.aggregate_run(tmp_path)
    assert not summary["complete_five_fold"]
    metric = next(m for m in summary["metrics"] if m["metric"] == "marginal.groups.continuous.mean_wd")
    assert metric["n_valid"] == 4
    assert metric["mean"] == pytest.approx(.2)
    assert metric["fold_values"]["2"] is None
    saved = pd.read_csv(tmp_path / "per_fold.csv")
    assert not np.isinf(saved["value"].to_numpy()).any()


def test_finite_extreme_values_have_stable_mean_and_std():
    df = pd.DataFrame([record(0, "wd", 1e308), record(1, "wd", -1e308)])
    result = agg.summarise(df, 2)[0]
    assert result["mean"] == 0
    assert result["std"] == pytest.approx(2**.5 * 1e308)


def rank_rows(*, datasets=("toy",), n_folds=5):
    return pd.DataFrame([record(f, "wd", value, model=model, dataset=ds)
                         for ds in datasets for f in range(n_folds) for model, value in (("a", .1), ("b", .2))])


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_overall_ranks_drop_whole_incomplete_dataset_not_single_cell(bad):
    rows = rank_rows(datasets=("complete", "partial"))
    mask = rows.dataset.eq("partial") & rows.model.eq("a") & rows.fold.eq(4)
    rows.loc[mask, "value"] = bad
    result = agg.average_ranks(rows, "wd", models=["a", "b"])
    assert result["overall"]["datasets"] == ["complete"]
    assert result["overall"]["n_cells"] == 5
    assert result["overall"]["n_cells_dropped"] == 5
    assert result["per_dataset"]["partial"]["excluded_models"] == ["a"]


def test_missing_fifth_fold_from_every_model_is_not_a_complete_cohort():
    result = agg.average_ranks(rank_rows(n_folds=4), "wd")
    assert result["overall"]["status"] == "not_applicable"
    assert result["overall"]["n_expected_folds"] == 5
    assert result["per_dataset"]["toy"]["models"] == []
    assert result["per_dataset"]["toy"]["excluded_models"] == ["a", "b"]
    assert agg.average_ranks(rank_rows(n_folds=4), "wd", n_expected=4)["overall"]["n_cells"] == 4


@pytest.mark.parametrize("metadata", [True, False])
def test_root_aggregation_recomputes_legacy_completeness_without_mutation(tmp_path, metadata):
    directory = tmp_path / "toy/lightsb/run-old/evaluation/old"
    directory.mkdir(parents=True)
    docs = [evaluated(k, utility_status="utility_fit_failed" if k == 2 else "ok") for k in range(5)]
    pd.DataFrame([r for doc in docs for r in doc["records"]]).to_csv(directory / "per_fold.csv", index=False)
    write_json(directory / "summary.json", {"dataset": "toy", "model": "lightsb", "n_expected": 5,
                                            "complete_five_fold": True, "metrics": []})
    if metadata:
        # Metadata must override a stale successful CSV too.
        successful = [r for k in range(5) for r in evaluated(k)["records"]]
        pd.DataFrame(successful).to_csv(directory / "per_fold.csv", index=False)
        save_folds(directory, docs)
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    result = agg.aggregate_root(tmp_path)
    assert len(result["incomplete_runs"]) == 1
    assert result["runs"][0]["generator_complete"]
    assert not result["runs"][0]["utility_complete"]
    assert not result["runs"][0]["complete_five_fold"]
    assert {path: path.read_bytes() for path in before} == before


def test_complete_summary_requires_every_expected_fold_and_ignores_stale_flags():
    docs = {k: evaluated(k) for k in range(5)}
    assert completion_summary(docs)["complete_five_fold"]
    docs.pop(4)
    summary = completion_summary(docs)
    assert not summary["complete_five_fold"]
    assert summary["fold_completion"]["4"]["problems"] == ["not_evaluated"]
    stale = evaluated(0, utility_status="utility_fit_failed")
    stale["completion"] = {"evaluation_complete": True, "problems": []}
    assert not fold_completion(stale)["evaluation_complete"]


def test_pipeline_aggregate_does_not_trust_stale_successful_task(tmp_path):
    from sbtab.experiments import pipeline
    result = pipeline.create_plan(tmp_path, datasets=["diabetes"], models=["lightsb"], smoke=True, device="cpu")
    plan = pipeline.load_plan(result["plan"])
    directory = tmp_path / "diabetes/lightsb/run-pipeline/evaluation/old"
    save_folds(directory, [evaluated(k, utility_status="utility_fit_failed" if k == 2 else "ok") for k in range(5)])
    task_path = tmp_path / "pipeline/tasks/00000.json"
    write_json(task_path, {**plan["tasks"][0], "plan_hash": plan["plan_hash"], "status": "ok",
                           "stages": {"metrics": {"evaluation_dir": str(directory)}}})
    before = task_path.read_bytes()
    result = pipeline.aggregate(result["plan"])
    assert not result["complete"]
    assert result["counts"] == {"evaluation_failed": 1}
    assert task_path.read_bytes() == before
    summary = read_json(tmp_path / "pipeline/summary.json")
    assert summary["tasks"][0]["fold_problems"] == {"2": ["utility_fit_failed"]}
