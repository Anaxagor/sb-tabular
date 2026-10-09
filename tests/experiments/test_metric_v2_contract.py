"""Canonical metric policy, saved provenance and strictly fold-local evaluation."""
from pathlib import Path
import hashlib

import pytest

from sbtab import evaluation
from sbtab.data.preprocessing import CommonPreprocessor
from sbtab.experiments import calculate_metrics, cross_validate, tune
from sbtab.experiments.experiment_common import load_yaml, read_json
from test_stages import make_dataset


METRICS = "configs/metrics/metrics_v2.yaml"


def test_metric_v2_rejects_historical_signed_gap_definition():
    current = evaluation.MetricConfig.from_document(load_yaml(METRICS))
    assert current.kl_total_bins == 50
    assert evaluation.METRIC_VERSION == "sbtab.metrics/2"
    with pytest.raises(ValueError, match="unsupported metric version"):
        evaluation.MetricConfig.from_document(load_yaml("configs/metrics/metrics_v1.yaml"))
    assert calculate_metrics.metric_meta("utility.synth.gaps.macro_f1.delta_pct") == (
        "lower_is_better", "percent absolute difference from real reference")


def test_later_fold_metrics_never_fit_preprocessing_or_probe_another_fold(tmp_path, monkeypatch):
    root, splits_path, _, _ = make_dataset(tmp_path, monkeypatch, n=120)
    splits = read_json(splits_path)
    tuned = tune.run("toy", "mixedsbm", splits_path, "configs/search_spaces/smoke/mixedsbm.yaml",
                     resume=False, smoke=True)
    selected = Path(tuned["run_dir"]) / "tuning" / "selected_config.json"
    cv = cross_validate.run("toy", "mixedsbm", selected, splits_path, root, smoke=True, folds=[1])

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("metrics must load the saved T_k transform, never fit another fold")
    monkeypatch.setattr(CommonPreprocessor, "fit", forbidden_fit)
    original = evaluation.resolve_utility_params
    calls = []

    def bounded_params(task, X, y, cats, cfg):
        calls.append(list(X.index))
        return original(task, X, y, cats, cfg, overrides={"iterations": 5, "depth": 2})
    monkeypatch.setattr(evaluation, "resolve_utility_params", bounded_params)
    result = calculate_metrics.run(cv["cv_run_manifest"], METRICS, folds=[1])
    out = Path(result["evaluation_dir"]) / "fold-1"
    metrics = read_json(out / "metrics.json")
    assert metrics["marginal"]["status"] == "ok"
    assert metrics["utility"]["status"] == "ok"
    assert calls == [splits["folds"][1]["train_row_ids"]]
    ctx = read_json(out / "metric_context.json")
    row_payload = "\n".join(map(str, splits["folds"][1]["train_row_ids"]))
    assert ctx["fit_row_hash"] == hashlib.sha256(row_payload.encode()).hexdigest()
    assert ctx["metric_version"] == "sbtab.metrics/2"
    utility = metrics["utility"]
    assert utility["reference"]["params"] == utility["tstr"]["params"]
    assert utility["reference"]["parameter_policy"] == "fixed_cpu_defaults_v1"
    gap = utility["tstr"]["gaps"]["macro_f1"]
    real, synthetic = gap["real"], gap["synth"]
    assert gap["abs_gap"] == pytest.approx(abs(real - synthetic))
    assert gap["delta_pct"] == pytest.approx(100 * abs(real - synthetic) / abs(real))
