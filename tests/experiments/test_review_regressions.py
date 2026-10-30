"""Failure-path and artifact-integrity regressions from the post-refactor review."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sbtab.adapters.base import ModelAdapter
from sbtab.adapters.representation import NativeMixedRepresentation
from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.data.preprocessing import CommonPreprocessor
from sbtab.experiments import aggregate_results as agg, calculate_metrics as metrics, cross_validate as cv
from sbtab.experiments import prepare_splits as ps, runner, tune
from sbtab.experiments.experiment_common import SeedLedger, StageError, load_protocol, read_json, write_json
from test_stages import make_dataset


@pytest.mark.parametrize("change", ["values", "row_ids", "swapped_row_ids", "membership", "duplicate"])
def test_saved_split_artifacts_detect_changes(tmp_path, monkeypatch, change):
    _, path, _, _ = make_dataset(tmp_path, monkeypatch)
    if change in ("values", "row_ids", "swapped_row_ids"):
        f = pd.read_parquet(path.parent / "data.parquet")
        if change == "swapped_row_ids":
            f.loc[[0, 1], "row_id"] = [1, 0]
        else:
            f.loc[0, "x" if change == "values" else "row_id"] += 10000
        f.to_parquet(path.parent / "data.parquet", index=False)
    else:
        s = read_json(path)
        ids = s["folds"][0]["train_row_ids"]
        ids[0] = s["V_row_ids"][0] if change == "membership" else ids[1]
        write_json(path, s)
    with pytest.raises(StageError):
        ps.load_split_artifacts(path)


def test_prepare_splits_refuses_changed_values_with_same_membership(tmp_path, monkeypatch):
    root, _, frame, schema = make_dataset(tmp_path, monkeypatch)
    frame.loc[0, "x"] += 123
    from sbtab.data.loading import frame_fingerprint
    monkeypatch.setattr(ps, "load_dataset", lambda *a: (frame, schema, {
        "fingerprint": frame_fingerprint(frame), "missing_counts": {}}))
    with pytest.raises(StageError, match="fingerprint|data"):
        ps.run("toy", load_protocol(smoke=True), root)


def tuned_cv(tmp_path, monkeypatch):
    root, path, _, schema = make_dataset(tmp_path, monkeypatch)
    tune.run("toy", "mixedsbm", path, "configs/search_spaces/smoke/mixedsbm.yaml",
             resume=False, smoke=True, run_id="review")
    selected = path.parent / "mixedsbm" / "review" / "tuning" / "selected_config.json"
    result = cv.run("toy", "mixedsbm", selected, path, root, smoke=True, folds=[0])
    assert result["n_ok"] == 1
    return root, path, schema, selected, Path(result["cv_run_manifest"])


def test_cv_reuse_refuses_changed_implementation(tmp_path, monkeypatch):
    root, path, _, selected, manifest = tuned_cv(tmp_path, monkeypatch)
    before = manifest.read_bytes()
    changed = {**cv.source_provenance(), "source_hash": "changed"}
    monkeypatch.setattr(cv, "source_provenance", lambda: changed)
    with pytest.raises(StageError, match="compatib|implementation|provenance"):
        cv.run("toy", "mixedsbm", selected, path, root, smoke=True, folds=[1])
    assert manifest.read_bytes() == before


def test_cv_reuse_refuses_missing_output(tmp_path, monkeypatch):
    root, path, _, selected, manifest = tuned_cv(tmp_path, monkeypatch)
    (manifest.parent / "fold-0" / "synthetic.parquet").unlink()
    with pytest.raises(StageError, match="artifact|synthetic"):
        cv.run("toy", "mixedsbm", selected, path, root, smoke=True, folds=[0])


def test_partial_and_failed_evaluations_are_not_complete(tmp_path):
    record = {"fold": 0, "metric": "wd", "value": 1.0, "status": "ok", "direction": "lower_is_better", "unit": "x"}
    write_json(tmp_path / "fold-0" / "metrics.json", {"records": [record]})
    summary = agg.aggregate_run(tmp_path)
    assert summary["n_folds_with_generator_output"] == 1
    assert not summary["complete_five_fold"]


def test_failed_fold_record_survives_metric_aggregation(tmp_path):
    fold = tmp_path / "cv" / "fold-0"
    write_json(fold / "manifest.json", {"status": "training_failed", "failure": {"message": "boom"}})
    out = tmp_path / "evaluation"
    metrics.evaluate_fold(0, None, None, None, tmp_path, out, None, {}, {"model": "toy"}, tmp_path)
    records = agg.load_fold_records(out)
    assert len(records) == 1
    assert records.iloc[0]["metric"] == "fold_status"
    assert records.iloc[0]["status"] == "training_failed"


def test_ranks_reject_ambiguous_runs_instead_of_picking_first():
    rows = [{"dataset": "toy", "fold": 0, "model": model, "run_id": run, "metric": "wd", "value": v,
             "status": "ok", "direction": "lower_is_better"}
            for model, run, v in [("a", "old", 1), ("a", "new", 5), ("b", "only", 3)]]
    with pytest.raises(StageError, match="ambiguous|multiple|duplicate"):
        agg.average_ranks(pd.DataFrame(rows), "wd")


def test_requested_missing_model_is_not_silently_dropped_from_ranks():
    rows = pd.DataFrame([{"dataset": "toy", "fold": 0, "model": m, "metric": "wd", "value": v,
                          "status": "ok", "direction": "lower_is_better"} for m, v in [("a", 1), ("b", 2)]])
    ranked = agg.average_ranks(rows, "wd", models=["a", "b", "missing"])
    assert ranked["overall"]["status"] == "not_applicable"
    assert "missing" in ranked["overall"]["models"]


@pytest.mark.parametrize("bad", [0.5, -0.5, np.nan, np.inf])
def test_native_state_decoder_does_not_truncate_invalid_ids(bad):
    schema = DatasetSchema("toy", (ColumnSpec("c", "categorical"),))
    rep = NativeMixedRepresentation(schema).fit(pd.DataFrame({"c": [0, 1]}))
    out, report = rep.decode(np.empty((1, 0)), np.array([[bad]]))
    assert out["c"].isna().all()
    assert report["invalid_state_rate"]["c"] == 1.0


def test_serialization_preserves_invalid_category_codes_for_validity(tmp_path):
    schema = DatasetSchema("toy", (ColumnSpec("c", "categorical"),))
    f = pd.DataFrame({"c": [0.5, -0.5, np.inf]}, index=pd.RangeIndex(3, name="synthetic_id"))
    runner.write_synthetic(tmp_path / "synthetic.parquet", f, schema)
    pd.testing.assert_frame_equal(runner.read_synthetic(tmp_path / "synthetic.parquet", schema), f)


def test_inverse_preprocessing_rejects_fractional_category_codes():
    schema = DatasetSchema("toy", (ColumnSpec("c", "categorical"),))
    pre = CommonPreprocessor(schema).fit(pd.DataFrame({"c": ["a", "b"]}))
    with pytest.raises(ValueError, match="integer|integral|fraction"):
        pre.inverse_transform(pd.DataFrame({"c": [0.5]}))


@pytest.mark.parametrize("stage", ["_prepare", "_build_model", "_fit_model", "_generate", "_decode"])
def test_failed_operations_keep_timing_and_invalidate_failed_fit(tmp_path, monkeypatch, stage):
    from sbtab.adapters.native import MixedSBMAdapter
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"),))
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 20)})
    def fail(*a, **kw):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(MixedSBMAdapter, stage, fail)
    rec = runner.fit_generate("mixedsbm", frame, schema, {"num_steps": 2, "n_stages": 1, "epochs_per_direction": 1},
                              SeedLedger(5), ("trial", 0), 10, tmp_path)
    assert rec["status"] == ("sampling_failed" if stage in ("_generate", "_decode") else "training_failed")
    key = {"_prepare": "generator_fit_seconds", "_fit_model": "generator_fit_seconds", "_build_model": "model_init_seconds",
           "_generate": "generation_seconds", "_decode": "inverse_transform_seconds"}[stage]
    assert rec["timing"][key] > 0


def test_failed_refit_cannot_sample_previous_or_partial_state(monkeypatch):
    from sbtab.adapters.neural import LightSBAdapter
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"),))
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 20)})
    adapter = LightSBAdapter().fit(frame, schema, {"max_iter": 1, "n_potentials": 2}, seed=0)
    monkeypatch.setattr(adapter, "_fit_model", lambda: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        adapter.fit(frame, schema, {"max_iter": 1, "n_potentials": 2}, seed=0)
    assert not adapter.fitted
    assert adapter._encoded is None
    with pytest.raises(RuntimeError, match="fit"):
        adapter.sample(1, seed=0)


def test_undefined_utility_score_status_is_preserved():
    result = list(
        metrics.flatten({"status": "ok", "scores": {"r2": None}, "score_status": {"r2": "undefined"}}, "utility.real"))
    assert ("utility.real.scores.r2", None, "undefined") in result


def test_utility_gap_metadata_has_correct_direction_and_units():
    assert metrics.metric_meta("utility.synth.gaps.macro_f1.delta_pct")[0] == "lower_is_better"
    assert metrics.metric_meta("utility.synth.gaps.r2.abs_gap")[0] == "lower_is_better"
    assert metrics.metric_meta("utility.synth.gaps.mae.synth") == ("lower_is_better", "raw target units")


def test_resume_refuses_missing_sampler_state(tmp_path, monkeypatch):
    root, path, _, _ = make_dataset(tmp_path, monkeypatch)
    space = "configs/search_spaces/smoke/mixedsbm.yaml"
    out = tune.run("toy", "mixedsbm", path, space, resume=False, smoke=True, max_new_trials=1)
    (Path(out["run_dir"]) / "tuning" / "sampler.pkl").unlink()
    with pytest.raises(StageError, match="sampler"):
        tune.run("toy", "mixedsbm", path, space, resume=True, smoke=True)


def test_utility_initialization_failure_keeps_fidelity_metrics(tmp_path, monkeypatch):
    _, _, _, _, manifest = tuned_cv(tmp_path, monkeypatch)
    from sbtab import evaluation
    def fail(*a, **kw):
        raise RuntimeError("utility initialization failed")
    monkeypatch.setattr(evaluation, "resolve_utility_params", fail)
    result = metrics.run(manifest, "configs/metrics/metrics_v2.yaml", folds=[0])
    saved = read_json(Path(result["evaluation_dir"]) / "fold-0" / "metrics.json")
    assert saved["marginal"]["status"] == "ok"
    assert saved["utility"]["status"] == "utility_fit_failed"
    assert any(r["metric"].startswith("generator.") for r in saved["records"])


@pytest.mark.parametrize("kind", ["ct", "dt"])
def test_ipf_dropout_fit_is_reproducible_and_preserves_caller_rng(kind):
    import torch
    from importlib import import_module
    module = import_module(f"sbtab.solvers.{'continuous_time' if kind == 'ct' else 'discrete_time'}.joint_distribution.mlp.ipf_dsb.solver")
    config = module.IPFDSBConfig(num_steps=3, horizon=0.5, hidden_units=8, n_layers=2, time_features=8,
                                 batch_size=8, cache_batches=1, ipf_iters=1, dropout=0.4, steps_per_phase=2)
    data = np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32)
    torch.manual_seed(812)
    before = torch.get_rng_state().clone()
    a = module.IPFDSBSolver(2, config).fit(data)
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(991)
    b = module.IPFDSBSolver(2, config).fit(data)
    np.testing.assert_array_equal(a.sample(8, seed=4), b.sample(8, seed=4))


def test_csbm_reload_retains_training_update_counts(tmp_path):
    import torch
    from sbtab.solvers.csbm import CSBMConfig, CSBMSolver
    s = CSBMSolver([2], [False], CSBMConfig(num_steps=2, num_outer_iterations=1, epochs=1,
                                         hidden_dim=8, emb_dim=4, time_dim=4)).fit(torch.tensor([[0], [1]] * 10))
    s.save_checkpoint(tmp_path / "model.pt")
    assert CSBMSolver.load_checkpoint(tmp_path / "model.pt").n_updates == s.n_updates > 0


@pytest.mark.parametrize("kind", ["csbm", "mixedsbm"])
def test_zero_training_budget_is_rejected(kind):
    from sbtab.solvers.csbm import CSBMConfig
    from sbtab.solvers.msbm import MixedSBMConfig
    with pytest.raises(ValueError, match="epochs"):
        CSBMConfig(epochs=0) if kind == "csbm" else MixedSBMConfig(epochs_per_direction=0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_timegrid_rejects_nonfinite_horizons_and_points(value):
    from sbtab.bridge.timegrid import TimeGrid
    with pytest.raises(ValueError):
        TimeGrid.uniform(3, horizon=value)
    with pytest.raises(ValueError):
        TimeGrid.from_points([0, 0.5, value])


def test_uniform_reference_does_not_allocate_quadratic_storage(monkeypatch):
    import torch
    from sbtab.bridge.reference import CategoricalReference
    from sbtab.bridge.timegrid import TimeGrid
    original = torch.eye
    def guarded(n, *a, **kw):
        assert n < 10000, "uniform kernel allocated a quadratic identity matrix"
        return original(n, *a, **kw)
    monkeypatch.setattr(torch, "eye", guarded)
    ref = CategoricalReference([10000], torch.tensor([False]), TimeGrid.uniform(2))
    p = ref._kernels[0].row("step", torch.tensor([0]), torch.tensor([3]))
    assert p.shape == (1, 10000)
    assert p.sum().item() == pytest.approx(1.0)


def test_conditional_partial_scores_retain_coverage_status_in_flat_records():
    from sbtab.evaluation import MetricContext, conditional_metrics
    schema = DatasetSchema("toy", (ColumnSpec("c", "categorical"), ColumnSpec("x", "continuous")))
    real = pd.DataFrame({"c": [0] * 20 + [1] * 20, "x": list(range(20)) * 2})
    summary, _ = conditional_metrics(MetricContext.fit(real, schema), real, real.iloc[:20])
    records = {name: status for name, _, status in metrics.flatten(summary, "conditional")}
    assert records["conditional.conditioners.c.wd_block.weighted_mean"] == "incomplete_conditional_coverage"
    assert records["conditional.conditioners.c.responses.x.weighted_mean"] == "incomplete_conditional_coverage"


@pytest.mark.parametrize("key,value", [("kl_total_bin", 30), ("kl_direction", "synthetic_to_real")])
def test_metric_config_does_not_silently_ignore_settings(key, value):
    from sbtab.evaluation import MetricConfig
    cfg = yaml.safe_load(Path("configs/metrics/metrics_v2.yaml").read_text())
    cfg[key] = value
    with pytest.raises(ValueError):
        MetricConfig.from_document(cfg)


@pytest.mark.parametrize("bad", [0.5, -0.5, np.nan, np.inf])
def test_baseline_vocabulary_decoder_rejects_invalid_state_ids(bad):
    from sbtab.baselines.encoding import decode_with_vocabulary
    with pytest.raises(ValueError):
        decode_with_vocabulary(np.array([bad]), ["a", "b"])


def test_failing_preprocessor_refit_invalidates_transform():
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"),))
    pre = CommonPreprocessor(schema).fit(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        pre.fit(pd.DataFrame({"x": [1.0, np.inf]}))
    assert not pre.fitted


@pytest.mark.parametrize("kind", ["csbm", "mixedsbm"])
def test_native_solvers_reject_fractional_training_categories(kind):
    import torch
    from sbtab.solvers.csbm import CSBMConfig, CSBMSolver
    from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver
    invalid = torch.tensor([[0.5], [1.0]])
    with pytest.raises(ValueError, match="integers"):
        if kind == "csbm":
            CSBMSolver([2], [False], CSBMConfig(num_steps=2)).fit(invalid)
        else:
            MixedSBMSolver(0, [2], [False], MixedSBMConfig(num_steps=2)).fit(None, invalid)


def test_mixed_dropout_fit_is_seeded_independently_of_callers():
    import torch
    from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver
    cfg = MixedSBMConfig(num_steps=2, fb_sequence=("b",), epochs_per_direction=1,
                         hidden_dim=8, n_layers=2, time_dim=4, dropout=0.5)
    a, b = [MixedSBMSolver(1, [], [], cfg) for _ in range(2)]
    data = torch.linspace(-1, 1, 20).reshape(-1, 1)
    torch.manual_seed(123)
    a.fit(data)
    torch.manual_seed(987)
    b.fit(data)
    before = torch.get_rng_state().clone()
    torch.testing.assert_close(a.sample(8, seed=1)[0], b.sample(8, seed=1)[0], rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
