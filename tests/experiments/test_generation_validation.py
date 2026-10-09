"""Failures must be diagnosed before invalid generators enter model selection."""
import numpy as np
import optuna
import pandas as pd
import pytest

from sbtab.adapters.base import ModelAdapter
from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.data.preprocessing import CommonPreprocessor
from sbtab.experiments import runner, tune
from sbtab.experiments.experiment_common import SeedLedger, read_json, write_json


class FaultAdapter(ModelAdapter):
    registry_id = "fault_test"
    DEFAULTS = {"mode": "ok"}

    def _prepare(self, train):
        pass

    def _build_model(self):
        self.loaded = False
        self.calls = 0

    def _fit_model(self):
        if self.config["mode"] == "bad_training":
            from sbtab.numerics import NumericalError
            raise NumericalError("non-finite loss", model="fault_test", step=3, direction="backward")

    def _generate(self, n, seed):
        self.calls += 1
        result = np.zeros((n, len(self.schema.columns)))
        mode = self.config["mode"]
        if (mode == "all_nan" or mode == "bad_final_probe" and n == 2
                or mode == "bad_original_probe" and self.calls > 1 and not self.loaded
                or mode == "bad_reloaded_probe" and self.loaded):
            result[:, 0] = np.nan
        elif mode == "bad_category":
            result[:, 1] = .5
        elif mode == "extreme":
            result[:, 0] = 1e100
        elif mode == "mismatch" and self.loaded:
            result[:, 0] = 2.0
        return result

    def _decode(self, generated):
        frame = pd.DataFrame(generated, columns=self.schema.column_order)
        if self.config["mode"] == "mixed_type_category":
            frame["c"] = ["broken"] + [0] * (len(frame) - 1)
        return frame

    def _save_model(self, directory):
        pass

    def _load_model(self, directory, state):
        if self.config["mode"] == "bad_load":
            raise OSError("broken checkpoint")
        self.loaded = True
        self.calls = 0


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(runner, "get_adapter_class", lambda _: FaultAdapter)
    monkeypatch.setattr(tune, "get_adapter_class", lambda _: FaultAdapter)
    schema = DatasetSchema("fault_test", (ColumnSpec("x", "continuous"), ColumnSpec("c", "categorical")))
    frame = pd.DataFrame({"x": np.linspace(-2, 2, 20), "c": ["a", "b"] * 10})
    return frame, schema


def run_fault(setup, tmp_path, mode, **kwargs):
    frame, schema = setup
    return runner.fit_generate("fault_test", frame, schema, {"mode": mode}, SeedLedger(5),
                               ("trial", 0), 10, tmp_path, **kwargs)


@pytest.mark.parametrize("verify_reload", [False, True])
@pytest.mark.parametrize("mode,reason", [("all_nan", "nonfinite_numeric_values"),
                                         ("bad_category", "unexpected_categorical_labels")])
def test_invalid_main_sample_never_passes_and_keeps_diagnostics(setup, tmp_path, mode, reason, verify_reload):
    rec = run_fault(setup, tmp_path, mode, verify_reload=verify_reload)
    assert rec["status"] == "invalid_generated_data"
    assert reason in rec["validity"]["reasons"]
    assert rec["failure"]["stage"] == "generation"
    assert rec["checkpoint"] == "checkpoint" and rec["reload_verified"] is None
    saved = runner.read_synthetic(tmp_path / "synthetic.parquet", setup[1])
    assert saved["x"].isna().all() if mode == "all_nan" else (saved["c"] == .5).all()


@pytest.mark.parametrize("mode,label", [("bad_original_probe", "original"), ("bad_reloaded_probe", "reloaded")])
def test_each_probe_is_validated_before_comparing(setup, tmp_path, mode, label):
    rec = run_fault(setup, tmp_path, mode)
    assert rec["validity"]["status"] == "ok"
    assert rec["status"] == "sampling_failed" and rec["failure_kind"] == "sampling_probe_failed"
    assert rec["checkpoint_loaded"] is True and rec["reload_verified"]["ok"] is False
    assert rec["sampling_probe"][label]["status"] == "invalid_generated_data"
    assert rec["sampling_probe"]["active_model"] == label


@pytest.mark.parametrize("mode,kind,loaded", [("bad_load", "checkpoint_load_failed", False),
                                            ("mismatch", "checkpoint_mismatch", True)])
def test_checkpoint_failure_causes_are_distinct(setup, tmp_path, mode, kind, loaded):
    rec = run_fault(setup, tmp_path, mode)
    assert rec["status"] == "sampling_failed" and rec["failure_kind"] == kind
    assert rec["checkpoint_loaded"] is loaded and rec["reload_verified"]["ok"] is False


def test_finite_extreme_data_is_flagged_without_clipping_or_rejection(setup, tmp_path):
    rec = run_fault(setup, tmp_path, "extreme")
    assert rec["status"] == "ok" and rec["reload_verified"]["ok"]
    assert rec["numerical_diagnostics"]["warnings"] == [{"column": "x", "reason": "extreme_finite_values", "count": 10}]
    assert rec["numerical_diagnostics"]["affects_validity_or_selection"] is False
    assert (runner.read_synthetic(tmp_path / "synthetic.parquet", setup[1])["x"] == 1e100).all()


def test_unserializable_category_keeps_invalid_data_as_primary_cause(setup, tmp_path):
    rec = run_fault(setup, tmp_path, "mixed_type_category")
    assert rec["status"] == "invalid_generated_data" and rec["failure_kind"] == "invalid_generated_data"
    assert rec["failure"]["type"] == "InvalidGeneratedData"
    assert rec["serialization_failure"]["stage"] == "synthetic_save"
    assert "unexpected_categorical_labels" in rec["validity"]["reasons"]


def test_structured_numerical_context_survives_runner_failure(setup, tmp_path):
    rec = run_fault(setup, tmp_path, "bad_training")
    assert rec["status"] == "training_failed"
    assert rec["failure"]["details"] == {"model": "fault_test", "step": 3, "direction": "backward"}


@pytest.mark.parametrize("mode,kind", [("bad_load", "checkpoint_load_failed"),
                                      ("bad_final_probe", "sampling_probe_failed"), ("ok", None)])
def test_final_selection_keeps_exact_minimum_and_distinguishes_load_from_probe(setup, tmp_path, mode, kind):
    frame, schema = setup
    study = optuna.create_study(direction="minimize")
    for trial, (value, trial_mode) in enumerate(((.1, mode), (.2, "ok"))):
        study.add_trial(optuna.trial.create_trial(value=value))
        d = tmp_path / f"trial-{trial:03d}"
        pre = CommonPreprocessor(schema).fit(frame)
        pre.save(d / "preprocessor")
        adapter = FaultAdapter().fit(pre.transform(frame), schema, {"mode": trial_mode}, seed=0)
        adapter.save_checkpoint(d / "checkpoint")
        write_json(d / "config.json", {"effective_config": adapter.config})
    write_json(tmp_path / "selected_config.json", {"stale": True})
    out = tune.write_best(study, tmp_path, 2, True, "fault_test", {})
    assert out["trial"] == 0 and out["objective"] == .1
    assert out["failure_kind"] == kind
    if kind is None:
        assert out["selection"] == "final" and out["reload_verified"] is True
        assert read_json(tmp_path / "selected_config.json")["trial"] == 0
    else:
        assert out["selection"] == "failed" and out["reload_verified"] is False
        assert not (tmp_path / "selected_config.json").exists()
    if mode == "bad_load":
        assert out["checkpoint_loaded"] is False and out["sampling_probe"] is None
    elif mode == "bad_final_probe":
        assert out["checkpoint_loaded"] is True and out["sampling_probe"]["ok"] is False
        assert "loaded; sampling_probe failed" in out["reason"]
