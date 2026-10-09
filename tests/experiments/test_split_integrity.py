"""Reject plausible but false split metadata before any tuning or evaluation."""
import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.data.loading import frame_fingerprint
from sbtab.experiments import prepare_splits as ps
from sbtab.experiments.experiment_common import Protocol, StageError, load_protocol, read_json, write_json


def toy():
    n = 120
    frame = pd.DataFrame({"x": np.arange(n, dtype=float), "c": ["a", "b", "c"] * (n // 3),
                          "y": ["no", "yes"] * (n // 2)}, index=pd.RangeIndex(n, name="row_id"))
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"), ColumnSpec("c", "categorical"),
                                   ColumnSpec("y", "categorical", role="target")), task="classification")
    return frame, schema


def build(frame, schema, protocol=None):
    return ps.build_splits(frame, schema, {"fingerprint": frame_fingerprint(frame), "missing_counts": {}},
                           protocol or load_protocol())[0]


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("section,key,value", [
    ("split", "stratify", False), ("split", "stratify", 1), ("split", "pool_order", "random"),
    ("split", "random_state", 42), ("split", "test_size", .2),
    ("split", "regression_strata_bins", 5), ("cv", "population", "all"),
    ("cv", "shuffle", 1), ("cv", "random_state", 5), ("cv", "n_generated", "len_test_fold"),
    ("tuning", "sampler", "random"), ("tuning", "direction", "maximize"), ("seeds", "base", 42),
    ("tuning", "n_generated", "len_train"), ("preprocessing", "tuning_fit_population", "all"),
    ("preprocessing", "cv_fit_population", "T"),
])
def test_production_and_smoke_enforce_the_same_data_and_evaluation_contract(tmp_path, smoke, section, key, value):
    data = copy.deepcopy(load_protocol(smoke=smoke).data)
    data[section][key] = value
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="canonical constants"):
        load_protocol(path, smoke=smoke)


@pytest.mark.parametrize("budget", [0, True, 2.5, 100])
def test_smoke_budget_is_explicitly_bounded(tmp_path, budget):
    data = copy.deepcopy(load_protocol(smoke=True).data)
    data["tuning"]["n_trials"] = budget
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="tuning.n_trials"):
        load_protocol(path, smoke=True)


def test_version_two_contract_cannot_omit_preprocessing_populations(tmp_path):
    data = copy.deepcopy(load_protocol().data)
    del data["preprocessing"]
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="explicit preprocessing"):
        load_protocol(path)


def test_unknown_kind_cannot_bypass_canonical_constraints(tmp_path):
    data = copy.deepcopy(load_protocol().data)
    data["kind"] = "custom"
    data["tuning"]["n_trials"] = 1
    data["split"]["random_state"] = 99
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="accepts only kind"):
        load_protocol(path)


def test_production_and_smoke_only_differ_in_trial_budget_and_identity():
    production, smoke = load_protocol(), load_protocol(smoke=True)
    for key in ("version", "eligibility", "split", "preprocessing", "cv", "seeds", "metrics_config"):
        assert production[key] == smoke[key]
    assert {k: v for k, v in production["tuning"].items() if k != "n_trials"} == {
        k: v for k, v in smoke["tuning"].items() if k != "n_trials"}


def test_same_size_self_consistent_wrong_holdout_is_rejected():
    frame, schema = toy()
    data = copy.deepcopy(load_protocol().data)
    data["split"]["random_state"] = 6
    wrong = build(frame, schema, Protocol("other seed", data))
    # All hashes and five KFold memberships are internally consistent, but these
    # are not the rows produced by the claimed seed. Hash checking alone passed.
    wrong["split"]["random_state"] = 5
    with pytest.raises(StageError, match="T/V membership"):
        ps.validate_split_artifacts(frame, wrong, schema)


@pytest.mark.parametrize("rare", [False, True])
def test_saved_support_status_is_recomputed_from_values(rare):
    frame, schema = toy()
    if rare:
        frame.loc[0, "c"] = "singleton"
    split = build(frame, schema)
    assert split["split_status"] == ("blocked_support" if rare else "ok")
    split["split_status"] = "ok" if rare else "blocked_support"
    with pytest.raises(StageError, match="support status mismatch"):
        ps.validate_split_artifacts(frame, split, schema)


def test_recomputed_alternative_seed_cannot_impersonate_the_frozen_protocol():
    frame, schema = toy()
    canonical = load_protocol()
    data = copy.deepcopy(canonical.data)
    data["split"]["random_state"] = 6
    wrong = build(frame, schema, Protocol("other seed", data))
    with pytest.raises(StageError, match="split.random_state differs"):
        ps.validate_split_artifacts(frame, wrong, schema, canonical)


def test_unrepaired_regression_artifacts_without_new_bin_field_still_verify():
    frame = pd.DataFrame({"x": np.arange(120, dtype=float), "y": np.arange(120, dtype=float) ** 2},
                         index=pd.RangeIndex(120, name="row_id"))
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"), ColumnSpec("y", "continuous", role="target")),
                           task="regression")
    split = build(frame, schema)
    split["split"].pop("regression_strata_bins")
    ps.validate_split_artifacts(frame, split, schema, load_protocol())


def test_saved_protocol_marker_is_checked_before_loading_data(tmp_path, monkeypatch):
    frame, schema = toy()
    monkeypatch.setattr(ps, "load_dataset", lambda *args: (frame, schema,
        {"fingerprint": frame_fingerprint(frame), "missing_counts": {}}))
    ps.run("toy", load_protocol(), tmp_path)
    marker = read_json(tmp_path / "protocol.json")
    marker["protocol"]["split"]["random_state"] = 6
    write_json(tmp_path / "protocol.json", marker)
    with pytest.raises(StageError, match="protocol.json does not match"):
        ps.load_split_artifacts(tmp_path / "toy" / "splits.json")


def test_stage_zero_cli_reports_blocked_support_as_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "run", lambda *args, **kwargs: {"split_status": "blocked_support"})
    assert ps.main(["--dataset", "toy", "--output-root", str(tmp_path), "--dry-run"]) == 1
