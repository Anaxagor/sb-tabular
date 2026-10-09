"""Versioned coverage repair must preserve rows, strata, ordinary KFold and auditability."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.data.loading import frame_fingerprint
from sbtab.data.support_split import validate_repair_rule
from sbtab.experiments import pipeline, prepare_splits as ps
from sbtab.experiments.experiment_common import Protocol, StageError, load_protocol, read_json, write_json


V3 = "configs/protocols/sbtab_8515_hpo100_cv5_v3.yaml"
SMOKE_V3 = "configs/protocols/sbtab_smoke_v3.yaml"


def assert_partition(frame, schema, old, new):
    assert new["n_T"] == old["n_T"] and new["n_V"] == old["n_V"]
    T, V = set(new["T_row_ids"]), set(new["V_row_ids"])
    assert T | V == set(frame.index) and not T & V
    labels = frame[schema.target]
    if schema.task != "classification":
        labels = pd.Series(np.digitize(labels, old["split"]["stratification"]["interior_edges"], right=True),
                           index=frame.index)
    for name in ("T", "V"):
        pd.testing.assert_series_equal(labels.loc[old[f"{name}_row_ids"]].value_counts().sort_index(),
                                       labels.loc[new[f"{name}_row_ids"]].value_counts().sort_index())
    pool = np.array(new["T_row_ids"])
    for fold, (tr, te) in zip(new["folds"], KFold(5, shuffle=True, random_state=42).split(pool)):
        assert fold["train_row_ids"] == pool[tr].tolist()
        assert fold["test_row_ids"] == pool[te].tolist()
        assert not ps.support_violations(frame, schema, pool[tr], pool[te], "test")
    assert not ps.support_violations(frame, schema, sorted(T), sorted(V), "validation")


@pytest.mark.parametrize("name", ["breast_cancer", "house_sales", "student_perf"])
def test_real_blocked_datasets_are_repaired_without_extra_filtering(name):
    v2, v3 = load_protocol(), load_protocol(V3)
    frame, schema, manifest, eligibility = ps.load_eligible_dataset(name, v3)
    assert v3.eligibility == v2.eligibility
    old, _ = ps.build_splits(frame, schema, manifest, v2)
    new, report = ps.build_splits(frame, schema, manifest, v3)
    again, _ = ps.build_splits(frame, schema, manifest, v3)
    assert old["split_status"] == "blocked_support" and new["split_status"] == "ok"
    assert new == again and report["support_repair"]["status"] == "repaired"
    assert len(frame) == eligibility["n_eligible_rows"] == new["n_rows"]
    assert_partition(frame, schema, old, new)
    for col in report["support_repair"]["validation_columns"]:
        from sbtab.data.eligibility import finite_levels
        assert set(finite_levels(frame.loc[new["V_row_ids"]], col, schema).dropna()) == set(
            finite_levels(frame, col, schema).dropna())
    if name == "breast_cancer":
        assert (frame.loc[new["V_row_ids"], "inv-nodes"] == "14-Dec").sum() == 1
        counts = [(frame.loc[fold["test_row_ids"], "inv-nodes"] == "14-Dec").sum() for fold in new["folds"]]
        assert sorted(counts) == [0, 0, 0, 1, 1]
    ps.validate_split_artifacts(frame, new, schema)


def test_healthy_split_is_unchanged_and_old_protocol_is_still_default():
    assert load_protocol().id == "sbtab_8515_hpo100_cv5_v2"
    v3 = load_protocol(V3)
    f, s, m, _ = ps.load_eligible_dataset("diabetes", v3)
    old, _ = ps.build_splits(f, s, m, load_protocol())
    new, _ = ps.build_splits(f, s, m, v3)
    assert new["membership_hash"] == old["membership_hash"]
    assert new["split"]["support_repair"]["audit"]["status"] == "unchanged"
    assert v3["tuning"]["n_trials"] == 100 and v3["cv"] == load_protocol()["cv"]
    assert load_protocol(SMOKE_V3, smoke=True)["split"] == v3["split"]


def test_singleton_is_blocked_instead_of_copied_dropped_or_encoded_globally():
    n = 100
    frame = pd.DataFrame({"c": ["rare"] + ["usual"] * (n - 1), "y": ["a", "b"] * (n // 2)},
                          index=pd.RangeIndex(n, name="row_id"))
    schema = DatasetSchema("singleton", (ColumnSpec("c", "categorical"),
        ColumnSpec("y", "categorical", role="target")), task="classification")
    manifest = {"fingerprint": frame_fingerprint(frame), "missing_counts": {}}
    split, report = ps.build_splits(frame, schema, manifest, load_protocol(V3))
    assert split["split_status"] == "blocked_support"
    assert set(split["T_row_ids"]) | set(split["V_row_ids"]) == set(frame.index)
    assert not report["support_repair"]["swaps"]
    with pytest.raises(StageError, match="blocked_support"):
        ps.require_ok(split)


def test_bounded_search_and_invalid_rules():
    v3 = load_protocol(V3)
    f, s, m, _ = ps.load_eligible_dataset("house_sales", v3)
    data = copy.deepcopy(v3.data)
    data["split"]["support_repair"]["max_candidate_evaluations"] = 1
    split, report = ps.build_splits(f, s, m, Protocol("bounded", data))
    assert split["split_status"] == "blocked_support"
    assert report["support_repair"]["candidate_evaluations"] == 1
    assert report["support_repair"]["status"] == "unresolved"
    rule = v3["split"]["support_repair"]
    for invalid in ({}, {**rule, "method": "other"}, {**rule, "max_swaps": True},
                    {**rule, "max_candidate_evaluations": 0}, {**rule, "unknown": 1}):
        with pytest.raises(ValueError):
            validate_repair_rule(invalid)


def test_pipeline_preparation_persists_and_verifies_repaired_memberships(tmp_path):
    result = pipeline.create_plan(tmp_path, datasets=["breast_cancer"], models=["csbm"],
                                  protocol_path=V3)
    assert pipeline.prepare(result["plan"])["counts"] == {"ok": 1}
    path = tmp_path / "breast_cancer/splits.json"
    frame, schema, splits = ps.load_split_artifacts(path)
    assert len(frame) == 283 and splits["split_status"] == "ok"
    with pytest.raises(StageError, match="different --output-root"):
        ps.run("breast_cancer", load_protocol(), tmp_path)
    corrupted = read_json(path)
    corrupted["split"]["support_repair"]["audit"]["swaps"][0]["to_training"] = -1
    write_json(path, corrupted)
    with pytest.raises(StageError, match="audit mismatch"):
        ps.load_split_artifacts(path)
