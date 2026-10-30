"""Dataset-eligibility rule (protocol v2): rows with a rare finite-support value are removed before splitting."""
import json

import numpy as np
import pandas as pd
import pytest

from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.data.eligibility import apply_eligibility, validate_rule
from sbtab.data.loading import frame_fingerprint
from sbtab.experiments import prepare_splits as ps
from sbtab.experiments.experiment_common import Protocol, StageError, load_protocol

RULE = {"min_value_count": 3, "columns": "finite_support"}
V1 = "configs/protocols/sbtab_8515_hpo100_cv5_v1.yaml"
V2 = "configs/protocols/sbtab_8515_hpo100_cv5_v2.yaml"


def table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"x": rng.normal(size=n), "k": rng.integers(0, 3, n).astype(float),
                      "c": rng.choice(["a", "b", "c"], n), "y": rng.choice(["no", "yes"], n, p=[0.7, 0.3])})
    f.index = pd.RangeIndex(n, name="row_id")
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"), ColumnSpec("k", "discrete"), ColumnSpec("c", "categorical"),
                                   ColumnSpec("y", "categorical", role="target")), task="classification")
    return f, schema


def manifest_of(frame, schema):
    return {"fingerprint": frame_fingerprint(frame), "missing_counts": {c: int(n) for c, n in frame.isna().sum().items() if n},
            "dataset": "toy", "source": {}, "n_rows": len(frame), "n_columns": frame.shape[1], "schema_hash": schema.hash(),
            "regime": schema.regime, "task": schema.task, "target": schema.target, "dropped_columns": [], "value_maps": {},
            "missing_policy": schema.missing_policy, "row_filtering": "none"}


# --------------------------------------------------------------------------- the rule
def test_rows_with_a_value_seen_fewer_than_three_times_are_removed_and_row_ids_survive():
    f, schema = table()
    f.loc[10, "c"] = "once"                       # 1 row
    f.loc[[20, 21], "c"] = "twice"                # 2 rows
    f.loc[[30, 31, 32], "c"] = "thrice"           # 3 rows: kept
    f.loc[40, "k"] = 99.0                         # rare value of a DISCRETE column (it is support-checked too)
    f.loc[50, "x"] = 1e9                          # continuous columns are never counted
    out, rep = apply_eligibility(f, schema, RULE)

    assert rep["removed_row_ids"] == [10, 20, 21, 40] and rep["n_removed_rows"] == 4 and rep["passes"] == 1
    assert sorted(set(f.index) - set(out.index)) == [10, 20, 21, 40]
    assert {30, 31, 32, 50} <= set(out.index)                                   # original ids, with gaps — not renumbered
    assert out.loc[31].equals(f.loc[31])
    got = {(r["column"], r["value"]): (r["count"], r["row_ids"]) for r in rep["removals"]}
    assert got == {("c", "once"): (1, [10]), ("c", "twice"): (2, [20, 21]), ("k", "99.0"): (1, [40])}
    assert rep["task_changed"] is False and rep["n_source_rows"] == 400 and rep["n_eligible_rows"] == 396
    for col in schema.finite_support:                                           # the postcondition the rule promises
        assert out[col].value_counts().min() >= 3
    json.dumps(rep, allow_nan=False)


def test_rule_is_iterated_to_a_fixed_point_and_does_not_depend_on_column_order():
    f, schema = table()
    f.loc[[5, 6, 7], "c"] = "rare3"               # exactly 3 rows ...
    f.loc[5, "k"] = 77.0                          # ... but one of them goes away because of ANOTHER column,
    out, rep = apply_eligibility(f, schema, RULE)  # which pushes "rare3" down to 2 -> removed in pass 2
    assert rep["passes"] == 2 and rep["removed_row_ids"] == [5, 6, 7]
    assert [(r["pass"], r["column"], r["value"]) for r in rep["removals"]] == [(1, "k", "77.0"), (2, "c", "rare3")]

    once, rep1 = apply_eligibility(f, schema, {**RULE, "iterate_to_fixed_point": False})
    assert rep1["removed_row_ids"] == [5] and (once["c"] == "rare3").sum() == 2   # without iteration the promise is broken

    reordered = DatasetSchema("toy", tuple(reversed(schema.columns)), task="classification")
    out2, _ = apply_eligibility(f[reordered.column_order], reordered, RULE)
    assert list(out2.index) == list(out.index)


def test_missing_categorical_is_a_level_missing_discrete_is_not_and_scope_can_exclude_discrete():
    f, _ = table()
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"), ColumnSpec("k", "discrete"), ColumnSpec("c", "categorical"),
                                   ColumnSpec("y", "categorical", role="target")), task="classification", missing_policy="impute")
    f.loc[[1, 2], "c"] = None                     # 2 rows with the level __missing__ -> removed
    f.loc[[3, 4], "k"] = np.nan                   # imputed later from training rows: not a level, rows stay
    f.loc[60, "k"] = 55.0
    out, rep = apply_eligibility(f, schema, RULE)
    assert rep["removed_row_ids"] == [1, 2, 60] and {3, 4} <= set(out.index)
    assert ("c", "__missing__") in {(r["column"], r["value"]) for r in rep["removals"]}

    only_cat, rep_cat = apply_eligibility(f, schema, {"min_value_count": 3, "columns": "categorical"})
    assert rep_cat["removed_row_ids"] == [1, 2] and 60 in only_cat.index          # discrete column left alone


def test_rare_target_classes_are_removed_and_flagged_as_a_changed_task():
    f, schema = table()
    f.loc[[8, 9], "y"] = "maybe"
    out, rep = apply_eligibility(f, schema, RULE)
    assert rep["target_values_removed"] == ["maybe"] and rep["task_changed"] is True
    assert set(out["y"]) == {"no", "yes"}


def test_threshold_one_and_v1_never_remove_a_row_and_v1_is_frozen():
    f, schema = table()
    f.loc[10, "c"] = "once"
    same, rep = apply_eligibility(f, schema, {"min_value_count": 1})
    assert same.equals(f) and rep["applied"] is False and rep["n_removed_rows"] == 0
    same, rep = apply_eligibility(f, schema, {})
    assert rep["n_removed_rows"] == 0

    v1 = load_protocol(V1)
    assert v1.id == "sbtab_8515_hpo100_cv5_v1" and v1.eligibility == {} and v1.hash().startswith("d661d3713a5f")
    v2 = load_protocol(V2)
    assert v2.id == "sbtab_8515_hpo100_cv5_v2" and v2.data["supersedes"] == v1.id and v2.hash() != v1.hash()
    assert v2.eligibility == {"min_value_count": 3, "columns": "finite_support", "iterate_to_fixed_point": True}
    assert {k: v2[k] for k in ("split", "tuning", "cv", "seeds")} == {k: v1[k] for k in ("split", "tuning", "cv", "seeds")}
    assert load_protocol(smoke=True).eligibility == v2.eligibility               # smoke mirrors production


def test_bad_rules_are_rejected_up_front():
    for bad in ({"min_value_count": 0}, {"min_value_count": 2.5}, {"min_value_count": True}, {"columns": "everything"},
                {"min_count": 3}):
        with pytest.raises(ValueError):
            validate_rule(bad)
    f, schema = table(n=12)
    f["c"] = [f"u{i}" for i in range(12)]                                         # every value is rare
    with pytest.raises(ValueError):
        apply_eligibility(f, schema, RULE)


# --------------------------------------------------------------------------- what the rule does NOT give
def test_three_rows_of_a_value_do_not_guarantee_support_coverage():
    """
    The rule is motivated by coverage, but a count cannot guarantee it under a fixed random split:
    put a value on 1 row of V and 2 rows of the SAME CV test fold. All three survive the rule, and
    the fold's training rows do not contain the value -> the dataset is (correctly) still blocked.
    """
    f, schema = table()
    proto = load_protocol(V2)
    base, _ = ps.build_splits(f, schema, manifest_of(f, schema), proto)
    fold = base["folds"][2]
    rows = [base["V_row_ids"][0]] + fold["test_row_ids"][:2]
    f.loc[rows, "c"] = "exactly3"                                                  # target untouched -> identical split

    out, rep = apply_eligibility(f, schema, proto.eligibility)
    assert rep["n_removed_rows"] == 0 and (out["c"] == "exactly3").sum() == 3       # passes the rule ...
    splits, report = ps.build_splits(out, schema, manifest_of(out, schema), proto)
    assert splits["V_row_ids"] == base["V_row_ids"]
    assert splits["split_status"] == "blocked_support"                             # ... and still fails coverage
    v = report["violations"]
    assert [(x["where"], x["column"], x["value"], x["count_in_dataset"]) for x in v] == [("fold_2", "c", "exactly3", 3)]
    assert v[0]["affected_row_ids"] == sorted(rows[1:])


# --------------------------------------------------------------------------- stage
def test_prepare_splits_applies_the_rule_before_splitting_and_records_it(tmp_path, monkeypatch):
    f, schema = table()
    f.loc[10, "c"] = "once"
    f.loc[[20, 21], "k"] = 9.0
    monkeypatch.setattr(ps, "load_dataset", lambda name, config_dir=None: (f, schema, manifest_of(f, schema)))

    v1 = ps.run("toy", load_protocol(V1), tmp_path / "v1", dry_run=True)
    assert v1["split_status"] == "blocked_support" and v1["n_removed_by_eligibility"] == 0 and v1["n_rows"] == 400

    out = ps.run("toy", load_protocol(V2), tmp_path / "v2")
    assert out["split_status"] == "ok" and out["n_source_rows"] == 400 and out["n_removed_by_eligibility"] == 3
    d = tmp_path / "v2" / "toy"
    rep, man, splits = (json.loads((d / n).read_text()) for n in ("eligibility_report.json", "dataset_manifest.json", "splits.json"))
    assert rep["removed_row_ids"] == [10, 20, 21]
    assert man["n_source_rows"] == 400 and man["n_rows"] == 397 and man["source_fingerprint"] != man["fingerprint"]
    assert "removed 3 of 400 rows before splitting" in man["row_filtering"] and man["eligibility"]["task_changed"] is False
    assert splits["eligibility_rule"] == load_protocol(V2).eligibility and splits["n_rows"] == 397

    members = set(splits["T_row_ids"]) | set(splits["V_row_ids"])
    assert members == set(f.index) - {10, 20, 21}                                   # D is the ELIGIBLE table; ids are original
    frame, _, _ = ps.load_split_artifacts(d / "splits.json")
    assert set(frame.index) == members and frame_fingerprint(frame) == man["fingerprint"]

    # the two protocols can never share an artifact root
    with pytest.raises(StageError):
        ps.run("toy", load_protocol(V1), tmp_path / "v2")


def test_a_dataset_emptied_by_the_rule_is_a_structured_failure(monkeypatch):
    f, schema = table(n=12)
    f["c"] = [f"u{i}" for i in range(12)]
    monkeypatch.setattr(ps, "load_dataset", lambda name, config_dir=None: (f, schema, manifest_of(f, schema)))
    with pytest.raises(StageError) as e:
        ps.load_eligible_dataset("toy", load_protocol(V2))
    assert "removed every row" in str(e.value)


# --------------------------------------------------------------------------- real data (documented facts)
@pytest.mark.integration
def test_documented_effect_on_real_datasets():
    proto = load_protocol(V2)
    # unblocked by removing ONE row
    frame, schema, man, rep = ps.load_eligible_dataset("stroke_prediction", proto)
    assert rep["removed_row_ids"] == [3116] and rep["removals"][0]["value"] == "Other" and not rep["task_changed"]
    assert ps.build_splits(frame, schema, man, proto)[0]["split_status"] == "ok"
    # the rule also removes a rare target CLASS, and needs a second pass here
    frame, schema, man, rep = ps.load_eligible_dataset("lymphography", proto)
    assert rep["target_values_removed"] == ["normal"] and rep["task_changed"] and rep["passes"] == 2 and rep["n_removed_rows"] == 10
    # still blocked: a value with exactly 3 rows fails coverage in one fold
    frame, schema, man, rep = ps.load_eligible_dataset("breast_cancer", proto)
    splits, report = ps.build_splits(frame, schema, man, proto)
    assert splits["split_status"] == "blocked_support" and report["blocking_columns"] == ["inv-nodes"]
    assert report["violations"][0]["value"] == "14-Dec" and report["violations"][0]["count_in_dataset"] == 3
