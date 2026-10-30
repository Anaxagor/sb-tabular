"""Protocol constants, split identity, category support, train-only preprocessing."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, train_test_split

from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.data.loading import frame_fingerprint
from sbtab.data.preprocessing import CommonPreprocessor, MissingValueError, UnseenCategoryError, row_id_hash
from sbtab.experiments import prepare_splits as ps
from sbtab.experiments.experiment_common import (
    Protocol, StageError, assert_production_constants, claim_output_root, load_protocol, load_yaml,
)


def manifest_of(frame):
    return {"fingerprint": frame_fingerprint(frame), "missing_counts": {c: int(n) for c, n in frame.isna().sum().items() if n}}


def clf_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"x": rng.normal(size=n), "k": rng.integers(0, 3, n).astype(float),
                      "c": rng.choice(["a", "b", "c"], n), "y": rng.choice(["no", "yes"], n, p=[0.7, 0.3])})
    f.index = pd.RangeIndex(n, name="row_id")
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"), ColumnSpec("k", "discrete"), ColumnSpec("c", "categorical"),
                                   ColumnSpec("y", "categorical", role="target")), task="classification")
    return f, schema


# --------------------------------------------------------------------------- protocol
def test_production_dry_run_resolves_to_the_canonical_constants():
    p = load_protocol()
    assert p.id == "sbtab_8515_hpo100_cv5_v4" and p.kind == "production"
    assert p["split"]["test_size"] == 0.15 and p["split"]["random_state"] == 5 and p["split"]["stratify"] is True
    assert p["tuning"]["n_trials"] == 100 and p["tuning"]["sampler_seed"] == 5 and p["tuning"]["n_jobs"] == 1
    assert p["tuning"]["pruner"] == "none" and p["tuning"]["direction"] == "minimize"
    assert (p["cv"]["n_splits"], p["cv"]["shuffle"], p["cv"]["random_state"], p["cv"]["population"]) == (5, True, 42, "T")
    assert p["tuning"]["n_generated"] == "len_validation"
    assert p["preprocessing"] == {"tuning_fit_population": "T", "cv_fit_population": "train_fold"}
    assert p["metrics_config"] == "configs/metrics/metrics_v2.yaml"


def test_smoke_is_a_separate_protocol_and_cannot_pose_as_production():
    prod, smoke = load_protocol(), load_protocol(smoke=True)
    assert smoke.kind == "smoke" and smoke.id != prod.id and smoke.hash() != prod.hash()
    assert smoke["tuning"]["n_trials"] < 100
    with pytest.raises(ValueError):
        load_protocol("configs/protocols/sbtab_smoke_v1.yaml")                  # smoke file without --smoke
    with pytest.raises(ValueError):
        load_protocol("configs/protocols/sbtab_8515_hpo100_cv5_v1.yaml", smoke=True)
    tampered = copy.deepcopy(prod.data)
    tampered["tuning"]["n_trials"] = 10                                         # a reduced budget labelled "production"
    with pytest.raises(ValueError):
        assert_production_constants(tampered)
    for key, value in (("split", {"test_size": 0.2}), ("cv", {"random_state": 0}), ("split", {"random_state": 42})):
        bad = copy.deepcopy(prod.data)
        bad[key].update(value)
        with pytest.raises(ValueError):
            assert_production_constants(bad)
        assert Protocol("x", bad).hash() != prod.hash()                         # any change -> a different protocol hash


def test_an_output_root_belongs_to_exactly_one_protocol(tmp_path):
    claim_output_root(tmp_path / "r", load_protocol())
    claim_output_root(tmp_path / "r", load_protocol())                          # idempotent
    with pytest.raises(StageError):
        claim_output_root(tmp_path / "r", load_protocol(smoke=True))


# --------------------------------------------------------------------------- split identity
def test_split_membership_is_identical_across_runs_and_matches_scikit_learn():
    frame, schema = clf_frame()
    proto = load_protocol()
    a, _ = ps.build_splits(frame, schema, manifest_of(frame), proto)
    b, _ = ps.build_splits(frame.copy(), schema, manifest_of(frame), proto)
    assert a["membership_hash"] == b["membership_hash"] and a["T_row_ids"] == b["T_row_ids"]

    T, V = set(a["T_row_ids"]), set(a["V_row_ids"])
    assert T | V == set(frame.index) and not T & V                               # T and V partition D
    assert len(V) == 60 and len(T) == 340                                        # scikit-learn's own rounding of 15 %
    # the ACTUAL scikit-learn membership, not another RNG with the same seed label
    t_ids, v_ids = train_test_split(frame.index.to_numpy(), test_size=0.15, random_state=5,
                                    stratify=frame["y"].astype(str).to_numpy())
    assert set(v_ids) == V
    pool = np.sort(t_ids)
    for fold, (tr, te) in zip(a["folds"], KFold(n_splits=5, shuffle=True, random_state=42).split(pool)):
        assert fold["test_row_ids"] == [int(i) for i in pool[te]] and fold["train_row_ids"] == [int(i) for i in pool[tr]]

    tests = [set(f["test_row_ids"]) for f in a["folds"]]
    assert len(tests) == 5 and set().union(*tests) == T                          # each T row is tested ...
    assert sum(len(t) for t in tests) == len(T)                                  # ... exactly once
    for f in a["folds"]:
        tr, te = set(f["train_row_ids"]), set(f["test_row_ids"])
        assert te <= T and tr <= T and not tr & te and tr | te == T
        assert not (tr | te) & V                                                 # no V row enters CV
    # stratification preserved the class proportion (70/30) on both sides
    assert abs((frame.loc[sorted(V), "y"] == "yes").mean() - (frame["y"] == "yes").mean()) < 0.02


def test_regression_uses_persisted_quantile_strata_with_repeated_edges_dropped():
    rng = np.random.default_rng(0)
    n = 300
    y = np.where(rng.random(n) < 0.5, 0.0, rng.normal(5, 1, n))                 # half the targets tie at 0 -> repeated edges
    f = pd.DataFrame({"x": rng.normal(size=n), "y": y}, index=pd.RangeIndex(n, name="row_id"))
    schema = DatasetSchema("reg", (ColumnSpec("x", "continuous"), ColumnSpec("y", "continuous", role="target")), task="regression")
    s, _ = ps.build_splits(f, schema, manifest_of(f), load_protocol())
    meta = s["split"]["stratification"]
    assert meta["kind"] == "target_quantiles" and meta["requested_bins"] == 10
    assert len(set(meta["edges"])) == len(meta["edges"]) and meta["n_strata"] < 10   # duplicates dropped
    assert s["split_status"] == "ok"


def test_missing_stratification_variable_is_an_error_not_an_unstratified_split():
    f = pd.DataFrame({"x": np.arange(50.0)}, index=pd.RangeIndex(50, name="row_id"))
    with pytest.raises(StageError):
        ps.build_splits(f, DatasetSchema("u", (ColumnSpec("x", "continuous"),)), manifest_of(f), load_protocol())


# --------------------------------------------------------------------------- support handling
def test_unseen_feature_category_is_detected_although_target_stratification_succeeds():
    frame, schema = clf_frame()
    splits, _ = ps.build_splits(frame, schema, manifest_of(frame), load_protocol())
    v_row = splits["V_row_ids"][0]
    frame.loc[v_row, "c"] = "only_in_V"                                          # target untouched -> same split
    again, report = ps.build_splits(frame, schema, manifest_of(frame), load_protocol())
    assert again["V_row_ids"] == splits["V_row_ids"]
    assert again["split_status"] == "blocked_support" and report["blocking_columns"] == ["c"]
    v = report["violations"][0]
    assert (v["where"], v["column"], v["value"], v["affected_row_ids"], v["count_in_dataset"]) == ("V_vs_T", "c", "only_in_V", [v_row], 1)
    assert report["value_counts_of_blocking_columns"]["c"]["only_in_V"] == 1
    with pytest.raises(StageError) as e:
        ps.require_ok(again)                                                     # the dataset is stopped before tuning
    assert e.value.status == "blocked_support"


def test_singleton_in_T_necessarily_fails_in_the_fold_that_tests_it_and_two_occurrences_can_pass():
    frame, schema = clf_frame()
    splits, _ = ps.build_splits(frame, schema, manifest_of(frame), load_protocol())
    fold0, fold1 = splits["folds"][0], splits["folds"][1]
    one = frame.copy()
    one.loc[fold0["test_row_ids"][0], "k"] = 99.0                                # a discrete value occurring once in T
    s1, r1 = ps.build_splits(one, schema, manifest_of(one), load_protocol())
    assert s1["split_status"] == "blocked_support"
    assert [(v["where"], v["column"], v["value"]) for v in r1["violations"]] == [
        ("fold_0", "k", "99.0"), ("T_vs_V", "k", "99.0")]
    assert r1["violations"][0]["affected_row_ids"] == [fold0["test_row_ids"][0]]

    two = frame.copy()                                                           # two occurrences in DIFFERENT test folds:
    two.loc[[fold0["test_row_ids"][0], fold1["test_row_ids"][0]], "k"] = 99.0    # each is in the other's training fold
    s2, _ = ps.build_splits(two, schema, manifest_of(two), load_protocol())
    assert s2["split_status"] == "ok"                                            # five occurrences are NOT required
    assert s2["n_rows"] == len(frame)                                            # and no row was dropped to get there


def test_written_manifests_are_immutable_and_reload_identically(tmp_path, monkeypatch):
    frame, schema = clf_frame()
    monkeypatch.setattr(ps, "load_dataset", lambda name, config_dir=None: (frame, schema, {
        **manifest_of(frame), "dataset": name, "source": {}, "n_rows": len(frame), "n_columns": 4,
        "schema_hash": schema.hash(), "regime": schema.regime, "task": schema.task, "target": schema.target,
        "dropped_columns": [], "value_maps": {}, "missing_policy": "reject", "row_filtering": "none"}))
    out = ps.run("toy", load_protocol(), tmp_path / "root")
    assert out["split_status"] == "ok"
    for name in ("dataset_manifest.json", "schema.json", "splits.json", "support_report.json", "data.parquet"):
        assert (tmp_path / "root" / "toy" / name).exists()
    f2, s2, sp = ps.load_split_artifacts(tmp_path / "root" / "toy" / "splits.json")
    assert s2.hash() == schema.hash() and frame_fingerprint(f2) == frame_fingerprint(frame)      # lossless local copy
    assert ps.run("toy", load_protocol(), tmp_path / "root")["note"].startswith("existing identical")
    frame.loc[0, "y"] = "yes" if frame.loc[0, "y"] == "no" else "no"
    frame.loc[1, "c"] = "zzz"
    with pytest.raises(StageError):
        ps.run("toy", load_protocol(), tmp_path / "root")                        # never silently overwritten


# --------------------------------------------------------------------------- train-only preprocessing
def test_preprocessor_sees_only_the_fit_rows_and_never_expands_a_vocabulary():
    frame, schema = clf_frame()
    splits, _ = ps.build_splits(frame, schema, manifest_of(frame), load_protocol())
    fold = splits["folds"][2]
    T_k, E_k, V = frame.loc[fold["train_row_ids"]], frame.loc[fold["test_row_ids"]], frame.loc[splits["V_row_ids"]]
    pre = CommonPreprocessor(schema).fit(T_k)
    assert pre.fit_row_hash == fold["train_hash"] == row_id_hash(T_k.index)      # saved fit-row hash equals T_k
    assert pre.fit_row_hash not in (fold["test_hash"], splits["V_hash"], splits["T_hash"])
    assert pre.means["x"] == pytest.approx(T_k["x"].mean()) and pre.scales["x"] == pytest.approx(T_k["x"].std(ddof=0))
    assert pre.means["x"] != pytest.approx(frame["x"].mean(), abs=1e-12)         # not the full-data statistic

    common = pre.transform(E_k)
    assert common["k"].tolist() == E_k["k"].tolist()                             # discrete: no scaling, no recoding
    back = pre.inverse_transform(common)
    assert np.allclose(back["x"], E_k["x"]) and (back[["c", "y"]].values == E_k[["c", "y"]].values).all()
    assert list(back.columns) == schema.column_order

    held = V.copy()
    held.iloc[0, held.columns.get_loc("c")] = "never_seen"
    vocab_before = list(pre.vocab["c"])
    with pytest.raises(UnseenCategoryError):
        pre.transform(held)
    assert pre.vocab["c"] == vocab_before

    again = CommonPreprocessor.from_dict(pre.to_dict(), schema)
    pd.testing.assert_frame_equal(again.transform(E_k), common)


def test_constants_missing_values_and_ordinal_vocabulary():
    schema = DatasetSchema("m", (ColumnSpec("x", "continuous"), ColumnSpec("const", "continuous"), ColumnSpec("k", "discrete"),
                                 ColumnSpec("size", "categorical", ordered_values=("S", "M", "L"))), missing_policy="impute")
    f = pd.DataFrame({"x": [1.0, np.nan, 3.0, 5.0], "const": [2.0] * 4, "k": [1.0, 2.0, np.nan, 4.0],
                      "size": ["L", None, "S", "L"]}, index=pd.RangeIndex(4, name="row_id"))
    pre = CommonPreprocessor(schema).fit(f)
    assert pre.constant["const"] and pre.scales["const"] == 1.0                  # zero variance handled deterministically
    assert pre.impute_values == {"x": 3.0, "const": 2.0, "k": 2.0}               # discrete: LOWER median, a support member
    assert pre.vocab["size"] == ["S", "L", "__missing__"]                        # declared order, then the missingness token
    out, counts = pre.transform(f, return_imputed_counts=True)
    assert counts == {"x": 1, "k": 1, "size": 1} and not out.isna().any().any()  # every imputation is counted
    assert out["const"].tolist() == [0.0] * 4

    strict = DatasetSchema("m", schema.columns, missing_policy="reject")
    with pytest.raises(MissingValueError):
        CommonPreprocessor(strict).fit(f)                                        # unspecified handling is rejected


def test_schema_rules():
    with pytest.raises(ValueError):                                              # an integer-stored class label is still nominal
        DatasetSchema("s", (ColumnSpec("y", "discrete", role="target"),), task="classification")
    s = DatasetSchema("s", (ColumnSpec("x", "continuous"), ColumnSpec("y", "categorical", role="target")), task="classification")
    assert s.categorical == ["y"] and s.continuous == ["x"] and s.regime == "mixed"      # target counted once, in its group
    assert s.without_target().regime == "continuous"
    assert DatasetSchema.from_dict(s.to_dict()).hash() == s.hash()
