"""
Section 9 (second half): conditional-distribution metrics. The per-level oracle is
recomputed here with plain boolean masks + scipy, independently of the grouped
implementation.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from sbtab.evaluation import (MetricConfig, MetricContext, association_metrics, conditional_metrics, json_safe,
                              marginal_metrics)

E_ABS_Z = math.sqrt(2.0 / math.pi)       # E|Z| for a standard normal


def _table(n, seed, mean_b=3.0, sd_a=1.0, sd_b=1.0, p_b=0.5):
    r = np.random.default_rng(seed)
    g = np.where(r.random(n) < p_b, "b", "a")
    x = np.where(g == "a", r.normal(0.0, sd_a, n), r.normal(mean_b, sd_b, n))
    d = np.where(g == "a", r.integers(0, 3, n), r.integers(1, 4, n))
    return pd.DataFrame({"g": g, "x": x, "d": d})


COLS = [("g", "categorical"), ("x", "continuous"), ("d", "discrete")]


def _rows(table, conditioner, response):
    t = table[(table["conditioner"] == conditioner) & (table["response"] == response)]
    return t.set_index("level")


def test_scores_match_a_boolean_mask_oracle_and_blocks_are_never_pooled(make_schema):
    schema = make_schema(COLS)
    train, real, synth = _table(2000, 1), _table(1500, 2), _table(1200, 3, mean_b=2.0)
    summary, table = conditional_metrics(MetricContext.fit(train, schema), real, synth)
    assert summary["status"] == "ok" and set(summary["conditioners"]) == {"g", "d"}

    for lvl in ("a", "b"):
        want = wasserstein_distance(real.loc[real.g == lvl, "x"], synth.loc[synth.g == lvl, "x"])
        assert _rows(table, "g", "x").loc[lvl, "score"] == pytest.approx(want, abs=1e-12)
        labels = sorted(set(real["d"]) | set(synth["d"]))
        p = real.loc[real.g == lvl, "d"].value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
        q = synth.loc[synth.g == lvl, "d"].value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
        assert _rows(table, "g", "d").loc[lvl, "score"] == pytest.approx(jensenshannon(p, q) ** 2, abs=1e-12)
        assert _rows(table, "g", "x").loc[lvl, "n_real"] == int((real.g == lvl).sum())
        assert _rows(table, "g", "x").loc[lvl, "n_synth"] == int((synth.g == lvl).sum())

    # macro / real-frequency weighted means, recomputed from the saved rows
    rows = _rows(table, "g", "x")
    resp = summary["conditioners"]["g"]["responses"]["x"]
    w = rows["real_level_mass"] / rows["real_level_mass"].sum()
    assert resp["macro_mean"] == pytest.approx(rows["score"].mean(), abs=1e-12)
    assert resp["weighted_mean"] == pytest.approx(float((rows["score"] * w).sum()), abs=1e-12)
    assert resp["eligible_mass"] == pytest.approx(1.0) and resp["n_eligible_levels"] == 2

    # WD block and JS block are separate; each averages only its own metric with equal column weights
    cond = summary["conditioners"]["g"]
    assert cond["wd_block"]["n_responses"] == 1 and cond["js_block"]["n_responses"] == 1
    assert cond["wd_block"]["macro_mean"] == pytest.approx(resp["macro_mean"])
    assert cond["js_block"]["macro_mean"] == pytest.approx(cond["responses"]["d"]["macro_mean"])
    assert cond["wd_block"]["macro_mean"] > 0.4 > cond["js_block"]["macro_mean"]        # ~0.5 vs ~0.0x
    assert set(table.loc[table["metric"] == "wd", "response_type"]) == {"continuous"}
    assert set(table.loc[table["metric"] == "js", "response_type"]) == {"discrete", "categorical"}
    flat = json.dumps(json_safe(summary))
    assert "pooled_mean" not in flat and "combined_score" not in flat
    # the discrete conditioner d: responses g (JS) and x (WD) -- two JS-capable columns never enter the WD block
    assert summary["conditioners"]["d"]["responses"]["g"]["metric"] == "js"
    assert summary["conditioners"]["d"]["responses"]["x"]["metric"] == "wd"


def test_mean_shift_is_detected(make_schema):
    schema = make_schema(COLS)
    train, real = _table(3000, 4), _table(3000, 5)
    good, shifted = _table(3000, 6), _table(3000, 7, mean_b=0.0)
    ctx = MetricContext.fit(train, schema)
    s_good, _ = conditional_metrics(ctx, real, good)
    s_bad, t_bad = conditional_metrics(ctx, real, shifted)
    # level b: N(3,1) vs N(0,1) -> WD = 3 (sampling error of two 1500-row means ~ 0.04); level a unchanged
    assert _rows(t_bad, "g", "x").loc["b", "score"] == pytest.approx(3.0, abs=0.2)
    assert _rows(t_bad, "g", "x").loc["a", "score"] < 0.15
    assert s_bad["conditioners"]["g"]["wd_block"]["macro_mean"] > 1.3
    assert s_good["conditioners"]["g"]["wd_block"]["macro_mean"] < 0.15


def test_equal_means_different_variances_are_caught_by_wd_but_not_by_eta_squared(make_schema):
    schema = make_schema([("g", "categorical"), ("x", "continuous")])

    def draw(n, seed, sd_a, sd_b):
        r = np.random.default_rng(seed)
        g = np.array(["a", "b"])[np.arange(n) % 2]
        return pd.DataFrame({"g": g, "x": np.where(g == "a", r.normal(0, sd_a, n), r.normal(0, sd_b, n))})
    train, real, synth = draw(8000, 8, 1.0, 3.0), draw(8000, 9, 1.0, 3.0), draw(8000, 10, 3.0, 1.0)   # swapped
    ctx = MetricContext.fit(train, schema)
    assoc, arr = association_metrics(ctx, real, synth)
    assert arr["eta2_real"][0, 0] < 0.002 and arr["eta2_synth"][0, 0] < 0.002        # conditional means agree
    assert assoc["blocks"]["eta_squared"]["max_abs_error"] < 0.002
    marg, _ = marginal_metrics(ctx, real, synth)
    assert marg["groups"]["continuous"]["mean_wd"] < 0.1                              # same 50/50 mixture
    summary, table = conditional_metrics(ctx, real, synth)
    # W1(N(0,1), N(0,9)) = (3 - 1) E|Z| = 1.596 ; 4000 rows per level -> error of a few 0.01
    for lvl in ("a", "b"):
        assert _rows(table, "g", "x").loc[lvl, "score"] == pytest.approx(2 * E_ABS_Z, abs=0.15)
    assert summary["conditioners"]["g"]["wd_block"]["weighted_mean"] == pytest.approx(2 * E_ABS_Z, abs=0.15)


def test_missing_minority_level_gives_missing_mass_and_incomplete_status(make_schema):
    schema = make_schema(COLS)
    r = np.random.default_rng(11)

    def with_rare(df, k):
        rare = pd.DataFrame({"g": "rare", "x": r.normal(10, 1, k), "d": r.integers(0, 3, k)})
        return pd.concat([df, rare], ignore_index=True)
    train, real = with_rare(_table(1900, 12), 100), with_rare(_table(950, 13), 50)
    synth = _table(1000, 14)                                     # the generator never produces "rare"
    summary, table = conditional_metrics(MetricContext.fit(train, schema), real, synth)
    cond = summary["conditioners"]["g"]
    assert summary["status"] == "incomplete_conditional_coverage" and cond["status"] == "incomplete_conditional_coverage"
    assert summary["incomplete_conditioners"] == ["g"]
    assert cond["missing_category_mass"] == pytest.approx(50 / 1000) and cond["missing_levels"] == ["rare"]
    assert cond["n_missing_levels"] == 1 and cond["eligible_mass"] == pytest.approx(950 / 1000)
    rare = _rows(table, "g", "x").loc["rare"]
    assert rare["status"] == "undefined" and np.isnan(rare["score"]) and bool(rare["missing_in_synth"])
    assert rare["n_real"] == 50 and rare["n_synth"] == 0
    # the partial mean is normalised over the REPORTED eligible mass, not silently over everything
    rows = _rows(table, "g", "x").drop(index="rare")
    w = rows["real_level_mass"] / rows["real_level_mass"].sum()
    resp = cond["responses"]["x"]
    assert resp["weighted_mean"] == pytest.approx(float((rows["score"] * w).sum()), abs=1e-12)
    assert resp["eligible_mass"] == pytest.approx(0.95) and resp["n_eligible_levels"] == 2
    # the conditioner d is complete: coverage is tracked per conditioner
    assert summary["conditioners"]["d"]["status"] == "ok"
    assert summary["conditioners"]["d"]["missing_category_mass"] == 0.0


def test_insufficient_counts_are_recorded_not_dropped(make_schema):
    schema = make_schema(COLS)
    r = np.random.default_rng(15)

    def with_level(df, name, k):
        extra = pd.DataFrame({"g": name, "x": r.normal(0, 1, k), "d": r.integers(0, 3, k)})
        return pd.concat([df, extra], ignore_index=True)
    train = with_level(with_level(_table(2000, 16), "few_real", 40), "few_synth", 40)
    real = with_level(with_level(_table(991, 17), "few_real", 9), "few_synth", 30)     # 9 < 10 real rows
    synth = with_level(with_level(_table(1000, 18), "few_real", 30), "few_synth", 9)   # 9 < 10 synthetic rows
    summary, table = conditional_metrics(MetricContext.fit(train, schema), real, synth)
    rows = _rows(table, "g", "x")
    assert len(rows) == 4
    for lvl in ("few_real", "few_synth"):
        assert rows.loc[lvl, "status"] == "insufficient_data" and np.isnan(rows.loc[lvl, "score"])
        assert not rows.loc[lvl, "missing_in_synth"]
    assert rows.loc["few_real", "n_real"] == 9 and rows.loc["few_synth", "n_synth"] == 9
    cond = summary["conditioners"]["g"]
    assert cond["status"] == "ok" and cond["n_insufficient_levels"] == 2 and cond["n_eligible_levels"] == 2
    assert cond["insufficient_mass"] == pytest.approx(39 / 1030)
    assert cond["eligible_mass"] == pytest.approx(991 / 1030) and cond["missing_category_mass"] == 0.0
    # exactly 10 on both sides IS eligible
    real10 = with_level(_table(990, 19), "ten", 10)
    synth10 = with_level(_table(990, 20), "ten", 10)
    _, t10 = conditional_metrics(MetricContext.fit(with_level(train, "ten", 20), schema), real10, synth10)
    assert _rows(t10, "g", "x").loc["ten", "status"] == "ok"
    # a configurable, versioned threshold
    ctx25 = MetricContext.fit(with_level(train, "ten", 20), schema, MetricConfig(conditional_min_rows=25))
    _, t25 = conditional_metrics(ctx25, real10, synth10)
    assert _rows(t25, "g", "x").loc["ten", "status"] == "insufficient_data"


def test_synthetic_only_levels_are_reported(make_schema):
    schema = make_schema(COLS)
    r = np.random.default_rng(21)
    train = pd.concat([_table(1000, 22), pd.DataFrame({"g": "c", "x": r.normal(size=30), "d": 1})], ignore_index=True)
    real = _table(800, 23)                                             # E_k happens to hold no "c"
    synth = pd.concat([_table(900, 24), pd.DataFrame({"g": "c", "x": r.normal(size=100), "d": 1})], ignore_index=True)
    synth.loc[:49, "d"] = 9                                            # discrete value outside the training support
    summary, table = conditional_metrics(MetricContext.fit(train, schema), real, synth)
    g = summary["conditioners"]["g"]
    assert g["synthetic_only_mass"] == pytest.approx(100 / 1000)
    assert g["synthetic_only_levels"] == [{"level": "c", "n_synth": 100, "synth_mass": 0.1, "in_training_support": True}]
    assert g["status"] == "ok" and g["missing_category_mass"] == 0.0
    d = summary["conditioners"]["d"]
    assert [lv["level"] for lv in d["synthetic_only_levels"]] == ["9"]
    assert d["synthetic_only_levels"][0]["in_training_support"] is False
    only = table[table["synthetic_only"]]
    assert set(zip(only["conditioner"], only["level"])) == {("g", "c"), ("d", "9")}
    assert (only["n_real"] == 0).all() and only["score"].isna().all() and (only["status"] == "undefined").all()
    # ... and they stay in the unconditional metrics
    marg, mt = marginal_metrics(MetricContext.fit(train, schema), real, synth)
    assert mt.set_index("column").loc["g", "js"] > 0.01
    assert mt.set_index("column").loc["d", "synth_unexpected_mass"] == pytest.approx(0.05)


def test_high_cardinality_conditioner_is_listed_as_excluded(make_schema):
    schema = make_schema([("c50", "categorical"), ("c51", "categorical"), ("d60", "discrete"), ("x", "continuous")])
    n = 51 * 50 * 60 // 30
    idx = np.arange(n)
    df = pd.DataFrame({"c50": idx % 50, "c51": idx % 51, "d60": idx % 60,
                       "x": np.random.default_rng(25).normal(size=n)})
    ctx = MetricContext.fit(df, schema)
    summary, table = conditional_metrics(ctx, df, df)
    assert list(summary["conditioners"]) == ["c50"]                    # exactly 50 training levels: kept
    excluded = {e["column"]: e for e in summary["excluded_conditioners"]}
    assert set(excluded) == {"c51", "d60"}
    assert excluded["c51"]["n_levels"] == 51 and excluded["c51"]["reason"] == "high_cardinality"
    assert excluded["d60"]["max_levels"] == 50
    assert set(table["conditioner"]) == {"c50"}
    assert {e["column"] for e in ctx.to_dict()["conditioning"]["excluded"]} == {"c51", "d60"}   # persisted
    # excluded columns are still RESPONSES of the remaining conditioner
    assert set(summary["conditioners"]["c50"]["responses"]) == {"c51", "d60", "x"}


def test_regression_target_strata_come_from_train(make_schema):
    schema = make_schema([("x", "continuous"), ("y", "continuous", "target")], task="regression")
    # 50 zeros then 1..50 ; numpy linear quantiles at 0,.2,.4,.6,.8,1 of the 100 sorted values:
    #   positions 0, 19.8, 39.6, 59.4, 79.2, 99 -> 0, 0, 0, 10.4, 30.2, 50
    # duplicates dropped -> (0, 10.4, 30.2, 50) ; exterior boundaries -> -inf / +inf
    y = np.array([0.0] * 50 + list(np.arange(1.0, 51.0)))
    r = np.random.default_rng(26)
    train = pd.DataFrame({"x": r.normal(size=100), "y": y})
    ctx = MetricContext.fit(train, schema)
    strata = [c for c in ctx.conditioning["conditioners"] if c["kind"] == "regression_target_strata"]
    assert len(strata) == 1 and strata[0]["column"] == "y" and strata[0]["id"] == "y::quantile_strata"
    assert strata[0]["interior_boundaries"] == pytest.approx([10.4, 30.2], abs=1e-12)
    assert strata[0]["n_levels"] == 3 and strata[0]["requested_strata"] == 5
    assert strata[0]["labels"][0].startswith("(-inf,") and strata[0]["labels"][-1].endswith("+inf)")
    json.dumps(ctx.to_dict(), allow_nan=False)                         # infinities are persisted as text

    # held-out / generated values far outside the training range fall into the exterior strata
    def draw(seed, n=600):
        rr = np.random.default_rng(seed)
        yy = np.concatenate([rr.uniform(-500, 0, n // 3), rr.uniform(11, 30, n // 3), rr.uniform(31, 9000, n // 3)])
        return pd.DataFrame({"x": rr.normal(size=n) + (yy > 30) * 2.0, "y": yy})
    real, synth = draw(27), draw(28)
    summary, table = conditional_metrics(ctx, real, synth)
    cond = summary["conditioners"]["y::quantile_strata"]
    assert cond["kind"] == "regression_target_strata" and cond["n_levels_real"] == 3
    rows = _rows(table, "y::quantile_strata", "x")
    assert list(rows["n_real"]) == [200, 200, 200] and list(rows["n_synth"]) == [200, 200, 200]
    assert "y" not in cond["responses"]                                # the target is not its own response
    lo = real["y"] <= 10.4
    want = wasserstein_distance(real.loc[lo, "x"], synth.loc[synth["y"] <= 10.4, "x"])
    assert rows.iloc[0]["score"] == pytest.approx(want, abs=1e-12)
    # strata are a property of the context: another E_k cannot move them
    before = json.dumps(ctx.to_dict()["conditioning"], sort_keys=True)
    conditional_metrics(ctx, draw(29), draw(30))
    assert json.dumps(ctx.to_dict()["conditioning"], sort_keys=True) == before

    # a value equal to the SAVED boundary belongs to the lower stratum (right-closed intervals)
    b = strata[0]["interior_boundaries"][0]
    edge = pd.DataFrame({"x": np.zeros(50), "y": [b] * 20 + [np.nextafter(b, np.inf)] * 30})
    _, t_edge = conditional_metrics(ctx, edge, edge)
    assert list(_rows(t_edge, "y::quantile_strata", "x")["n_real"]) == [20, 30]


def test_shuffled_pairings_damage_conditionals_but_not_marginals(make_schema):
    schema = make_schema(COLS)
    train, real = _table(3000, 31), _table(3000, 32)
    r = np.random.default_rng(33)
    shuffled = pd.DataFrame({c: r.permutation(real[c].to_numpy()) for c in real.columns})
    ctx = MetricContext.fit(train, schema)
    marg, _ = marginal_metrics(ctx, real, shuffled)
    assert marg["groups"]["continuous"]["mean_wd"] == pytest.approx(0.0, abs=1e-12)
    assert marg["groups"]["finite_combined"]["mean_js"] == pytest.approx(0.0, abs=1e-12)
    s_shuf, _ = conditional_metrics(ctx, real, shuffled)
    s_same, _ = conditional_metrics(ctx, real, real.iloc[r.permutation(len(real))].reset_index(drop=True))
    assert s_same["conditioners"]["g"]["wd_block"]["weighted_mean"] == pytest.approx(0.0, abs=1e-12)
    assert s_same["conditioners"]["g"]["js_block"]["weighted_mean"] == pytest.approx(0.0, abs=1e-12)
    # x | g: N(0,1) / N(3,1) against the 50/50 mixture -> WD around 1.5
    assert s_shuf["conditioners"]["g"]["wd_block"]["weighted_mean"] > 1.0
    assert s_shuf["conditioners"]["g"]["js_block"]["weighted_mean"] > 0.02


def test_label_recoding_leaves_conditional_metrics_unchanged(make_schema):
    schema = make_schema(COLS)
    train, real, synth = _table(1500, 34), _table(1200, 35), _table(1000, 36, mean_b=2.5, p_b=0.3)
    recode = lambda df: df.assign(g=df["g"].map({"a": "zz", "b": "aa"}))          # also flips the sort order
    s1, t1 = conditional_metrics(MetricContext.fit(train, schema), real, synth)
    s2, t2 = conditional_metrics(MetricContext.fit(recode(train), schema), recode(real), recode(synth))
    for cid in ("g", "d"):
        for blk in ("wd_block", "js_block"):
            for key in ("macro_mean", "weighted_mean"):
                assert s1["conditioners"][cid][blk][key] == pytest.approx(s2["conditioners"][cid][blk][key], abs=1e-12)
    assert s1["conditioners"]["g"]["wd_block"]["macro_mean"] > 0.1


def test_invalid_and_not_applicable_cases(make_schema, collect_statuses):
    schema = make_schema(COLS)
    train, real = _table(500, 37), _table(500, 38)
    bad = real.copy()
    bad.loc[0, "x"] = np.inf
    summary, table = conditional_metrics(MetricContext.fit(train, schema), real, bad)
    assert summary["status"] == "invalid_generated_data" and len(table) == 0 and summary["conditioners"] == {}
    cont = make_schema([("a", "continuous"), ("b", "continuous")])
    df = pd.DataFrame(np.random.default_rng(39).normal(size=(50, 2)), columns=["a", "b"])
    s2, t2 = conditional_metrics(MetricContext.fit(df, cont), df, df)
    assert s2["status"] == "not_applicable" and len(t2) == 0 and s2["secondary_across_conditioners"] is None
    json.dumps(json_safe(s2), allow_nan=False)
