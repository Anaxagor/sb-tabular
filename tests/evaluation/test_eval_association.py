"""
Section 9 (first half): dependence metrics. Oracles: numpy / scipy / sklearn
reference functions and hand-computed numbers.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

from sbtab.evaluation import MetricContext, association_metrics, marginal_metrics
from sbtab.evaluation.association import offdiag_summary


def _dependent_table(n, seed):
    r = np.random.default_rng(seed)
    g = r.integers(0, 3, n)
    x1 = r.normal(size=n) + 1.5 * g
    return pd.DataFrame({
        "x1": x1, "x2": 0.8 * x1 + 0.3 * r.normal(size=n), "x3": r.normal(size=n),
        "d1": r.poisson(1 + 2 * g), "d2": (x1 > 1).astype(int) + r.integers(0, 2, n), "d3": r.integers(0, 5, n),
        "c1": np.array(["lo", "mid", "hi"])[g], "c2": np.where(r.random(n) < 0.8, g, r.integers(0, 3, n)),
        "c3": r.choice(["p", "q"], size=n),
    })


SCHEMA_COLS = [("x1", "continuous"), ("x2", "continuous"), ("x3", "continuous"),
               ("d1", "discrete"), ("d2", "discrete"), ("d3", "discrete"),
               ("c1", "categorical"), ("c2", "categorical"), ("c3", "categorical")]


def test_matrices_match_reference_implementations(make_schema):
    schema = make_schema(SCHEMA_COLS)
    train, real, synth = _dependent_table(800, 1), _dependent_table(600, 2), _dependent_table(500, 3)
    summary, arr = association_metrics(MetricContext.fit(train, schema), real, synth)
    assert summary["status"] == "ok"
    for name, df in (("real", real), ("synth", synth)):
        assert np.allclose(arr[f"pearson_{name}"], np.corrcoef(df[["x1", "x2", "x3"]].to_numpy().T), atol=1e-12)
        assert np.allclose(arr[f"spearman_{name}"], spearmanr(df[["d1", "d2", "d3"]].to_numpy()).statistic, atol=1e-12)
        cats = ["c1", "c2", "c3"]
        for i, a in enumerate(cats):
            for j, b in enumerate(cats):
                want = 1.0 if i == j else normalized_mutual_info_score(df[a], df[b], average_method="arithmetic")
                assert arr[f"nmi_{name}"][i, j] == pytest.approx(want, abs=1e-10)
        cross = spearmanr(df[["x1", "x2", "x3", "d1", "d2", "d3"]].to_numpy()).statistic[:3, 3:]
        assert np.allclose(arr[f"spearman_cross_{name}"], cross, atol=1e-12)
    assert list(arr["pearson_columns"]) == ["x1", "x2", "x3"] and list(arr["nmi_columns"]) == ["c1", "c2", "c3"]
    assert np.allclose(arr["pearson_diff"], arr["pearson_synth"] - arr["pearson_real"])
    assert np.allclose(arr["spearman_cross_abs_error"], np.abs(arr["spearman_cross_synth"] - arr["spearman_cross_real"]))
    # entropies (natural log) are saved
    p = real["c3"].value_counts(normalize=True).to_numpy()
    assert summary["blocks"]["nmi"]["real_entropy"]["c3"] == pytest.approx(-(p * np.log(p)).sum(), abs=1e-12)


def test_normalised_rmse_divides_by_d_times_d_minus_one():
    real = np.eye(3)
    synth = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
    out = offdiag_summary(real, synth)
    # off-diagonal squared sum = 2 * (0.01 + 0.04 + 0.09) = 0.28 ; d (d - 1) = 6
    assert out["offdiag_rmse"] == pytest.approx(math.sqrt(0.28 / 6), abs=1e-15)
    assert out["frobenius"] == pytest.approx(math.sqrt(0.28), abs=1e-15)
    assert out["offdiag_rmse"] != pytest.approx(math.sqrt(0.28 / 9))      # not d^2
    assert out["offdiag_rmse"] != pytest.approx(math.sqrt(0.28 / 3))      # not the d(d-1)/2 unordered pairs


def test_block_summary_uses_the_same_normaliser(make_schema):
    schema = make_schema(SCHEMA_COLS)
    train, real, synth = _dependent_table(800, 4), _dependent_table(600, 5), _dependent_table(500, 6)
    synth["x2"] = np.random.default_rng(0).normal(size=len(synth))       # break one dependence
    summary, arr = association_metrics(MetricContext.fit(train, schema), real, synth)
    R, S = np.corrcoef(real[["x1", "x2", "x3"]].to_numpy().T), np.corrcoef(synth[["x1", "x2", "x3"]].to_numpy().T)
    d = 3
    sq = sum((S[i, j] - R[i, j]) ** 2 for i in range(d) for j in range(d) if i != j)
    assert summary["blocks"]["pearson"]["offdiag_rmse"] == pytest.approx(math.sqrt(sq / (d * (d - 1))), abs=1e-12)
    assert summary["blocks"]["pearson"]["frobenius"] == pytest.approx(math.sqrt(sq), abs=1e-12)  # common diagonal
    assert np.all(np.diag(arr["pearson_real"]) == 1.0) and np.all(np.diag(arr["pearson_synth"]) == 1.0)
    assert np.all(np.diag(arr["nmi_real"]) == 1.0) and np.all(np.diag(arr["nmi_diff"]) == 0.0)
    assert summary["blocks"]["pearson"]["offdiag_rmse"] > 0.3


def test_eta_squared_hand_example(make_schema):
    schema = make_schema([("x", "continuous"), ("g", "categorical")])
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "g": ["a", "a", "a", "b", "b", "b"]})
    # grand mean 3.5, SS_total = 17.5 ; group means 2 and 5 -> SS_between = 3*1.5^2 + 3*1.5^2 = 13.5
    flat = pd.DataFrame({"x": [1.0, 4.0, 2.0, 5.0, 3.0, 6.0], "g": ["a", "a", "a", "b", "b", "b"]})
    # group means (7/3, 14/3) -> SS_between = 3*(7/6)^2*2 = 49/6
    summary, arr = association_metrics(MetricContext.fit(df, schema), df, flat)
    assert arr["eta2_real"][0, 0] == pytest.approx(13.5 / 17.5, abs=1e-14)
    assert arr["eta2_synth"][0, 0] == pytest.approx((49 / 6) / 17.5, abs=1e-14)
    assert arr["eta2_abs_error"][0, 0] == pytest.approx(13.5 / 17.5 - (49 / 6) / 17.5, abs=1e-14)
    assert summary["blocks"]["eta_squared"]["mean_abs_error"] == pytest.approx(arr["eta2_abs_error"][0, 0])
    assert list(arr["eta2_numeric_columns"]) == ["x"] and list(arr["eta2_nominal_columns"]) == ["g"]


def test_label_permutation_leaves_nominal_metrics_unchanged(make_schema):
    schema = make_schema(SCHEMA_COLS)
    train, real, synth = _dependent_table(700, 7), _dependent_table(600, 8), _dependent_table(500, 9)
    synth["c2"] = np.random.default_rng(1).permutation(synth["c2"].to_numpy())
    maps = {"c1": {"lo": "B", "mid": "C", "hi": "A"}, "c2": {0: 2, 1: 0, 2: 1}, "c3": {"p": "q", "q": "p"}}

    def recode(df):
        out = df.copy()
        for c, m in maps.items():
            out[c] = df[c].map(m)
        return out
    s1, a1 = association_metrics(MetricContext.fit(train, schema), real, synth)
    s2, a2 = association_metrics(MetricContext.fit(recode(train), schema), recode(real), recode(synth))
    for key in ("nmi_real", "nmi_synth", "nmi_diff", "eta2_real", "eta2_synth", "eta2_abs_error",
                "nmi_real_entropy", "nmi_synth_entropy"):
        assert np.allclose(a1[key], a2[key], rtol=0, atol=1e-12), key
    assert s1["blocks"]["nmi"]["offdiag_rmse"] == pytest.approx(s2["blocks"]["nmi"]["offdiag_rmse"], abs=1e-12)
    assert s1["blocks"]["nmi"]["offdiag_rmse"] > 0.05            # the comparison is not vacuous


def test_constant_column_conventions(make_schema):
    schema = make_schema(SCHEMA_COLS)
    train, real = _dependent_table(600, 10), _dependent_table(500, 11)
    synth = _dependent_table(500, 12)
    synth["x2"] = 0.1            # a non-representable constant: exact-zero std cannot be relied upon
    synth["d2"] = 4
    synth["c1"] = "lo"
    synth["c2"] = 0
    summary, arr = association_metrics(MetricContext.fit(train, schema), real, synth)
    b = summary["blocks"]
    assert b["pearson"]["synth_constant_columns"] == ["x2"] and b["pearson"]["real_constant_columns"] == []
    assert np.all(arr["pearson_synth"][1, [0, 2]] == 0.0) and np.all(arr["pearson_synth"][[0, 2], 1] == 0.0)
    assert arr["pearson_synth"][1, 1] == 1.0
    assert b["spearman"]["synth_constant_columns"] == ["d2"]
    assert np.all(arr["spearman_synth"][1, [0, 2]] == 0.0)
    # collapse of the strong real x1-x2 dependence stays visible
    assert abs(arr["pearson_diff"][0, 1]) > 0.8 and b["pearson"]["offdiag_rmse"] > 0.4
    # NMI: 0 for every pair with a constant column, INCLUDING constant/constant ...
    assert b["nmi"]["synth_constant_columns"] == ["c1", "c2"]
    assert arr["nmi_synth"][0, 1] == 0.0 and arr["nmi_synth"][0, 2] == 0.0 and arr["nmi_synth"][1, 2] == 0.0
    assert b["nmi"]["synth_entropy"]["c1"] == 0.0
    # ... whereas the library convention would report a perfectly preserved dependence
    assert normalized_mutual_info_score(synth["c1"], synth["c2"], average_method="arithmetic") == 1.0
    # eta^2: zero-variance numeric -> 0 + flag ; cross Spearman with the constant discrete column -> 0
    assert "x2" in b["eta_squared"]["synth_zero_variance_columns"]
    assert np.all(arr["eta2_synth"][list(arr["eta2_numeric_columns"]).index("x2")] == 0.0)
    assert np.all(arr["spearman_cross_synth"][:, 1] == 0.0) and np.all(arr["spearman_cross_synth"][1, :] == 0.0)
    assert set(b["spearman_cross"]["synth_constant_columns"]) == {"x2", "d2"}


def test_fewer_than_two_columns_is_not_applicable(make_schema):
    schema = make_schema([("x", "continuous"), ("d", "discrete"), ("c", "categorical")])
    r = np.random.default_rng(14)
    df = pd.DataFrame({"x": r.normal(size=100), "d": r.integers(0, 3, 100), "c": r.choice(["a", "b"], 100)})
    summary, arr = association_metrics(MetricContext.fit(df, schema), df, df)
    for name in ("pearson", "spearman", "nmi"):
        blk = summary["blocks"][name]
        assert blk["status"] == "not_applicable" and blk["n_columns"] == 1
        assert blk["offdiag_rmse"] is None and blk["frobenius"] is None       # None, never 0
        assert f"{name}_real" not in arr
    assert summary["blocks"]["eta_squared"]["status"] == "ok"
    assert summary["blocks"]["spearman_cross"]["status"] == "ok"
    only_cont = make_schema([("a", "continuous"), ("b", "continuous")])
    s2, a2 = association_metrics(MetricContext.fit(df.rename(columns={"x": "a", "d": "b"})[["a", "b"]].astype(float),
                                                   only_cont),
                                 df.rename(columns={"x": "a", "d": "b"})[["a", "b"]].astype(float),
                                 df.rename(columns={"x": "a", "d": "b"})[["a", "b"]].astype(float))
    assert s2["blocks"]["pearson"]["status"] == "ok" and s2["blocks"]["pearson"]["offdiag_rmse"] == 0.0
    for name in ("spearman", "nmi", "eta_squared", "spearman_cross"):
        assert s2["blocks"][name]["status"] == "not_applicable"
    assert s2["blocks"]["eta_squared"]["mean_abs_error"] is None


def test_shuffled_pairings_keep_marginals_and_break_associations(make_schema):
    schema = make_schema(SCHEMA_COLS)
    train, real = _dependent_table(1500, 15), _dependent_table(1500, 16)
    r = np.random.default_rng(17)
    shuffled = pd.DataFrame({c: r.permutation(real[c].to_numpy()) for c in real.columns})
    faithful = real.iloc[r.permutation(len(real))].reset_index(drop=True)          # rows moved jointly
    ctx = MetricContext.fit(train, schema)

    m_shuf, _ = marginal_metrics(ctx, real, shuffled)
    for g in ("continuous", "discrete", "categorical"):
        for k, v in m_shuf["groups"][g].items():
            if k.startswith("mean_"):
                assert v == pytest.approx(0.0, abs=1e-12)                          # same multiset per column

    a_shuf, _ = association_metrics(ctx, real, shuffled)
    a_true, _ = association_metrics(ctx, real, faithful)
    for name in ("pearson", "spearman", "nmi"):
        assert a_true["blocks"][name]["offdiag_rmse"] == pytest.approx(0.0, abs=1e-12)
        assert a_shuf["blocks"][name]["offdiag_rmse"] > 0.15, name
    assert a_true["blocks"]["eta_squared"]["max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert a_shuf["blocks"]["eta_squared"]["max_abs_error"] > 0.4
    assert a_shuf["blocks"]["spearman_cross"]["max_abs_error"] > 0.3


def test_high_cardinality_nominal_is_listed_not_silently_dropped_from_eta(make_schema):
    schema = make_schema([("x", "continuous"), ("small", "categorical"), ("big", "categorical")])
    r = np.random.default_rng(18)
    n = 600
    df = pd.DataFrame({"x": r.normal(size=n), "small": r.choice(["a", "b"], n), "big": np.arange(n) % 60})
    summary, arr = association_metrics(MetricContext.fit(df, schema), df, df)
    blk = summary["blocks"]["eta_squared"]
    assert blk["nominal_columns"] == ["small"]
    assert blk["excluded_high_cardinality_nominal_columns"] == ["big"]
    assert arr["eta2_real"].shape == (1, 1)


def test_invalid_generated_data_propagates(make_schema):
    schema = make_schema(SCHEMA_COLS)
    train, real = _dependent_table(300, 19), _dependent_table(300, 20)
    synth = real.copy()
    synth.loc[0, "x1"] = np.nan
    summary, arr = association_metrics(MetricContext.fit(train, schema), real, synth)
    assert summary["status"] == "invalid_generated_data" and arr == {}
    assert summary["blocks"]["pearson"]["status"] == "invalid_generated_data"
    assert summary["blocks"]["pearson"]["offdiag_rmse"] is None


def test_single_column_table_has_no_association_block(make_schema):
    schema = make_schema([("x", "continuous")])
    df = pd.DataFrame({"x": np.random.default_rng(21).normal(size=30)})
    summary, arr = association_metrics(MetricContext.fit(df, schema), df, df)
    assert summary["status"] == "not_applicable" and arr == {}
    assert all(b["status"] == "not_applicable" for b in summary["blocks"].values())
