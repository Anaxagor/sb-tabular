"""
Section 8: tuning objective and final marginal metrics.

Oracles are hand-computed numbers, scipy reference functions or numpy code written
here independently of sbtab.evaluation (np.histogram, explicit formulas).
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon

from sbtab.evaluation import (MetricConfig, MetricContext, check_validity, json_safe, marginal_metrics,
                              tuning_objective)
from sbtab.evaluation._common import hist_counts

LOG2 = math.log(2.0)


def _row(table, col):
    return table.set_index("column").loc[col]


# --------------------------------------------------------------------------- JS
def test_js_hand_values(make_schema):
    schema = make_schema([("c", "categorical"), ("d", "categorical")])
    # c: p = (1/2, 1/2, 0), q = (0, 1/2, 1/2)  ->  m = (1/4, 1/2, 1/4)
    #    KL(p||m) = 1/2 log 2 + 1/2 log 1 = 1/2 log 2 = KL(q||m)   =>  JS = 1/2 log 2
    # d: disjoint supports  =>  JS = log 2 (the maximum)
    real = pd.DataFrame({"c": ["a", "a", "b", "b"], "d": ["u", "u", "u", "v"]})
    synth = pd.DataFrame({"c": ["b", "c", "c", "b"], "d": ["w", "w", "x", "w"]})
    out = tuning_objective(real, synth, schema)
    assert out["status"] == "ok"
    assert out["per_column"]["c"]["value"] == pytest.approx(0.5 * LOG2, abs=1e-14)
    assert out["per_column"]["d"]["value"] == pytest.approx(LOG2, abs=1e-14)
    assert out["mean_js"] == pytest.approx(0.75 * LOG2, abs=1e-14)
    assert out["mean_wd"] is None and out["objective"] == pytest.approx(0.75 * LOG2, abs=1e-14)


def test_js_is_the_divergence_not_scipys_distance(make_schema):
    rng = np.random.default_rng(11)
    schema = make_schema([("c", "categorical")])
    real = pd.DataFrame({"c": rng.choice(list("abcdef"), size=500, p=[.3, .25, .2, .15, .07, .03])})
    synth = pd.DataFrame({"c": rng.choice(list("bcdefg"), size=700, p=[.1, .2, .3, .2, .1, .1])})
    labels = sorted(set(real["c"]) | set(synth["c"]))
    p = real["c"].value_counts(normalize=True).reindex(labels, fill_value=0.0).to_numpy()
    q = synth["c"].value_counts(normalize=True).reindex(labels, fill_value=0.0).to_numpy()
    distance = jensenshannon(p, q)                     # scipy: SQUARE ROOT of the divergence, natural log
    got = tuning_objective(real, synth, schema)["per_column"]["c"]["value"]
    assert got == pytest.approx(distance ** 2, abs=1e-12)
    assert abs(got - distance) > 0.05                  # would catch returning the distance
    assert 0.0 < got <= LOG2


def test_support_is_aligned_by_label_not_by_position(make_schema):
    schema = make_schema([("c", "categorical")])
    real = pd.DataFrame({"c": ["a"] * 30 + ["b"] * 70})
    synth = pd.DataFrame({"c": ["b"] * 70 + ["a"] * 30})        # other order of first appearance
    assert tuning_objective(real, synth, schema)["per_column"]["c"]["value"] == 0.0
    ctx = MetricContext.fit(real, schema)
    summary, table = marginal_metrics(ctx, real, synth)
    assert _row(table, "c")["js"] == 0.0 and _row(table, "c")["kl"] == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- KL
def _smoothed_kl(p, q, eps):
    b = len(p)
    ps = [(x + eps / b) / (1 + eps) for x in p]
    qs = [(x + eps / b) / (1 + eps) for x in q]
    return sum(a * math.log(a / c) for a, c in zip(ps, qs))


@pytest.mark.parametrize("eps", [1e-6, 0.3])
def test_discrete_kl_smoothing_formula(make_schema, eps):
    schema = make_schema([("d", "discrete")])
    train = pd.DataFrame({"d": [0, 1]})
    real = pd.DataFrame({"d": [0, 0, 0, 1]})             # p = (3/4, 1/4)
    synth = pd.DataFrame({"d": [0, 0, 1, 1]})            # q = (1/2, 1/2)
    ctx = MetricContext.fit(train, schema, MetricConfig(kl_smoothing_mass=eps))
    summary, table = marginal_metrics(ctx, real, synth)
    # B = 3 bins: the training support {0, 1} plus the unexpected-value bin
    expected = _smoothed_kl([0.75, 0.25, 0.0], [0.5, 0.5, 0.0], eps)
    assert _row(table, "d")["kl_n_bins"] == 3
    assert _row(table, "d")["kl"] == pytest.approx(expected, abs=1e-14)
    assert summary["kl_smoothing_mass"] == eps and summary["kl_direction"] == "KL(real||synthetic)"
    if eps == 0.3:
        # written out: p_s = (.85, .35, .1)/1.3, q_s = (.6, .6, .1)/1.3
        by_hand = (.85 * math.log(.85 / .6) + .35 * math.log(.35 / .6) + .1 * math.log(1.0)) / 1.3
        assert expected == pytest.approx(by_hand, abs=1e-15)
        unsmoothed = .75 * math.log(1.5) + .25 * math.log(.5)
        assert abs(expected - unsmoothed) > 0.02         # smoothing really is applied
    # direction: KL(synth||real) is a different number
    assert _row(table, "d")["kl"] != pytest.approx(_smoothed_kl([0.5, 0.5, 0.0], [0.75, 0.25, 0.0], eps), abs=1e-6)


def test_kl_is_finite_and_large_when_synthetic_drops_a_supported_value(make_schema):
    schema = make_schema([("d", "discrete")])
    train = pd.DataFrame({"d": [0, 1] * 50})
    synth = pd.DataFrame({"d": [0] * 100})
    ctx = MetricContext.fit(train, schema)
    _, table = marginal_metrics(ctx, train, synth)
    expected = _smoothed_kl([0.5, 0.5, 0.0], [1.0, 0.0, 0.0], 1e-6)     # ~ 0.5*log(0.5/3.3e-7) - 0.35
    assert expected > 6.0
    assert _row(table, "d")["kl"] == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- WD
def test_wd_of_a_shift_equals_the_shift(make_schema):
    rng = np.random.default_rng(3)
    schema = make_schema([("x", "continuous")])
    real = pd.DataFrame({"x": rng.normal(size=400)})
    synth = pd.DataFrame({"x": real["x"].to_numpy()[::-1] - 0.75})
    out = tuning_objective(real, synth, schema)
    assert out["per_column"]["x"]["value"] == pytest.approx(0.75, abs=1e-12)
    assert out["objective"] == pytest.approx(0.75, abs=1e-12) and out["mean_js"] is None
    ctx = MetricContext.fit(real, schema)
    summary, table = marginal_metrics(ctx, real, synth)
    assert _row(table, "x")["wd"] == pytest.approx(0.75, abs=1e-12)
    assert summary["groups"]["continuous"]["mean_wd"] == pytest.approx(0.75, abs=1e-12)


# --------------------------------------------------------------------------- histogram bins
def _oracle_hist_kl(train_x, real_x, synth_x, eps=1e-6):
    """np.histogram on 48 equal bins over [train min, train max] + explicit tails."""
    lo, hi = train_x.min(), train_x.max()
    edges = np.array([lo + i * (hi - lo) / 48 for i in range(49)])
    edges[-1] = hi

    def probs(x):
        inner, _ = np.histogram(x[(x >= lo) & (x <= hi)], bins=edges)      # numpy closes the last bin on the right
        c = np.concatenate([[np.sum(x < lo)], inner, [np.sum(x > hi)]]).astype(float)
        assert c.sum() == len(x)
        p = c / c.sum()
        return (p + eps / 50) / (1 + eps)
    p, q = probs(real_x), probs(synth_x)
    return float(np.sum(p * np.log(p / q))), edges


def test_edges_come_from_train_and_are_identical_for_every_model(make_schema):
    rng = np.random.default_rng(5)
    schema = make_schema([("x", "continuous")])
    train = pd.DataFrame({"x": rng.normal(size=1000)})
    real = pd.DataFrame({"x": rng.normal(size=600) * 1.4})            # wider than train: real tails are non-empty
    model_a = pd.DataFrame({"x": rng.normal(size=800) * 0.5 + 0.3})
    model_b = pd.DataFrame({"x": rng.standard_t(df=2, size=900) * 2.0})
    ctx = MetricContext.fit(train, schema)
    edges_before = ctx.hist["x"]["edges"].copy()
    assert len(edges_before) == 49                                     # 48 interior intervals
    assert edges_before[0] == train["x"].min() and edges_before[-1] == train["x"].max()

    (_, ta), (_, tb) = marginal_metrics(ctx, real, model_a), marginal_metrics(ctx, real, model_b)
    assert np.array_equal(ctx.hist["x"]["edges"], edges_before)        # no model (nor E_k) moved the edges
    assert np.array_equal(np.array(ctx.to_dict()["histograms"]["x"]["edges"]), edges_before)
    for table, model in ((ta, model_a), (tb, model_b)):
        kl, edges = _oracle_hist_kl(train["x"].to_numpy(), real["x"].to_numpy(), model["x"].to_numpy())
        assert np.allclose(edges, edges_before, rtol=0, atol=1e-12)
        assert _row(table, "x")["kl"] == pytest.approx(kl, rel=1e-10)
        assert _row(table, "x")["kl_n_bins"] == 50
    assert _row(ta, "x")["real_underflow_mass"] == pytest.approx(np.mean(real["x"] < train["x"].min()))
    assert _row(ta, "x")["real_overflow_mass"] == pytest.approx(np.mean(real["x"] > train["x"].max()))
    assert _row(ta, "x")["real_overflow_mass"] > 0


def test_fifty_bins_and_tail_mass_is_scored_not_ignored(make_schema):
    rng = np.random.default_rng(6)
    schema = make_schema([("x", "continuous")])
    train = pd.DataFrame({"x": rng.normal(size=500)})
    far = pd.DataFrame({"x": train["x"].max() + 100.0 + rng.random(300)})
    ctx = MetricContext.fit(train, schema)
    summary, table = marginal_metrics(ctx, train, far)
    r = _row(table, "x")
    assert r["kl_n_bins"] == 50 and summary["kl_total_bins"] == 50
    assert r["synth_overflow_mass"] == 1.0 and r["synth_underflow_mass"] == 0.0
    # all synthetic mass sits in the overflow bin where real has only the smoothing mass:
    # KL = sum_i p_i log(p_i / (eps/50)) >= log(50/eps) - log(48) = 17.7 - 3.9
    assert r["kl"] > math.log(50 / 1e-6) - math.log(48) - 1e-6
    assert summary["status"] == "ok"                                   # valid numbers, terrible score

    below = pd.DataFrame({"x": train["x"].min() - 5.0 - rng.random(300)})
    _, t2 = marginal_metrics(ctx, train, below)
    assert _row(t2, "x")["synth_underflow_mass"] == 1.0


def test_both_training_extrema_are_interior(make_schema):
    rng = np.random.default_rng(7)
    schema = make_schema([("x", "continuous")])
    train = pd.DataFrame({"x": rng.normal(size=257)})
    ctx = MetricContext.fit(train, schema)
    edges = ctx.hist["x"]["edges"]
    lo, hi = train["x"].min(), train["x"].max()
    counts = hist_counts(np.array([lo, hi]), edges)
    assert counts.shape == (50,)
    assert counts[0] == 0 and counts[-1] == 0 and counts[1] == 1 and counts[48] == 1
    tails = hist_counts(np.array([np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)]), edges)
    assert tails[0] == 1 and tails[-1] == 1 and tails[1:-1].sum() == 0
    _, table = marginal_metrics(ctx, train, train)
    assert _row(table, "x")["real_underflow_mass"] == 0.0 and _row(table, "x")["real_overflow_mass"] == 0.0
    assert _row(table, "x")["kl"] == pytest.approx(0.0, abs=1e-15)


def test_constant_training_column(make_schema):
    schema = make_schema([("x", "continuous"), ("z", "continuous")])
    rng = np.random.default_rng(8)
    train = pd.DataFrame({"x": np.full(50, 3.0), "z": rng.normal(size=50)})
    ctx = MetricContext.fit(train, schema)
    h = ctx.hist["x"]
    assert h["constant"] is True and ctx.hist["z"]["constant"] is False
    assert len(h["edges"]) == 49 and h["edges"][0] == 2.5 and h["edges"][-1] == 3.5      # unit width around 3
    assert np.allclose(np.diff(h["edges"]), 1.0 / 48, atol=1e-15)
    summary, table = marginal_metrics(ctx, train, train)
    assert summary["constant_training_columns"] == ["x"] and bool(_row(table, "x")["train_constant"]) is True
    assert _row(table, "x")["kl"] == pytest.approx(0.0, abs=1e-15)
    moved = train.assign(x=10.0)
    _, t2 = marginal_metrics(ctx, train, moved)
    assert _row(t2, "x")["synth_overflow_mass"] == 1.0 and _row(t2, "x")["kl"] > 10.0
    assert _row(t2, "x")["wd"] == pytest.approx(7.0)


# --------------------------------------------------------------------------- invalid generated data
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_never_score_perfectly(make_schema, bad_value):
    rng = np.random.default_rng(9)
    schema = make_schema([("x", "continuous"), ("c", "categorical")])
    train = pd.DataFrame({"x": rng.normal(size=300), "c": rng.choice(["a", "b"], size=300)})
    synth = train.copy()                                  # a filter-then-score implementation would see a perfect copy
    synth.loc[[3, 17], "x"] = bad_value
    ctx = MetricContext.fit(train, schema)
    validity = check_validity(synth, schema, ctx)
    assert validity["status"] == "invalid_generated_data"
    assert validity["nonfinite_rate"]["x"] == pytest.approx(2 / 300) and validity["n_rows"] == 300
    assert validity["n_invalid_rows"] == 2

    summary, table = marginal_metrics(ctx, train, synth)
    assert summary["status"] == "invalid_generated_data"
    g = summary["groups"]
    assert g["continuous"]["mean_wd"] is None and g["continuous"]["mean_kl"] is None
    assert g["categorical"]["mean_kl"] is None and g["categorical"]["mean_js"] is None
    assert summary["tuning_objective_equivalent"] is None
    assert _row(table, "x")["status"] == "invalid_generated_data" and np.isnan(_row(table, "x")["kl"])

    obj = tuning_objective(train, synth, schema)
    assert obj["status"] == "invalid_generated_data" and obj["objective"] is None
    assert obj["mean_wd"] is None and obj["mean_js"] is None
    json.dumps(json_safe(summary), allow_nan=False)
    json.dumps(json_safe(obj), allow_nan=False)


def test_out_of_vocabulary_category_is_invalid_and_lands_in_the_unexpected_bin(make_schema):
    rng = np.random.default_rng(10)
    schema = make_schema([("c", "categorical"), ("x", "continuous")])
    train = pd.DataFrame({"c": rng.choice(["a", "b", "c"], size=400), "x": rng.normal(size=400)})
    synth = train.copy()
    synth.loc[:39, "c"] = "zzz"                           # 10% labels outside the training vocabulary
    ctx = MetricContext.fit(train, schema)
    validity = check_validity(synth, schema, ctx)
    assert validity["status"] == "invalid_generated_data"
    assert "unexpected_categorical_labels" in validity["reasons"]
    assert validity["unexpected_value_rate"]["c"] == pytest.approx(0.1)
    summary, table = marginal_metrics(ctx, train, synth)
    assert summary["status"] == "invalid_generated_data" and summary["groups"]["categorical"]["mean_kl"] is None
    r = _row(table, "c")
    assert r["status"] == "invalid_generated_data"
    assert r["synth_unexpected_mass"] == pytest.approx(0.1) and r["kl"] > 0.05 and r["js"] > 0.01
    # without the training context (tuning) the unknown label still costs JS through the label union
    obj = tuning_objective(train, synth, schema)
    assert obj["per_column"]["c"]["value"] > 0.01 and obj["objective"] > 0.01


def test_out_of_support_discrete_value_is_scored_through_the_unexpected_bin(make_schema):
    schema = make_schema([("d", "discrete")])
    train = pd.DataFrame({"d": [0, 1, 2, 3] * 100})
    synth = train.copy()
    synth.loc[:79, "d"] = 17                               # a legal number that training never showed
    ctx = MetricContext.fit(train, schema)
    validity = check_validity(synth, schema, ctx)
    assert validity["status"] == "ok" and validity["unexpected_value_rate"]["d"] == pytest.approx(0.2)
    summary, table = marginal_metrics(ctx, train, synth)
    r = _row(table, "d")
    assert r["kl_n_bins"] == 5 and r["synth_unexpected_mass"] == pytest.approx(0.2) and r["real_unexpected_mass"] == 0
    # KL(real||synth): real mass 1/4 per value against 0.2, 0.2, 0.2, 0.2 (+0.2 unexpected)... rows 0..79 hold
    # 20 of each value, so synth = (80, 80, 80, 80, 80)/400 and KL = log(0.25/0.2) up to the 1e-6 smoothing
    assert r["kl"] == pytest.approx(math.log(1.25), abs=1e-4)
    assert summary["status"] == "ok" and summary["groups"]["discrete"]["mean_kl"] > 0.2


def test_missing_column_and_empty_table_are_invalid(make_schema):
    schema = make_schema([("x", "continuous"), ("c", "categorical")])
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0], "c": ["a", "b", "a"]})
    ctx = MetricContext.fit(train, schema)
    v = check_validity(train[["x"]], schema, ctx)
    assert v["status"] == "invalid_generated_data" and v["missing_columns"] == ["c"]
    summary, _ = marginal_metrics(ctx, train, train[["x"]])
    assert summary["status"] == "invalid_generated_data" and summary["groups"]["continuous"]["mean_wd"] is None
    v0 = check_validity(train.iloc[:0], schema, ctx)
    assert v0["status"] == "invalid_generated_data" and "no_rows" in v0["reasons"]
    assert tuning_objective(train, train.iloc[:0], schema)["objective"] is None


# --------------------------------------------------------------------------- grouping rules
def test_mixed_objective_uses_one_combined_finite_mean(make_schema):
    schema = make_schema([("x", "continuous"), ("d1", "discrete"), ("d2", "discrete"), ("c", "categorical")])
    n = 40
    real = pd.DataFrame({"x": np.linspace(-1, 1, n), "d1": [0, 1] * (n // 2), "d2": [5, 6, 7, 8] * (n // 4),
                         "c": ["a"] * n})
    synth = real.copy()
    synth["x"] = real["x"] + 0.5                           # WD = 0.5
    synth["c"] = "b"                                       # JS = log 2 ; d1, d2 identical -> JS = 0
    out = tuning_objective(real, synth, schema)
    combined = (0.0 + 0.0 + LOG2) / 3                      # ONE mean over the 3 finite columns
    separate = (0.0 + 0.0) / 2 + LOG2 / 1                  # what summing two group means would give
    assert out["mean_js"] == pytest.approx(combined, abs=1e-14)
    assert out["objective"] == pytest.approx(0.5 + combined, abs=1e-12)
    assert abs(out["objective"] - (0.5 + separate)) > 0.4

    train = pd.concat([real, synth], ignore_index=True)     # the vocabulary of c must contain "b" to be valid
    ctx = MetricContext.fit(train, schema)
    summary, _ = marginal_metrics(ctx, real, synth)
    g = summary["groups"]
    assert g["discrete"]["mean_js"] == 0.0 and g["discrete"]["n_columns"] == 2
    assert g["categorical"]["mean_js"] == pytest.approx(LOG2) and g["categorical"]["n_columns"] == 1
    assert g["finite_combined"]["mean_js"] == pytest.approx(combined) and g["finite_combined"]["n_columns"] == 3
    assert summary["tuning_objective_equivalent"] == pytest.approx(out["objective"], abs=1e-12)


def test_target_is_counted_exactly_once_in_its_declared_group(make_schema):
    schema = make_schema([("x", "continuous"), ("c", "categorical"), ("y", "categorical", "target")],
                         task="classification")
    n = 20
    real = pd.DataFrame({"x": np.arange(n, dtype=float), "c": ["a", "b"] * (n // 2), "y": [0] * n})
    synth = real.assign(y=1)                               # only the target differs: JS(y) = log 2
    out = tuning_objective(real, synth, schema)
    assert sorted(out["per_column"]) == ["c", "x", "y"] and out["n_finite"] == 2 and out["n_continuous"] == 1
    assert out["mean_js"] == pytest.approx(LOG2 / 2)       # (0 + log 2) / 2 : y once, not zero times, not twice
    ctx = MetricContext.fit(real, schema)
    summary, table = marginal_metrics(ctx, real, synth.assign(y=0))
    assert summary["groups"]["categorical"]["columns"] == ["c", "y"]
    assert summary["groups"]["continuous"]["columns"] == ["x"] and summary["groups"]["discrete"]["n_columns"] == 0
    assert list(table["column"]) == ["x", "c", "y"] and int(table["is_target"].sum()) == 1

    reg = make_schema([("x", "continuous"), ("y", "continuous", "target")], task="regression")
    real_r = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)})
    out_r = tuning_objective(real_r, real_r.assign(y=real_r["y"] + 2.0), reg)
    assert out_r["mean_wd"] == pytest.approx(1.0)          # (0 + 2) / 2: the continuous target is in the WD mean


def test_empty_groups_are_none_not_zero(make_schema):
    rng = np.random.default_rng(12)
    cont = make_schema([("a", "continuous"), ("b", "continuous")])
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    summary, _ = marginal_metrics(MetricContext.fit(df, cont), df, df + 0.1)
    for name in ("discrete", "categorical", "finite_combined"):
        g = summary["groups"][name]
        assert g["status"] == "not_applicable" and g["n_columns"] == 0
        assert g["mean_kl"] is None and g["mean_js"] is None
    assert summary["groups"]["continuous"]["status"] == "ok"
    obj = tuning_objective(df, df + 0.1, cont)
    assert obj["regime"] == "continuous" and obj["mean_js"] is None
    assert obj["objective"] == pytest.approx(obj["mean_wd"]) == pytest.approx(0.1)

    disc = make_schema([("d", "discrete"), ("c", "categorical")])
    dd = pd.DataFrame({"d": rng.integers(0, 4, 100), "c": rng.choice(["u", "v"], 100)})
    s2, _ = marginal_metrics(MetricContext.fit(dd, disc), dd, dd)
    g = s2["groups"]["continuous"]
    assert g["status"] == "not_applicable" and g["mean_wd"] is None and g["mean_kl"] is None
    assert s2["groups"]["discrete"]["mean_kl"] == pytest.approx(0.0, abs=1e-15)
    obj2 = tuning_objective(dd, dd, disc)
    assert obj2["regime"] == "discrete" and obj2["mean_wd"] is None and obj2["objective"] == 0.0


def test_label_recoding_leaves_marginal_metrics_unchanged(make_schema):
    rng = np.random.default_rng(13)
    schema = make_schema([("c", "categorical"), ("k", "categorical")])
    def draw(n, p):
        return pd.DataFrame({"c": rng.choice(["a", "b", "c"], size=n, p=p), "k": rng.choice([0, 1, 2, 3], size=n)})
    train, real, synth = draw(500, [.5, .3, .2]), draw(400, [.5, .3, .2]), draw(300, [.2, .3, .5])
    recode_c, recode_k = {"a": "zeta", "b": "alpha", "c": "mid"}, {0: 3, 1: 0, 2: 1, 3: 2}
    def recode(df):
        return pd.DataFrame({"c": df["c"].map(recode_c), "k": df["k"].map(recode_k)})
    s1, t1 = marginal_metrics(MetricContext.fit(train, schema), real, synth)
    s2, t2 = marginal_metrics(MetricContext.fit(recode(train), schema), recode(real), recode(synth))
    assert np.allclose(t1[["kl", "js"]].to_numpy(), t2[["kl", "js"]].to_numpy(), rtol=0, atol=1e-14)
    assert s1["groups"]["categorical"]["mean_kl"] == pytest.approx(s2["groups"]["categorical"]["mean_kl"], abs=1e-14)
    assert s1["groups"]["categorical"]["mean_kl"] > 0.05
