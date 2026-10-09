"""
Section 10: mixed-space product-kernel MMD.

Oracles: (i) a dense O(n^2) numpy implementation written here with explicit
broadcasting; (ii) exact population values obtained by ENUMERATING the finite state
space under the documented kernel, with a Monte-Carlo tolerance derived from the
enumerated first-order (Hoeffding) variance of the estimator.
"""
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from sbtab.evaluation import MetricConfig, MetricContext, json_safe, marginal_metrics, mmd_metrics
from sbtab.evaluation.mmd import mmd2_blockwise


# --------------------------------------------------------------------------- dense oracle
def dense_kernel(ZA, CA, ZB, CB, h):
    expo = np.zeros((len(ZA), len(ZB)))
    if ZA.shape[1]:
        expo += ((ZA[:, None, :] - ZB[None, :, :]) ** 2).sum(-1) / (2.0 * h * h)
    if CA.shape[1]:
        expo += (CA[:, None, :] != CB[None, :, :]).sum(-1) / max(CA.shape[1], 1)
    return np.exp(-expo)


def dense_mmd2_unbiased(ZA, CA, ZB, CB, h):
    m, n = len(ZA), len(ZB)
    kaa, kbb, kab = dense_kernel(ZA, CA, ZA, CA, h), dense_kernel(ZB, CB, ZB, CB, h), dense_kernel(ZA, CA, ZB, CB, h)
    return ((kaa.sum() - np.trace(kaa)) / (m * (m - 1)) + (kbb.sum() - np.trace(kbb)) / (n * (n - 1))
            - 2.0 * kab.sum() / (m * n))


def oracle_representation(df, train, cont, disc, nom):
    """z / c built here from the raw frames: discrete columns scaled by TRAIN mean and population std."""
    cols = [df[c].to_numpy(float) for c in cont]
    for c in disc:
        sd = train[c].to_numpy(float).std()
        cols.append((df[c].to_numpy(float) - train[c].to_numpy(float).mean()) / (sd if sd > 0 else 1.0))
    Z = np.column_stack(cols) if cols else np.empty((len(df), 0))
    C = np.column_stack([df[c].astype(str).to_numpy() for c in nom]) if nom else np.empty((len(df), 0), dtype=object)
    return Z, C


def _mixed(n, seed, shift=0.0):
    r = np.random.default_rng(seed)
    g = r.integers(0, 3, n)
    return pd.DataFrame({"x1": r.normal(size=n) + g + shift, "x2": r.normal(size=n),
                         "d": r.poisson(2 + g), "c1": np.array(["a", "b", "c"])[g], "c2": r.integers(0, 4, n),
                         "y": (g + r.integers(0, 2, n)) % 3})


MIXED_COLS = [("x1", "continuous"), ("x2", "continuous"), ("d", "discrete"), ("c1", "categorical"),
              ("c2", "categorical"), ("y", "categorical", "target")]


def test_blockwise_equals_dense():
    r = np.random.default_rng(0)
    ZA, ZB = r.normal(size=(57, 3)), r.normal(size=(43, 3)) + 0.3
    CA, CB = r.integers(0, 3, size=(57, 2)).astype(float), r.integers(0, 3, size=(43, 2)).astype(float)
    want = dense_mmd2_unbiased(ZA, CA, ZB, CB, 1.7)
    for block in (1, 7, 16, 57, 1000):                  # 7 -> 9x9 / 7x7 / 9x7 blocks with ragged last blocks
        got = mmd2_blockwise(ZA, CA, ZB, CB, 1.7, block_size=block)["mmd2_unbiased"]
        assert got == pytest.approx(want, abs=1e-10)
    # pure regimes: an absent factor is omitted
    e = np.empty((57, 0)), np.empty((43, 0))
    assert mmd2_blockwise(ZA, e[0], ZB, e[1], 1.7, 7)["mmd2_unbiased"] == pytest.approx(
        dense_mmd2_unbiased(ZA, e[0], ZB, e[1], 1.7), abs=1e-10)
    assert mmd2_blockwise(e[0], CA, e[1], CB, None, 7)["mmd2_unbiased"] == pytest.approx(
        dense_mmd2_unbiased(e[0], CA, e[1], CB, None), abs=1e-10)


def test_metric_reproduces_from_saved_ids_bandwidth_and_scales(make_schema):
    schema = make_schema(MIXED_COLS, task="classification")
    train, real, synth = _mixed(700, 1), _mixed(520, 2), _mixed(430, 3, shift=0.5)
    config = MetricConfig(mmd_max_rows=150, mmd_block_size=32)
    ctx = MetricContext.fit(train, schema, config)
    real_ids, synth_ids = 10_000 + np.arange(len(real)), 50_000 + np.arange(len(synth))
    out = mmd_metrics(ctx, real, synth, real_row_ids=real_ids, synth_row_ids=synth_ids)
    assert out["status"] == "ok" and out["subsample_size"] == 150 and out["seeds"] == [0, 1, 2]
    assert sorted(out["subsets"]) == ["0", "1", "2"]

    for name, cont, disc, nom in (("full", ["x1", "x2"], ["d"], ["c1", "c2", "y"]),
                                  ("features", ["x1", "x2"], ["d"], ["c1", "c2"])):
        k = out["kernels"][name]
        assert k["nominal_columns"] == nom and len(k["mmd2_unbiased"]) == 3
        h = k["bandwidth"]
        for i, seed in enumerate(("0", "1", "2")):
            ids_r, ids_s = np.array(out["subsets"][seed]["real_ids"]), np.array(out["subsets"][seed]["synth_ids"])
            assert len(ids_r) == len(ids_s) == 150 and len(set(ids_r)) == 150 and len(set(ids_s)) == 150
            assert set(ids_r) <= set(real_ids) and set(ids_s) <= set(synth_ids)
            ZA, CA = oracle_representation(real.iloc[ids_r - 10_000], train, cont, disc, nom)
            ZB, CB = oracle_representation(synth.iloc[ids_s - 50_000], train, cont, disc, nom)
            assert k["mmd2_unbiased"][i] == pytest.approx(dense_mmd2_unbiased(ZA, CA, ZB, CB, h), abs=1e-10)
        assert k["mmd2_unbiased_mean"] == pytest.approx(np.mean(k["mmd2_unbiased"]), abs=1e-15)
        assert k["mmd2_unbiased_subsampling_std"] == pytest.approx(np.std(k["mmd2_unbiased"], ddof=1), abs=1e-15)
        assert k["mmd2_unbiased_subsampling_std"] > 0
    assert out["subsets"]["0"]["real_ids"] != out["subsets"]["1"]["real_ids"]
    assert out["kernels"]["full"]["mmd2_unbiased_mean"] != out["kernels"]["features"]["mmd2_unbiased_mean"]
    assert "subsampling variability" in out["variability_note"]
    json.dumps(json_safe(out), allow_nan=False)


def test_bandwidth_and_scales_depend_on_training_rows_only(make_schema):
    schema = make_schema(MIXED_COLS, task="classification")
    train = _mixed(300, 4)
    ctx = MetricContext.fit(train, schema)
    snapshot = json.dumps(ctx.to_dict(), sort_keys=True)
    # oracle: median positive pairwise distance over ALL training rows (300 <= 1024), by explicit broadcasting
    for name, nom in (("full", ["c1", "c2", "y"]), ("features", ["c1", "c2"])):
        Z, _ = oracle_representation(train, train, ["x1", "x2"], ["d"], nom)
        D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(Z), k=1)]
        meta = ctx.mmd[name]
        assert meta["bandwidth"] == pytest.approx(np.median(D[D > 0]), rel=1e-12)
        assert meta["bandwidth_row_positions"] == list(range(300)) and meta["bandwidth_degenerate"] is False
        assert meta["discrete_scaler"]["mean"]["d"] == pytest.approx(train["d"].mean())
        assert meta["discrete_scaler"]["std"]["d"] == pytest.approx(train["d"].to_numpy(float).std())
        assert "x1" not in meta["discrete_scaler"]["mean"]           # continuous coordinates are NOT rescaled
    h = ctx.mmd["full"]["bandwidth"]
    a = mmd_metrics(ctx, _mixed(200, 5), _mixed(200, 6))
    b = mmd_metrics(ctx, _mixed(150, 7, shift=9.0) * 1, _mixed(260, 8, shift=-40.0))   # wildly different E_k / G_k
    assert a["kernels"]["full"]["bandwidth"] == h == b["kernels"]["full"]["bandwidth"]
    assert json.dumps(ctx.to_dict(), sort_keys=True) == snapshot
    assert b["kernels"]["full"]["mmd2_unbiased_mean"] > 10 * abs(a["kernels"]["full"]["mmd2_unbiased_mean"])


def test_bandwidth_row_selection_is_capped_deterministic_and_saved(make_schema):
    schema = make_schema([("x", "continuous"), ("d", "discrete")])
    r = np.random.default_rng(9)
    train = pd.DataFrame({"x": r.normal(size=3000), "d": r.integers(0, 6, 3000)})
    ids = np.array([f"row-{i}" for i in range(3000)])
    ctx = MetricContext.fit(train, schema, train_row_ids=ids)
    meta = ctx.mmd["full"]
    pos = meta["bandwidth_row_positions"]
    assert len(pos) == 1024 == meta["bandwidth_n_rows"] and len(set(pos)) == 1024 and pos == sorted(pos)
    assert meta["bandwidth_row_ids"] == [f"row-{i}" for i in pos]
    sub = train.iloc[pos]
    Z = np.column_stack([sub["x"], (sub["d"] - train["d"].mean()) / train["d"].to_numpy(float).std()])
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(Z), k=1)]
    assert meta["bandwidth"] == pytest.approx(np.median(D[D > 0]), rel=1e-12)
    assert MetricContext.fit(train, schema, train_row_ids=ids).mmd == ctx.mmd             # deterministic
    assert ctx.mmd["features"]["status"] == "not_applicable"                              # no target in the schema


def test_degenerate_bandwidth_and_zero_std(make_schema):
    schema = make_schema([("x", "continuous"), ("d", "discrete"), ("c", "categorical")])
    train = pd.DataFrame({"x": [0.25] * 20, "d": [3] * 20, "c": ["a", "b"] * 10})
    ctx = MetricContext.fit(train, schema)
    meta = ctx.mmd["full"]
    assert meta["bandwidth"] == 1.0 and meta["bandwidth_degenerate"] is True
    assert meta["discrete_scaler"]["std"]["d"] == 1.0 and meta["discrete_scaler"]["zero_std_columns"] == ["d"]
    out = mmd_metrics(ctx, train, train.assign(d=4))     # d differs by exactly one (unit) scale
    assert out["kernels"]["full"]["bandwidth_degenerate"] is True
    ZA, CA = np.column_stack([train["x"], train["d"] - 3.0]), train[["c"]].to_numpy()
    ZB = np.column_stack([train["x"], np.full(20, 1.0)])
    assert out["kernels"]["full"]["mmd2_unbiased_mean"] == pytest.approx(
        dense_mmd2_unbiased(ZA, CA, ZB, CA, 1.0), abs=1e-12)


# --------------------------------------------------------------------------- enumeration oracles
def population_mmd(states, P, Q, kernel, n):
    """
    Exact population MMD^2 for finite-state P and Q, and a standard deviation for the
    unbiased estimator on n + n iid rows:

      first order (Hoeffding projection):  (4/n) [ Var_P w + Var_Q w ],  w(s) = sum_t (P - Q)(t) k(s, t)
      remainder: the degenerate second-order parts of the three kernel averages,
                 <= 2 Var k/(n(n-1)) twice and 4 Var k/n^2 once, with Var k <= ((kmax - kmin)/2)^2 (Popoviciu).
    """
    K = np.array([[kernel(s, t) for t in states] for s in states])
    p, q = np.array([P.get(s, 0.0) for s in states]), np.array([Q.get(s, 0.0) for s in states])
    mmd2 = p @ K @ p + q @ K @ q - 2 * p @ K @ q
    w = K @ (p - q)
    var1 = (4.0 / n) * ((p @ w ** 2 - (p @ w) ** 2) + (q @ w ** 2 - (q @ w) ** 2))
    vmax = ((K.max() - K.min()) / 2.0) ** 2
    var2 = 4.0 * vmax / (n * (n - 1)) + 4.0 * vmax / n ** 2
    return float(mmd2), math.sqrt(var1 + var2)


def test_two_bit_dependence_invisible_to_marginals_is_seen_by_the_product_kernel(make_schema):
    schema = make_schema([("b1", "categorical"), ("b2", "categorical")])
    n, r = 2000, np.random.default_rng(123)                       # 2000 + 2000 rows, numpy seed 123

    def draw_p(k):                                                # P: uniform on {00, 11}
        b = r.integers(0, 2, k)
        return pd.DataFrame({"b1": b, "b2": b})

    def draw_q(k):                                                # Q: two independent fair bits
        return pd.DataFrame({"b1": r.integers(0, 2, k), "b2": r.integers(0, 2, k)})
    train, real, synth = draw_p(1000), draw_p(n), draw_q(n)
    ctx = MetricContext.fit(train, schema)
    assert ctx.mmd["full"]["bandwidth"] is None                   # no numerical factor in a purely nominal table

    # every marginal agrees: JS ~ (p_hat - q_hat)^2 / 2 with sd(p_hat - q_hat) = sqrt(2 * 0.25 / 2000) = 0.0158,
    # so 5 sd gives JS < 0.0032
    marg, table = marginal_metrics(ctx, real, synth)
    assert table["js"].max() < 0.0032

    states = list(itertools.product([0, 1], repeat=2))
    kernel = lambda s, t: math.exp(-sum(a != b for a, b in zip(s, t)) / 2.0)       # C = 2 nominal columns
    P, Q = {(0, 0): 0.5, (1, 1): 0.5}, {s: 0.25 for s in states}
    pop, sd = population_mmd(states, P, Q, kernel, n)
    # closed form of the same enumeration: (1 + e^-1)/2 - ((1 + e^-1/2)/2)^2
    assert pop == pytest.approx((1 + math.exp(-1)) / 2 - ((1 + math.exp(-0.5)) / 2) ** 2, abs=1e-15)
    assert pop > 0.038 and 5 * sd < 0.01

    out = mmd_metrics(ctx, real, synth)
    est = out["kernels"]["full"]["mmd2_unbiased"]
    assert out["subsample_size"] == n and len(set(est)) == 1      # uncapped: the three subsets coincide
    assert out["kernels"]["full"]["mmd2_unbiased_subsampling_std"] == 0.0
    assert abs(est[0] - pop) < 5 * sd                             # 5 sd of the estimator, derived above
    assert est[0] > 0.03
    # same-size real-real floor is an order of magnitude below the matched real-synthetic value
    floor = out["kernels"]["full"]["floor"]
    assert floor["size"] == 1000 and abs(floor["real_real_mean"]) < 0.005 < floor["real_synth_matched_mean"]

    # the same table declared as ordered DISCRETE bits: RBF factor on train-standardised coordinates
    dschema = make_schema([("b1", "discrete"), ("b2", "discrete")])
    dctx = MetricContext.fit(train, dschema)
    h = dctx.mmd["full"]["bandwidth"]
    sg = train["b1"].to_numpy(float).std()                        # b1 == b2 in P: one scale for both coordinates
    # training rows only take the states 00 and 11, so the single positive distance is sqrt(2)/sg
    assert h == pytest.approx(math.sqrt(2.0) / sg, rel=1e-12)
    kd = lambda s, t: math.exp(-sum(((a - b) / sg) ** 2 for a, b in zip(s, t)) / (2 * h * h))
    pop_d, sd_d = population_mmd(states, P, Q, kd, n)
    est_d = mmd_metrics(dctx, real, synth)["kernels"]["full"]["mmd2_unbiased_mean"]
    assert abs(est_d - pop_d) < 5 * sd_d and est_d > 5 * sd_d
    assert marginal_metrics(dctx, real, synth)[1]["js"].max() < 0.0032


def test_mixed_xor_structure_against_enumeration(make_schema):
    """x (continuous, +-1), c (nominal bit), t = [x > 0] XOR c (nominal). Every pair of columns is independent
    under BOTH P (t determined) and Q (t an independent fair bit): only the three-way joint differs."""
    schema = make_schema([("x", "continuous"), ("c", "categorical"), ("t", "categorical")])
    n, r = 2000, np.random.default_rng(321)                       # 2000 + 2000 rows, numpy seed 321

    def draw(k, xor):
        x = r.choice([-1.0, 1.0], size=k)
        c = r.integers(0, 2, k)
        t = ((x > 0).astype(int) ^ c) if xor else r.integers(0, 2, k)
        return pd.DataFrame({"x": x, "c": c, "t": t})
    train, real, synth = draw(1000, True), draw(n, True), draw(n, False)
    ctx = MetricContext.fit(train, schema)
    h = ctx.mmd["full"]["bandwidth"]
    assert h == 2.0                                               # the only positive distance between x values

    states = list(itertools.product([-1.0, 1.0], [0, 1], [0, 1]))
    kernel = lambda s, u: math.exp(-(s[0] - u[0]) ** 2 / (2 * h * h) - ((s[1] != u[1]) + (s[2] != u[2])) / 2.0)
    P = {s: 0.25 for s in states if s[2] == (int(s[0] > 0) ^ s[1])}
    Q = {s: 0.125 for s in states}
    pop, sd = population_mmd(states, P, Q, kernel, n)
    # by hand, with a = e^{-1/2} for every differing coordinate (x: (2)^2/(2*2^2) = 1/2; c, t: 1/2 each):
    #   E_PP k = (1 + 3 a^2)/4  (two P-states differ in exactly 0 or 2 coordinates),  E_QQ k = E_PQ k = ((1+a)/2)^3
    a = math.exp(-0.5)
    assert pop == pytest.approx((1 + 3 * a * a) / 4 - ((1 + a) / 2) ** 3, abs=1e-15)       # = 0.00761
    assert 5 * sd < pop / 2                # sd = 0.00065 at n = 2000: the 5 sd band separates MMD^2 from 0

    out = mmd_metrics(ctx, real, synth)
    est = out["kernels"]["full"]["mmd2_unbiased_mean"]
    assert abs(est - pop) < 5 * sd and est > pop / 2
    # marginals cannot see it: |WD| = 2 |p_hat - q_hat| < 2 * 5 * 0.0158 ; JS < 0.0032 (see the two-bit test)
    _, table = marginal_metrics(ctx, real, synth)
    assert table["wd"].max() < 0.16 and table["js"].max() < 0.0032
    # a SUM of per-column kernels is blind to it as well (its population MMD^2 is exactly zero)
    ksum = lambda s, u: (math.exp(-(s[0] - u[0]) ** 2 / (2 * h * h)) + math.exp(-(s[1] != u[1]) / 2.0)
                         + math.exp(-(s[2] != u[2]) / 2.0))
    assert population_mmd(states, P, Q, ksum, n)[0] == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- sign, floor, availability
def test_unbiased_estimate_is_signed_and_never_clipped(make_schema):
    # identical two-point samples {0, 1}, h = 1: k = exp(-1/2)
    #   within terms: 2k/(2*1) each ; cross term: 2 (2 + 2k)/4  ->  MMD^2_u = 2k - (1 + k) = k - 1 < 0
    Z = np.array([[0.0], [1.0]])
    E = np.empty((2, 0))
    got = mmd2_blockwise(Z, E, Z, E, 1.0, block_size=1)
    assert got["mmd2_unbiased"] == pytest.approx(math.exp(-0.5) - 1.0, abs=1e-15)
    assert got["mmd2_biased"] == pytest.approx(0.0, abs=1e-15) and got["mmd2_biased"] >= 0.0

    schema = make_schema(MIXED_COLS, task="classification")
    ctx = MetricContext.fit(_mixed(400, 40), schema)
    values = [mmd_metrics(ctx, _mixed(300, 100 + i), _mixed(300, 200 + i))["kernels"]["full"] for i in range(8)]
    unbiased = [v["mmd2_unbiased_mean"] for v in values]
    assert min(unbiased) < 0 < max(unbiased)                     # same distribution: both signs occur
    assert all(v["mmd2_biased_mean"] >= 0 for v in values)       # the separately named presentation estimate
    assert abs(np.mean(unbiased)) < 0.003


def test_matched_real_real_floor(make_schema):
    schema = make_schema(MIXED_COLS, task="classification")
    train, real, synth = _mixed(500, 41), _mixed(301, 42), _mixed(400, 43, shift=1.0)
    ctx = MetricContext.fit(train, schema, MetricConfig(mmd_block_size=64))
    out = mmd_metrics(ctx, real, synth)
    assert out["floor_size"] == 150                              # floor(301 / 2)
    for seed in ("0", "1", "2"):
        f = out["floor_subsets"][seed]
        a, b, g = set(f["real_a_ids"]), set(f["real_b_ids"]), f["synth_ids"]
        assert len(a) == len(b) == len(g) == 150 and not (a & b)  # two DISJOINT halves of E_k, matched size
    floor = out["kernels"]["full"]["floor"]
    assert floor["status"] == "ok" and floor["size"] == 150 and len(floor["real_real_mmd2_unbiased"]) == 3
    # reproduce seed 0 from the saved ids
    f = out["floor_subsets"]["0"]
    cont, disc, nom = ["x1", "x2"], ["d"], ["c1", "c2", "y"]
    ZA, CA = oracle_representation(real.iloc[f["real_a_ids"]], train, cont, disc, nom)
    ZB, CB = oracle_representation(real.iloc[f["real_b_ids"]], train, cont, disc, nom)
    ZG, CG = oracle_representation(synth.iloc[f["synth_ids"]], train, cont, disc, nom)
    h = out["kernels"]["full"]["bandwidth"]
    assert floor["real_real_mmd2_unbiased"][0] == pytest.approx(dense_mmd2_unbiased(ZA, CA, ZB, CB, h), abs=1e-10)
    assert floor["real_synth_matched_mmd2_unbiased"][0] == pytest.approx(
        dense_mmd2_unbiased(ZA, CA, ZG, CG, h), abs=1e-10)
    assert floor["real_synth_matched_mean"] > 10 * abs(floor["real_real_mean"])
    assert "features" in out["kernels"] and out["kernels"]["features"]["floor"]["size"] == 150


def test_fewer_than_two_rows_is_unavailable(make_schema):
    schema = make_schema(MIXED_COLS, task="classification")
    train = _mixed(200, 44)
    ctx = MetricContext.fit(train, schema)
    three = mmd_metrics(ctx, _mixed(3, 45), _mixed(50, 46))       # floor(3/2) = 1 row per side
    assert three["status"] == "ok" and three["subsample_size"] == 3
    assert three["kernels"]["full"]["floor"]["status"] == "insufficient_data"
    assert "real_real_mean" not in three["kernels"]["full"]["floor"]
    one = mmd_metrics(ctx, _mixed(1, 47), _mixed(50, 48))
    assert one["status"] == "insufficient_data" and one["kernels"]["full"]["status"] == "insufficient_data"
    assert "mmd2_unbiased_mean" not in one["kernels"]["full"]
    bad = _mixed(50, 49)
    bad.loc[0, "x1"] = np.nan
    inv = mmd_metrics(ctx, _mixed(50, 50), bad)
    assert inv["status"] == "invalid_generated_data" and "mmd2_unbiased_mean" not in inv["kernels"]["full"]
    json.dumps(json_safe([three, one, inv]), allow_nan=False)


def test_nominal_label_permutation_leaves_mmd_unchanged(make_schema):
    schema = make_schema(MIXED_COLS, task="classification")
    train, real, synth = _mixed(300, 51), _mixed(250, 52), _mixed(250, 53, shift=0.4)
    recode = lambda df: df.assign(c1=df["c1"].map({"a": "c", "b": "a", "c": "b"}), y=df["y"].map({0: 2, 1: 0, 2: 1}))
    a = mmd_metrics(MetricContext.fit(train, schema), real, synth)
    b = mmd_metrics(MetricContext.fit(recode(train), schema), recode(real), recode(synth))
    for name in ("full", "features"):
        assert a["kernels"][name]["mmd2_unbiased"] == pytest.approx(b["kernels"][name]["mmd2_unbiased"], abs=1e-13)


def test_floor_is_matched_even_when_the_generated_table_is_small(make_schema):
    schema = make_schema(MIXED_COLS, task="classification")
    ctx = MetricContext.fit(_mixed(300, 60), schema)
    out = mmd_metrics(ctx, _mixed(400, 61), _mixed(90, 62))
    assert out["subsample_size"] == 90 and out["floor_size"] == 90 and out["floor_size_limited_by_synth"] is True
    f = out["floor_subsets"]["0"]
    assert len(f["real_a_ids"]) == len(f["real_b_ids"]) == len(f["synth_ids"]) == 90
    assert mmd_metrics(ctx, _mixed(400, 61), _mixed(900, 63))["floor_size_limited_by_synth"] is False
