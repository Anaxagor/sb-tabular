"""MetricContext persistence, configuration versioning, JSON safety and the status vocabulary."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import sbtab.evaluation as ev
from sbtab.evaluation import (METRIC_VERSION, STATUSES, MetricConfig, MetricContext, association_metrics,
                              check_validity, conditional_metrics, json_safe, marginal_metrics, mmd_metrics,
                              tuning_objective)


def _table(n, seed, shift=0.0):
    r = np.random.default_rng(seed)
    g = r.integers(0, 4, n)
    return pd.DataFrame({
        "x1": r.normal(size=n) + 0.7 * g + shift, "x2": r.exponential(size=n),
        "d1": r.poisson(1 + g), "d2": r.integers(0, 3, n),
        "c1": np.array(["n", "e", "s", "w"])[g], "c2": r.choice([10, 20, 30], size=n),
        "y": 2.0 * g + r.normal(size=n),
    })


COLS = [("x1", "continuous"), ("x2", "continuous"), ("d1", "discrete"), ("d2", "discrete"),
        ("c1", "categorical"), ("c2", "categorical"), ("y", "continuous", "target")]


def test_public_api_names_and_constants():
    assert METRIC_VERSION == "sbtab.metrics/2"
    assert STATUSES == ("ok", "not_applicable", "insufficient_data", "incomplete_conditional_coverage", "undefined",
                        "invalid_generated_data", "training_failed", "sampling_failed", "utility_fit_failed",
                        "blocked_support")
    for name in ("METRIC_VERSION", "STATUSES", "MetricConfig", "MetricContext", "check_validity",
                 "tuning_objective", "marginal_metrics", "association_metrics", "conditional_metrics",
                 "mmd_metrics", "resolve_utility_params", "utility_reference", "utility_tstr", "json_safe"):
        assert hasattr(ev, name), name
    c = MetricConfig()
    assert (c.kl_total_bins, c.kl_smoothing_mass, c.conditional_max_levels, c.conditional_min_rows,
            c.regression_target_strata, c.mmd_bandwidth_max_rows, c.mmd_max_rows, c.mmd_seeds, c.mmd_block_size,
            c.utility_seed, c.utility_thread_count) == (50, 1e-6, 50, 10, 5, 1024, 2048, (0, 1, 2), 1024, 0, 4)


def test_config_roundtrip_and_hash():
    c = MetricConfig(kl_total_bins=30, mmd_seeds=[5, 6])
    assert c.mmd_seeds == (5, 6)
    d = json.loads(json.dumps(c.to_dict()))
    assert MetricConfig.from_dict(d) == c and MetricConfig.from_dict(d).hash() == c.hash()
    assert c.hash() != MetricConfig().hash() and MetricConfig().hash() == MetricConfig().hash()
    assert MetricConfig(kl_smoothing_mass=1e-5).hash() != MetricConfig().hash()
    with pytest.raises(ValueError):
        MetricConfig.from_dict({"kl_total_bins": 50, "not_a_metric_constant": 1})


def test_json_safe():
    raw = {"a": np.float64("nan"), "b": [np.inf, -np.inf, np.float32(1.5)], "c": np.int64(3), "d": np.bool_(True),
           "e": np.array([[1.0, np.nan]]), "f": (1, 2), np.int64(7): "numpy key", "g": None, "h": {"i": pd.NA},
           "j": pd.Series([1, 2]), "k": float("nan")}
    out = json_safe(raw)
    text = json.dumps(out, allow_nan=False)
    back = json.loads(text)
    assert back == {"a": None, "b": [None, None, 1.5], "c": 3, "d": True, "e": [[1.0, None]], "f": [1, 2],
                    "7": "numpy key", "g": None, "h": {"i": None}, "j": [1, 2], "k": None}
    assert isinstance(out["c"], int) and isinstance(out["d"], bool)


def test_context_roundtrip_gives_identical_metrics(make_schema, collect_statuses):
    schema = make_schema(COLS, task="regression")
    train, real, synth = _table(900, 1), _table(500, 2), _table(450, 3, shift=0.4)
    config = MetricConfig(mmd_max_rows=120, mmd_block_size=50)
    ids = np.arange(len(train)) * 3 + 11
    ctx = MetricContext.fit(train, schema, config, train_row_ids=ids)

    d = ctx.to_dict()
    text = json.dumps(d, allow_nan=False, sort_keys=True)          # standards-compliant JSON
    ctx2 = MetricContext.from_dict(json.loads(text), schema)
    assert ctx2.spec_hash() == ctx.spec_hash()
    assert json.dumps(ctx2.to_dict(), allow_nan=False, sort_keys=True) == text

    assert d["metric_version"] == METRIC_VERSION and d["schema_hash"] == schema.hash()
    assert d["config_hash"] == config.hash() and d["config"]["mmd_max_rows"] == 120 and d["n_train"] == 900
    assert d["fit_row_hash"] == hashlib.sha256("\n".join(str(i) for i in ids.tolist()).encode()).hexdigest()
    assert d["kl"] == {"direction": "KL(real||synthetic)", "total_bins": 50, "interior_bins": 48,
                       "smoothing_mass": 1e-6, "log_base": "e",
                       "finite_columns": "training support + one unexpected-value bin"}
    assert set(d["histograms"]) == {"x1", "x2", "y"} and set(d["support"]) == {"d1", "d2", "c1", "c2"}
    assert d["support"]["c1"] == ["e", "n", "s", "w"] and d["support"]["c2"] == [10, 20, 30]
    assert [c["id"] for c in d["conditioning"]["conditioners"]] == ["d1", "d2", "c1", "c2", "y::quantile_strata"]
    strata = d["conditioning"]["conditioners"][-1]
    assert strata["interior_boundaries"] == pytest.approx(list(np.quantile(train["y"], [0.2, 0.4, 0.6, 0.8])), abs=1e-12)
    for col in ("x1", "x2", "y"):                                       # ALL training rows, nothing else
        assert d["histograms"][col]["edges"][0] == train[col].min()
        assert d["histograms"][col]["edges"][-1] == train[col].max()
    assert d["mmd"]["full"]["continuous_columns"] == ["x1", "x2", "y"]
    assert d["mmd"]["features"]["continuous_columns"] == ["x1", "x2"]
    assert d["mmd"]["full"]["bandwidth"] != d["mmd"]["features"]["bandwidth"]        # separately saved metadata

    summaries = []
    for fn in (marginal_metrics, association_metrics, conditional_metrics):
        (s1, extra1), (s2, extra2) = fn(ctx, real, synth), fn(ctx2, real, synth)
        assert json.dumps(json_safe(s1), allow_nan=False, sort_keys=True) == \
            json.dumps(json_safe(s2), allow_nan=False, sort_keys=True)
        if isinstance(extra1, pd.DataFrame):
            pd.testing.assert_frame_equal(extra1, extra2)
        else:
            assert extra1.keys() == extra2.keys() and all(np.array_equal(extra1[k], extra2[k]) for k in extra1)
        summaries.append(s1)
    m1, m2 = mmd_metrics(ctx, real, synth), mmd_metrics(ctx2, real, synth)
    assert json.dumps(json_safe(m1), sort_keys=True, allow_nan=False) == \
        json.dumps(json_safe(m2), sort_keys=True, allow_nan=False)
    summaries += [m1, tuning_objective(real, synth, schema), check_validity(synth, schema, ctx)]

    found = [s for summary in summaries for s in collect_statuses(json_safe(summary))]
    assert found and set(found) <= set(STATUSES)
    assert all("status" in s for s in summaries)
    # association arrays are directly usable with np.savez (no object arrays)
    _, arrays = association_metrics(ctx, real, synth)
    assert all(a.dtype != object for a in arrays.values())


def test_context_is_bound_to_schema_rows_and_version(make_schema):
    schema = make_schema(COLS, task="regression")
    train = _table(300, 4)
    ctx = MetricContext.fit(train, schema)
    d = ctx.to_dict()
    assert d["row_id_source"] == "row_position"
    assert d["fit_row_hash"] == hashlib.sha256("\n".join(str(i) for i in range(300)).encode()).hexdigest()
    other_rows = MetricContext.fit(train, schema, train_row_ids=np.arange(300) + 1)
    assert other_rows.to_dict()["fit_row_hash"] != d["fit_row_hash"] and other_rows.spec_hash() != ctx.spec_hash()
    assert MetricContext.fit(train, schema, MetricConfig(kl_total_bins=20)).spec_hash() != ctx.spec_hash()
    other_schema = make_schema([c for c in COLS if c[0] != "x2"], task="regression")
    with pytest.raises(ValueError, match="schema"):
        MetricContext.from_dict(d, other_schema)
    with pytest.raises(ValueError, match="version"):
        MetricContext.from_dict({**d, "metric_version": "sbtab.metrics/0"}, schema)
    with pytest.raises(ValueError):
        MetricContext.fit(train.assign(x1=np.nan), schema)                 # training rows must be clean
    with pytest.raises(ValueError):
        MetricContext.fit(train, schema, train_row_ids=np.arange(10))


def test_everything_is_learned_from_training_rows_only(make_schema):
    schema = make_schema(COLS, task="regression")
    train = _table(400, 5)
    ctx = MetricContext.fit(train, schema)
    before = json.dumps(ctx.to_dict(), sort_keys=True)
    for seed, shift in ((6, 0.0), (7, 25.0)):
        real, synth = _table(200, seed, shift), _table(200, seed + 50, -shift)
        marginal_metrics(ctx, real, synth)
        association_metrics(ctx, real, synth)
        conditional_metrics(ctx, real, synth)
        mmd_metrics(ctx, real, synth)
    assert json.dumps(ctx.to_dict(), sort_keys=True) == before
    # a real table with broken values is a protocol error, not something to score
    with pytest.raises(ValueError):
        marginal_metrics(ctx, _table(50, 8).assign(x1=np.inf), _table(50, 9))
