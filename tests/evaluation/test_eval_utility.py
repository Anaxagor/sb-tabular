"""
Section 11: TSTR utility. CatBoost is kept tiny (a few shallow trees) so the file runs in seconds.
"""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import f1_score, mean_absolute_error, r2_score

from sbtab.evaluation import (MetricConfig, json_safe, resolve_utility_params, utility_gap, utility_reference,
                              utility_tstr)

TINY = {"iterations": 25, "depth": 3, "learning_rate": 0.3, "random_seed": 0, "thread_count": 2, "task_type": "CPU"}
CATS = ["cat"]


def _clf_data(n, seed, classes=3):
    r = np.random.default_rng(seed)
    cat = r.integers(0, 3, n)                                   # nominal predictor stored as INTEGER codes
    num = r.normal(size=n)
    cnt = r.poisson(2.0, n)
    score = (cat == 1) * 2.0 + num + 0.3 * r.normal(size=n)
    y = np.digitize(score, [0.3, 1.6]) if classes == 3 else (score > 0.8).astype(int)
    return pd.DataFrame({"num": num, "cnt": cnt, "cat": cat}), pd.Series(y, name="label")


def _reg_data(n, seed):
    r = np.random.default_rng(seed)
    cat = r.integers(0, 3, n)
    num = r.normal(size=n)
    y = 5.0 + 2.0 * num + 3.0 * (cat == 2) + 0.2 * r.normal(size=n)
    return pd.DataFrame({"num": num, "cnt": r.poisson(2.0, n), "cat": cat}), pd.Series(y, name="price")


# --------------------------------------------------------------------------- sign conventions (hand examples)
def test_gap_sign_conventions():
    hi = utility_gap(0.80, 0.60, higher_is_better=True)         # F1 / R^2: synthetic worse -> positive
    assert hi["abs_gap"] == pytest.approx(0.20) and hi["delta_pct"] == pytest.approx(25.0)
    better = utility_gap(0.50, 0.75, higher_is_better=True)     # synthetic better -> negative
    assert better["abs_gap"] == pytest.approx(-0.25) and better["delta_pct"] == pytest.approx(-50.0)
    lo = utility_gap(2.0, 3.0, higher_is_better=False)          # MAE / RMSE / MAPE: e_synth - e_real
    assert lo["abs_gap"] == pytest.approx(1.0) and lo["delta_pct"] == pytest.approx(50.0)
    lo_better = utility_gap(4.0, 3.0, higher_is_better=False)
    assert lo_better["abs_gap"] == pytest.approx(-1.0) and lo_better["delta_pct"] == pytest.approx(-25.0)
    neg = utility_gap(-0.5, -1.0, higher_is_better=True)        # |s_real| in the denominator keeps the sign
    assert neg["abs_gap"] == pytest.approx(0.5) and neg["delta_pct"] == pytest.approx(100.0)
    assert not hi["near_zero_denominator"] and not neg["near_zero_denominator"]
    tiny = utility_gap(0.0, 0.1, higher_is_better=True)         # denominator floored at 1e-8 and flagged
    assert tiny["near_zero_denominator"] is True and tiny["delta_pct"] == pytest.approx(100 * (-0.1) / 1e-8)
    undefined = utility_gap(None, 0.3, higher_is_better=True)
    assert undefined["status"] == "undefined" and undefined["delta_pct"] is None and undefined["abs_gap"] is None


# --------------------------------------------------------------------------- dispatch, leakage, identical handling
def test_dispatch_and_target_is_never_a_predictor():
    Xtr, ytr = _clf_data(300, 1)
    Xte, yte = _clf_data(200, 2)
    leaky_train, leaky_test = Xtr.assign(label=ytr), Xte.assign(label=yte)      # the caller forgot to strip y
    ref = utility_reference("classification", leaky_train, ytr, leaky_test, yte, CATS, TINY)
    assert ref["status"] == "ok" and ref["model"] == "CatBoostClassifier"
    assert ref["model_feature_names"] == ["num", "cnt", "cat"] and "label" not in ref["feature_names"]
    assert ref["scores"]["macro_f1"] < 0.999                    # a leaked label would be learned perfectly
    assert set(ref["scores"]) == {"macro_f1"}
    # nominal predictor stored as integers is passed through cat_features; the count stays numeric
    assert ref["model_cat_feature_indices"] == [2] and ref["cat_features"] == ["cat"]

    Xr, yr = _reg_data(300, 3)
    Xrt, yrt = _reg_data(200, 4)
    reg = utility_reference("regression", Xr.assign(price=yr), yr, Xrt, yrt, CATS, TINY)
    assert reg["status"] == "ok" and reg["model"] == "CatBoostRegressor"
    assert reg["model_feature_names"] == ["num", "cnt", "cat"]
    assert set(reg["scores"]) == {"r2", "mae", "rmse", "mape"}
    with pytest.raises(ValueError):
        utility_reference("ranking", Xr, yr, Xrt, yrt, CATS, TINY)


def test_regression_scores_match_sklearn_on_the_returned_predictions():
    Xtr, ytr = _reg_data(400, 5)
    Xte, yte = _reg_data(250, 6)
    ref = utility_reference("regression", Xtr, ytr, Xte, yte, CATS, TINY)
    pred = ref["predictions"]
    assert isinstance(pred, np.ndarray) and pred.shape == (250,)
    assert ref["scores"]["r2"] == pytest.approx(r2_score(yte, pred), abs=1e-12)
    assert ref["scores"]["mae"] == pytest.approx(mean_absolute_error(yte, pred), abs=1e-12)
    assert ref["scores"]["rmse"] == pytest.approx(math.sqrt(np.mean((yte.to_numpy() - pred) ** 2)), abs=1e-12)
    assert ref["scores"]["mape"] == pytest.approx(100 * np.mean(np.abs(yte - pred) / np.abs(yte)), abs=1e-10)
    assert 1.0 < ref["scores"]["mape"] < 60.0                   # percent, not a fraction
    assert ref["scores"]["r2"] > 0.8 and "nonpositive_real_r2" not in ref["flags"]


def test_reference_is_reused_bit_identically_and_handling_is_identical():
    Xtr, ytr = _reg_data(400, 7)
    Xte, yte = _reg_data(250, 8)
    Xs, ys = _reg_data(400, 9)
    ys = ys + np.random.default_rng(0).normal(0, 2.0, len(ys))  # a noisier "generator"
    ref = utility_reference("regression", Xtr, ytr, Xte, yte, CATS, TINY)
    frozen = copy.deepcopy(ref)
    out = utility_tstr("regression", Xs, ys, Xte, yte, CATS, TINY, reference=ref)
    assert json.dumps(json_safe(ref), sort_keys=True) == json.dumps(json_safe(frozen), sort_keys=True)  # untouched
    assert out["scores_real"] == frozen["scores"]                # reused, not refitted
    assert out["params"] == ref["params"] and out["test_hash"] == ref["test_hash"]
    assert out["feature_names"] == ref["feature_names"] and out["n_test"] == ref["n_test"] == 250
    assert np.array_equal(out["y_test"], ref["y_test"])
    # determinism of the reference itself
    again = utility_reference("regression", Xtr, ytr, Xte, yte, CATS, TINY)
    assert np.array_equal(again["predictions"], ref["predictions"]) and again["scores"] == ref["scores"]

    # gaps follow the documented signs, computed here from the raw scores
    for m, higher in (("r2", True), ("mae", False), ("rmse", False), ("mape", False)):
        real, synth = out["scores_real"][m], out["scores_synth"][m]
        gap = (real - synth) if higher else (synth - real)
        assert out["gaps"][m]["abs_gap"] == pytest.approx(gap, abs=1e-12)
        assert out["gaps"][m]["delta_pct"] == pytest.approx(100 * gap / max(abs(real), 1e-8), abs=1e-9)
        assert out["gaps"][m]["abs_gap"] > 0                     # the noisy generator IS worse on every metric

    with pytest.raises(ValueError, match="same test rows"):
        utility_tstr("regression", Xs, ys, Xte.iloc[:-1], yte.iloc[:-1], CATS, TINY, reference=ref)
    with pytest.raises(ValueError, match="identical parameters"):
        utility_tstr("regression", Xs, ys, Xte, yte, CATS, {**TINY, "depth": 4}, reference=ref)
    with pytest.raises(ValueError, match="different task"):
        utility_tstr("classification", Xs, ys, Xte, yte, CATS, TINY, reference=ref)


def test_undefined_regression_cases():
    Xtr, ytr = _reg_data(300, 10)
    Xte, yte = _reg_data(100, 11)
    zero = yte.copy()
    zero.iloc[5] = 0.0
    ref = utility_reference("regression", Xtr, ytr, Xte, zero, CATS, TINY)
    assert ref["scores"]["mape"] is None and ref["score_status"]["mape"] == "undefined"
    assert "undefined_zero_target" in ref["flags"]
    assert ref["scores"]["mae"] > 0 and ref["scores"]["rmse"] > 0 and ref["scores"]["r2"] is not None
    out = utility_tstr("regression", Xtr, ytr, Xte, zero, CATS, TINY, reference=ref)
    assert out["gaps"]["mape"]["status"] == "undefined" and out["gaps"]["mape"]["delta_pct"] is None
    assert out["gaps"]["mae"]["status"] == "ok" and "undefined_zero_target" in out["flags"]

    const = pd.Series(np.full(len(yte), 7.0), name="price")
    cref = utility_reference("regression", Xtr, ytr, Xte, const, CATS, TINY)
    assert cref["scores"]["r2"] is None and cref["score_status"]["r2"] == "undefined"
    assert "undefined_constant_target" in cref["flags"] and cref["scores"]["mae"] is not None
    json.dumps(json_safe([ref, out, cref]), allow_nan=False)

    # a real-trained model that is worse than the mean: R^2 <= 0 is flagged (the % gap is then hard to read)
    r = np.random.default_rng(12)
    noise_train = pd.Series(r.normal(size=len(Xtr)), name="price")
    noise_test = pd.Series(r.normal(size=len(Xte)), name="price")
    bad = utility_reference("regression", Xtr, noise_train, Xte, noise_test, CATS, TINY)
    assert bad["scores"]["r2"] <= 0 and "nonpositive_real_r2" in bad["flags"]
    assert "nonpositive_real_r2" in utility_tstr("regression", Xtr, noise_train, Xte, noise_test, CATS, TINY,
                                                 reference=bad)["flags"]


def test_single_class_synthetic_labels_fail_the_utility_fit_only():
    Xtr, ytr = _clf_data(300, 13)
    Xte, yte = _clf_data(200, 14)
    ref = utility_reference("classification", Xtr, ytr, Xte, yte, CATS, TINY)
    Xs, _ = _clf_data(300, 15)
    ys = pd.Series(np.ones(300, dtype=int), name="label")
    out = utility_tstr("classification", Xs, ys, Xte, yte, CATS, TINY, reference=ref)
    assert out["status"] == "utility_fit_failed" and out["error"]
    cov = out["class_coverage"]
    assert cov["present"] == [1] and cov["missing"] == [0, 2] and cov["n_present"] == 1 and cov["n_universe"] == 3
    assert cov["coverage"] == pytest.approx(1 / 3) and cov["counts"] == {"0": 0, "1": 300, "2": 0}
    assert out["scores_synth"]["macro_f1"] is None and out["gaps"]["macro_f1"]["status"] == "undefined"
    assert out["scores_real"] == ref["scores"] and out["predictions"] is None     # the reference is retained
    assert out["model_feature_names"] is None and out["fit_seconds"] is None      # same keys as a successful fit
    assert ref["fit_seconds"] > 0 and ref["predict_seconds"] > 0
    json.dumps(json_safe(out), allow_nan=False)


def test_macro_f1_uses_the_fixed_real_label_universe():
    Xtr, ytr = _clf_data(500, 16)
    Xte, yte = _clf_data(300, 17)
    ref = utility_reference("classification", Xtr, ytr, Xte, yte, CATS, TINY)
    assert ref["label_universe"] == [0, 1, 2] and ref["class_coverage"]["missing"] == []
    keep = (ytr != 2).to_numpy()                                 # the generator never emits class 2
    out = utility_tstr("classification", Xtr[keep], ytr[keep], Xte, yte, CATS, TINY, reference=ref)
    assert out["status"] == "ok"
    assert out["class_coverage"]["missing"] == [2] and out["class_coverage"]["coverage"] == pytest.approx(2 / 3)
    pred = out["predictions"]
    assert set(pred) <= {0, 1}
    per_class = f1_score(yte, pred.astype(int), labels=[0, 1, 2], average=None, zero_division=0)
    assert per_class[2] == 0.0
    assert out["scores_synth"]["macro_f1"] == pytest.approx(per_class.sum() / 3, abs=1e-12)       # class 2 counts as 0
    present_only = f1_score(yte, pred.astype(int), labels=[0, 1], average="macro", zero_division=0)
    assert out["scores_synth"]["macro_f1"] < present_only - 0.1                                   # not averaged away
    assert out["gaps"]["macro_f1"]["abs_gap"] > 0.1
    # an explicit universe is honoured and must agree with the reference
    wide = utility_reference("classification", Xtr, ytr, Xte, yte, CATS, TINY, label_universe=[0, 1, 2, 3])
    assert wide["scores"]["macro_f1"] == pytest.approx(ref["scores"]["macro_f1"] * 3 / 4, abs=1e-12)
    with pytest.raises(ValueError, match="label universe"):
        utility_tstr("classification", Xtr, ytr, Xte, yte, CATS, TINY, reference=ref, label_universe=[0, 1])


def test_string_labels_and_float_coded_categories():
    Xtr, ytr = _clf_data(300, 18, classes=2)
    Xte, yte = _clf_data(200, 19, classes=2)
    names = {0: "no", 1: "yes"}
    ref = utility_reference("classification", Xtr, ytr.map(names), Xte, yte.map(names), CATS, TINY)
    assert ref["label_universe"] == ["no", "yes"] and set(ref["predictions"]) <= {"no", "yes"}
    assert ref["scores"]["macro_f1"] == pytest.approx(
        f1_score(yte.map(names), ref["predictions"], labels=["no", "yes"], average="macro"), abs=1e-12)
    # integer labels give the same model: the score cannot depend on how labels are spelled
    assert utility_reference("classification", Xtr, ytr, Xte, yte, CATS, TINY)["scores"] == ref["scores"]
    # a generator that emits category codes as floats (1.0) is the same category as the real 1
    out = utility_tstr("classification", Xtr.assign(cat=Xtr["cat"].astype(float)), ytr.map(names), Xte,
                       yte.map(names), CATS, TINY, reference=ref)
    assert out["status"] == "ok" and out["scores_synth"] == ref["scores"]
    assert out["gaps"]["macro_f1"]["abs_gap"] == 0.0


# --------------------------------------------------------------------------- default resolution
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_resolved_defaults_reproduce_the_default_model(task):
    (Xtr, ytr), (Xte, yte) = (_clf_data(300, 20), _clf_data(100, 21)) if task == "classification" \
        else (_reg_data(300, 20), _reg_data(100, 21))
    config = MetricConfig(utility_thread_count=2)
    params = resolve_utility_params(task, Xtr, ytr, CATS, config, overrides={"iterations": 30})
    json.dumps(params, allow_nan=False)                           # JSON-safe as returned
    assert params["iterations"] == 30 and params["random_seed"] == 0 and params["thread_count"] == 2
    assert params["task_type"] == "CPU"
    for resolved in ("learning_rate", "depth", "l2_leaf_reg", "loss_function", "bootstrap_type", "border_count"):
        assert resolved in params                                 # automatic defaults are now explicit numbers
    for dropped in ("verbose", "logging_level", "allow_writing_files", "train_dir", "class_names", "classes_count",
                    "eval_metric", "use_best_model"):
        assert dropped not in params
    assert params["loss_function"] == ("MultiClass" if task == "classification" else "RMSE")

    cls = CatBoostClassifier if task == "classification" else CatBoostRegressor
    default = cls(iterations=30, random_seed=0, thread_count=2, verbose=False, allow_writing_files=False)
    default.fit(Xtr.assign(cat=Xtr["cat"].astype(str)), ytr, cat_features=CATS)
    want = np.asarray(default.predict(Xte.assign(cat=Xte["cat"].astype(str)))).reshape(-1)
    ref = utility_reference(task, Xtr, ytr, Xte, yte, CATS, params)
    assert ref["status"] == "ok"
    if task == "classification":
        assert np.array_equal(ref["predictions"].astype(int), want.astype(int))
    else:
        assert np.allclose(ref["predictions"], want, rtol=0, atol=1e-12)
    # the same dict serves every later real / synthetic fit
    out = utility_tstr(task, Xtr, ytr, Xte, yte, CATS, json.loads(json.dumps(params)), reference=ref)
    assert out["status"] == "ok" and out["scores_synth"] == ref["scores"]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_a_reference_cached_as_json_is_reused_bit_identically(task):
    (Xtr, ytr), (Xte, yte), (Xs, ys) = [(_clf_data if task == "classification" else _reg_data)(n, s)
                                        for n, s in ((300, 30), (150, 31), (300, 32))]
    ref = utility_reference(task, Xtr, ytr, Xte, yte, CATS, TINY)
    stored = {k: v for k, v in ref.items() if k != "predictions"}          # predictions go to their own file
    cached = json.loads(json.dumps(json_safe(stored), allow_nan=False))
    live = utility_tstr(task, Xs, ys, Xte, yte, CATS, TINY, reference=ref)
    from_cache = utility_tstr(task, Xs, ys, Xte, yte, CATS, json.loads(json.dumps(TINY)), reference=cached)
    assert from_cache["status"] == "ok"
    assert from_cache["scores_real"] == live["scores_real"] == ref["scores"]
    assert from_cache["scores_synth"] == live["scores_synth"] and from_cache["gaps"] == live["gaps"]
    assert np.array_equal(from_cache["predictions"], live["predictions"])
