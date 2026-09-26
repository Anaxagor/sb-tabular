"""
TSTR utility (spec section 11): train on synthetic, test on real, against a
real-trained reference that uses the SAME task handling, parameters and test rows.

  * classification: CatBoostClassifier, macro F1 over the fixed real-training label
    universe (zero_division=0);  regression: CatBoostRegressor, R^2 / MAE / RMSE / MAPE;
  * the target is never a predictor; every nominal predictor goes through
    ``cat_features`` (as a canonical string), ordered/count predictors stay numeric;
  * hyper-parameters are not tuned: ``resolve_utility_params`` resolves CatBoost's
    automatic defaults ONCE on the first real training fold and the saved values are
    reused verbatim for every real/synthetic fit of that dataset;
  * no eval set, CPU, fixed seed and thread count, ``verbose=False`` and
    ``allow_writing_files=False`` (logging/file controls are not parameters).

Signs: every gap is positive when synthetic training is WORSE.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._common import json_safe, numeric_array, support_codes, to_native
from .spec import METRIC_VERSION, OK, UNDEFINED, UTILITY_FIT_FAILED, MetricConfig

GAP_DENOMINATOR_FLOOR = 1e-8
HIGHER_IS_BETTER = {"macro_f1": True, "r2": True, "mae": False, "rmse": False, "mape": False}
TASK_METRICS = {"classification": ("macro_f1",), "regression": ("r2", "mae", "rmse", "mape")}

# get_all_params() entries that are NOT training hyper-parameters:
#   data descriptors of the fold they were read from, and evaluation / early-stopping
#   controls (no eval set is ever passed).
_DROPPED_PARAMS = {"class_names", "classes_count", "eval_metric", "eval_fraction", "use_best_model",
                   "best_model_min_trees"}
_RUNTIME_KWARGS = {"verbose": False, "allow_writing_files": False}


def _model_class(task: str):
    from catboost import CatBoostClassifier, CatBoostRegressor
    if task == "classification":
        return CatBoostClassifier
    if task == "regression":
        return CatBoostRegressor
    raise ValueError(f"unknown task {task!r}")


# --------------------------------------------------------------------------- predictors / labels
def _canonical_category(s: pd.Series) -> pd.Series:
    """Nominal predictor as a string label; integer-valued numbers print as integers so that
    1, 1.0 and numpy ints are one category in the real, synthetic and test tables alike."""
    if s.dtype.kind in "iub":
        return s.astype(np.int64).astype(str)
    if s.dtype.kind == "f":
        x = s.to_numpy(dtype=np.float64)
        if np.all(np.isfinite(x)) and np.all(x == np.round(x)):
            return pd.Series(x.astype(np.int64).astype(str), index=s.index, name=s.name)
        return s.astype(str)

    def _one(v: Any) -> str:
        v = to_native(v)
        if isinstance(v, float) and math.isfinite(v) and v == int(v):
            return str(int(v))
        return str(v)
    return s.map(_one)


def _prepare_X(X: pd.DataFrame, y: Any, cat_features: Sequence[str], feature_names: Optional[List[str]] = None
               ) -> pd.DataFrame:
    X = pd.DataFrame(X).copy()
    X.columns = [str(c) for c in X.columns]
    target_name = getattr(y, "name", None)
    if target_name is not None and str(target_name) in X.columns:
        X = X.drop(columns=[str(target_name)])          # y is never a predictor
    if feature_names is not None:
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"predictor columns missing: {missing}")
        X = X[feature_names]
    for c in cat_features:
        if c in X.columns:
            X[c] = _canonical_category(X[c])
    return X.reset_index(drop=True)


def _cat_in(X: pd.DataFrame, cat_features: Sequence[str]) -> List[str]:
    return [str(c) for c in cat_features if str(c) in X.columns]


def _label_universe(label_universe: Optional[Sequence[Any]], y_train: Any) -> List[Any]:
    if label_universe is not None:
        return [to_native(v) for v in label_universe]
    labs = [to_native(v) for v in pd.unique(np.asarray(y_train).reshape(-1))]
    try:
        return sorted(labs)
    except TypeError:
        return sorted(labs, key=lambda v: (type(v).__name__, str(v)))


def _class_coverage(y: Any, universe: List[Any]) -> Dict[str, Any]:
    codes = support_codes(np.asarray(y).reshape(-1), universe)
    counts = np.bincount(codes[codes >= 0], minlength=len(universe))
    present = [u for u, c in zip(universe, counts) if c > 0]
    return {"label_universe": list(universe), "n_universe": len(universe),
            "present": present, "missing": [u for u, c in zip(universe, counts) if c == 0],
            "n_present": len(present), "coverage": len(present) / max(len(universe), 1),
            "counts": {str(u): int(c) for u, c in zip(universe, counts)},
            "n_out_of_universe": int((codes < 0).sum())}


def _test_hash(X_test: pd.DataFrame, y_test: Any) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, X_test.columns)).encode())
    h.update(pd.util.hash_pandas_object(X_test, index=False).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(pd.Series(np.asarray(y_test).reshape(-1)), index=False).to_numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- parameters
def resolve_utility_params(task: str, X_first_fold: pd.DataFrame, y_first_fold, cat_features: List[str],
                           config: MetricConfig = MetricConfig(), overrides: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
    """
    Fit ONE default model on the first real training fold and return its effective
    training parameters as plain constructor kwargs (JSON-safe). Re-fitting with the
    returned dict reproduces the default fit exactly.

    ``overrides`` exists for bounded smoke/unit runs (e.g. ``{"iterations": 20}``);
    benchmark runs leave it empty -- hyper-parameters are not tuned.
    """
    cls = _model_class(task)
    base = {"random_seed": int(config.utility_seed), "thread_count": int(config.utility_thread_count),
            "task_type": "CPU", **(overrides or {})}
    X = _prepare_X(X_first_fold, y_first_fold, cat_features)
    y = np.asarray(y_first_fold).reshape(-1)
    if task == "classification":           # same label handling as every later fit
        y = support_codes(y, _label_universe(None, y))
    else:
        y = numeric_array(y)
    model = cls(**base, **_RUNTIME_KWARGS)
    model.fit(X, y, cat_features=_cat_in(X, cat_features))
    accepted = set(inspect.signature(cls.__init__).parameters)
    params = {k: v for k, v in model.get_all_params().items() if k in accepted and k not in _DROPPED_PARAMS}
    params.update(base)                    # thread_count is not echoed by get_all_params()
    return json_safe(params)


def _fit(task: str, X: pd.DataFrame, y: np.ndarray, cat_features: Sequence[str], params: Dict[str, Any]):
    model = _model_class(task)(**{**params, **_RUNTIME_KWARGS})
    model.fit(X, y, cat_features=_cat_in(X, cat_features))       # no eval_set: nothing can stop early
    return model


# --------------------------------------------------------------------------- scores
def _classification_scores(y_true_codes: np.ndarray, pred_codes: np.ndarray, n_labels: int
                           ) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true_codes, pred_codes, labels=list(range(n_labels)), average="macro", zero_division=0)
    return {"macro_f1": float(f1)}, {"macro_f1": OK}, []


def _regression_scores(y: np.ndarray, pred: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    err = pred - y
    scores: Dict[str, Any] = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err * err).mean()))}
    status = {"mae": OK, "rmse": OK}
    flags: List[str] = []
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if y.max() == y.min() or ss_tot == 0.0:
        scores["r2"], status["r2"] = None, UNDEFINED
        flags.append("undefined_constant_target")
    else:
        scores["r2"], status["r2"] = float(1.0 - float((err * err).sum()) / ss_tot), OK
    if np.any(y == 0):
        scores["mape"], status["mape"] = None, UNDEFINED
        flags.append("undefined_zero_target")
    else:
        scores["mape"], status["mape"] = float(100.0 * np.mean(np.abs(err) / np.abs(y))), OK
    return scores, status, flags


def utility_gap(real: Optional[float], synth: Optional[float], higher_is_better: bool) -> Dict[str, Any]:
    """
    higher-is-better s:  abs_gap = s_real - s_synth,  delta_pct = 100 (s_real - s_synth) / max(|s_real|, 1e-8)
    lower-is-better  e:  abs_gap = e_synth - e_real,  delta_pct = 100 (e_synth - e_real) / max(|e_real|, 1e-8)
    Positive = synthetic training is worse. Undefined inputs give an undefined gap.
    """
    out = {"real": real, "synth": synth, "higher_is_better": bool(higher_is_better), "abs_gap": None,
           "delta_pct": None, "near_zero_denominator": None, "status": UNDEFINED}
    if real is None or synth is None or not (math.isfinite(real) and math.isfinite(synth)):
        return out
    gap = (real - synth) if higher_is_better else (synth - real)
    out.update(abs_gap=float(gap), delta_pct=float(100.0 * gap / max(abs(real), GAP_DENOMINATOR_FLOOR)),
               near_zero_denominator=bool(abs(real) < GAP_DENOMINATOR_FLOOR), status=OK)
    return out


# --------------------------------------------------------------------------- one predictor
def _evaluate(task: str, X_fit, y_fit, X_test, y_test, cat_features, params, universe: Optional[List[Any]],
              feature_names: Optional[List[str]]) -> Dict[str, Any]:
    Xf = _prepare_X(X_fit, y_fit, cat_features, feature_names)
    names = list(Xf.columns)
    Xt = _prepare_X(X_test, y_test, cat_features, names)
    out: Dict[str, Any] = {
        "status": OK, "task": task, "metric_version": METRIC_VERSION, "model": _model_class(task).__name__,
        "n_fit": int(len(Xf)), "n_test": int(len(Xt)), "feature_names": names,
        "cat_features": _cat_in(Xf, cat_features), "params": json_safe(dict(params)),
        "test_hash": _test_hash(Xt, y_test), "scores": {m: None for m in TASK_METRICS[task]},
        "score_status": {m: UTILITY_FIT_FAILED for m in TASK_METRICS[task]}, "flags": [],
        "predictions": None, "y_test": np.asarray(y_test).reshape(-1), "error": None,
        "label_universe": None, "class_coverage": None,
        "model_feature_names": None, "model_cat_feature_indices": None,
        "fit_seconds": None, "predict_seconds": None,
    }
    clock = {}

    def _timed_fit(y_codes_or_values):
        t0 = time.perf_counter()
        fitted = _fit(task, Xf, y_codes_or_values, cat_features, params)
        clock["fit"] = time.perf_counter() - t0
        return fitted

    def _timed_predict(fitted):
        t0 = time.perf_counter()
        raw = fitted.predict(Xt)
        clock["predict"] = time.perf_counter() - t0
        return raw
    try:
        if task == "classification":
            out["label_universe"] = list(universe)
            out["class_coverage"] = _class_coverage(y_fit, universe)
            k = len(universe)
            fit_codes = support_codes(np.asarray(y_fit).reshape(-1), universe)
            test_codes = support_codes(np.asarray(y_test).reshape(-1), universe)
            # labels outside the fixed universe stay in the fit as one extra, always-wrong class (never dropped)
            fit_codes = np.where(fit_codes < 0, k, fit_codes)
            if np.unique(fit_codes).size < 2:
                raise ValueError("training labels contain a single class")
            model = _timed_fit(fit_codes)
            pred_codes = np.asarray(_timed_predict(model)).reshape(-1).astype(np.int64)
            lookup = np.array(list(universe) + [None], dtype=object)
            out["predictions"] = lookup[np.clip(pred_codes, 0, k)]
            out["scores"], out["score_status"], out["flags"] = _classification_scores(test_codes, pred_codes, k)
        else:
            yf = numeric_array(np.asarray(y_fit).reshape(-1))
            yt = numeric_array(np.asarray(y_test).reshape(-1))
            if not np.all(np.isfinite(yt)):
                raise ValueError("held-out regression target is not finite")
            if not np.all(np.isfinite(yf)):
                raise ValueError("training regression target is not finite")
            model = _timed_fit(yf)
            pred = np.asarray(_timed_predict(model), dtype=np.float64).reshape(-1)
            out["predictions"] = pred
            out["scores"], out["score_status"], out["flags"] = _regression_scores(yt, pred)
        out["model_feature_names"] = [str(c) for c in model.feature_names_]
        out["model_cat_feature_indices"] = [int(i) for i in model.get_cat_feature_indices()]
        out["fit_seconds"], out["predict_seconds"] = float(clock["fit"]), float(clock["predict"])
    except Exception as exc:                                   # the learner could not fit this table
        out.update(status=UTILITY_FIT_FAILED, error=f"{type(exc).__name__}: {exc}"[:500])
    return out


def utility_reference(task: str, X_train, y_train, X_test, y_test, cat_features, params,
                      label_universe=None) -> Dict[str, Any]:
    """Real-trained predictor. Compute once per (dataset, fold) and pass to every ``utility_tstr`` call."""
    universe = _label_universe(label_universe, y_train) if task == "classification" else None
    out = _evaluate(task, X_train, y_train, X_test, y_test, cat_features, params, universe, None)
    out["role"] = "real_reference"
    if out["status"] == OK and task == "regression" and out["scores"]["r2"] is not None \
            and out["scores"]["r2"] <= 0.0:
        out["flags"].append("nonpositive_real_r2")
    return out


def utility_tstr(task: str, X_synth, y_synth, X_test, y_test, cat_features, params, reference: Dict[str, Any],
                 label_universe=None) -> Dict[str, Any]:
    """Synthetic-trained predictor on the SAME test rows / params / label universe as ``reference``."""
    if reference.get("task") != task:
        raise ValueError("reference was computed for a different task")
    if json_safe(dict(params)) != reference["params"]:
        raise ValueError("real and synthetic predictors must use identical parameters")
    universe = None
    if task == "classification":
        universe = [to_native(v) for v in label_universe] if label_universe is not None \
            else list(reference["label_universe"])
        if universe != list(reference["label_universe"]):
            raise ValueError("label universe differs from the reference")
    if _test_hash(_prepare_X(X_test, y_test, cat_features, reference["feature_names"]), y_test) \
            != reference["test_hash"]:
        raise ValueError("real and synthetic predictors must be evaluated on the same test rows")
    out = _evaluate(task, X_synth, y_synth, X_test, y_test, cat_features, params, universe,
                    reference["feature_names"])
    out["role"] = "tstr"
    out["scores_synth"] = out.pop("scores")
    out["score_status_synth"] = out.pop("score_status")
    out["scores_real"] = dict(reference["scores"])
    out["score_status_real"] = dict(reference["score_status"])
    out["reference_status"] = reference["status"]
    out["gaps"] = {m: utility_gap(reference["scores"].get(m), out["scores_synth"].get(m), HIGHER_IS_BETTER[m])
                   for m in TASK_METRICS[task]}
    for f in reference.get("flags", []):
        if f not in out["flags"]:
            out["flags"].append(f)
    near_zero = [m for m, g in out["gaps"].items() if g["near_zero_denominator"]]
    if near_zero:
        out["flags"].append("near_zero_denominator:" + ",".join(near_zero))
    return out
