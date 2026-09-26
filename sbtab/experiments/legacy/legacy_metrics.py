"""
LEGACY metric and helper definitions -- metric version ``legacy/0``.

FROZEN. This module holds exactly one copy of every helper that used to be
copy-pasted across the per-model scripts in ``sbtab/experiments/legacy/``.
It exists ONLY so that the historical numbers produced by those scripts stay
interpretable. It is NOT the canonical protocol: the canonical metric package is
``sbtab.evaluation`` (metric version ``sbtab.metrics/1``) and the canonical stages
are ``python -m sbtab.experiments.{prepare_splits,tune,cross_validate,
calculate_metrics,aggregate_results}``. Nothing in here imports the canonical
package and the canonical package must never import this module.

The definitions below are intentionally NOT "fixed". Every known defect of the
historical definition is preserved, because changing a definition under the same
version string would silently change the meaning of already-published numbers.
Numbers computed here must never be mixed, tabulated or ranked together with
``sbtab.metrics/1`` numbers.

How each legacy definition differs from the canonical ``sbtab.metrics/1`` one
-----------------------------------------------------------------------------

``avg_kl_hist`` / ``avg_kl_hist_autocols``  (marginal KL(real || synthetic))
    * Histogram edges are ``np.linspace(lo, hi, n_bins + 1)`` with ``lo``/``hi``
      taken from the UNION of the REAL (test fold) and the SYNTHETIC sample, so
      the bin grid depends on the model being evaluated and one synthetic outlier
      rescales every bin. Canonical: 48 interior + 1 underflow + 1 overflow bins,
      edges fitted on the TRAINING rows only.
    * ``eps = 1e-12`` is added to every RAW COUNT before normalisation (so the
      effective smoothing depends on the number of rows). Canonical: a total
      smoothing mass of 1e-6 spread uniformly over the bins.
    * The function default is ``n_bins=50`` but EVERY legacy CLI passed
      ``--n-bins-kl`` with default ``20``; all tracked historical KL values are
      therefore 20-bin values unless a run overrode the flag (no run recorded it).
    * A column whose pooled range is degenerate or non-finite contributes 0.0.
    * Every column is treated as continuous; there is no categorical KL.

``avg_wd`` / ``avg_wd_autocols`` / ``average_wd`` / ``average_wd_processed*``
    * Mean of per-column 1-D ``scipy.stats.wasserstein_distance`` computed in
      WHATEVER SPACE THE CALLER PASSES. The callers passed the pipeline-scaled
      frame, but the old schema excluded the TARGET column from scaling, so the
      target enters the mean in RAW UNITS and can dominate it (e.g. house prices).
      One-hot columns, when present, are averaged in like continuous ones.
    * ``avg_wd`` with an empty column list returns NaN (``np.mean([])``) plus a
      RuntimeWarning; the ``*_autocols`` / ``*_processed*`` variants raise.

``corr_frobenius_fillna0`` / ``corr_frobenius_raw`` / ``corr_frobenius_fillna0_autocols``
    * Pearson correlation only, Frobenius norm of the difference, un-normalised
      (grows with the number of columns). No rank or categorical association.
    * TWO historical behaviours existed and are kept apart on purpose:
      ``*_fillna0`` replaces NaN correlations (constant column) by 0;
      ``*_raw`` does not, so a single constant column in either sample makes the
      whole value NaN. ``dsbm_metrics``, ``light_sb_metrics``, ``stasy_metrics``
      and ``tapfn_metrics`` used ``raw``; the other seven scripts used ``fillna0``.

``utility_delta_r2_percent_*``  (train-on-synthetic, test-on-real)
    * ALWAYS a regressor scored with R^2, even when the task is classification
      (the "target" of several datasets is an arbitrary numeric feature column).
    * No ``cat_features`` are passed: every column is a plain float feature.
    * The real model is seeded with ``seed`` and the synthetic model with
      ``seed + 1`` -- the two fits never share a seed.
    * The synthetic model is trained on ``len(test fold)`` rows while the real
      model is trained on the ~4x larger train fold, so the comparison is
      confounded by sample size.
    * ``delta% = (R2_synth - R2_real) / (|R2_real| + 1e-12) * 100`` explodes
      when ``R2_real`` is near zero.
    * SILENT FALLBACK: if building ``CatBoostRegressor`` fails the helper returns
      an sklearn ``HistGradientBoostingRegressor`` without logging or recording
      it, so a historical number does not say which learner produced it. Two
      fallback behaviours existed: ``except Exception`` (seven
      ``calculating_metrics`` scripts) and ``except ImportError`` (the four
      ``joint_*``/``structural_*`` scripts). Only import + construction are inside
      the ``try``; a failure during ``fit`` was never caught by either.
    * ``*_numeric_target`` (ctgan, tabddpm) coerces the three target vectors with
      ``pd.to_numeric(errors="raise")`` before anything is fitted;
      ``*_raw_target`` passes the target through untouched.

``sliced_wasserstein``
    * Restored verbatim from ``de5acc9:sbtab/evaluation/metrics/statistical.py``.
      Both samples are mean-centred first (a pure location shift scores 0), the
      256 projections come from the GLOBAL, UNSEEDED torch RNG (values are not
      reproducible), and the two samples must have the same number of rows.
      There is no canonical counterpart under this name.

Non-metric helpers consolidated here (``TARGET_COL_BY_DATASET``,
``resolve_target_col_*``, ``load_best_params``, ``export_trials_csv*``,
``build_transforms*``, ``common_numeric_cols``) are documented at their
definitions. Where historical copies differed in behaviour each behaviour keeps
its own NAME and every script imports the variant it historically used; two
behaviours are never merged behind one name.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score

if TYPE_CHECKING:  # annotations only -- keep this module importable on its own
    import optuna

    from sbtab.data.schema import TabularSchema
    from sbtab.transforms.pipeline import TransformPipeline


LEGACY_METRIC_VERSION = "legacy/0"

LEGACY_WARNING = (
    "LEGACY SCRIPT ({script}): frozen historical code, metric definitions "
    f"'{LEGACY_METRIC_VERSION}'. This is NOT the canonical protocol (it uses an 80/20 "
    "seed-42 holdout, 50 Optuna trials, KFold on 100% of the rows and len(test fold) "
    "synthetic rows). It is superseded by `python -m sbtab.experiments.<stage>` "
    "(prepare_splits, tune, cross_validate, calculate_metrics, aggregate_results). "
    "Results it produces must NEVER be mixed with 'sbtab.metrics/1' results. "
    "It has not been run end-to-end on this branch; see sbtab/experiments/legacy/README.md."
)

_WARNED_SCRIPTS: set = set()


def print_legacy_warning(script: str) -> None:
    """Print the legacy banner to stderr, once per process and script name."""
    if script in _WARNED_SCRIPTS:
        return
    _WARNED_SCRIPTS.add(script)
    print("[WARNING] " + LEGACY_WARNING.format(script=script), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Dataset -> "target" column map (one copy; 15 historical copies)
# ---------------------------------------------------------------------------
# All 15 copies held the same nine key/value pairs. Fourteen shared the key order
# below. ``tapfn_metrics`` used a different key order, and because every script
# builds its ``--datasets`` default from ``.keys()`` that order is observable; it
# is preserved separately in ``TAPFN_DEFAULT_DATASET_ORDER``.
# Note " shares" (leading space) is the real column name in that dataset.
TARGET_COL_BY_DATASET: Dict[str, str] = {
    "german_credit": "duration",
    "online_news_popularity": " shares",
    "covertype": "Horizontal_Distance_To_Hydrology",
    "online_shoppers": "ProductRelated",
    "bank_marketing": "pdays",
    "bank_loan": "Income",
    "diabetes": "target",
    "california_housing": "MedHouseVal",
    "king_county_housing": "price",
}

TAPFN_DEFAULT_DATASET_ORDER: Tuple[str, ...] = (
    "covertype",
    "online_shoppers",
    "bank_marketing",
    "bank_loan",
    "diabetes",
    "california_housing",
    "online_news_popularity",
    "king_county_housing",
    "german_credit",
)


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

def common_numeric_cols(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    *,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """Columns present in both frames and numeric in both, in ``real`` order.

    Historical name ``_common_numeric_cols`` (3 copies). ``tabddpm_metrics`` had no
    ``exclude_cols`` parameter; that copy is exactly the ``exclude_cols=None`` case.
    """
    exclude = set(exclude_cols or [])
    cols = [c for c in real.columns if c in synth.columns and c not in exclude]
    return [
        c for c in cols
        if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(synth[c])
    ]


def _autocols(real: pd.DataFrame, synth: pd.DataFrame, cols: Optional[List[str]], what: str) -> List[str]:
    metric_cols = common_numeric_cols(real, synth) if cols is None else cols
    if not metric_cols:
        raise ValueError(f"No common numeric columns for {what}.")
    return metric_cols


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------

def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Mean per-column 1-D Wasserstein distance; ``cols`` is required.

    9 historical copies. An empty ``cols`` yields NaN (not an error).
    """
    return float(np.mean([wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols]))


# Historical name used by dsbm_tuning / ipf_dsb_tuning / lightsb_optuna_tune (3 copies).
# Those copies built the list with an explicit loop and ``float(...)`` per element,
# which is value-identical to ``avg_wd`` (same float64 inputs to the same ``np.mean``).
average_wd = avg_wd


def avg_wd_autocols(real: pd.DataFrame, synth: pd.DataFrame, cols: Optional[List[str]] = None) -> float:
    """``avg_wd`` as written in ctgan_metrics / tabddpm_metrics (2 copies).

    ``cols=None`` selects ``common_numeric_cols``; an empty selection RAISES.
    """
    return avg_wd(real, synth, _autocols(real, synth, cols, "Wasserstein distance"))


def _average_wd_processed(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    exclude_cols: Optional[List[str]],
    echo_columns: bool,
) -> float:
    metric_cols = common_numeric_cols(real, synth, exclude_cols=exclude_cols)
    if not metric_cols:
        raise ValueError("No common numeric columns available for Wasserstein metric.")

    wds = []
    for c in metric_cols:
        if echo_columns:
            print(c)
        wds.append(float(wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy())))
    return float(np.mean(wds))


def average_wd_processed(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    *,
    exclude_cols: Optional[List[str]] = None,
) -> float:
    """Tuning objective of ctgan_tuning: mean WD over all common numeric processed columns."""
    return _average_wd_processed(real, synth, exclude_cols, echo_columns=False)


def average_wd_processed_echo_columns(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Tuning objective of tabddpm_tuning.

    Same value as ``average_wd_processed(real, synth)``; the historical copy had no
    ``exclude_cols`` and contained a left-over ``print(c)`` that writes every column
    name to stdout on every trial. The print is preserved.
    """
    return _average_wd_processed(real, synth, None, echo_columns=True)


# ---------------------------------------------------------------------------
# Histogram KL
# ---------------------------------------------------------------------------

def avg_kl_hist(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    cols: List[str],
    n_bins: int = 50,
    eps: float = 1e-12,
) -> float:
    """
    Histogram-based marginal KL divergence: KL(p_real || p_synth) averaged over columns.
    Shared bins per feature from combined min/max.

    9 historical copies (they differed only in docstring/line layout).
    """
    kls: List[float] = []
    for c in cols:
        r = real[c].to_numpy()
        s = synth[c].to_numpy()

        lo = float(np.min([np.min(r), np.min(s)]))
        hi = float(np.max([np.max(r), np.max(s)]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            kls.append(0.0)
            continue

        bins = np.linspace(lo, hi, n_bins + 1)
        pr, _ = np.histogram(r, bins=bins, density=False)
        ps, _ = np.histogram(s, bins=bins, density=False)

        pr = pr.astype(np.float64) + eps
        ps = ps.astype(np.float64) + eps
        pr /= pr.sum()
        ps /= ps.sum()

        kls.append(float(np.sum(pr * (np.log(pr) - np.log(ps)))))

    return float(np.mean(kls))


def avg_kl_hist_autocols(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    cols: Optional[List[str]] = None,
    n_bins: int = 50,
    eps: float = 1e-12,
) -> float:
    """``avg_kl_hist`` as written in ctgan_metrics / tabddpm_metrics (2 copies).

    ``cols=None`` selects ``common_numeric_cols``; an empty selection RAISES.
    """
    return avg_kl_hist(real, synth, _autocols(real, synth, cols, "KL metric"), n_bins=n_bins, eps=eps)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def corr_frobenius_fillna0(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Pearson-Frobenius with NaN correlations replaced by 0.

    Used by dsb_metrics, joint_continuous/joint_discrete/structural_continuous/
    structural_discrete_metrics (5 copies).
    """
    rc = real[cols].corr().fillna(0).to_numpy()
    sc = synth[cols].corr().fillna(0).to_numpy()
    return float(np.linalg.norm(rc - sc, ord="fro"))


def corr_frobenius_raw(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Pearson-Frobenius WITHOUT ``fillna``: any constant column makes the result NaN.

    Used by dsbm_metrics, light_sb_metrics, stasy_metrics, tapfn_metrics (4 copies).
    """
    rc = real[cols].corr().to_numpy()
    sc = synth[cols].corr().to_numpy()
    return float(np.linalg.norm(rc - sc, ord="fro"))


def corr_frobenius_fillna0_autocols(
    real: pd.DataFrame, synth: pd.DataFrame, cols: Optional[List[str]] = None
) -> float:
    """``corr_frobenius_fillna0`` as written in ctgan_metrics / tabddpm_metrics (2 copies).

    ``cols=None`` selects ``common_numeric_cols``; an empty selection RAISES.
    """
    return corr_frobenius_fillna0(real, synth, _autocols(real, synth, cols, "correlation metric"))


# ---------------------------------------------------------------------------
# Utility regressor (R2)
# ---------------------------------------------------------------------------

def _make_regressor(random_state: int, fallback_on: Type[BaseException]):
    try:
        from catboost import CatBoostRegressor  # type: ignore
        return CatBoostRegressor(
            depth=8,
            learning_rate=0.1,
            iterations=500,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )
    except fallback_on:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=8,
            learning_rate=0.1,
            max_iter=500,
        )


def make_regressor_broad_fallback(random_state: int):
    """CatBoost, silently replaced by HistGradientBoosting on ANY exception (``except Exception``).

    The seven ``calculating_metrics/*`` scripts.
    """
    return _make_regressor(random_state, Exception)


def make_regressor_importerror_fallback(random_state: int):
    """CatBoost, silently replaced by HistGradientBoosting only on ``ImportError``.

    The four ``joint_*`` / ``structural_*`` scripts.
    """
    return _make_regressor(random_state, ImportError)


def _utility_delta_r2_percent(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    train_synth: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
    *,
    numeric_target: bool,
    make_regressor,
) -> Tuple[float, float, float]:
    # Statement order follows the historical copies exactly (it only matters for
    # which exception surfaces first on malformed input).
    if numeric_target:
        ytr = pd.to_numeric(train_real[target_col], errors="raise").to_numpy()
        yte = pd.to_numeric(test_real[target_col], errors="raise").to_numpy()
        ys = pd.to_numeric(train_synth[target_col], errors="raise").to_numpy()

        Xtr = train_real[feature_cols].to_numpy()
        Xte = test_real[feature_cols].to_numpy()
        Xs = train_synth[feature_cols].to_numpy()
    else:
        Xtr = train_real[feature_cols].to_numpy()
        ytr = train_real[target_col].to_numpy()
        Xte = test_real[feature_cols].to_numpy()
        yte = test_real[target_col].to_numpy()

    reg_real = make_regressor(seed)
    reg_real.fit(Xtr, ytr)
    r2_real = float(r2_score(yte, reg_real.predict(Xte)))

    if not numeric_target:
        Xs = train_synth[feature_cols].to_numpy()
        ys = train_synth[target_col].to_numpy()
    reg_syn = make_regressor(seed + 1)
    reg_syn.fit(Xs, ys)
    r2_syn = float(r2_score(yte, reg_syn.predict(Xte)))

    delta = (r2_syn - r2_real) / (abs(r2_real) + 1e-12) * 100.0
    return float(delta), float(r2_real), float(r2_syn)


def utility_delta_r2_percent_raw_target(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    train_synth: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
) -> Tuple[float, float, float]:
    """
    Utility metric: delta% between R2 scores on real test.

    R2_real  : model trained on real-train, evaluated on real-test
    R2_synth : model trained on synth-train, evaluated on real-test

    delta% = (R2_synth - R2_real) / (abs(R2_real) + 1e-12) * 100

    Variant of dsb/dsbm/light_sb/stasy/tapfn_metrics (5 copies): target passed
    through untouched, ``make_regressor_broad_fallback``.
    """
    return _utility_delta_r2_percent(
        train_real, test_real, train_synth, feature_cols, target_col, seed,
        numeric_target=False, make_regressor=make_regressor_broad_fallback,
    )


def utility_delta_r2_percent_raw_target_importerror_fallback(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    train_synth: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
) -> Tuple[float, float, float]:
    """Variant of the four ``joint_*``/``structural_*`` scripts: target untouched,
    ``make_regressor_importerror_fallback``."""
    return _utility_delta_r2_percent(
        train_real, test_real, train_synth, feature_cols, target_col, seed,
        numeric_target=False, make_regressor=make_regressor_importerror_fallback,
    )


def utility_delta_r2_percent_numeric_target(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    train_synth: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
) -> Tuple[float, float, float]:
    """Variant of ctgan_metrics / tabddpm_metrics (2 copies): the three target vectors
    go through ``pd.to_numeric(errors="raise")`` first; ``make_regressor_broad_fallback``."""
    return _utility_delta_r2_percent(
        train_real, test_real, train_synth, feature_cols, target_col, seed,
        numeric_target=True, make_regressor=make_regressor_broad_fallback,
    )


# ---------------------------------------------------------------------------
# Sliced Wasserstein -- verbatim from de5acc9:sbtab/evaluation/metrics/statistical.py
# ---------------------------------------------------------------------------
# That module never existed in the ancestry of this branch (the commits touching it are
# reachable only from the unmerged remote branches origin/feat/igor and
# origin/feat/evaluation-metrics) and, as committed at de5acc9, could not even be
# imported (NameError: name 'pd' is not defined -- a later function in the same file
# used ``pd.DataFrame`` in an annotation without importing pandas). The function
# below does not itself need pandas, so nothing had to be added to it.

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_proj: int = 256) -> float:
    """
    Cut-off Wasserstein distance (SWD).
    Evaluates the similarity of the joint distribution of all features.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    # Centering the data
    Xc, Yc = X_t - X_t.mean(0), Y_t - Y_t.mean(0)

    # Generating random projection
    thetas = torch.randn(n_proj, X_t.shape[1])
    thetas = thetas / thetas.norm(dim=1, keepdim=True)

    sw2 = 0.0
    for theta in thetas:
        # Projecting multidimensional data onto a line
        x1 = Xc @ theta
        y1 = Yc @ theta
        # We consider 1 day to be a weekend because of the schedules
        x1, _ = torch.sort(x1)
        y1, _ = torch.sort(y1)
        sw2 += F.mse_loss(x1, y1, reduction="mean")

    return float(sw2 / n_proj)


# ---------------------------------------------------------------------------
# Target-column resolution -- TWO historical behaviours, kept apart
# ---------------------------------------------------------------------------

def resolve_target_col_whitespace_tolerant(df: pd.DataFrame, ds_name: str) -> str:
    """
    Be robust to datasets whose column names contain leading/trailing spaces.

    ctgan_metrics / tabddpm_metrics (2 copies). Signature ``(df, ds_name)``.
    An unmapped dataset is a ``KeyError``; a mapped target is matched exactly first
    and then after ``str.strip()`` on both sides; otherwise ``ValueError``.
    """
    raw_target = TARGET_COL_BY_DATASET[ds_name]
    if raw_target in df.columns:
        return raw_target

    stripped_map = {str(c).strip(): c for c in df.columns}
    key = raw_target.strip()
    if key in stripped_map:
        return stripped_map[key]

    raise ValueError(
        f"Target column {raw_target!r} for dataset {ds_name!r} not found. "
        f"Available columns: {list(df.columns)}"
    )


def resolve_target_col_last_column_fallback(ds_name: str, df: pd.DataFrame, strict: bool) -> str:
    """light_sb_metrics / stasy_metrics / tapfn_metrics (3 copies). Signature ``(ds_name, df, strict)``.

    A mapped target must match EXACTLY (no whitespace tolerance). An unmapped
    dataset silently uses the LAST column as the target unless ``strict``.
    """
    if ds_name in TARGET_COL_BY_DATASET:
        t = TARGET_COL_BY_DATASET[ds_name]
        if t not in df.columns:
            raise ValueError(
                f"Mapped target '{t}' not found in dataset '{ds_name}'. "
                f"Columns: {df.columns.tolist()}"
            )
        return t
    if strict:
        raise KeyError(
            f"No target mapping for dataset '{ds_name}'. "
            f"Add it to TARGET_COL_BY_DATASET or disable --strict-targets."
        )
    t = df.columns[-1]
    print(f"[WARN] No target mapping for '{ds_name}'. Falling back to last column: '{t}'")
    return t


# ---------------------------------------------------------------------------
# Best-params loading / Optuna export
# ---------------------------------------------------------------------------

def load_best_params(best_json_path: Path) -> Dict:
    """8 identical historical copies."""
    data = json.loads(best_json_path.read_text(encoding="utf-8"))
    if "best_params" in data:
        return dict(data["best_params"])
    return {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}


def _export_trials_csv(study: "optuna.Study", out_csv: Path, user_attr_columns: Tuple[str, ...]) -> None:
    rows = []
    for tr in study.trials:
        row = {
            "trial_number": tr.number,
            "state": str(tr.state),
            "value": tr.value,
            **{k: tr.user_attrs.get(k) for k in user_attr_columns},
            **tr.params,
        }
        # store exception if present
        if "exception" in tr.user_attrs:
            row["exception"] = tr.user_attrs["exception"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def export_trials_csv(study: "optuna.Study", out_csv: Path) -> None:
    """Export all trials to CSV for offline analysis (5 copies)."""
    _export_trials_csv(study, out_csv, ())


def export_trials_csv_composite(study: "optuna.Study", out_csv: Path) -> None:
    """tabddpm_mixed_data_tuning variant: adds the ``wd_num_mean`` and
    ``js_disc_cat_mean`` user attributes as columns right after ``value``."""
    _export_trials_csv(study, out_csv, ("wd_num_mean", "js_disc_cat_mean"))


# ---------------------------------------------------------------------------
# Pipeline selection
# ---------------------------------------------------------------------------

def _build_transforms(schema: "TabularSchema", missing_strategy: str, drop_message: str) -> "TransformPipeline":
    from sbtab.transforms.pipeline import TransformPipeline

    if schema.has_categorical:
        if missing_strategy == "drop":
            raise ValueError(drop_message)
        return TransformPipeline.default_impute_scale_encode()

    if missing_strategy == "drop":
        return TransformPipeline.default_dropna_and_scale()
    return TransformPipeline.default_impute_and_scale()


def build_transforms(schema: "TabularSchema", *, missing_strategy: str) -> "TransformPipeline":
    """
    Select pipeline according to the actual TransformPipeline API.

    ctgan_metrics, tabddpm_metrics, ctgan_tuning (3 copies).
    """
    return _build_transforms(
        schema,
        missing_strategy,
        "missing_strategy='drop' is not supported for datasets with categorical features "
        "under the current TransformPipeline API. Use 'impute'.",
    )


def build_transforms_tabddpm_tuning(schema: "TabularSchema", *, missing_strategy: str) -> "TransformPipeline":
    """tabddpm_tuning copy: same branches as ``build_transforms``; only the text of the
    ``ValueError`` raised for ``missing_strategy="drop"`` on categorical data differs."""
    return _build_transforms(
        schema,
        missing_strategy,
        "missing_strategy='drop' is not supported for mixed/categorical datasets by the current "
        "TransformPipeline API. Use 'impute'.",
    )


def build_transforms_mixed(
    schema: "TabularSchema", *, missing_strategy: str, cat_encoding: str
) -> "TransformPipeline":
    """tabddpm_mixed_data_tuning variant: ``cat_encoding`` in {"onehot", "integer"} and a
    feature-detected ``default_drop_scale_integer_encode`` for ``missing_strategy="drop"``."""
    from sbtab.transforms.pipeline import TransformPipeline

    if schema.has_categorical:
        if missing_strategy == "drop":
            # if your repo exposes a mixed drop+integer pipeline, use it; otherwise fail loudly
            if hasattr(TransformPipeline, "default_drop_scale_integer_encode"):
                return TransformPipeline.default_drop_scale_integer_encode()
            raise ValueError(
                "missing_strategy='drop' is not supported for mixed data with the current "
                "TransformPipeline API unless default_drop_scale_integer_encode() exists."
            )

        if cat_encoding == "onehot":
            return TransformPipeline.default_impute_scale_encode()
        if cat_encoding == "integer":
            return TransformPipeline.default_impute_scale_integer_encode()
        raise ValueError(f"Unknown cat_encoding={cat_encoding!r}")

    if missing_strategy == "drop":
        return TransformPipeline.default_dropna_and_scale()
    return TransformPipeline.default_impute_and_scale()


__all__ = [
    "LEGACY_METRIC_VERSION",
    "LEGACY_WARNING",
    "print_legacy_warning",
    "TARGET_COL_BY_DATASET",
    "TAPFN_DEFAULT_DATASET_ORDER",
    "common_numeric_cols",
    "avg_wd",
    "average_wd",
    "avg_wd_autocols",
    "average_wd_processed",
    "average_wd_processed_echo_columns",
    "avg_kl_hist",
    "avg_kl_hist_autocols",
    "corr_frobenius_fillna0",
    "corr_frobenius_raw",
    "corr_frobenius_fillna0_autocols",
    "make_regressor_broad_fallback",
    "make_regressor_importerror_fallback",
    "utility_delta_r2_percent_raw_target",
    "utility_delta_r2_percent_raw_target_importerror_fallback",
    "utility_delta_r2_percent_numeric_target",
    "sliced_wasserstein",
    "resolve_target_col_whitespace_tolerant",
    "resolve_target_col_last_column_fallback",
    "load_best_params",
    "export_trials_csv",
    "export_trials_csv_composite",
    "build_transforms",
    "build_transforms_tabddpm_tuning",
    "build_transforms_mixed",
]
