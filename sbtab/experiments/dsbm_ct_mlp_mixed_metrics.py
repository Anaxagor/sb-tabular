"""
K-fold evaluation: DSBM + CT + MLP + Joint on mixed/categorical tabular data.

Follows the same pattern as light_sb_mixed_metrics.py:
  - Uses datasets_mixed.pkl with df.attrs metadata
    (feature_types, target_variable, task_type)
  - Pipeline: TransformPipeline.default_impute_scale_encode() for mixed,
    default_dropna_and_scale() for continuous-only
  - DSBM is fitted on [encoded_features | numeric_target]
  - After sampling, features are inverse_transformed to original space for metrics

Final metrics per fold:
  Mixed data:
    avg_wd_cont       — Mean WD over continuous feature cols (original scale)
    avg_kl_disc_cat   — Mean KL over discrete + categorical feature cols (original space)
    corr_frob         — Frobenius norm of correlation-matrix difference (encoded space, all cols)
    delta_r2_percent  — % change R2_synth vs R2_real  (regression)
    delta_f1_percent  — % change F1_synth vs F1_real  (classification)

  Pure discrete/categorical:
    avg_kl_all        — Mean KL over all feature cols (original space)
    corr_frob         — same as above
    delta_f1_percent  — classification utility

Usage:
  python -m sbtab.experiments.dsbm_ct_mlp_mixed_metrics \\
      --pickle sbtab/data/datasets/datasets_mixed.pkl \\
      --best-json-dir dsbm_ct_mlp_mixed_optuna_results
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from sbtab.data.schema import TabularSchema
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import (
    IMFDSBMConfig,
    IMFDSBMSolver,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Mean marginal Wasserstein-1 over continuous cols (original scale)."""
    if not cols:
        return float("nan")
    return float(np.mean([
        wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols
    ]))


def _kl_col(real_s: pd.Series, synth_s: pd.Series, eps: float = 1e-12) -> float:
    all_cats = sorted(set(real_s.astype(str).unique()) | set(synth_s.astype(str).unique()))
    rc = real_s.astype(str).value_counts()
    sc = synth_s.astype(str).value_counts()
    p = np.array([rc.get(c, 0) for c in all_cats], dtype=float) + eps
    q = np.array([sc.get(c, 0) for c in all_cats], dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def avg_kl_discrete(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Mean KL(real||synth) over discrete/categorical cols in original-value space."""
    if not cols:
        return float("nan")
    return float(np.mean([_kl_col(real[c], synth[c]) for c in cols]))


def corr_frobenius(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Frobenius norm of the difference between Pearson correlation matrices."""
    rc = real[cols].corr().fillna(0.0).to_numpy()
    sc = synth[cols].corr().fillna(0.0).to_numpy()
    return float(np.linalg.norm(rc - sc, ord="fro"))


# ---------------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------------

def _make_classifier(random_state: int):
    try:
        from catboost import CatBoostClassifier  # type: ignore
        return CatBoostClassifier(
            depth=6, learning_rate=0.1, iterations=300,
            random_seed=random_state, verbose=False,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.1, max_iter=300, random_state=random_state,
        )


def _make_regressor(random_state: int):
    try:
        from catboost import CatBoostRegressor  # type: ignore
        return CatBoostRegressor(
            depth=8, learning_rate=0.1, iterations=500,
            loss_function="RMSE", random_seed=random_state, verbose=False,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.1, max_iter=500, random_state=random_state,
        )


def utility_delta_f1_percent(
    X_train_real: np.ndarray, X_test_real: np.ndarray, X_train_synth: np.ndarray,
    y_train_real: np.ndarray, y_test_real: np.ndarray, y_train_synth: np.ndarray,
    seed: int,
) -> Tuple[float, float, float]:
    from sklearn.metrics import f1_score

    clf_real = _make_classifier(seed)
    clf_real.fit(X_train_real, y_train_real.astype(int))
    f1_real = float(f1_score(
        y_test_real.astype(int), clf_real.predict(X_test_real),
        average="macro", zero_division=0,
    ))

    clf_syn = _make_classifier(seed + 1)
    clf_syn.fit(X_train_synth, y_train_synth.astype(int))
    f1_syn = float(f1_score(
        y_test_real.astype(int), clf_syn.predict(X_test_real),
        average="macro", zero_division=0,
    ))

    delta = (f1_syn - f1_real) / (abs(f1_real) + 1e-12) * 100.0
    return float(delta), float(f1_real), float(f1_syn)


def utility_delta_r2_percent(
    X_train_real: np.ndarray, X_test_real: np.ndarray, X_train_synth: np.ndarray,
    y_train_real: np.ndarray, y_test_real: np.ndarray, y_train_synth: np.ndarray,
    seed: int,
) -> Tuple[float, float, float]:
    from sklearn.metrics import r2_score

    reg_real = _make_regressor(seed)
    reg_real.fit(X_train_real, y_train_real)
    r2_real = float(r2_score(y_test_real, reg_real.predict(X_test_real)))

    reg_syn = _make_regressor(seed + 1)
    reg_syn.fit(X_train_synth, y_train_synth)
    r2_syn = float(r2_score(y_test_real, reg_syn.predict(X_test_real)))

    delta = (r2_syn - r2_real) / (abs(r2_real) + 1e-12) * 100.0
    return float(delta), float(r2_real), float(r2_syn)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_best_params(best_json_dir: Optional[str], ds_name: str) -> Optional[Dict]:
    if not best_json_dir:
        return None
    p = Path(best_json_dir) / f"{ds_name}_best.json"
    if not p.exists():
        print(f"  [INFO] Best-params file not found: {p}")
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("best_params", {})


def build_cfg(args, best_params: Optional[Dict], seed: int) -> IMFDSBMConfig:
    bp = best_params or {}

    def _get(key, default):
        return bp[key] if key in bp else default

    imf_len = int(_get("imf_len", args.imf_len))
    if imf_len % 2 == 0:
        imf_len += 1
    fb_sequence = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

    grad_clip_val = float(_get("grad_clip", args.grad_clip))
    grad_clip = grad_clip_val if grad_clip_val > 0 else None

    return IMFDSBMConfig(
        fb_sequence=fb_sequence,
        num_steps=int(_get("num_steps", args.num_steps)),
        sigma=float(_get("sigma", args.sigma)),
        eps=float(_get("eps", args.eps)),
        first_coupling=str(_get("first_coupling", args.first_coupling)),
        inner_iters=int(_get("inner_iters", args.inner_iters)),
        batch_size=int(_get("batch_size", args.batch_size)),
        lr=float(_get("lr", args.lr)),
        weight_decay=float(_get("weight_decay", args.weight_decay)),
        grad_clip=grad_clip,
        noise=bool(_get("noise", args.noise)),
        device=args.device,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="K-fold evaluation: DSBM+CT+MLP+joint on mixed/categorical tabular data"
    )
    ap.add_argument("--pickle", type=str,
                    default="sbtab/data/datasets/datasets_mixed.pkl")
    ap.add_argument("--outdir", type=str, default="dsbm_ct_mlp_mixed_kfold_eval")
    ap.add_argument("--best-json-dir", type=str, default=None, dest="best_json_dir",
                    help="Directory with {dataset}_best.json from tuning.")
    ap.add_argument("--datasets", type=str, default="ALL")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-size", type=int, default=0, dest="max_train_size",
                    help="Subsample train fold to at most N rows (0 = no limit).")

    # IMFDSBMConfig hyperparameters
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--num-steps", type=int, default=1000, dest="num_steps")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--imf-len", type=int, default=5, dest="imf_len")
    ap.add_argument("--first-coupling", type=str, default="ref", dest="first_coupling",
                    choices=["ref", "ind"])
    ap.add_argument("--noise", type=lambda x: x.lower() != "false", default=True)
    ap.add_argument("--inner-iters", type=int, default=2000, dest="inner_iters")
    ap.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay")
    ap.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip",
                    help="Gradient clipping norm (0 to disable)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--verbose-every", type=int, default=0, dest="verbose_every")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        all_data: Dict[str, pd.DataFrame] = pickle.load(f)

    if args.datasets.strip().upper() == "ALL":
        ds_list = list(all_data.keys())
    else:
        ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
        missing = [d for d in ds_list if d not in all_data]
        if missing:
            raise KeyError(f"Datasets not found in pickle: {missing}")

    global_rows: List[Dict] = []

    for ds_name in ds_list:
        print("\n" + "=" * 100)
        print(f"DATASET: {ds_name}  [DSBM+CT+MLP+joint mixed]")
        print("=" * 100)

        df_raw = all_data[ds_name].copy()

        # --- Feature type info from attrs ---
        feature_types = df_raw.attrs.get("feature_types", {})
        target_col: str = df_raw.attrs.get("target_variable", df_raw.columns[-1])
        task_type: str = df_raw.attrs.get("task_type", "regression")

        continuous_cols: List[str] = list(feature_types.get("continuous", []))
        discrete_cols: List[str] = list(feature_types.get("discrete", []))
        categorical_cols: List[str] = list(feature_types.get("categorical", []))
        feature_cols: List[str] = continuous_cols + discrete_cols + categorical_cols
        disc_cat_cols = discrete_cols + categorical_cols

        is_mixed = len(continuous_cols) > 0

        print(
            f"  target={target_col}  task={task_type}  "
            f"cont={len(continuous_cols)}  disc={len(discrete_cols)}  "
            f"cat={len(categorical_cols)}  rows={len(df_raw)}"
        )
        print(f"  is_mixed={is_mixed}")

        if len(feature_cols) < 1:
            print(f"[SKIP] No feature columns.")
            continue

        # --- Pre-encode target to numeric ---
        target_le: Optional[LabelEncoder] = None
        target_series = df_raw[target_col]
        if (
            pd.api.types.is_object_dtype(target_series)
            or pd.api.types.is_bool_dtype(target_series)
            or isinstance(target_series.dtype, pd.CategoricalDtype)
        ):
            target_le = LabelEncoder()
            df_raw = df_raw.copy()
            df_raw[target_col] = target_le.fit_transform(
                target_series.astype(str)
            ).astype(float)
            if task_type == "regression":
                task_type = "classification"
                print(f"  [NOTE] target was non-numeric; overriding task_type to 'classification'")
        else:
            df_raw[target_col] = pd.to_numeric(df_raw[target_col], errors="coerce")

        n_classes = (
            int(df_raw[target_col].dropna().nunique())
            if task_type == "classification"
            else None
        )

        # --- Schema (features only) ---
        schema = TabularSchema(
            continuous_cols=continuous_cols,
            discrete_cols=discrete_cols,
            categorical_cols=categorical_cols,
        )

        # --- Solver config ---
        best_params = load_best_params(args.best_json_dir, ds_name)
        cfg = build_cfg(args, best_params, seed=args.seed)

        # --- K-Fold loop ---
        kf = KFold(n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.seed)
        fold_rows: List[Dict] = []

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(df_raw)))):
            print(f"\n--- Fold {fold_id + 1}/{args.n_splits} ---")

            df_train = df_raw.iloc[train_idx].copy().reset_index(drop=True)
            df_test = df_raw.iloc[test_idx].copy().reset_index(drop=True)

            if args.max_train_size > 0 and len(df_train) > args.max_train_size:
                rng = np.random.default_rng(args.seed + fold_id)
                idx = rng.choice(len(df_train), size=args.max_train_size, replace=False)
                df_train = df_train.iloc[idx].reset_index(drop=True)

            # -- Transform features only (fit on train)
            df_feat_train = df_train[feature_cols].copy()
            df_feat_test = df_test[feature_cols].copy()

            if is_mixed or categorical_cols:
                pipe = TransformPipeline.default_impute_scale_encode()
            else:
                pipe = TransformPipeline.default_dropna_and_scale()
            pipe.fit(df_feat_train, schema)

            train_feat_enc = pipe.transform(df_feat_train).reset_index(drop=True)
            test_feat_enc = pipe.transform(df_feat_test).reset_index(drop=True)
            encoded_feat_cols = list(train_feat_enc.columns)

            # -- Build solver input: encoded features + numeric target
            train_target = df_train[target_col].to_numpy()
            test_target = df_test[target_col].to_numpy()

            train_enc = pd.concat(
                [train_feat_enc, pd.DataFrame({target_col: train_target})], axis=1
            )
            test_enc = pd.concat(
                [test_feat_enc, pd.DataFrame({target_col: test_target})], axis=1
            )
            encoded_all_cols: List[str] = encoded_feat_cols + [target_col]

            # -- Fit DSBM with per-fold seed
            cfg_fold = build_cfg(args, best_params, seed=args.seed + fold_id)
            dim = train_enc.shape[1]
            solver = IMFDSBMSolver(dim=dim, cfg=cfg_fold)
            solver.fit(train_enc)

            # -- Sample synthetic
            x_synth = solver.sample(
                n=len(test_enc),
                seed=args.seed + 1000 + fold_id,
            )
            synth_enc = pd.DataFrame(x_synth, columns=encoded_all_cols)

            # -- Inverse-transform feature columns to original space
            synth_feat_enc = synth_enc[encoded_feat_cols].copy()
            test_feat_orig = pipe.inverse_transform(test_feat_enc)
            synth_feat_orig = pipe.inverse_transform(synth_feat_enc)

            # -- Corr distance in encoded space (all cols including target)
            m_corr = corr_frobenius(test_enc, synth_enc, encoded_all_cols)

            # -- Distribution metrics in original space
            row: Dict = {
                "dataset": ds_name,
                "fold": fold_id,
                "n_train": len(df_train),
                "n_test": len(df_test),
                "corr_frob": m_corr,
            }

            if is_mixed:
                row["avg_wd_cont"] = avg_wd(test_feat_orig, synth_feat_orig, continuous_cols)
                row["avg_kl_disc_cat"] = avg_kl_discrete(test_feat_orig, synth_feat_orig, disc_cat_cols)
            else:
                row["avg_kl_all"] = avg_kl_discrete(test_feat_orig, synth_feat_orig, feature_cols)

            # -- Utility metric
            X_tr = train_feat_enc.to_numpy()
            X_te = test_feat_enc.to_numpy()
            X_sy = synth_feat_enc.to_numpy()
            y_tr = train_target
            y_te = test_target
            y_sy = synth_enc[target_col].to_numpy()

            if task_type == "classification":
                y_sy_cls = np.clip(np.round(y_sy).astype(int), 0, n_classes - 1)
                delta, score_real, score_syn = utility_delta_f1_percent(
                    X_tr, X_te, X_sy,
                    y_tr, y_te, y_sy_cls,
                    seed=args.seed + fold_id,
                )
                row["delta_f1_percent"] = delta
                row["f1_real"] = score_real
                row["f1_synth"] = score_syn
            else:
                delta, score_real, score_syn = utility_delta_r2_percent(
                    X_tr, X_te, X_sy,
                    y_tr, y_te, y_sy,
                    seed=args.seed + fold_id,
                )
                row["delta_r2_percent"] = delta
                row["r2_real"] = score_real
                row["r2_synth"] = score_syn

            fold_rows.append(row)

            _metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)
            )
            print(f"  {_metric_str}")

        # --- Aggregate and save ---
        fold_df = pd.DataFrame(fold_rows)
        fold_csv = outdir / f"{ds_name}_fold_metrics.csv"
        fold_df.to_csv(fold_csv, index=False)

        def _ms(s: pd.Series) -> Tuple[float, float]:
            return float(s.mean()), float(s.std(ddof=0))

        metric_cols_present = [
            c for c in [
                "avg_wd_cont", "avg_kl_disc_cat", "avg_kl_all",
                "corr_frob",
                "delta_f1_percent", "f1_real", "f1_synth",
                "delta_r2_percent", "r2_real", "r2_synth",
            ]
            if c in fold_df.columns
        ]

        summary: Dict = {
            "dataset": ds_name,
            "target_col": target_col,
            "task_type": task_type,
            "is_mixed": is_mixed,
            "solver": "DSBM+CT+MLP+joint (mixed)",
            "n_splits": args.n_splits,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "best_params_source": args.best_json_dir,
            "config": asdict(cfg),
            "metrics_mean": {},
            "metrics_std": {},
        }
        for key in metric_cols_present:
            mu, sd = _ms(fold_df[key])
            summary["metrics_mean"][key] = mu
            summary["metrics_std"][key] = sd

        summary_json = outdir / f"{ds_name}_kfold_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        global_row: Dict = {
            "dataset": ds_name,
            "target_col": target_col,
            "task_type": task_type,
            "is_mixed": is_mixed,
        }
        for key in metric_cols_present:
            global_row[f"{key}_mean"] = summary["metrics_mean"][key]
            global_row[f"{key}_std"] = summary["metrics_std"][key]
        global_row["fold_csv"] = str(fold_csv)
        global_row["summary_json"] = str(summary_json)
        global_rows.append(global_row)

        print(f"\nSaved: {fold_csv}")
        print(f"Saved: {summary_json}")

    if global_rows:
        sort_col = next(
            (c for c in ["avg_wd_cont_mean", "avg_kl_disc_cat_mean", "avg_kl_all_mean"]
             if c in global_rows[0]),
            None,
        )
        global_df = pd.DataFrame(global_rows)
        if sort_col:
            global_df = global_df.sort_values(sort_col, ascending=True)
        global_csv = outdir / "kfold_summary_all_datasets.csv"
        global_df.to_csv(global_csv, index=False)
        print("\n" + "=" * 100)
        print(f"DONE. Global summary: {global_csv}")
        print("=" * 100)
    else:
        print("\nNo datasets evaluated.")


if __name__ == "__main__":
    main()
