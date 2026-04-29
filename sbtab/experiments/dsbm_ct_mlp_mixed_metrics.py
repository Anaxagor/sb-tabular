"""
K-fold evaluation: DSBM + Continuous-Time + MLP + Joint on **mixed** data.

Handles datasets with continuous, discrete, and categorical columns:
  - Uses TabularSchema.infer_from_dataframe() for auto column classification
  - Uses TransformPipeline.default_impute_scale_encode() for mixed data
    (TypeAwareImputer -> ContinuousStandardScaler -> OneHotRepresentation)
  - Falls back to default_dropna_and_scale() for continuous-only data
  - Computes metrics in the encoded (numeric) space
  - Additionally computes categorical accuracy after inverse_transform
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from sbtab.data.schema import TabularSchema
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import (
    IMFDSBMSolver,
    IMFDSBMConfig,
)
from sbtab.evaluation.metrics.statistical import sliced_wasserstein


# ----------------------------
# Dataset -> target column map
# ----------------------------

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


# ----------------------------
# Utility regressor (R2)
# ----------------------------

def make_regressor(random_state: int):
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            depth=8, learning_rate=0.1, iterations=500,
            loss_function="RMSE", random_seed=random_state, verbose=False,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            random_state=random_state, max_depth=8, learning_rate=0.1, max_iter=500,
        )


# ----------------------------
# Transform pipeline builder
# ----------------------------

def build_transforms(schema: TabularSchema, *, missing_strategy: str) -> TransformPipeline:
    """
    Select the appropriate pipeline based on schema content.
    """
    if schema.has_categorical:
        if missing_strategy == "drop":
            raise ValueError(
                "missing_strategy='drop' is not supported for mixed/categorical datasets. "
                "Use 'impute'."
            )
        return TransformPipeline.default_impute_scale_encode()

    if missing_strategy == "impute":
        return TransformPipeline.default_impute_and_scale()
    return TransformPipeline.default_dropna_and_scale()


# ----------------------------
# Metrics
# ----------------------------

def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Average 1-D Wasserstein distance across numeric columns."""
    return float(np.mean([
        wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols
    ]))


def avg_kl_hist(
    real: pd.DataFrame, synth: pd.DataFrame, cols: List[str],
    n_bins: int = 50, eps: float = 1e-12,
) -> float:
    kls: List[float] = []
    for c in cols:
        r, s = real[c].to_numpy(), synth[c].to_numpy()
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


def corr_frobenius(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    rc = real[cols].corr().fillna(0).to_numpy()
    sc = synth[cols].corr().fillna(0).to_numpy()
    return float(np.linalg.norm(rc - sc, ord="fro"))


def categorical_match_rate(
    real: pd.DataFrame, synth: pd.DataFrame, cat_cols: List[str],
) -> float:
    """
    For each categorical column, compute the fraction of synthetic values
    that belong to the set of categories observed in the real data.
    Returns the average across all categorical columns.
    """
    if not cat_cols:
        return float("nan")
    rates: List[float] = []
    for c in cat_cols:
        if c not in real.columns or c not in synth.columns:
            continue
        real_cats = set(real[c].dropna().unique())
        if not real_cats:
            rates.append(1.0)
            continue
        synth_vals = synth[c].dropna()
        if len(synth_vals) == 0:
            rates.append(0.0)
            continue
        match = synth_vals.isin(real_cats).sum()
        rates.append(float(match) / float(len(synth_vals)))
    return float(np.mean(rates)) if rates else float("nan")


def utility_delta_r2_percent(
    train_real: pd.DataFrame, test_real: pd.DataFrame, train_synth: pd.DataFrame,
    feature_cols: List[str], target_col: str, seed: int,
) -> Tuple[float, float, float]:
    Xtr, ytr = train_real[feature_cols].to_numpy(), train_real[target_col].to_numpy()
    Xte, yte = test_real[feature_cols].to_numpy(), test_real[target_col].to_numpy()

    reg_real = make_regressor(seed)
    reg_real.fit(Xtr, ytr)
    r2_real = float(r2_score(yte, reg_real.predict(Xte)))

    Xs, ys = train_synth[feature_cols].to_numpy(), train_synth[target_col].to_numpy()
    reg_syn = make_regressor(seed + 1)
    reg_syn.fit(Xs, ys)
    r2_syn = float(r2_score(yte, reg_syn.predict(Xte)))

    delta = (r2_syn - r2_real) / (abs(r2_real) + 1e-12) * 100.0
    return float(delta), float(r2_real), float(r2_syn)


# ----------------------------
# Params loading -> Config
# ----------------------------

def load_best_params(best_json_path: Path) -> Dict:
    data = json.loads(best_json_path.read_text(encoding="utf-8"))
    if "best_params" in data:
        return dict(data["best_params"])
    return {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}


def build_config_from_best(best: Dict, seed: int, device: str) -> IMFDSBMConfig:
    imf_len = int(best.get("imf_len", 5))
    if imf_len % 2 == 0:
        imf_len += 1
    fb_sequence = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

    return IMFDSBMConfig(
        fb_sequence=fb_sequence,
        num_steps=int(best.get("num_steps", 1000)),
        sigma=float(best.get("sigma", 0.10)),
        eps=float(best.get("eps", 1e-3)),
        first_coupling=str(best.get("first_coupling", "ref")),
        inner_iters=int(best.get("inner_iters", 2000)),
        batch_size=int(best.get("batch_size", 256)),
        lr=float(best.get("lr", 1e-4)),
        weight_decay=float(best.get("weight_decay", 0.0)),
        grad_clip=float(best.get("grad_clip", 1.0)),
        noise=bool(best.get("noise", True)),
        device=device,
        seed=seed,
    )


def build_config(args, seed: int) -> IMFDSBMConfig:
    imf_len = int(args.imf_len)
    if imf_len % 2 == 0:
        imf_len += 1
    fb_sequence = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

    return IMFDSBMConfig(
        fb_sequence=fb_sequence,
        num_steps=int(args.num_steps),
        sigma=float(args.sigma),
        eps=float(args.eps),
        first_coupling=args.first_coupling,
        inner_iters=int(args.inner_iters),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip) if args.grad_clip > 0 else None,
        noise=bool(args.noise),
        device=args.device,
        seed=seed,
    )


# ----------------------------
# Main experiment
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="K-fold evaluation: DSBM+CT+MLP+joint on mixed data"
    )
    ap.add_argument("--pickle", type=str, required=True,
                    help="Path to pickle with Dict[str, pd.DataFrame]")
    ap.add_argument("--outdir", type=str, default="dsbm_ct_mlp_mixed_kfold_eval")
    ap.add_argument("--best-json-dir", type=str, default=None, dest="best_json_dir",
                    help="Directory with {dataset}_best.json from tuning. "
                         "If provided, params are loaded from there; CLI hyperparams are ignored.")
    ap.add_argument("--datasets", type=str, default=",".join(TARGET_COL_BY_DATASET.keys()))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bins-kl", type=int, default=20)
    ap.add_argument("--max-train-size", type=int, default=20000, dest="max_train_size",
                    help="Max training samples per fold (subsample if exceeded).")
    ap.add_argument("--max-folds", type=int, default=0,
                    help="Max folds to run per dataset (0 = all)")
    ap.add_argument("--missing-strategy", type=str, default="impute",
                    choices=["impute", "drop"], dest="missing_strategy",
                    help="How to handle missing values. 'impute' required for categorical data.")

    # Solver hyperparameters (used when --best-json-dir is not provided)
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--num-steps", type=int, default=1000, dest="num_steps")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--imf-len", type=int, default=5, dest="imf_len",
                    help="IMF sequence length (will be forced to odd)")
    ap.add_argument("--first-coupling", type=str, default="ref", dest="first_coupling",
                    choices=["ref", "ind"])
    ap.add_argument("--noise", type=lambda x: x.lower() != "false", default=True)

    # MLP training hyperparameters
    ap.add_argument("--inner-iters", type=int, default=2000, dest="inner_iters")
    ap.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay")
    ap.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip",
                    help="Gradient clipping norm (0 to disable)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data: Dict[str, pd.DataFrame] = pickle.load(f)

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    missing_ds = [d for d in ds_list if d not in my_data]
    if missing_ds:
        raise KeyError(f"Missing dataset keys in pickle: {missing_ds}")

    missing_targets = [d for d in ds_list if d not in TARGET_COL_BY_DATASET]
    if missing_targets:
        raise KeyError(f"Target column not specified for: {missing_targets}")

    global_rows: List[Dict] = []
    max_folds = args.max_folds if args.max_folds > 0 else args.n_splits

    for ds_name in ds_list:
        existing_csv = outdir / f"{ds_name}_fold_metrics.csv"
        if existing_csv.is_file():
            print(f"\n[SKIP] {ds_name}: {existing_csv} already exists")
            prev = pd.read_csv(existing_csv)
            summary_mean = prev.mean(numeric_only=True).to_dict()
            global_rows.append({
                "dataset": ds_name,
                "avg_kl_mean": summary_mean.get("avg_kl", 0),
                "avg_wd_mean": summary_mean.get("avg_wd", 0),
                "swd_mean": summary_mean.get("swd", 0),
                "corr_frob_mean": summary_mean.get("corr_frob", 0),
                "cat_match_rate_mean": summary_mean.get("cat_match_rate", float("nan")),
                "delta_r2_percent_mean": summary_mean.get("delta_r2_percent", 0),
                "r2_real_mean": summary_mean.get("r2_real", 0),
                "r2_synth_mean": summary_mean.get("r2_synth", 0),
            })
            continue

        print("\n" + "=" * 100)
        print(f"DATASET: {ds_name}  [DSBM+CT+MLP+joint mixed-data]")
        print("=" * 100)

        df = my_data[ds_name].copy()

        target_col = TARGET_COL_BY_DATASET[ds_name]
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in '{ds_name}'. "
                f"Available: {df.columns.tolist()}"
            )

        # Infer schema with auto column classification
        schema = TabularSchema.infer_from_dataframe(df, target_col=target_col)
        print(f"  continuous : {schema.continuous_cols}")
        print(f"  discrete   : {schema.discrete_cols}")
        print(f"  categorical: {schema.categorical_cols}")
        print(f"  target     : {schema.target_col}")

        # Original categorical column names (before encoding)
        original_cat_cols = list(schema.categorical_cols)

        # Build appropriate transform pipeline
        transforms = build_transforms(schema, missing_strategy=args.missing_strategy)

        if args.best_json_dir is not None:
            best_json_path = Path(args.best_json_dir) / f"{ds_name}_best.json"
            if not best_json_path.exists():
                raise FileNotFoundError(f"Best params JSON not found: {best_json_path}")
            best_params = load_best_params(best_json_path)
            cfg = build_config_from_best(best_params, seed=args.seed, device=args.device)
            print(f"  Loaded best params from: {best_json_path.name}")
        else:
            cfg = build_config(args, seed=args.seed)

        print(f"  Config: sigma={cfg.sigma}, num_steps={cfg.num_steps}, "
              f"fb_seq={cfg.fb_sequence}, first_coupling={cfg.first_coupling}, "
              f"noise={cfg.noise}, inner_iters={cfg.inner_iters}, "
              f"batch_size={cfg.batch_size}, lr={cfg.lr}, device={cfg.device}")

        kf = KFold(n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.seed)
        idx = np.arange(len(df))
        fold_rows: List[Dict] = []

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(idx)):
            if fold_id >= max_folds:
                break
            print(f"\n--- Fold {fold_id + 1}/{max_folds} ---")

            df_train_raw = df.iloc[train_idx].copy()
            df_test_raw = df.iloc[test_idx].copy()

            # Fit pipeline on train, transform both
            pipe = build_transforms(schema, missing_strategy=args.missing_strategy)
            pipe.fit(df_train_raw, schema)

            train_scaled = pipe.transform(df_train_raw)
            test_scaled = pipe.transform(df_test_raw)

            # After encoding, columns may have expanded (one-hot)
            encoded_cols = list(train_scaled.columns)
            print(f"  encoded cols={len(encoded_cols)}, "
                  f"train={len(train_scaled)}, test={len(test_scaled)}")

            # Determine feature columns in encoded space (exclude target if present)
            if target_col in encoded_cols:
                encoded_feature_cols = [c for c in encoded_cols if c != target_col]
            else:
                encoded_feature_cols = list(encoded_cols)

            # Subsample training data if needed
            if args.max_train_size > 0 and len(train_scaled) > args.max_train_size:
                rng_sub = np.random.default_rng(args.seed + fold_id + 7777)
                sub_idx = rng_sub.choice(len(train_scaled), size=args.max_train_size, replace=False)
                train_for_model = train_scaled.iloc[sub_idx].reset_index(drop=True)
                print(f"  [subsample] {len(train_scaled)} -> {args.max_train_size} rows for solver")
            else:
                train_for_model = train_scaled

            # Train DSBM in encoded space
            model = IMFDSBMSolver(dim=len(encoded_cols), cfg=cfg)
            model.fit(train_for_model)

            x_synth = model.sample(n=len(test_scaled), seed=args.seed + 1000 + fold_id)
            synth_scaled = pd.DataFrame(x_synth, columns=encoded_cols)

            # --- Metrics in encoded space ---
            m_kl = avg_kl_hist(test_scaled, synth_scaled, cols=encoded_cols, n_bins=args.n_bins_kl)
            m_wd = avg_wd(test_scaled, synth_scaled, cols=encoded_cols)
            m_corr = corr_frobenius(test_scaled, synth_scaled, cols=encoded_cols)
            m_swd = sliced_wasserstein(
                test_scaled[encoded_cols].to_numpy(),
                synth_scaled[encoded_cols].to_numpy(),
            )

            # --- Utility metric (R2) in encoded space ---
            if target_col in encoded_cols:
                util_delta, r2_real, r2_syn = utility_delta_r2_percent(
                    train_real=train_scaled,
                    test_real=test_scaled,
                    train_synth=synth_scaled,
                    feature_cols=encoded_feature_cols,
                    target_col=target_col,
                    seed=args.seed + fold_id,
                )
            else:
                util_delta, r2_real, r2_syn = float("nan"), float("nan"), float("nan")

            # --- Categorical match rate (inverse transform to original space) ---
            m_cat_match = float("nan")
            if original_cat_cols:
                try:
                    test_inv = pipe.inverse_transform(test_scaled)
                    synth_inv = pipe.inverse_transform(synth_scaled)
                    m_cat_match = categorical_match_rate(test_inv, synth_inv, original_cat_cols)
                except Exception as e:
                    print(f"  [WARN] inverse_transform failed for cat_match_rate: {e}")

            fold_rows.append({
                "dataset": ds_name,
                "fold": fold_id,
                "n_train": len(train_scaled),
                "n_test": len(test_scaled),
                "n_encoded_cols": len(encoded_cols),
                "avg_kl": float(m_kl),
                "avg_wd": float(m_wd),
                "corr_frob": float(m_corr),
                "swd": float(m_swd),
                "cat_match_rate": float(m_cat_match),
                "delta_r2_percent": float(util_delta),
                "r2_real": float(r2_real),
                "r2_synth": float(r2_syn),
            })

            print(f"  avg_KL={m_kl:.6f}  avg_WD={m_wd:.6f}  SWD={m_swd:.4f}  "
                  f"corr_F={m_corr:.6f}  cat_match={m_cat_match:.4f}  "
                  f"deltaR2%={util_delta:.3f}")

        fold_df = pd.DataFrame(fold_rows)
        fold_csv = outdir / f"{ds_name}_fold_metrics.csv"
        fold_df.to_csv(fold_csv, index=False)

        def mean_std(s: pd.Series) -> Tuple[float, float]:
            return float(s.mean()), float(s.std(ddof=0))

        summary = {
            "dataset": ds_name,
            "target_col": target_col,
            "solver": "DSBM+CT+MLP+joint (mixed)",
            "schema": {
                "continuous": schema.continuous_cols,
                "discrete": schema.discrete_cols,
                "categorical": schema.categorical_cols,
            },
            "n_splits": int(args.n_splits),
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "config": asdict(cfg),
            "metrics_mean": {},
            "metrics_std": {},
        }

        metric_keys = [
            "avg_kl", "avg_wd", "corr_frob", "swd",
            "cat_match_rate", "delta_r2_percent", "r2_real", "r2_synth",
        ]
        for key in metric_keys:
            mu, sd = mean_std(fold_df[key])
            summary["metrics_mean"][key] = mu
            summary["metrics_std"][key] = sd

        summary_json = outdir / f"{ds_name}_kfold_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        global_rows.append({
            "dataset": ds_name,
            "target_col": target_col,
            "n_continuous": len(schema.continuous_cols),
            "n_discrete": len(schema.discrete_cols),
            "n_categorical": len(schema.categorical_cols),
            **{f"{k}_mean": summary["metrics_mean"][k] for k in metric_keys},
            **{f"{k}_std": summary["metrics_std"][k] for k in metric_keys},
            "fold_csv": str(fold_csv),
            "summary_json": str(summary_json),
        })

        print(f"\n  Saved fold metrics:    {fold_csv}")
        print(f"  Saved dataset summary: {summary_json}")

    global_df = pd.DataFrame(global_rows).sort_values("avg_wd_mean", ascending=True)
    global_csv = outdir / "kfold_summary_all_datasets.csv"
    global_df.to_csv(global_csv, index=False)

    print("\n" + "=" * 100)
    print("DONE. Global summary saved:")
    print(global_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
