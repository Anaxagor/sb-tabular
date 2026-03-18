from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from sbtab.data.schema import TabularSchema
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.baselines.tabddpm.model import TabDDPMWrapper, TabDDPMConfig


# ----------------------------
# Dataset -> target column map
# ----------------------------

TARGET_COL_BY_DATASET: Dict[str, str] = {
    "german_credit": 'duration',
    "online_news_popularity": " shares",
    "covertype": "Horizontal_Distance_To_Hydrology",
    "online_shoppers": "ProductRelated",
    "bank_marketing": "pdays",
    "bank_loan": "Income",
    "diabetes": "target",
    "california_housing": "MedHouseVal",
    "king_county_housing": "price"
}


# ----------------------------
# Utility regressor (R2)
# ----------------------------

def make_regressor(random_state: int):
    """
    Prefer CatBoost if installed; else fallback to sklearn.
    """
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
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=8,
            learning_rate=0.1,
            max_iter=500,
        )


# ----------------------------
# Metrics
# ----------------------------

def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    return float(np.mean([wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols]))


def avg_kl_hist(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    cols: List[str],
    n_bins: int = 50,
    eps: float = 1e-12,
) -> float:
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


def corr_frobenius(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    rc = real[cols].corr().fillna(0).to_numpy()
    sc = synth[cols].corr().fillna(0).to_numpy()
    return float(np.linalg.norm(rc - sc, ord="fro"))


def utility_delta_r2_percent(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    train_synth: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
) -> Tuple[float, float, float]:
    Xtr = train_real[feature_cols].to_numpy()
    ytr = train_real[target_col].to_numpy()
    Xte = test_real[feature_cols].to_numpy()
    yte = test_real[target_col].to_numpy()

    reg_real = make_regressor(seed)
    reg_real.fit(Xtr, ytr)
    r2_real = float(r2_score(yte, reg_real.predict(Xte)))

    Xs = train_synth[feature_cols].to_numpy()
    ys = train_synth[target_col].to_numpy()
    reg_syn = make_regressor(seed + 1)
    reg_syn.fit(Xs, ys)
    r2_syn = float(r2_score(yte, reg_syn.predict(Xte)))

    delta = (r2_syn - r2_real) / (abs(r2_real) + 1e-12) * 100.0
    return float(delta), float(r2_real), float(r2_syn)


# ----------------------------
# Best params loading -> config
# ----------------------------

def load_best_params(best_json_path: Path) -> Dict:
    data = json.loads(best_json_path.read_text(encoding="utf-8"))
    if "best_params" in data:
        return dict(data["best_params"])
    return {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}


def build_tabddpm_config_from_best(best: Dict, seed: int, device: str) -> TabDDPMConfig:
    """
    Maps Optuna JSON parameters to TabDDPMConfig.
    """
    return TabDDPMConfig(
        n_epochs=int(best.get("n_iter", 1000)),
        num_timesteps=int(best.get("num_timesteps", 1000)),
        batch_size=int(best.get("batch_size", 4096)),
        lr=float(best.get("lr", 0.001)),
        seed=seed,
        device=device
    )


# ----------------------------
# Main experiment
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, required=True, help="../data/datasets/datasets_continuous_only.pkl")
    ap.add_argument("--best_json_dir", type=str, required=True, help="tabddpm_optuna_results")
    ap.add_argument("--outdir", type=str, default="tabddpm_kfold_eval")

    ap.add_argument("--datasets", type=str, default=",".join(TARGET_COL_BY_DATASET.keys()))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-bins-kl", type=int, default=20)

    args = ap.parse_args()

    best_dir = Path(args.best_json_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data: Dict[str, pd.DataFrame] = pickle.load(f)

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    
    global_rows = []

    for ds_name in ds_list:
        if ds_name not in my_data: continue
        print("\n" + "=" * 100)
        print(f"TABDDPM EVALUATION: {ds_name}")
        print("=" * 100)

        df = my_data[ds_name].copy()
        cols = list(df.columns)

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        target_col = TARGET_COL_BY_DATASET[ds_name]
        feature_cols = [c for c in cols if c != target_col]

        # Load best params
        best_json_path = best_dir / f"{ds_name}_best.json"
        if not best_json_path.exists():
            print(f"[WARN] Best params JSON not found for {ds_name}, using defaults.")
            best_params_raw = {}
        else:
            best_params_raw = load_best_params(best_json_path)
            
        cfg = build_tabddpm_config_from_best(best_params_raw, args.seed, args.device)

        kf = KFold(n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.seed)
        idx = np.arange(len(df))

        fold_rows = []

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(idx)):
            print(f"\n--- Fold {fold_id+1}/{args.n_splits} ---")

            df_train_raw = df.iloc[train_idx].copy()
            df_test_raw = df.iloc[test_idx].copy()

            # Preprocess per fold: fit on train only (Supervisor style)
            schema = TabularSchema(feature_cols=cols)
            pipe = TransformPipeline.default_continuous_dropna()
            pipe.fit(df_train_raw, schema)

            train_scaled = pipe.transform(df_train_raw)
            test_scaled = pipe.transform(df_test_raw)

            # Train TabDDPM baseline
            model = TabDDPMWrapper(cfg)
            model.fit(train_scaled)

            # Sample synthetic dataset of size equal to test fold size
            x_synth = model.sample(n=len(test_scaled), seed=args.seed + fold_id)
            synth_scaled = pd.DataFrame(x_synth, columns=cols)

            # Metrics on PREPROCESSED test fold
            m_kl = avg_kl_hist(test_scaled, synth_scaled, cols=cols, n_bins=args.n_bins_kl)
            m_wd = avg_wd(test_scaled, synth_scaled, cols=cols)
            m_corr = corr_frobenius(test_scaled, synth_scaled, cols=cols)

            # Utility on PREPROCESSED data
            util_delta, r2_real, r2_syn = utility_delta_r2_percent(
                train_real=train_scaled,
                test_real=test_scaled,
                train_synth=synth_scaled,
                feature_cols=feature_cols,
                target_col=target_col,
                seed=args.seed + fold_id,
            )

            fold_rows.append(
                {
                    "dataset": ds_name,
                    "fold": fold_id,
                    "avg_kl": float(m_kl),
                    "avg_wd": float(m_wd),
                    "corr_frob": float(m_corr),
                    "delta_r2_percent": float(util_delta),
                    "r2_real": float(r2_real),
                    "r2_synth": float(r2_syn),
                }
            )

            print(f"avg_KL={m_kl:.6f}  avg_WD={m_wd:.6f}  corr_F={m_corr:.6f}  deltaR2%={util_delta:.3f}")

        # Save fold metrics
        fold_df = pd.DataFrame(fold_rows)
        fold_csv = outdir / f"{ds_name}_fold_metrics.csv"
        fold_df.to_csv(fold_csv, index=False)

        # Summary
        summary = {
            "dataset": ds_name,
            "best_params": asdict(cfg),
            "metrics_mean": fold_df[["avg_kl", "avg_wd", "corr_frob", "delta_r2_percent", "r2_real", "r2_synth"]].mean().to_dict(),
            "metrics_std": fold_df[["avg_kl", "avg_wd", "corr_frob", "delta_r2_percent", "r2_real", "r2_synth"]].std(ddof=0).to_dict(),
        }

        summary_json = outdir / f"{ds_name}_kfold_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        global_rows.append(
            {
                "dataset": ds_name,
                "avg_kl_mean": summary["metrics_mean"]["avg_kl"],
                "avg_kl_std": summary["metrics_std"]["avg_kl"],
                "avg_wd_mean": summary["metrics_mean"]["avg_wd"],
                "avg_wd_std": summary["metrics_std"]["avg_wd"],
                "corr_frob_mean": summary["metrics_mean"]["corr_frob"],
                "corr_frob_std": summary["metrics_std"]["corr_frob"],
                "delta_r2_percent_mean": summary["metrics_mean"]["delta_r2_percent"],
                "delta_r2_percent_std": summary["metrics_std"]["delta_r2_percent"],
                "r2_real_mean": summary["metrics_mean"]["r2_real"],
                "r2_synth_mean": summary["metrics_mean"]["r2_synth"],
            }
        )

    # Global summary CSV
    global_df = pd.DataFrame(global_rows).sort_values("avg_wd_mean", ascending=True)
    global_csv = outdir / "kfold_summary_all_datasets.csv"
    global_df.to_csv(global_csv, index=False)

    print("\n" + "=" * 100)
    print(f"DONE. Global summary saved: {global_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()