from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from sbtab.data.schema import TabularSchema
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.continuous_time.feature_wise.boosting.ipf_dsb.solver import StructuralContinuousBoostedSolver, StructuralContinuousBoostedConfig

from sbtab.models.boosted.catboost_continuous_scalar import CatBoostContinuousScalarConfig
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
# Metrics
# ----------------------------

def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    return float(np.mean([wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols]))


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


def build_structural_config(best: Dict, seed: int) -> StructuralContinuousBoostedConfig:
    cat_cfg = CatBoostContinuousScalarConfig(
        iterations=int(best.get("iterations", 2000)),
        depth=int(best.get("depth", 8)),
        learning_rate=float(best.get("learning_rate", 0.05)),
        l2_leaf_reg=float(best.get("l2_leaf_reg", 3.0)),
        task_type=best.get("task_type", "CPU"),
        feature_mode="x_x0_t",
    )
    return StructuralContinuousBoostedConfig(
        num_steps=int(best.get("num_steps", 30)),
        ipf_iters=int(best.get("ipf_iters", 5)),
        alpha_ou=float(best.get("alpha_ou", 1.0)),
        n_bins=int(best.get("n_bins", 5)),
        seed=seed,
        catboost=cat_cfg,
    )


# ----------------------------
# Main experiment
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, required=True, help="path to datasets_continuous_only.pkl")
    ap.add_argument("--best_json_dir", type=str, required=True, help="dir with <dataset>_best.json")
    ap.add_argument("--outdir", type=str, default="structural_continuous_kfold_eval")
    ap.add_argument("--datasets", type=str, default=",".join(TARGET_COL_BY_DATASET.keys()))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bins-kl", type=int, default=20)
    ap.add_argument("--max-folds", type=int, default=0, help="Max folds to run per dataset (0 = all)")
    args = ap.parse_args()

    best_dir = Path(args.best_json_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data: Dict[str, pd.DataFrame] = pickle.load(f)

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    global_rows: List[Dict] = []

    max_folds = args.max_folds if args.max_folds > 0 else args.n_splits

    for ds_name in ds_list:
        if ds_name not in my_data:
            continue

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
                "delta_r2_percent_mean": summary_mean.get("delta_r2_percent", 0),
                "r2_real_mean": summary_mean.get("r2_real", 0),
                "r2_synth_mean": summary_mean.get("r2_synth", 0),
            })
            continue

        print("\n" + "=" * 100)
        print(f"STRUCTURAL CONTINUOUS BOOSTED | DATASET: {ds_name}")
        print("=" * 100)

        df = my_data[ds_name].copy()
        cols = list(df.columns)
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        target_col = TARGET_COL_BY_DATASET[ds_name]
        feature_cols = [c for c in cols if c != target_col]

        best_json_path = best_dir / f"{ds_name}_best.json"
        best_params = load_best_params(best_json_path) if best_json_path.exists() else {}
        cfg = build_structural_config(best_params, seed=args.seed)

        kf = KFold(n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.seed)
        fold_rows: List[Dict] = []

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(df)))):
            if fold_id >= max_folds:
                break
            print(f"\n--- Fold {fold_id + 1}/{max_folds} ---")

            df_train_raw = df.iloc[train_idx].copy()
            df_test_raw = df.iloc[test_idx].copy()

            schema = TabularSchema(feature_cols=cols)
            pipe = TransformPipeline.default_continuous_dropna()
            pipe.fit(df_train_raw, schema)

            train_scaled = pipe.transform(df_train_raw)
            test_scaled = pipe.transform(df_test_raw)

            train_df = pd.DataFrame(train_scaled, columns=cols) if not isinstance(train_scaled, pd.DataFrame) else train_scaled
            test_df = pd.DataFrame(test_scaled, columns=cols) if not isinstance(test_scaled, pd.DataFrame) else test_scaled

            model = StructuralContinuousBoostedSolver(cfg=cfg)
            model.fit(train_df)

            synth_df = model.sample(n=len(test_df))

            m_kl = avg_kl_hist(test_df, synth_df, cols=cols, n_bins=args.n_bins_kl)
            m_wd = avg_wd(test_df, synth_df, cols=cols)
            m_corr = corr_frobenius(test_df, synth_df, cols=cols)
            m_swd = sliced_wasserstein(test_df.to_numpy(), synth_df.to_numpy())

            util_delta, r2_real, r2_syn = utility_delta_r2_percent(
                train_real=train_df, test_real=test_df, train_synth=synth_df,
                feature_cols=feature_cols, target_col=target_col, seed=args.seed + fold_id,
            )

            fold_rows.append({
                "dataset": ds_name, "fold": fold_id,
                "avg_kl": m_kl, "avg_wd": m_wd, "corr_frob": m_corr, "swd": m_swd,
                "delta_r2_percent": util_delta, "r2_real": r2_real, "r2_synth": r2_syn,
            })
            print(f"avg_KL={m_kl:.6f}  avg_WD={m_wd:.6f}  SWD={m_swd:.4f}  deltaR2%={util_delta:.3f}")

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(outdir / f"{ds_name}_fold_metrics.csv", index=False)

        summary = {
            "dataset": ds_name,
            "metrics_mean": fold_df.mean(numeric_only=True).to_dict(),
            "metrics_std": fold_df.std(ddof=0, numeric_only=True).to_dict(),
            "solver_config": {
                "num_steps": cfg.num_steps, "ipf_iters": cfg.ipf_iters,
                "alpha_ou": cfg.alpha_ou, "n_bins": cfg.n_bins, "seed": cfg.seed,
            },
        }
        (outdir / f"{ds_name}_kfold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        global_rows.append({
            "dataset": ds_name,
            "avg_kl_mean": summary["metrics_mean"]["avg_kl"],
            "avg_wd_mean": summary["metrics_mean"]["avg_wd"],
            "swd_mean": summary["metrics_mean"]["swd"],
            "corr_frob_mean": summary["metrics_mean"]["corr_frob"],
            "delta_r2_percent_mean": summary["metrics_mean"]["delta_r2_percent"],
            "r2_real_mean": summary["metrics_mean"]["r2_real"],
            "r2_synth_mean": summary["metrics_mean"]["r2_synth"],
        })

    global_df = pd.DataFrame(global_rows).sort_values("avg_wd_mean", ascending=True)
    global_df.to_csv(outdir / "kfold_summary_all_datasets.csv", index=False)
    print(f"\nDONE. Global summary saved to {outdir}")


if __name__ == "__main__":
    main()
