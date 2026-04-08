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
from sbtab.models.sb.light_sb import LightSBPotentialConfig
from sbtab.solvers.light_sb.config import LightSBConfig
from sbtab.solvers.light_sb.solver import LightSBSolver


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


def make_regressor(random_state: int):
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


def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    return float(
        np.mean([wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols])
    )


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
    rc = real[cols].corr().to_numpy()
    sc = synth[cols].corr().to_numpy()
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


def resolve_target_col(ds_name: str, df: pd.DataFrame, strict: bool) -> str:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="sbtab/data/datasets/datasets_continuous_only.pkl")
    ap.add_argument("--outdir", type=str, default="light_sb_kfold_eval")
    ap.add_argument("--datasets", type=str, default=",".join(TARGET_COL_BY_DATASET.keys()))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bins-kl", type=int, default=20)
    ap.add_argument("--strict-targets", action="store_true")

    # LightSBPotentialConfig params
    ap.add_argument("--n-potentials", type=int, default=50)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--no-diagonal", action="store_true")
    ap.add_argument("--sampling-batch-size", type=int, default=256)
    ap.add_argument("--s-diagonal-init", type=float, default=0.1)

    # LightSBConfig params
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=10_000)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--use-sde-sampling", action="store_true")
    ap.add_argument("--n-euler-steps", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--verbose-every", type=int, default=0)

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
            raise KeyError(f"Missing dataset keys in pickle: {missing}")

    global_rows = []

    for ds_name in ds_list:
        print("\n" + "=" * 100)
        print(f"DATASET: {ds_name}")
        print("=" * 100)

        df = all_data[ds_name].copy()
        cols = list(df.columns)

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if len(cols) < 2:
            print(f"[SKIP] Dataset '{ds_name}' has <2 columns.")
            continue

        target_col = resolve_target_col(ds_name, df, strict=args.strict_targets)
        feature_cols = [c for c in cols if c != target_col]
        if len(feature_cols) < 1:
            print(f"[SKIP] Dataset '{ds_name}' has no features.")
            continue

        print(f"Target: {target_col}  |  #features: {len(feature_cols)}  |  rows: {len(df)}")

        potential_cfg = LightSBPotentialConfig(
            n_potentials=args.n_potentials,
            epsilon=args.epsilon,
            is_diagonal=not args.no_diagonal,
            sampling_batch_size=args.sampling_batch_size,
            S_diagonal_init=args.s_diagonal_init,
        )
        cfg = LightSBConfig(
            potential=potential_cfg,
            lr=args.lr,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            grad_clip=args.grad_clip,
            use_sde_sampling=args.use_sde_sampling,
            n_euler_steps=args.n_euler_steps,
            device=args.device,
            seed=args.seed,
            verbose_every=args.verbose_every,
        )

        kf = KFold(n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.seed)
        fold_rows = []

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(df)))):
            print(f"\n--- Fold {fold_id + 1}/{args.n_splits} ---")

            df_train_raw = df.iloc[train_idx].copy()
            df_test_raw = df.iloc[test_idx].copy()

            schema = TabularSchema(feature_cols=cols)
            pipe = TransformPipeline.default_continuous_dropna()
            pipe.fit(df_train_raw, schema)

            train_scaled = pipe.transform(df_train_raw)
            test_scaled = pipe.transform(df_test_raw)

            solver = LightSBSolver(dim=len(cols), cfg=cfg)
            solver.fit(train_scaled)

            synth_scaled = solver.sample_df(
                n=len(test_scaled),
                seed=args.seed + 1000 + fold_id,
                use_sde_sampling=args.use_sde_sampling,
            )

            m_kl = avg_kl_hist(test_scaled, synth_scaled, cols=cols, n_bins=args.n_bins_kl)
            m_wd = avg_wd(test_scaled, synth_scaled, cols=cols)
            m_corr = corr_frobenius(test_scaled, synth_scaled, cols=cols)

            util_delta, r2_real, r2_syn = utility_delta_r2_percent(
                train_real=train_scaled,
                test_real=test_scaled,
                train_synth=synth_scaled,
                feature_cols=feature_cols,
                target_col=target_col,
                seed=args.seed + fold_id,
            )

            fold_rows.append({
                "dataset": ds_name,
                "fold": fold_id,
                "n_train": len(train_scaled),
                "n_test": len(test_scaled),
                "avg_kl": float(m_kl),
                "avg_wd": float(m_wd),
                "corr_frob": float(m_corr),
                "delta_r2_percent": float(util_delta),
                "r2_real": float(r2_real),
                "r2_synth": float(r2_syn),
            })

            print(
                f"avg_KL={m_kl:.6f}  avg_WD={m_wd:.6f}  "
                f"corr_F={m_corr:.6f}  deltaR2%={util_delta:.3f}"
            )

        fold_df = pd.DataFrame(fold_rows)
        fold_csv = outdir / f"{ds_name}_fold_metrics.csv"
        fold_df.to_csv(fold_csv, index=False)

        def mean_std(s: pd.Series) -> Tuple[float, float]:
            return float(s.mean()), float(s.std(ddof=0))

        summary: dict = {
            "dataset": ds_name,
            "target_col": target_col,
            "n_splits": int(args.n_splits),
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "potential_config": asdict(potential_cfg),
            "solver_config": {
                k: v for k, v in asdict(cfg).items() if k != "potential"
            },
            "metrics_mean": {},
            "metrics_std": {},
        }
        for key in ["avg_kl", "avg_wd", "corr_frob", "delta_r2_percent", "r2_real", "r2_synth"]:
            mu, sd = mean_std(fold_df[key])
            summary["metrics_mean"][key] = mu
            summary["metrics_std"][key] = sd

        summary_json = outdir / f"{ds_name}_kfold_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        global_rows.append({
            "dataset": ds_name,
            "target_col": target_col,
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
            "fold_csv": str(fold_csv),
            "summary_json": str(summary_json),
        })

        print(f"\nSaved: {fold_csv}")
        print(f"Saved: {summary_json}")

    if global_rows:
        global_df = pd.DataFrame(global_rows).sort_values("avg_wd_mean", ascending=True)
        global_csv = outdir / "kfold_summary_all_datasets.csv"
        global_df.to_csv(global_csv, index=False)
        print("\n" + "=" * 100)
        print(f"DONE. Global summary: {global_csv}")
        print("=" * 100)
    else:
        print("\nNo datasets evaluated.")


if __name__ == "__main__":
    main()
