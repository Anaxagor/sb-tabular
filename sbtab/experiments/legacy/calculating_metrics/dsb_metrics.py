"""
LEGACY -- frozen historical script. Metric definitions: ``legacy/0``.

NOT the canonical protocol. The legacy protocol tuned on an 80/20 seed-42 holdout
with 50 Optuna trials, evaluated with KFold on 100 % of the rows, and sampled only
len(test fold) synthetic rows. This script is superseded by
``python -m sbtab.experiments.<stage>`` (prepare_splits, tune, cross_validate,
calculate_metrics, aggregate_results). Results it produces must NEVER be mixed with
``sbtab.metrics/1`` results. It is kept only so that historical numbers remain
interpretable; all metric helpers live in ``sbtab.experiments.legacy.legacy_metrics``.
Import / ``--help`` status on this branch: ``sbtab/experiments/legacy/README.md``
(it has NOT been run end-to-end on this branch).

Historical purpose: 5-fold evaluation of the continuous-time MLP IPF-DSB solver.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from sbtab.data.schema import TabularSchema
from sbtab.transforms.pipeline import TransformPipeline

from sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver import IPFDSBSolver, IPFDSBConfig
from sbtab.experiments.legacy.legacy_metrics import (
    TARGET_COL_BY_DATASET,
    avg_kl_hist,
    avg_wd,
    corr_frobenius_fillna0 as corr_frobenius,
    load_best_params,
    print_legacy_warning,
    utility_delta_r2_percent_raw_target as utility_delta_r2_percent,
)


# ----------------------------
# Best params loading -> IPFDSBConfig
# ----------------------------


def build_dsb_config_from_best(best: Dict, seed: int, device: str) -> IPFDSBConfig:
    return IPFDSBConfig(
        ipf_iters=int(best.get("ipf_iters", 10)),
        num_steps=int(best.get("N", 48)), 
        batch_size=int(best.get("batch_size", 2048)),
        hidden_units = int(best.get("hidden_units", 256)),
        lr=float(best.get("lr", 3e-4)),
        time_features=int(best.get("time_features", 32)),
        alpha_ou=float(best.get("alpha_ou", 1.0)),
        device=device,
        seed=seed,
    )


# ----------------------------
# Main experiment
# ----------------------------

def main() -> None:
    print_legacy_warning("sbtab.experiments.legacy.calculating_metrics.dsb_metrics")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="sbtab/data/datasets/datasets_continuous_only.pkl")
    ap.add_argument("--best_json_dir", type=str, default="sbtab/experiments/legacy/tuning_script/dsb_optuna_results/")
    ap.add_argument("--outdir", type=str, default="dsb_kfold_eval")

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
        print("\n" + "=" * 100)
        print(f"DATASET: {ds_name}")
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
            print(f"[WARN] Best params not found for {ds_name}, using defaults.")
            best_params = {}
        else:
            best_params = load_best_params(best_json_path)
            
        cfg = build_dsb_config_from_best(best_params, seed=args.seed, device=args.device)

        kf = KFold(n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.seed)
        idx = np.arange(len(df))

        fold_rows = []

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(idx)):
            print(f"\n--- Fold {fold_id+1}/{args.n_splits} ---")

            df_train_raw = df.iloc[train_idx].copy()
            df_test_raw = df.iloc[test_idx].copy()

            # Preprocessing via sbtab pipeline
            schema = TabularSchema.infer_from_dataframe(df=df, target_col=TARGET_COL_BY_DATASET[ds_name])
            pipe = TransformPipeline.default_dropna_and_scale()
            pipe.fit(df_train_raw, schema)

            train_scaled = pipe.transform(df_train_raw)
            test_scaled = pipe.transform(df_test_raw)

            # Fit DSB Solver
            model = IPFDSBSolver(dim=len(cols), cfg=cfg)
            model.fit(train_scaled)

            # Sample synthetic data
            x_synth = model.sample(n=len(test_scaled), seed=args.seed + fold_id)
            synth_scaled = pd.DataFrame(x_synth, columns=cols)

            # Calculate Metrics
            m_kl = avg_kl_hist(test_scaled, synth_scaled, cols=cols, n_bins=args.n_bins_kl)
            m_wd = avg_wd(test_scaled, synth_scaled, cols=cols)
            m_corr = corr_frobenius(test_scaled, synth_scaled, cols=cols)

            # Utility evaluation
            util_delta, r2_real, r2_syn = utility_delta_r2_percent(
                train_real=train_scaled,
                test_real=test_scaled,
                train_synth=synth_scaled,
                feature_cols=feature_cols,
                target_col=target_col,
                seed=args.seed + fold_id,
            )

            fold_rows.append({
                "dataset": ds_name, "fold": fold_id,
                "avg_kl": m_kl, "avg_wd": m_wd, "corr_frob": m_corr,
                "delta_r2_percent": util_delta, "r2_real": r2_real, "r2_synth": r2_syn,
            })

            print(f"avg_KL={m_kl:.6f}  avg_WD={m_wd:.6f}  corr_F={m_corr:.6f}  deltaR2%={util_delta:.3f}")

        # Summary for current dataset
        fold_df = pd.DataFrame(fold_rows)
        fold_csv = outdir / f"{ds_name}_fold_metrics.csv"
        fold_df.to_csv(fold_csv, index=False)

        summary = {
            "dataset": ds_name, "metrics_mean": {}, "metrics_std": {},
            "solver_config": asdict(cfg)
        }

        for key in ["avg_kl", "avg_wd", "corr_frob", "delta_r2_percent", "r2_real", "r2_synth"]:
            summary["metrics_mean"][key] = float(fold_df[key].mean())
            summary["metrics_std"][key] = float(fold_df[key].std(ddof=0))

        summary_json = outdir / f"{ds_name}_kfold_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        global_rows.append({
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
        })

    # Global CSV summary
    global_df = pd.DataFrame(global_rows).sort_values("avg_wd_mean", ascending=True)
    global_df.to_csv(outdir / "kfold_summary_all_datasets.csv", index=False)
    print(f"\nGlobal summary saved to: {outdir / 'kfold_summary_all_datasets.csv'}")


if __name__ == "__main__":
    main()