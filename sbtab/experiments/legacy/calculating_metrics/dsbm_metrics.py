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

Historical purpose: 5-fold evaluation of the continuous-time MLP IMF-DSBM solver (produced ``dsbm_kfold_eval/``).
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from sbtab.data.schema import TabularSchema
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import IMFDSBMSolver, IMFDSBMConfig
from sbtab.experiments.legacy.legacy_metrics import (
    TARGET_COL_BY_DATASET,
    avg_kl_hist,
    avg_wd,
    corr_frobenius_raw as corr_frobenius,
    load_best_params,
    print_legacy_warning,
    utility_delta_r2_percent_raw_target as utility_delta_r2_percent,
)


# ----------------------------
# Best params loading -> config
# ----------------------------


def build_dsbm_config_from_best(best: Dict, seed: int, device: str) -> IMFDSBMConfig:
    """
    Reconstruct IMFDSBMConfig from tuning best params.
    Expected keys: sigma, num_steps, eps, inner_iters, lr, batch_size, imf_len, first_coupling, noise
    """
    sigma = float(best["sigma"])
    num_steps = int(best["num_steps"])
    eps = float(best["eps"])
    inner_iters = int(best["inner_iters"])
    lr = float(best["lr"])
    batch_size = int(best["batch_size"])
    first_coupling = str(best["first_coupling"])
    noise = bool(best["noise"])

    imf_len = int(best.get("imf_len", 5))
    if imf_len % 2 == 0:
        imf_len += 1
    fb_sequence = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

    return IMFDSBMConfig(
        fb_sequence=fb_sequence,        # type: ignore[arg-type]
        num_steps=num_steps,
        sigma=sigma,
        eps=eps,
        first_coupling=first_coupling,  # type: ignore[arg-type]
        inner_iters=inner_iters,
        batch_size=batch_size,
        lr=lr,
        weight_decay=0.0,
        grad_clip=1.0,
        noise=noise,
        device=device,
        seed=seed,
    )


# ----------------------------
# Main experiment
# ----------------------------

def main() -> None:
    print_legacy_warning("sbtab.experiments.legacy.calculating_metrics.dsbm_metrics")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="sbtab/data/datasets/datasets_continuous_only.pkl")
    ap.add_argument("--best_json_dir", type=str,  default="sbtab/experiments/legacy/tuning_script/dsbm_optuna_results/")
    ap.add_argument("--outdir", type=str, default="dsbm_kfold_eval")

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
    missing_ds = [d for d in ds_list if d not in my_data]
    if missing_ds:
        raise KeyError(f"Missing dataset keys in pickle: {missing_ds}")

    missing_targets = [d for d in ds_list if d not in TARGET_COL_BY_DATASET]
    if missing_targets:
        raise KeyError(f"Target column not specified for datasets: {missing_targets}. "
                       f"Add them to TARGET_COL_BY_DATASET in this script.")

    global_rows = []

    for ds_name in ds_list:
        print("\n" + "=" * 100)
        print(f"DATASET: {ds_name}")
        print("=" * 100)

        df = my_data[ds_name].copy()
        cols = list(df.columns)

        # Safety numeric cast (continuous-only expected)
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        target_col = TARGET_COL_BY_DATASET[ds_name]
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in dataset '{ds_name}'. "
                f"Available columns: {df.columns.tolist()}"
            )

        feature_cols = [c for c in cols if c != target_col]
        if len(feature_cols) < 1:
            raise ValueError(f"Dataset '{ds_name}' has no features after removing target '{target_col}'.")

        # Load best params
        best_json_path = best_dir / f"{ds_name}_best.json"
        if not best_json_path.exists():
            raise FileNotFoundError(f"Best params JSON not found: {best_json_path}")
        best_params = load_best_params(best_json_path)
        cfg = build_dsbm_config_from_best(best_params, seed=args.seed, device=args.device)

        print(f"Target column: {target_col}  (#features={len(feature_cols)})")
        print(f"Best params loaded: {best_json_path.name}")
        print(f"DSBM config: sigma={cfg.sigma}, steps={cfg.num_steps}, inner_iters={cfg.inner_iters}, lr={cfg.lr}, "
              f"batch={cfg.batch_size}, fb_sequence={cfg.fb_sequence}, first_coupling={cfg.first_coupling}, noise={cfg.noise}")

        kf = KFold(n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.seed)
        idx = np.arange(len(df))

        fold_rows = []

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(idx)):
            print(f"\n--- Fold {fold_id+1}/{args.n_splits} ---")

            df_train_raw = df.iloc[train_idx].copy()
            df_test_raw = df.iloc[test_idx].copy()

            # Preprocess per fold: fit on train only
            schema = TabularSchema.infer_from_dataframe(df, target_col=TARGET_COL_BY_DATASET[ds_name])
            print("\nInferred schema:")
            print("  continuous:", schema.continuous_cols)
            print("  discrete  :", schema.discrete_cols)
            print("  categorical:", schema.categorical_cols)
            pipe = TransformPipeline.default_dropna_and_scale()
            pipe.fit(df_train_raw, schema)

            train_scaled = pipe.transform(df_train_raw)
            test_scaled = pipe.transform(df_test_raw)

            # Train DSBM on preprocessed train
            model = IMFDSBMSolver(dim=len(cols), cfg=cfg)
            model.fit(train_scaled)

            # Sample synthetic dataset of size equal to test fold size
            x_synth = model.sample(
                n=len(test_scaled),
                seed=args.seed + 1000 + fold_id,
                steps=cfg.num_steps,
            )
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
                    "n_train": len(train_scaled),
                    "n_test": len(test_scaled),
                    "avg_kl": float(m_kl),
                    "avg_wd": float(m_wd),
                    "corr_frob": float(m_corr),
                    "delta_r2_percent": float(util_delta),
                    "r2_real": float(r2_real),
                    "r2_synth": float(r2_syn),
                }
            )

            print(f"avg_KL={m_kl:.6f}  avg_WD={m_wd:.6f}  corr_F={m_corr:.6f}  deltaR2%={util_delta:.3f}")

        # Save per-dataset fold metrics
        fold_df = pd.DataFrame(fold_rows)
        fold_csv = outdir / f"{ds_name}_fold_metrics.csv"
        fold_df.to_csv(fold_csv, index=False)

        # Summary stats
        def mean_std(s: pd.Series) -> Tuple[float, float]:
            return float(s.mean()), float(s.std(ddof=0))

        summary = {
            "dataset": ds_name,
            "target_col": target_col,
            "n_splits": int(args.n_splits),
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "best_params_path": str(best_json_path),
            "best_params": best_params,
            "dsbm_config": asdict(cfg),
            "metrics_mean": {},
            "metrics_std": {},
        }

        for key in ["avg_kl", "avg_wd", "corr_frob", "delta_r2_percent", "r2_real", "r2_synth"]:
            mu, sd = mean_std(fold_df[key])
            summary["metrics_mean"][key] = mu
            summary["metrics_std"][key] = sd

        summary_json = outdir / f"{ds_name}_kfold_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        global_rows.append(
            {
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
            }
        )

        print(f"\nSaved fold metrics:   {fold_csv}")
        print(f"Saved dataset summary:{summary_json}")

    # Global summary CSV
    global_df = pd.DataFrame(global_rows).sort_values("avg_wd_mean", ascending=True)
    global_csv = outdir / "kfold_summary_all_datasets.csv"
    global_df.to_csv(global_csv, index=False)

    print("\n" + "=" * 100)
    print("DONE. Global summary saved:")
    print(global_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()