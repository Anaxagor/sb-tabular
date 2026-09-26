#!/usr/bin/env python3
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

Historical purpose: 5-fold evaluation of the CTGAN baseline (KL / WD / Pearson-Frobenius / delta-R2).

Original description (kept for context):
  K-fold evaluation for the SDV-based CTGAN wrapper under the repository's mixed-type logic:
  TabularSchema.infer_from_dataframe(..., target_col=...), TabularDataModule.prepare_kfold/get_fold
  (preprocessing fitted on each train fold only), schema + fitted transforms passed into
  CTGANWrapper.fit(...), evaluation on the processed fold representation.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd

from sbtab.baselines.ctgan.model import CTGANConfig, CTGANWrapper
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema
from sbtab.data.splits import SplitConfigKFold
from sbtab.experiments.legacy.legacy_metrics import (
    TARGET_COL_BY_DATASET,
    avg_kl_hist_autocols as avg_kl_hist,
    avg_wd_autocols as avg_wd,
    build_transforms,
    common_numeric_cols as _common_numeric_cols,
    corr_frobenius_fillna0_autocols as corr_frobenius,
    load_best_params,
    print_legacy_warning,
    resolve_target_col_whitespace_tolerant as resolve_target_col,
    utility_delta_r2_percent_numeric_target as utility_delta_r2_percent,
)


def build_ctgan_config_from_best(best: Dict, seed: int, device: str) -> CTGANConfig:
    gen_w = int(best.get("gen_disc_width", best.get("gen_width", 512)))
    disc_w = int(best.get("gen_disc_width", best.get("disc_width", 512)))

    return CTGANConfig(
        embedding_dim=int(best.get("embedding_dim", 128)),
        generator_dim=(gen_w, gen_w),
        discriminator_dim=(disc_w, disc_w),
        generator_lr=float(best.get("generator_lr", 2e-4)),
        discriminator_lr=float(best.get("discriminator_lr", 2e-4)),
        batch_size=int(best.get("batch_size", 500)),
        epochs=int(best.get("epochs", 300)),
        pac=int(best.get("pac", 10)),
        enable_gpu=(device == "cuda"),
        seed=seed,
        verbose=False,
    )


def main() -> None:
    print_legacy_warning("sbtab.experiments.legacy.calculating_metrics.ctgan_metrics")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="sbtab/data/datasets/datasets_continuous_only.pkl")
    ap.add_argument("--best_json_dir", type=str, default="sbtab/experiments/legacy/tuning_script/ctgan_optuna_results/")
    ap.add_argument("--outdir", type=str, default="ctgan_kfold_eval")

    ap.add_argument("--datasets", type=str, default=",".join(TARGET_COL_BY_DATASET.keys()))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-bins-kl", type=int, default=20)
    ap.add_argument(
        "--missing-strategy",
        type=str,
        default="impute",
        choices=["impute", "drop"],
        help="Which TransformPipeline variant to use. "
             "'impute' -> default_impute_and_scale/default_impute_scale_encode; "
             "'drop' -> default_dropna_and_scale (continuous/discrete-only).",
    )

    args = ap.parse_args()

    best_dir = Path(args.best_json_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data: Dict[str, pd.DataFrame] = pickle.load(f)

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    global_rows = []

    for ds_name in ds_list:
        if ds_name not in my_data:
            print(f"[WARN] Dataset {ds_name!r} not found in pickle. Skipping.")
            continue

        print("\n" + "=" * 100)
        print(f"CTGAN EVALUATION: {ds_name}")
        print("=" * 100)

        df = my_data[ds_name].copy()
        target_col = resolve_target_col(df, ds_name)

        schema = TabularSchema.infer_from_dataframe(df, target_col=target_col)
        transforms = build_transforms(schema, missing_strategy=args.missing_strategy)

        dm = TabularDataModule(
            df=df,
            schema=schema,
            transforms=transforms,
            reset_index=True,
        )
        dm.prepare_kfold(
            SplitConfigKFold(
                n_splits=args.n_splits,
                shuffle=args.shuffle,
                random_state=args.seed,
            )
        )

        best_json_path = best_dir / f"{ds_name}_best.json"
        if best_json_path.exists():
            best_params = load_best_params(best_json_path)
        else:
            print(f"[WARN] Best params JSON not found for {ds_name}, using defaults.")
            best_params = {}

        cfg = build_ctgan_config_from_best(best_params, args.seed, args.device)

        fold_rows = []

        for fold_id in range(args.n_splits):
            print(f"\n--- Fold {fold_id + 1}/{args.n_splits} ---")
            fold = dm.get_fold(fold_id)

            train_proc = fold.train
            test_proc = fold.test
            fitted_transforms = fold.transforms

            model = CTGANWrapper(cfg)
            model.fit(train_proc, schema=schema, transforms=fitted_transforms)

            synth_proc = model.sample(n=len(test_proc), seed=args.seed + fold_id)

            exclude_for_metrics = [c for c in [schema.id_col] if c is not None]
            metric_cols = _common_numeric_cols(test_proc, synth_proc, exclude_cols=exclude_for_metrics)

            m_kl = avg_kl_hist(test_proc, synth_proc, cols=metric_cols, n_bins=args.n_bins_kl)
            m_wd = avg_wd(test_proc, synth_proc, cols=metric_cols)
            m_corr = corr_frobenius(test_proc, synth_proc, cols=metric_cols)

            exclude_for_utility = {c for c in [schema.target_col, schema.id_col] if c is not None}
            feature_cols_proc = [c for c in train_proc.columns if c not in exclude_for_utility]

            util_delta, r2_real, r2_syn = utility_delta_r2_percent(
                train_real=train_proc,
                test_real=test_proc,
                train_synth=synth_proc,
                feature_cols=feature_cols_proc,
                target_col=target_col,
                seed=args.seed + fold_id,
            )

            fold_rows.append(
                {
                    "dataset": ds_name,
                    "fold": fold_id,
                    "avg_kl_processed": float(m_kl),
                    "avg_wd_processed": float(m_wd),
                    "corr_frob_processed": float(m_corr),
                    "delta_r2_percent": float(util_delta),
                    "r2_real": float(r2_real),
                    "r2_synth": float(r2_syn),
                }
            )

            print(
                f"avg_KL={m_kl:.6f}  "
                f"avg_WD={m_wd:.6f}  "
                f"corr_F={m_corr:.6f}  "
                f"deltaR2%={util_delta:.3f}"
            )

        fold_df = pd.DataFrame(fold_rows)
        fold_csv = outdir / f"{ds_name}_fold_metrics.csv"
        fold_df.to_csv(fold_csv, index=False)

        summary = {
            "dataset": ds_name,
            "target_col": target_col,
            "schema": {
                "continuous_cols": schema.continuous_cols,
                "discrete_cols": schema.discrete_cols,
                "categorical_cols": schema.categorical_cols,
                "target_col": schema.target_col,
                "id_col": schema.id_col,
            },
            "best_params": asdict(cfg),
            "metrics_mean": fold_df[
                ["avg_kl_processed", "avg_wd_processed", "corr_frob_processed", "delta_r2_percent", "r2_real", "r2_synth"]
            ].mean().to_dict(),
            "metrics_std": fold_df[
                ["avg_kl_processed", "avg_wd_processed", "corr_frob_processed", "delta_r2_percent", "r2_real", "r2_synth"]
            ].std(ddof=0).to_dict(),
        }

        summary_json = outdir / f"{ds_name}_kfold_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        global_rows.append(
            {
                "dataset": ds_name,
                "avg_kl_processed_mean": summary["metrics_mean"]["avg_kl_processed"],
                "avg_kl_processed_std": summary["metrics_std"]["avg_kl_processed"],
                "avg_wd_processed_mean": summary["metrics_mean"]["avg_wd_processed"],
                "avg_wd_processed_std": summary["metrics_std"]["avg_wd_processed"],
                "corr_frob_processed_mean": summary["metrics_mean"]["corr_frob_processed"],
                "corr_frob_processed_std": summary["metrics_std"]["corr_frob_processed"],
                "delta_r2_percent_mean": summary["metrics_mean"]["delta_r2_percent"],
                "delta_r2_percent_std": summary["metrics_std"]["delta_r2_percent"],
                "r2_real_mean": summary["metrics_mean"]["r2_real"],
                "r2_synth_mean": summary["metrics_mean"]["r2_synth"],
            }
        )

    global_df = pd.DataFrame(global_rows).sort_values("avg_wd_processed_mean", ascending=True)
    global_csv = outdir / "kfold_summary_all_datasets.csv"
    global_df.to_csv(global_csv, index=False)

    print("\n" + "=" * 100)
    print(f"DONE. Global summary saved: {global_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
