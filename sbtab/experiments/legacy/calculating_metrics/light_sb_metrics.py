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

Historical purpose: 5-fold evaluation of the LightSB solver.
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
from sbtab.models.sb.light_sb import LightSBPotentialConfig
from sbtab.solvers.light_sb.config import LightSBConfig
from sbtab.solvers.light_sb.solver import LightSBSolver
from sbtab.experiments.legacy.legacy_metrics import (
    TARGET_COL_BY_DATASET,
    avg_kl_hist,
    avg_wd,
    corr_frobenius_raw as corr_frobenius,
    print_legacy_warning,
    resolve_target_col_last_column_fallback as resolve_target_col,
    utility_delta_r2_percent_raw_target as utility_delta_r2_percent,
)


def main() -> None:
    print_legacy_warning("sbtab.experiments.legacy.calculating_metrics.light_sb_metrics")
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
