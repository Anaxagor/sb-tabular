#!/usr/bin/env python3
from __future__ import annotations

"""
K-fold evaluation for the SDV-based CTGAN wrapper under the repository's new mixed-type logic.

Main updates vs the old script:
  - uses TabularSchema.infer_from_dataframe(..., target_col=...)
  - uses the actual TransformPipeline API:
        * default_dropna_and_scale()
        * default_impute_and_scale()
        * default_impute_scale_encode()
  - uses TabularDataModule.prepare_kfold/get_fold so preprocessing is fitted on each train fold only
  - passes both schema and fitted split-specific transforms into CTGANWrapper.fit(...)
  - evaluates on the processed fold representation returned by the DataModule

Assumptions:
  - the updated CTGANWrapper is used (the SDV-based version that:
      * accepts schema + transforms
      * inverse-transforms processed data back to a raw-like table for SDV
      * samples raw-like rows and maps them back to the same processed layout as fit input)
"""

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

from sbtab.baselines.ctgan.model import CTGANConfig, CTGANWrapper
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema
from sbtab.data.splits import SplitConfigKFold
from sbtab.transforms.pipeline import TransformPipeline


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


def _common_numeric_cols(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    *,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    exclude = set(exclude_cols or [])
    cols = [c for c in real.columns if c in synth.columns and c not in exclude]
    return [
        c for c in cols
        if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(synth[c])
    ]


def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: Optional[List[str]] = None) -> float:
    metric_cols = _common_numeric_cols(real, synth) if cols is None else cols
    if not metric_cols:
        raise ValueError("No common numeric columns for Wasserstein distance.")
    return float(np.mean([
        wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy())
        for c in metric_cols
    ]))


def avg_kl_hist(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    cols: Optional[List[str]] = None,
    n_bins: int = 50,
    eps: float = 1e-12,
) -> float:
    metric_cols = _common_numeric_cols(real, synth) if cols is None else cols
    if not metric_cols:
        raise ValueError("No common numeric columns for KL metric.")

    kls: List[float] = []
    for c in metric_cols:
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


def corr_frobenius(real: pd.DataFrame, synth: pd.DataFrame, cols: Optional[List[str]] = None) -> float:
    metric_cols = _common_numeric_cols(real, synth) if cols is None else cols
    if not metric_cols:
        raise ValueError("No common numeric columns for correlation metric.")

    rc = real[metric_cols].corr().fillna(0.0).to_numpy()
    sc = synth[metric_cols].corr().fillna(0.0).to_numpy()
    return float(np.linalg.norm(rc - sc, ord="fro"))


def utility_delta_r2_percent(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    train_synth: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
) -> Tuple[float, float, float]:
    ytr = pd.to_numeric(train_real[target_col], errors="raise").to_numpy()
    yte = pd.to_numeric(test_real[target_col], errors="raise").to_numpy()
    ys = pd.to_numeric(train_synth[target_col], errors="raise").to_numpy()

    Xtr = train_real[feature_cols].to_numpy()
    Xte = test_real[feature_cols].to_numpy()
    Xs = train_synth[feature_cols].to_numpy()

    reg_real = make_regressor(seed)
    reg_real.fit(Xtr, ytr)
    r2_real = float(r2_score(yte, reg_real.predict(Xte)))

    reg_syn = make_regressor(seed + 1)
    reg_syn.fit(Xs, ys)
    r2_syn = float(r2_score(yte, reg_syn.predict(Xte)))

    delta = (r2_syn - r2_real) / (abs(r2_real) + 1e-12) * 100.0
    return float(delta), float(r2_real), float(r2_syn)


def load_best_params(best_json_path: Path) -> Dict:
    data = json.loads(best_json_path.read_text(encoding="utf-8"))
    if "best_params" in data:
        return dict(data["best_params"])
    return {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}


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


def resolve_target_col(df: pd.DataFrame, ds_name: str) -> str:
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


def build_transforms(schema: TabularSchema, *, missing_strategy: str) -> TransformPipeline:
    if schema.has_categorical:
        if missing_strategy == "drop":
            raise ValueError(
                "missing_strategy='drop' is not supported for datasets with categorical features "
                "under the current TransformPipeline API. Use 'impute'."
            )
        return TransformPipeline.default_impute_scale_encode()

    if missing_strategy == "drop":
        return TransformPipeline.default_dropna_and_scale()
    return TransformPipeline.default_impute_and_scale()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="sbtab/data/datasets/datasets_continuous_only.pkl")
    ap.add_argument("--best_json_dir", type=str, default="sbtab/experiments/tuning_script/ctgan_optuna_results/")
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
