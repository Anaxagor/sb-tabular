#!/usr/bin/env python3
from __future__ import annotations

"""
Optuna tuning for TabDDPM under the current mixed-type repository logic.

Updated using the actual TransformPipeline API:
  - TransformPipeline.default_dropna_and_scale()
  - TransformPipeline.default_impute_and_scale()
  - TransformPipeline.default_impute_scale_encode()

Key repository assumptions reflected here:
  1. TabularSchema.infer_from_dataframe(...) splits columns into continuous / discrete / categorical.
  2. TabularDataModule applies only stateless global-safe transforms before splitting and
     then fits the pipeline on the train subset only for each holdout split.
  3. TabDDPMWrapper is trained on the *already-preprocessed* DataFrame and should receive both:
       - schema
       - the fitted split-specific transforms object
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.stats import wasserstein_distance

from sbtab.baselines.tabddpm.model import TabDDPMConfig, TabDDPMWrapper
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema
from sbtab.data.splits import SplitConfigHoldout
from sbtab.transforms.pipeline import TransformPipeline


DEFAULT_DATASETS = [
    "diabetes",
    "online_news_popularity",
    "king_county_housing",
    "bank_loan",
    "bank_marketing",
    "online_shoppers",
    "covertype",
    "german_credit",
    "california_housing",
]

TARGET_COL_BY_DATASET: Dict[str, str] = {
    "german_credit":'duration',
    "online_news_popularity": " shares",
    "covertype": "Horizontal_Distance_To_Hydrology",
    "online_shoppers": "ProductRelated",
    "bank_marketing": "pdays",
    "bank_loan": "Income",
    "diabetes": "target",
    "california_housing": "MedHouseVal",
    "king_county_housing": "price"
    

}


def average_wd_processed(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """
    Average 1D Wasserstein distance across all numeric columns in the processed representation.

    In the new module logic, the holdout split returned by TabularDataModule is already transformed.
    After `default_impute_scale_encode()`, categorical features become one-hot numeric columns and can
    also be included in this processed-space comparison.
    """
    common_cols = [c for c in real.columns if c in synth.columns]
    metric_cols = [
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(synth[c])
    ]
    if not metric_cols:
        raise ValueError("No common numeric columns available for Wasserstein metric.")

    wds = []
    for c in metric_cols:
        print(c)
        wds.append(float(wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy())))
    return float(np.mean(wds))


def export_trials_csv(study: optuna.Study, out_csv: Path) -> None:
    rows = []
    for tr in study.trials:
        row = {
            "trial_number": tr.number,
            "state": str(tr.state),
            "value": tr.value,
            **tr.params,
        }
        if "exception" in tr.user_attrs:
            row["exception"] = tr.user_attrs["exception"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def build_transforms(schema: TabularSchema, *, missing_strategy: str) -> TransformPipeline:
    """
    Select the actual pipeline constructor exposed by the repository.

    Missing strategies:
      - "impute" : keep rows, impute missing values
      - "drop"   : drop rows with missing values (only supported for continuous-only tuning)
    """
    if schema.has_categorical:
        # For mixed data, the explicit pipeline in pipeline.py is:
        #   TypeAwareImputer -> ContinuousStandardScaler -> CategoricalRepresentationTransform(one_hot)
        if missing_strategy == "drop":
            raise ValueError(
                "missing_strategy='drop' is not supported for mixed/categorical datasets by the current "
                "TransformPipeline API. Use 'impute'."
            )
        return TransformPipeline.default_impute_scale_encode()

    # Continuous/discrete-only datasets
    if missing_strategy == "drop":
        return TransformPipeline.default_dropna_and_scale()
    return TransformPipeline.default_impute_and_scale()


def make_objective_for_dataset(
    train_proc: pd.DataFrame,
    val_proc: pd.DataFrame,
    *,
    seed: int,
    device: str,
    schema: TabularSchema,
    fitted_transforms,
):
    """
    Objective factory for one dataset and one fixed holdout split.
    """
    def objective(trial: optuna.Trial) -> float:
        # Search space aligned with TabDDPM paper defaults
        n_epochs = trial.suggest_categorical("n_epochs", [5000, 10000, 20000])
        num_timesteps = trial.suggest_categorical("num_timesteps", [100, 1000])
        batch_size = trial.suggest_categorical("batch_size", [256, 4096])

        n_layers = trial.suggest_int("n_layers", 2, 8)
        layer_size = trial.suggest_categorical("layer_size", [128, 256, 512, 1024])
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)

        cfg = TabDDPMConfig(
            n_epochs=int(n_epochs),
            num_timesteps=int(num_timesteps),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=1e-4,
            d_layers=[layer_size] * n_layers,
            dropout=0.0,
            scheduler="cosine",
            device=device,
            seed=seed,
        )

        # try:
        model = TabDDPMWrapper(cfg=cfg)
        model.fit(train_proc, schema=schema, transforms=fitted_transforms)

        synth_df = model.sample(n=len(val_proc), seed=seed + 123)
        score = average_wd_processed(val_proc, synth_df)

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

        # except optuna.TrialPruned:
        #     raise
        # except Exception as e:
        #     trial.set_user_attr("exception", repr(e))
            #return float("inf")

    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="C:/Users/Anaxagor/Documents/projects/sb-tabular/sbtab/data/datasets/datasets_continuous_only.pkl")
    ap.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=0, help="Seconds per dataset (0 => no timeout)")
    ap.add_argument("--storage", type=str, default="sqlite:///tabddpm_optuna.db")
    ap.add_argument("--study-prefix", type=str, default="tabddpm")

    ap.add_argument("--outdir", type=str, default="tabddpm_optuna_results")
    ap.add_argument("--export-trials", action="store_true")
    ap.add_argument(
        "--missing-strategy",
        type=str,
        default="impute",
        choices=["impute", "drop"],
        help="Which TransformPipeline variant to use. "
             "'impute' -> default_impute_and_scale/default_impute_scale_encode; "
             "'drop' -> default_dropna_and_scale (continuous-only).",
    )

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data: Dict[str, pd.DataFrame] = pickle.load(f)

    dataset_keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    missing = [k for k in dataset_keys if k not in my_data]
    if missing:
        raise KeyError(f"These dataset keys are missing in pickle: {missing}")

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    summary_rows = []

    for ds_name in dataset_keys:
        print("\n" + "=" * 90)
        print(f"DATASET: {ds_name}")
        print("=" * 90)

        df = my_data[ds_name].copy()
        if df.shape[1] < 2:
            raise ValueError(f"Dataset '{ds_name}' has <2 columns; cannot tune TabDDPM.")

        # New schema logic: infer mixed feature groups from the raw DataFrame
        schema = TabularSchema.infer_from_dataframe(df, target_col=TARGET_COL_BY_DATASET[ds_name])
        transforms = build_transforms(schema, missing_strategy=args.missing_strategy)

        dm = TabularDataModule(
            df=df,
            schema=schema,
            transforms=transforms,
            reset_index=True,
        )
        dm.prepare_holdout(
            SplitConfigHoldout(
                val_size=args.test_size,
                shuffle=True,
                random_state=args.seed,
            )
        )
        holdout = dm.get_holdout()

        train_proc = holdout.train
        val_proc = holdout.val
        fitted_transforms = holdout.transforms

        print(
            "Schema: "
            f"continuous={len(schema.continuous_cols)}, "
            f"discrete={len(schema.discrete_cols)}, "
            f"categorical={len(schema.categorical_cols)}"
        )
        print(f"Pipeline: {transforms.__class__.__name__}")
        print(f"Train size (processed): {len(train_proc)}")
        print(f"Val size   (processed): {len(val_proc)}")
        print(f"Processed columns: {len(train_proc.columns)}")

        study_name = f"{args.study_prefix}__{ds_name}"
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage if args.storage != ":memory:" else None,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        objective = make_objective_for_dataset(
            train_proc=train_proc,
            val_proc=val_proc,
            seed=args.seed,
            device=args.device,
            schema=schema,
            fitted_transforms=fitted_transforms,
        )

        t0 = time.time()
        study.optimize(
            objective,
            n_trials=int(args.n_trials),
            timeout=(args.timeout if args.timeout > 0 else None),
            gc_after_trial=True,
            show_progress_bar=True,
        )
        elapsed = time.time() - t0

        best = study.best_trial
        print("\n--- BEST RESULT ---")
        print(f"Dataset: {ds_name}")
        print(f"Best avg WD (processed space): {best.value}")
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print(f"Trials: {len(study.trials)}  Elapsed: {elapsed:.1f}s")

        best_json = {
            "dataset": ds_name,
            "best_avg_wd_processed": float(best.value),
            "best_trial": int(best.number),
            "n_trials": int(len(study.trials)),
            "elapsed_sec": float(elapsed),
            "best_params": dict(best.params),
            "missing_strategy": args.missing_strategy,
        }
        (outdir / f"{ds_name}_best.json").write_text(json.dumps(best_json, indent=2), encoding="utf-8")

        if args.export_trials:
            export_trials_csv(study, outdir / f"{ds_name}_trials.csv")

        summary_rows.append(
            {
                "dataset": ds_name,
                "best_avg_wd_processed": float(best.value),
                "best_trial": int(best.number),
                "n_trials": int(len(study.trials)),
                "elapsed_sec": float(elapsed),
                "missing_strategy": args.missing_strategy,
                **best.params,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("best_avg_wd_processed", ascending=True)
    out_csv = outdir / "tabddpm_optuna_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY (sorted by best_avg_wd_processed)")
    print("=" * 90)
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(summary_df)
    print(f"\nSaved summary CSV to: {out_csv}")
    print(f"Saved per-dataset best JSON files to: {outdir}")


if __name__ == "__main__":
    main()
