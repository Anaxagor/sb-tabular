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

Historical purpose: Optuna tuning of the TabDDPM baseline (objective: mean processed-space WD on an 80/20 holdout).

Original description (kept for context):
  Optuna tuning for TabDDPM under the mixed-type repository logic. TabularDataModule fits the
  pipeline on the train subset of a single holdout split; TabDDPMWrapper is trained on the
  already-preprocessed DataFrame and receives the schema and the fitted transforms.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sbtab.baselines.tabddpm.model import TabDDPMConfig, TabDDPMWrapper
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema
from sbtab.data.splits import SplitConfigHoldout
from sbtab.experiments.legacy.legacy_metrics import (
    TARGET_COL_BY_DATASET,
    average_wd_processed_echo_columns as average_wd_processed,
    build_transforms_tabddpm_tuning as build_transforms,
    export_trials_csv,
    print_legacy_warning,
)


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
    print_legacy_warning("sbtab.experiments.legacy.tuning_script.tabddpm_tuning")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="sbtab/data/datasets/datasets_continuous_only.pkl")
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
