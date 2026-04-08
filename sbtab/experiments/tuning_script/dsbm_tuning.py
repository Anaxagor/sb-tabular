
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from scipy.stats import wasserstein_distance

from sbtab.data.schema import TabularSchema
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.splits import SplitConfigHoldout
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import IMFDSBMSolver, IMFDSBMConfig


DEFAULT_DATASETS = [
    "diabetes",
    "online_news_popularity",
    "king_county_housing",
    "bank_loan",
    "bank_marketing",
    "online_shoppers",
    "covertype",
    "german_credit",
    "california_housing"
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



def average_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Average 1D Wasserstein distance across columns."""
    wds = []
    for c in cols:
        wds.append(float(wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy())))
    return float(np.mean(wds))


def export_trials_csv(study: optuna.Study, out_csv: Path) -> None:
    """Export all trials to CSV for offline analysis."""
    rows = []
    for tr in study.trials:
        row = {
            "trial_number": tr.number,
            "state": str(tr.state),
            "value": tr.value,
            **tr.params,
        }
        # store exception if present
        if "exception" in tr.user_attrs:
            row["exception"] = tr.user_attrs["exception"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def make_objective_for_dataset(
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    cols: List[str],
    seed: int,
    device: str,
):
    """
    Objective function factory that reuses a FIXED train/test split and FIXED preprocessing.

    Inputs:
      - train_scaled: transformed train (used for fitting generative model)
      - test_scaled: transformed test (used to set sample size and to compute metric)
      - inv_pipe: pipeline fitted on raw train (for inverse transform to original scale)
      - cols: column order
    """

    def objective(trial: optuna.Trial) -> float:
        # --- hyperparameter search space ---
        sigma = trial.suggest_float("sigma", 0.03, 0.30, log=True)
        num_steps = trial.suggest_int("num_steps", 200, 2000, log=True)
        eps = trial.suggest_float("eps", 1e-4, 5e-3, log=True)

        inner_iters = trial.suggest_int("inner_iters", 500, 4000, log=True)
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        imf_len = trial.suggest_int("imf_len", 3, 7, step=2)  # odd => b,f,b,...
        fb_sequence = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

        first_coupling = trial.suggest_categorical("first_coupling", ["ref", "ind"])
        noise = trial.suggest_categorical("noise", [True, False])

        cfg = IMFDSBMConfig(
            fb_sequence=fb_sequence,                     # type: ignore[arg-type]
            num_steps=int(num_steps),
            sigma=float(sigma),
            eps=float(eps),
            first_coupling=first_coupling,               # type: ignore[arg-type]
            inner_iters=int(inner_iters),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=0.0,
            grad_clip=1.0,
            noise=bool(noise),
            device=device,
            seed=seed,
        )

        try:
            model = IMFDSBMSolver(dim=len(cols), cfg=cfg)
            model.fit(train_scaled)

            n_synth = len(test_scaled)
            x_synth = model.sample(n=n_synth, seed=seed + 123, steps=int(num_steps))
            synth_scaled = pd.DataFrame(x_synth, columns=cols)
            score = average_wd(test_scaled, synth_scaled, cols)

            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            trial.set_user_attr("exception", repr(e))
            return float("inf")

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
    ap.add_argument("--storage", type=str, default="sqlite:///dsbm_multi_optuna.db")
    ap.add_argument("--study-prefix", type=str, default="dsbm_imf")

    ap.add_argument("--outdir", type=str, default="dsbm_optuna_results", help="Folder for CSV summaries")
    ap.add_argument("--export-trials", action="store_true", help="Export per-trial CSV for each dataset")

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
        cols = list(df.columns)  # IMPORTANT: use all columns

        if len(cols) < 2:
            raise ValueError(f"Dataset '{ds_name}' has <2 columns; cannot tune DSBM.")

        # Safety: ensure numeric; datasets are said to already be continuous-only.
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        schema = TabularSchema.infer_from_dataframe(
        df,
        target_col=TARGET_COL_BY_DATASET[ds_name])
        print("\nInferred schema:")
        print("  continuous:", schema.continuous_cols)
        print("  discrete  :", schema.discrete_cols)
        print("  categorical:", schema.categorical_cols)

        transforms = TransformPipeline.default_dropna_and_scale()

        # Single split (train/test) via DataModule holdout
        dm = TabularDataModule(df=df, schema=schema, transforms=transforms)
        dm.prepare_holdout(SplitConfigHoldout(val_size=args.test_size, shuffle=True, random_state=args.seed))
        holdout = dm.get_holdout()

        train_scaled = holdout.train
        test_scaled = holdout.val


        print(f"Columns: {len(cols)}")
        print(f"Train size (scaled): {len(train_scaled)}")
        print(f"Test size  (scaled): {len(test_scaled)}")

            

        # Create study per dataset
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
            train_scaled=train_scaled,
            test_scaled=test_scaled,
            cols=cols,
            seed=args.seed,
            device=args.device,
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
        print(f"Best avg WD: {best.value}")
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print(f"Trials: {len(study.trials)}  Elapsed: {elapsed:.1f}s")

        # Save per-dataset summary JSON
        best_json = {
            "dataset": ds_name,
            "best_avg_wd": float(best.value),
            "best_trial": int(best.number),
            "n_trials": int(len(study.trials)),
            "elapsed_sec": float(elapsed),
            "best_params": dict(best.params),
        }
        (outdir / f"{ds_name}_best.json").write_text(json.dumps(best_json, indent=2), encoding="utf-8")

        if args.export_trials:
            export_trials_csv(study, outdir / f"{ds_name}_trials.csv")

        summary_rows.append(
            {
                "dataset": ds_name,
                "best_avg_wd": float(best.value),
                "best_trial": int(best.number),
                "n_trials": int(len(study.trials)),
                "elapsed_sec": float(elapsed),
                **best.params,
            }
        )

    # Global summary CSV across all datasets
    summary_df = pd.DataFrame(summary_rows).sort_values("best_avg_wd", ascending=True)
    out_csv = outdir / "dsbm_optuna_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY (sorted by best_avg_wd)")
    print("=" * 90)
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(summary_df)
    print(f"\nSaved summary CSV to: {out_csv}")
    print(f"Saved per-dataset best JSON files to: {outdir}")


if __name__ == "__main__":
    main()
