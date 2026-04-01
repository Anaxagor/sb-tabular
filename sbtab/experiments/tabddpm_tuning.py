from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List

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
from sbtab.baselines.tabddpm.model import TabDDPMWrapper, TabDDPMConfig


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
        if "exception" in tr.user_attrs:
            row["exception"] = tr.user_attrs["exception"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def make_objective_for_dataset(
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    inv_pipe: TransformPipeline,
    cols: List[str],
    seed: int,
    device: str,
    schema: TabularSchema,
):
    """
    Objective function factory for TabDDPM tuning.
    Uses fixed train/test split and fixed preprocessing.
    """
    def objective(trial: optuna.Trial) -> float:
        # --- hyperparameter search space (TabDDPM spec) ---
        n_iter = trial.suggest_int("n_iter", 500, 1000, step=100)
        num_timesteps = trial.suggest_int("num_timesteps", 100, 600, step=50)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        cfg = TabDDPMConfig(
            n_epochs=int(n_iter),
            num_timesteps=int(num_timesteps),
            batch_size=int(batch_size),
            lr=0.001,
            weight_decay=1e-4,
            seed=seed,
            device=device,
        )

        try:
            model = TabDDPMWrapper(cfg=cfg)
            model.fit(train_scaled, schema=schema)

            n_synth = len(test_scaled)
            synth_df = model.sample(n=n_synth, seed=seed + 123)

            score = average_wd(test_scaled, synth_df, cols)

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
    ap.add_argument(
        "--pickle",
        type=str,
        default="C:/Users/Anaxagor/Documents/projects/sb-tabular/sbtab/data/datasets/datasets_continuous_only.pkl",
    )
    ap.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=0, help="Seconds per dataset (0 => no timeout)")
    ap.add_argument("--storage", type=str, default="sqlite:///tabddpm_optuna.db")
    ap.add_argument("--study-prefix", type=str, default="tabddpm")

    ap.add_argument("--outdir", type=str, default="tabddpm_optuna_results", help="Folder for CSV summaries")
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
        cols = list(df.columns)

        if len(cols) < 2:
            raise ValueError(f"Dataset '{ds_name}' has <2 columns; cannot tune TabDDPM.")

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        schema = TabularSchema(feature_cols=cols)
        transforms = TransformPipeline.default_continuous_dropna()

        dm = TabularDataModule(df=df, schema=schema, transforms=transforms, reset_index=True)
        dm.prepare_holdout(SplitConfigHoldout(val_size=args.test_size, shuffle=True, random_seed=args.seed))
        holdout = dm.get_holdout()

        train_scaled = holdout.train
        test_scaled = holdout.val

        sp = dm._holdout_split  # type: ignore[attr-defined]
        train_raw = dm.df_clean.iloc[sp.train_idx].copy()  # type: ignore[attr-defined]

        inv_pipe = TransformPipeline.default_continuous_dropna()
        inv_pipe.fit(train_raw, schema)

        print(f"Columns: {len(cols)}")
        print(f"Train size (scaled): {len(train_scaled)}")
        print(f"Test size  (scaled): {len(test_scaled)}")

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
            inv_pipe=inv_pipe,
            cols=cols,
            seed=args.seed,
            device=args.device,
            schema=schema,
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

    summary_df = pd.DataFrame(summary_rows).sort_values("best_avg_wd", ascending=True)
    out_csv = outdir / "tabddpm_optuna_summary.csv"
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
