"""
Optuna tuning: DSBM + CT + MLP + Joint on **mixed** (continuous + categorical) data.

Key differences from the continuous-only variant:
  - Uses TabularSchema.infer_from_dataframe() to auto-classify columns
  - Uses TransformPipeline.default_impute_scale_encode() which applies:
      TypeAwareImputer -> ContinuousStandardScaler -> OneHotRepresentation
  - After one-hot encoding, all columns become numeric and IMFDSBMSolver
    operates in the expanded numeric space
  - Metric (average WD) is computed on the encoded (numeric) columns
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import (
    IMFDSBMSolver,
    IMFDSBMConfig,
)


# Default datasets — same as continuous-only; the script auto-detects mixed columns.
# Replace or extend with datasets that actually contain categorical features.
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


def average_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Average 1-D Wasserstein distance across numeric columns."""
    return float(np.mean([
        wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols
    ]))


def export_trials_csv(study: optuna.Study, out_csv: Path) -> None:
    rows = []
    for tr in study.trials:
        row = {"trial_number": tr.number, "state": str(tr.state), "value": tr.value, **tr.params}
        if "exception" in tr.user_attrs:
            row["exception"] = tr.user_attrs["exception"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def build_transforms(schema: TabularSchema, *, missing_strategy: str) -> TransformPipeline:
    """
    Select the appropriate pipeline based on schema content.

    - If categorical columns exist: impute + scale + one-hot encode
    - Otherwise: drop NaN + scale (or impute + scale)
    """
    if schema.has_categorical:
        if missing_strategy == "drop":
            raise ValueError(
                "missing_strategy='drop' is not supported for mixed/categorical datasets. "
                "Use 'impute'."
            )
        return TransformPipeline.default_impute_scale_encode()

    if missing_strategy == "impute":
        return TransformPipeline.default_impute_and_scale()
    return TransformPipeline.default_dropna_and_scale()


def make_objective(
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    cols: List[str],
    seed: int,
    max_train_size: int,
    device: str,
):
    def objective(trial: optuna.Trial) -> float:
        # --- SB hyperparameters ---
        sigma        = trial.suggest_float("sigma", 0.03, 0.50, log=True)
        num_steps    = trial.suggest_int("num_steps", 50, 1000, step=50)
        eps          = trial.suggest_float("eps", 1e-4, 5e-3, log=True)
        imf_len      = trial.suggest_int("imf_len", 1, 4) * 2 + 1   # 3,5,7,9
        first_coupling = trial.suggest_categorical("first_coupling", ["ref", "ind"])
        noise        = trial.suggest_categorical("noise", [True, False])

        # --- MLP training hyperparameters ---
        inner_iters  = trial.suggest_int("inner_iters", 500, 5000, step=500)
        batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512])
        lr           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3)
        grad_clip    = trial.suggest_float("grad_clip", 0.5, 5.0)

        fb_sequence: Tuple[str, ...] = tuple(
            "b" if i % 2 == 0 else "f" for i in range(imf_len)
        )

        cfg = IMFDSBMConfig(
            fb_sequence=fb_sequence,
            num_steps=num_steps,
            sigma=sigma,
            eps=eps,
            first_coupling=first_coupling,
            inner_iters=inner_iters,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            noise=noise,
            device=device,
            seed=seed,
        )

        try:
            train_fit = train_scaled
            if max_train_size > 0 and len(train_scaled) > max_train_size:
                rng = np.random.default_rng(seed + trial.number)
                idx = rng.choice(len(train_scaled), size=max_train_size, replace=False)
                train_fit = train_scaled.iloc[idx].reset_index(drop=True)

            # dim = number of columns AFTER encoding (one-hot expands categoricals)
            model = IMFDSBMSolver(dim=len(cols), cfg=cfg)
            model.fit(train_fit)

            x_synth = model.sample(n=len(test_scaled), seed=seed + 123)
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
    ap = argparse.ArgumentParser(
        description="Optuna tuning: DSBM+CT+MLP+joint on mixed (continuous+categorical) data"
    )
    ap.add_argument("--pickle", type=str,
                    default="sbtab/data/datasets/datasets_continuous_only.pkl",
                    help="Path to pickle with Dict[str, pd.DataFrame]. "
                         "Can contain mixed-type DataFrames.")
    ap.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-size", type=int, default=5000, dest="max_train_size")
    ap.add_argument("--device", type=str, default="cpu",
                    help="torch device for MLP training (cpu / cuda / mps)")
    ap.add_argument("--missing-strategy", type=str, default="impute",
                    choices=["impute", "drop"], dest="missing_strategy",
                    help="How to handle missing values. 'impute' required for categorical data.")

    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=0, help="Seconds per dataset (0 = no limit)")
    ap.add_argument("--storage", type=str, default="sqlite:///dsbm_ct_mlp_mixed_optuna.db")
    ap.add_argument("--study-prefix", type=str, default="dsbm_ct_mlp_mixed")

    ap.add_argument("--outdir", type=str, default="dsbm_ct_mlp_mixed_optuna_results")
    ap.add_argument("--export-trials", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data: Dict[str, pd.DataFrame] = pickle.load(f)

    dataset_keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    missing = [k for k in dataset_keys if k not in my_data]
    if missing:
        raise KeyError(f"Missing dataset keys: {missing}")

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    summary_rows = []

    for ds_name in dataset_keys:
        print("\n" + "=" * 90)
        print(f"DATASET: {ds_name}  [DSBM+CT+MLP+joint mixed-data tuning]")
        print("=" * 90)

        df = my_data[ds_name].copy()

        # Infer schema: auto-classifies columns into continuous / discrete / categorical
        target_col = TARGET_COL_BY_DATASET.get(ds_name)
        schema = TabularSchema.infer_from_dataframe(df, target_col=target_col)

        print(f"  continuous : {schema.continuous_cols}")
        print(f"  discrete   : {schema.discrete_cols}")
        print(f"  categorical: {schema.categorical_cols}")
        print(f"  target     : {schema.target_col}")

        # Build appropriate transform pipeline
        transforms = build_transforms(schema, missing_strategy=args.missing_strategy)

        dm = TabularDataModule(df=df, schema=schema, transforms=transforms, reset_index=True)
        dm.prepare_holdout(SplitConfigHoldout(
            val_size=args.test_size, shuffle=True, random_seed=args.seed
        ))
        holdout = dm.get_holdout()

        train_scaled = holdout.train
        test_scaled  = holdout.val

        # After encoding, columns may have expanded (one-hot)
        cols = list(train_scaled.columns)
        print(f"  encoded cols={len(cols)}, train={len(train_scaled)}, test={len(test_scaled)}")

        study_name = f"{args.study_prefix}__{ds_name}"
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage if args.storage != ":memory:" else None,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        objective = make_objective(
            train_scaled=train_scaled,
            test_scaled=test_scaled,
            cols=cols,
            seed=args.seed,
            max_train_size=args.max_train_size,
            device=args.device,
        )

        t0 = time.time()
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout if args.timeout > 0 else None,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        elapsed = time.time() - t0

        best = study.best_trial
        print(f"\n  Best avg WD: {best.value:.6f}")
        print(f"  Best params: {best.params}")
        print(f"  Trials: {len(study.trials)}  Elapsed: {elapsed:.1f}s")

        best_json = {
            "dataset": ds_name,
            "solver": "DSBM+CT+MLP+joint (mixed)",
            "schema": {
                "continuous": schema.continuous_cols,
                "discrete": schema.discrete_cols,
                "categorical": schema.categorical_cols,
                "target": schema.target_col,
            },
            "n_encoded_cols": len(cols),
            "best_avg_wd": float(best.value),
            "best_trial": int(best.number),
            "n_trials": int(len(study.trials)),
            "elapsed_sec": float(elapsed),
            "best_params": dict(best.params),
        }
        (outdir / f"{ds_name}_best.json").write_text(
            json.dumps(best_json, indent=2), encoding="utf-8"
        )

        if args.export_trials:
            export_trials_csv(study, outdir / f"{ds_name}_trials.csv")

        summary_rows.append({
            "dataset": ds_name,
            "n_continuous": len(schema.continuous_cols),
            "n_discrete": len(schema.discrete_cols),
            "n_categorical": len(schema.categorical_cols),
            "n_encoded_cols": len(cols),
            "best_avg_wd": float(best.value),
            "best_trial": int(best.number),
            "n_trials": int(len(study.trials)),
            "elapsed_sec": float(elapsed),
            **best.params,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("best_avg_wd")
    out_csv = outdir / "ct_mlp_mixed_optuna_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("TUNING DONE. Summary:")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(summary_df)
    print(f"\nSaved to: {out_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
