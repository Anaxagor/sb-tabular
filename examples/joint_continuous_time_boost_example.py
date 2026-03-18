

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Option A: sklearn version (no Kaggle needed)
from sklearn.datasets import fetch_california_housing

# --- sb-tabular imports ---
from sbtab.data.schema import TabularSchema
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.splits import SplitConfigHoldout
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.continuous_time.joint_distribution.boosting.imf_dsbm.solver import IMFDSBMContinuousJointCatBoostConfig, IMFDSBMContinuousJointCatBoostSolver
from sbtab.models.boosted.catboost_continuous_joint import CatBoostContinuousField, CatBoostContinuousFieldConfig


def load_california_housing_sklearn() -> pd.DataFrame:
    """
    Loads California Housing as a DataFrame with 8 features + target.
    Target column is named "MedianHouseValue" for consistency with Kaggle.
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()

    # sklearn naming:
    # MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseVal
    df = df.rename(columns={"MedHouseVal": "MedianHouseValue"})
    return df


def load_california_housing_kaggle_csv(csv_path: str) -> pd.DataFrame:
    """
    Kaggle dataset `california-housing-prices` often provides `housing.csv`
    with columns:
      longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
      population, households, median_income, median_house_value, ocean_proximity

    NOTE: This repository currently assumes fully continuous data, so you must
    drop/encode 'ocean_proximity' before using this.
    """
    df = pd.read_csv(csv_path)

    # Keep only continuous columns (drop ocean_proximity)
    # Rename columns to match a consistent schema if you want.
    if "ocean_proximity" in df.columns:
        df = df.drop(columns=["ocean_proximity"])

    # Optional renaming to the sklearn-style names (not required)
    rename = {
        "median_income": "MedInc",
        "housing_median_age": "HouseAge",
        "total_rooms": "AveRooms",        # note: not exactly "average", but ok as an example
        "total_bedrooms": "AveBedrms",    # same note
        "population": "Population",
        "households": "AveOccup",         # not exactly; again example
        "latitude": "Latitude",
        "longitude": "Longitude",
        "median_house_value": "MedianHouseValue",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Ensure float dtype
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main() -> None:
    # -------------------------------
    # 1) Load data
    # -------------------------------
    # Choose ONE:
    df = load_california_housing_sklearn()

    # OR Kaggle CSV:
    # df = load_california_housing_kaggle_csv("housing.csv")

    # Columns for sklearn-loaded version:
    feature_cols = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    target_col = "MedianHouseValue"

    # For *joint* modeling, include the target as a feature
    joint_cols = feature_cols + [target_col]

    # Keep only the joint columns
    df = df[joint_cols].copy()

    # -------------------------------
    # 2) Define schema + transforms
    # -------------------------------
    schema = TabularSchema(feature_cols=joint_cols)

    # Default pipeline: DropMissingRows -> StandardScaler (continuous-only)
    transforms = TransformPipeline.default_continuous_dropna()

    # -------------------------------
    # 3) Create DataModule + holdout split
    # -------------------------------
    dm = TabularDataModule(df=df, schema=schema, transforms=transforms, reset_index=True)

    holdout_cfg = SplitConfigHoldout(val_size=0.2, shuffle=True, random_seed=42)
    dm.prepare_holdout(holdout_cfg)

    holdout = dm.get_holdout()
    train_df = holdout.train
    val_df = holdout.val

    print("Train shape (transformed):", train_df.shape)
    print("Val shape   (transformed):", val_df.shape)

    # -------------------------------
    # 4) Train IPF+DSB solver (SB model)
    # -------------------------------
    dim = len(joint_cols)

    cb_cfg = CatBoostContinuousFieldConfig(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    task_type="CPU",
    verbose=False,
)

    cfg = IMFDSBMContinuousJointCatBoostConfig(
        fb_sequence=("b", "f", "b", "f", "b"),
        num_steps=10,
        sigma=0.10,
        eps=1e-3,
        first_coupling="ind",
        n_noise_per_pair=2,
        noise=True,
        seed=42,
        field=cb_cfg,
    )

    model = IMFDSBMContinuousJointCatBoostSolver(dim=train_df.shape[1], cfg=cfg).fit(train_df)
    x_synth = model.sample(n=5000, seed=123)
    

    # -------------------------------
    # 5) Generate synthetic samples (in transformed space)
    # -------------------------------
    n_synth = len(val_df)  # e.g., match validation size
    x_synth = model.sample(n=n_synth, seed=123)

    synth_df_scaled = pd.DataFrame(x_synth, columns=joint_cols)

    # -------------------------------
    # 6) Inverse transform back to original scale
    # -------------------------------
    # IMPORTANT:
    # DataModule fits transforms fold-wise internally, but we need the *same* fitted
    # pipeline used for this holdout split to invert samples.
    #
    # Easiest approach:
    # - create a fresh pipeline and fit it on *train_raw after global dropna*.
    # Here, dm.df_clean is already "missing-clean" (global), but still in original scale
    # only if transforms.transform was not applied pre-fit.
    #
    # In our DataModule implementation, df_clean is missing-clean and still may already be
    # transformed depending on your pipeline behavior.
    #
    # The most robust way: fit a *new* pipeline on the ORIGINAL train subset before scaling.
    #
    # Since our dm.get_holdout() returns already transformed frames, we rebuild train_raw from dm.df_clean
    # and refit a pipeline the same way dm.get_holdout does.

    # Rebuild raw holdout split indices:
    # (This assumes your TabularDataModule keeps _holdout_split; it does.)
    sp = dm._holdout_split  # type: ignore[attr-defined]
    train_raw = dm.df_clean.iloc[sp.train_idx].copy()  # type: ignore[attr-defined]

    inv_pipe = TransformPipeline.default_continuous_dropna()
    inv_pipe.fit(train_raw, schema)
    synth_df = inv_pipe.inverse_transform(synth_df_scaled)

    # -------------------------------
    # 7) Simple sanity checks
    # -------------------------------
    def summarize(name: str, real: pd.DataFrame, fake: pd.DataFrame) -> None:
        print(f"\n=== {name} summary (real vs synth) ===")
        real_mean = real.mean()
        fake_mean = fake.mean()
        real_std = real.std(ddof=0)
        fake_std = fake.std(ddof=0)

        out = pd.DataFrame(
            {
                "real_mean": real_mean,
                "synth_mean": fake_mean,
                "real_std": real_std,
                "synth_std": fake_std,
            }
        )
        print(out)

        # Correlation Frobenius norm
        rc = real.corr().to_numpy()
        fc = fake.corr().to_numpy()
        frob = np.linalg.norm(rc - fc, ord="fro")
        print("\nCorrelation Frobenius ||Corr(real)-Corr(synth)||_F =", float(frob))

    # Compare on original scale (more interpretable)
    val_raw = inv_pipe.inverse_transform(val_df)
    summarize("Holdout VAL", val_raw, synth_df)

    # -------------------------------
    # 8) Save synthetic data
    # -------------------------------
    out_path = "california_housing_synth_sb.csv"
    synth_df.to_csv(out_path, index=False)
    print("\nSaved synthetic samples to:", out_path)
    print("Head:\n", synth_df.head())


if __name__ == "__main__":
    main()
