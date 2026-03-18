from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing

# --- sb-tabular imports ---
from sbtab.data.schema import TabularSchema
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.splits import SplitConfigHoldout
from sbtab.transforms.pipeline import TransformPipeline

# Adjust this import path to match your repo layout if needed.
from sbtab.solvers.discrete_time.feature_wise.boosting.imf_dsbm_featurewise_boost.solver import (
    FeaturewiseDSBMBoostConfig,
    FeaturewiseDSBMBoostSolver,
)
from sbtab.models.boosted.catboost_discrete_scalar import CatBoostScalarConfig


def load_california_housing_sklearn() -> pd.DataFrame:
    """
    Loads California Housing as a DataFrame with 8 features + target.
    Target column is named "MedianHouseValue" for consistency.
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"MedHouseVal": "MedianHouseValue"})
    return df


def load_california_housing_kaggle_csv(csv_path: str) -> pd.DataFrame:
    """
    Optional Kaggle CSV loader.
    Drops non-continuous 'ocean_proximity' if present.
    """
    df = pd.read_csv(csv_path)

    if "ocean_proximity" in df.columns:
        df = df.drop(columns=["ocean_proximity"])

    rename = {
        "median_income": "MedInc",
        "housing_median_age": "HouseAge",
        "total_rooms": "AveRooms",
        "total_bedrooms": "AveBedrms",
        "population": "Population",
        "households": "AveOccup",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "median_house_value": "MedianHouseValue",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main() -> None:
    # -------------------------------
    # 1) Load data
    # -------------------------------
    df = load_california_housing_sklearn()
    # Or:
    # df = load_california_housing_kaggle_csv("housing.csv")

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

    # IMPORTANT:
    # FeaturewiseDSBMBoostSolver is autoregressive, so column order matters.
    # Putting the target last means it will be generated conditioned on the 8 features.
    feature_order = feature_cols + [target_col]

    df = df[feature_order].copy()

    # -------------------------------
    # 2) Define schema + transforms
    # -------------------------------
    schema = TabularSchema(feature_cols=feature_order)
    transforms = TransformPipeline.default_continuous_dropna()

    # -------------------------------
    # 3) Create DataModule + holdout split
    # -------------------------------
    dm = TabularDataModule(
        df=df,
        schema=schema,
        transforms=transforms,
        reset_index=True,
    )

    holdout_cfg = SplitConfigHoldout(val_size=0.2, shuffle=True, random_seed=42)
    dm.prepare_holdout(holdout_cfg)

    holdout = dm.get_holdout()
    train_df = holdout.train[feature_order].copy()
    val_df = holdout.val[feature_order].copy()

    print("Train shape (transformed):", train_df.shape)
    print("Val shape   (transformed):", val_df.shape)

    # -------------------------------
    # 4) Train Featurewise DSBM solver
    # -------------------------------
    cb_cfg = CatBoostScalarConfig(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        verbose=False,
        task_type="GPU",  # uncomment if supported in your CatBoost config/env
    )

    # Optional explicit context map.
    # If omitted, the solver defaults to "all earlier columns in feature_order".
    # With feature_order = features + target, the target will automatically
    # use all feature columns as context.
    context_cols_map = None

    cfg = FeaturewiseDSBMBoostConfig(
        fb_sequence=("b", "f", "b", "f", "b"),
        num_steps=10,
        sigma=0.10,
        eps=1e-3,
        first_coupling="ind",
        n_noise_per_pair=2,
        noise=True,
        feature_order=feature_order,
        context_cols_map=context_cols_map,
        catboost=cb_cfg,
        seed=42,
    )

    model = FeaturewiseDSBMBoostSolver(cfg=cfg).fit(train_df)

    # -------------------------------
    # 5) Generate synthetic samples (transformed space)
    # -------------------------------
    n_synth = len(val_df)
    synth_df_scaled = model.sample_df(n=n_synth, seed=123)[feature_order]

    # -------------------------------
    # 6) Inverse transform back to original scale
    # -------------------------------
    # Rebuild raw training subset used by holdout split so we can fit an inverse pipeline.
    sp = dm._holdout_split  # type: ignore[attr-defined]
    train_raw = dm.df_clean.iloc[sp.train_idx].copy()  # type: ignore[attr-defined]

    inv_pipe = TransformPipeline.default_continuous_dropna()
    inv_pipe.fit(train_raw, schema)

    synth_df = inv_pipe.inverse_transform(synth_df_scaled)
    val_raw = inv_pipe.inverse_transform(val_df)

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

        rc = real.corr().to_numpy()
        fc = fake.corr().to_numpy()
        frob = np.linalg.norm(rc - fc, ord="fro")
        print("\nCorrelation Frobenius ||Corr(real)-Corr(synth)||_F =", float(frob))

    summarize("Holdout VAL", val_raw, synth_df)

    # -------------------------------
    # 8) Save synthetic data
    # -------------------------------
    out_path = "california_housing_synth_featurewise_sb.csv"
    synth_df.to_csv(out_path, index=False)

    print("\nSaved synthetic samples to:", out_path)
    print("Head:\n", synth_df.head())


if __name__ == "__main__":
    main()