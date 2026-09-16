from __future__ import annotations

import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, f1_score
from scipy.stats import wasserstein_distance
from catboost import CatBoostRegressor, CatBoostClassifier

from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema, classify_feature_type
from sbtab.evaluation.metrics import Metrics
from sbtab.transforms.drop_cols import DropDataCols
from sbtab.transforms.missing import DropMissingRows
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.ForestDiffusion import ForestDiffusionModel
from sbtab.experiments.tuning_script.dataset_registry import DATASETS_PATH, DATASET_KEYS, KEY_NORMALIZER


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def align_x_y(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    combined = pd.concat([X, y.to_frame('__target__')], axis=1).dropna()
    return combined.drop('__target__', axis=1), combined['__target__']


def tstr_evaluate(train_real: pd.DataFrame, test_real: pd.DataFrame, train_synth: pd.DataFrame,
                  target_col: str, task_type: str, all_cat_discrete: list, fold_seed: int) -> dict:
    if target_col is None or target_col not in train_real.columns:
        if task_type == "classification":
            return {'tstr_f1_real': np.nan, 'tstr_f1_synth': np.nan, 'tstr_f1_deviation_%': np.nan,
                    'tstr_f1_diff_raw': np.nan}
        return {'tstr_r2_real': np.nan, 'tstr_r2_synth': np.nan, 'tstr_r2_deviation_%': np.nan,
                'tstr_r2_diff_raw': np.nan}

    X_train_real, y_train_real = align_x_y(train_real.drop(columns=[target_col]), train_real[target_col])
    X_train_synth, y_train_synth = align_x_y(train_synth.drop(columns=[target_col]), train_synth[target_col])
    X_test, y_test = align_x_y(test_real.drop(columns=[target_col]), test_real[target_col])

    if task_type == "classification":
        y_train_real, y_train_synth, y_test = y_train_real.astype(str), y_train_synth.astype(str), y_test.astype(str)
    else:
        y_train_real, y_train_synth, y_test = y_train_real.astype(float), y_train_synth.astype(float), y_test.astype(
            float)

    cat_features = [c for c in all_cat_discrete if c != target_col and c in X_train_real.columns]
    if cat_features:
        for df in [X_train_real, X_train_synth, X_test]:
            df[cat_features] = df[cat_features].dropna().astype(str)

    cb_params = {"random_seed": fold_seed, "verbose": 0, "thread_count": -1}
    metrics = {}

    if task_type == 'classification':
        model_real = CatBoostClassifier(**cb_params, cat_features=cat_features).fit(X_train_real, y_train_real)
        model_synth = CatBoostClassifier(**cb_params, cat_features=cat_features).fit(X_train_synth, y_train_synth)
        f1_real = f1_score(y_test, model_real.predict(X_test), average='macro')
        f1_synth = f1_score(y_test, model_synth.predict(X_test), average='macro')
        dev = ((f1_real - f1_synth) / max(abs(f1_real), 1e-9)) * 100
        metrics.update({'tstr_f1_real': f1_real, 'tstr_f1_synth': f1_synth, 'tstr_f1_deviation_%': dev,
                        'tstr_f1_diff_raw': f1_real - f1_synth})
    else:
        model_real = CatBoostRegressor(**cb_params, cat_features=cat_features).fit(X_train_real, y_train_real)
        model_synth = CatBoostRegressor(**cb_params, cat_features=cat_features).fit(X_train_synth, y_train_synth)
        r2_real = r2_score(y_test, model_real.predict(X_test))
        r2_synth = r2_score(y_test, model_synth.predict(X_test))
        r2_dev = ((r2_real - r2_synth) / max(abs(r2_real), 1e-9)) * 100
        metrics.update({'tstr_r2_real': r2_real, 'tstr_r2_synth': r2_synth, 'tstr_r2_deviation_%': r2_dev,
                        'tstr_r2_diff_raw': r2_real - r2_synth})
    return metrics


def compute_metrics(test_real_df, synth_df, test_num_scaled, synth_num_scaled, clean_num, clean_disc, clean_cat):
    metrics = {}
    all_discrete_cat = clean_disc + clean_cat
    if clean_num:
        wds = [wasserstein_distance(test_num_scaled[:, i], synth_num_scaled[:, i]) for i in
               range(synth_num_scaled.shape[1])]
        metrics['Continuous_Mean_WD'] = np.mean(wds)
        metrics['Continuous_Mean_KL_50bins'] = Metrics.compute_kl_histogram_continuous(test_real_df, synth_df,
                                                                                       clean_num, bins=50)
        metrics['Continuous_Corr_Pearson'] = Metrics.compute_corr_distance_for_columns(test_real_df, synth_df,
                                                                                       clean_num, method='pearson')
        metrics['mmd'] = Metrics.compute_mmd_numpy(test_num_scaled, synth_num_scaled)
    else:
        metrics['Continuous_Mean_WD'] = metrics['Continuous_Mean_KL_50bins'] = metrics['Continuous_Corr_Pearson'] = \
        metrics['mmd'] = np.nan

    if all_discrete_cat:
        metrics['Discrete_Cat_Mean_KL'] = Metrics.average_kl_discrete(test_real_df, synth_df, all_discrete_cat)
    else:
        metrics['Discrete_Cat_Mean_KL'] = np.nan

    if clean_disc:
        metrics['Discrete_Mean_KL'] = Metrics.average_kl_discrete(test_real_df, synth_df, clean_disc)
        metrics['Discrete_Corr_Spearman'] = Metrics.compute_corr_distance_for_columns(test_real_df, synth_df,
                                                                                      clean_disc, method='spearman')
    else:
        metrics['Discrete_Mean_KL'] = metrics['Discrete_Corr_Spearman'] = np.nan

    if clean_cat:
        metrics['Categorical_Mean_KL'] = Metrics.average_kl_discrete(test_real_df, synth_df, clean_cat)
        metrics['Categorical_NMI'] = Metrics.compute_nmi_distance_matrix(test_real_df, synth_df, clean_cat)
    else:
        metrics['Categorical_Mean_KL'] = metrics['Categorical_NMI'] = np.nan
    return metrics


def main():
    seed = 42
    seed_everything(seed)
    use_gpu = torch.cuda.is_available()

    results_dir = Path("../experiments/tuning_script/forestdiff_optuna_results")
    output_csv = "cv_forestdiff_results.csv"

    with open(DATASETS_PATH, "rb") as f:
        all_data_raw = pickle.load(f)

    all_data = {KEY_NORMALIZER(k): v for k, v in all_data_raw.items()}

    all_cv_results = []

    for ds_name in DATASET_KEYS:
        ds_key = KEY_NORMALIZER(ds_name)
        print(f"\nEvaluating ForestDiffusion on {ds_name}")

        if ds_key not in all_data:
            print(f"  Skipping {ds_name}: dataset not found in pickle")
            continue

        df_raw = all_data[ds_key]

        target_col = df_raw.attrs.get('target_variable')
        task_type = df_raw.attrs.get('task_type', 'classification')

        param_file = results_dir / f"{ds_key}_final_metrics.json"
        train_pkl = results_dir / f"{ds_key}_train.pkl"

        if not param_file.exists() or not train_pkl.exists():
            print(f"  Skipping {ds_name}: required files not found")
            continue

        with open(param_file) as f:
            tuning_info = json.load(f)
        best_params = tuning_info["best_params"]

        with open(train_pkl, "rb") as f:
            train_df = pickle.load(f)

        schema = TabularSchema.infer_from_dataframe(train_df, target_col=target_col)
        dm = TabularDataModule(df=train_df, schema=schema,
                               transforms=TransformPipeline(transforms=[DropMissingRows(), DropDataCols()]))
        clean_df = dm.get_clean_df().reset_index(drop=True)

        pure_cat_cols = list(schema.categorical_cols)
        discrete_cols = list(schema.discrete_cols)
        num_cols = list(schema.continuous_cols)

        if target_col:
            target_col_type = classify_feature_type(df_raw[target_col])
            if target_col_type == "continuous":
                num_cols.append(target_col)
            elif target_col_type == "discrete":
                discrete_cols.append(target_col)
            elif target_col_type == "categorical":
                pure_cat_cols.append(target_col)

        bad_cols = [c for c in clean_df.columns if clean_df[c].nunique() <= 1]
        if bad_cols:
            clean_df = clean_df.drop(columns=bad_cols)
            pure_cat_cols = [c for c in pure_cat_cols if c not in bad_cols]
            discrete_cols = [c for c in discrete_cols if c not in bad_cols]
            num_cols = [c for c in num_cols if c not in bad_cols]


        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        y_stratify = clean_df[target_col].values if target_col and task_type == "classification" else None

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(clean_df, y_stratify)):
            print(f"  Fold {fold_id + 1}/5")
            seed_everything(seed)

            train_fold = clean_df.iloc[train_idx].copy().reset_index(drop=True)
            test_fold = clean_df.iloc[test_idx].copy().reset_index(drop=True)

            clean_num = [c for c in num_cols if c in train_fold.columns and train_fold[c].std() > 1e-5]
            clean_cat = [c for c in pure_cat_cols if c in train_fold.columns and train_fold[c].nunique() > 1]
            clean_disc = [c for c in discrete_cols if c in train_fold.columns and train_fold[c].nunique() > 1]
            all_cat_discrete = clean_cat + clean_disc

            train_fold = train_fold.dropna(subset=clean_num + all_cat_discrete).reset_index(drop=True)
            test_fold = test_fold.dropna(subset=clean_num + all_cat_discrete).reset_index(drop=True)

            scaler = StandardScaler()
            if clean_num:
                train_num_scaled = scaler.fit_transform(train_fold[clean_num].to_numpy())
                test_num_scaled = scaler.transform(test_fold[clean_num].to_numpy())
            else:
                train_num_scaled = np.empty((len(train_fold), 0))
                test_num_scaled = np.empty((len(test_fold), 0))

            if clean_cat:
                fold_encoder = OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1)
                train_cat = fold_encoder.fit_transform(train_fold[clean_cat])
                train_cat = np.clip(train_cat, 0, None)  # Убираем -1 для train
                cat_categories = {col: list(fold_encoder.categories_[i]) for i, col in enumerate(clean_cat)}
            else:
                train_cat = np.empty((len(train_fold), 0), dtype=int)
                cat_categories = {}

            train_disc = train_fold[clean_disc].to_numpy(dtype=np.float64) if clean_disc else np.empty(
                (len(train_fold), 0))

            X_train = np.hstack([train_num_scaled, train_disc, train_cat]) if clean_num else (
                np.hstack([train_disc, train_cat]) if clean_disc else train_cat)

            num_dim = train_num_scaled.shape[1]
            disc_dim = train_disc.shape[1]
            cat_indexes = list(range(num_dim + disc_dim, X_train.shape[1])) if clean_cat else []
            int_indexes = list(range(num_dim, num_dim + disc_dim)) if clean_disc else []

            model = ForestDiffusionModel(
                X=X_train, model='xgboost', diffusion_type=best_params["diffusion_type"],
                n_t=best_params["n_t"], duplicate_K=best_params["duplicate_K"],
                max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                eta=best_params["eta"], reg_lambda=0.0, reg_alpha=0.0,
                cat_indexes=cat_indexes, int_indexes=int_indexes, seed=seed + fold_id,
                n_jobs=1 if use_gpu else -1,
                gpu_hist=use_gpu
            )

            synth_X = model.generate(batch_size=len(train_fold))
            num_dim = train_num_scaled.shape[1]
            disc_dim = train_disc.shape[1]

            synth_num_scaled = synth_X[:, :num_dim] if num_dim > 0 else np.empty((synth_X.shape[0], 0))
            synth_disc = synth_X[:, num_dim:num_dim + disc_dim] if disc_dim > 0 else np.empty((synth_X.shape[0], 0))
            synth_cat = synth_X[:, num_dim + disc_dim:].astype(int) if clean_cat else np.empty((synth_X.shape[0], 0),
                                                                                               dtype=int)

            synth_df = pd.DataFrame(index=range(len(train_fold)))
            if clean_num:
                synth_num_orig = scaler.inverse_transform(synth_num_scaled)
                for i, col in enumerate(clean_num): synth_df[col] = synth_num_orig[:, i]
            else:
                synth_num_orig = np.empty((len(train_fold), 0))

            if clean_disc:
                for i, col in enumerate(clean_disc): synth_df[col] = synth_disc[:, i].round().astype(int)

            if clean_cat:
                synth_cat_orig = []
                for i, col in enumerate(clean_cat):
                    cats = cat_categories[col]
                    col_data = synth_cat[:, i]
                    out_range = (col_data < 0) | (col_data >= len(cats))
                    is_numeric = len(cats) > 0 and isinstance(cats[0], (int, float, np.integer, np.floating))
                    ext_cats = np.array(cats + ([-999] if is_numeric else ["UNKNOWN_GEN"]),
                                        dtype=object if not is_numeric else None)
                    col_data[out_range] = len(cats)
                    synth_df[col] = ext_cats[col_data]
                    synth_cat_orig.append(ext_cats[col_data])
                synth_cat_orig = np.array(synth_cat_orig).T if len(synth_cat_orig) > 0 else np.empty(
                    (len(train_fold), 0), dtype=object)
            else:
                synth_cat_orig = np.empty((len(train_fold), 0), dtype=object)

            train_real_final = pd.concat([train_fold[clean_num], train_fold[all_cat_discrete]], axis=1).reset_index(
                drop=True)
            test_real_final = pd.concat([test_fold[clean_num], test_fold[all_cat_discrete]], axis=1).reset_index(
                drop=True)
            synth_df = synth_df[train_real_final.columns]

            fold_metrics = compute_metrics(test_real_final, synth_df, test_num_scaled, synth_num_scaled, clean_num,
                                           clean_disc, clean_cat)
            tstr_metrics = tstr_evaluate(train_real_final, test_real_final, synth_df, target_col, task_type,
                                         all_cat_discrete, seed + fold_id)

            all_metrics = {**fold_metrics, **tstr_metrics, "fold": fold_id, "dataset": ds_name, "model": "forestdiff"}
            all_cv_results.append(all_metrics)

    if all_cv_results:
        final_df = pd.DataFrame(all_cv_results)
        final_df.to_csv(output_csv, index=False)
        print(f"\nSaved CV results to {output_csv}")


if __name__ == "__main__":
    main()