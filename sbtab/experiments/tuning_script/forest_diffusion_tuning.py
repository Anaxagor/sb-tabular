from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema, classify_feature_type
from sbtab.data.splits import SplitConfigHoldout
from sbtab.evaluation.metrics import Metrics
from sbtab.experiments.tuning_script.dataset_registry import DATASETS_PATH, DATASET_KEYS, KEY_NORMALIZER
from sbtab.transforms.drop_cols import DropDataCols
from sbtab.transforms.missing import DropMissingRows
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.ForestDiffusion import ForestDiffusionModel


def seed_everything(seed: int) -> None:
    """Fixes all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_forestdiff_objective(train_num_np, train_cat_np, val_num_np, val_cat_np,
                              cardinalities, cat_indexes, int_indexes, seed, device, ds_name):
    cont_dim = train_num_np.shape[1] if train_num_np.size > 0 else 0

    def objective(trial: optuna.trial.Trial):
        seed_everything(seed + trial.number)

        n_rows = train_num_np.shape[0] + train_cat_np.shape[0]
        max_materialised = 4_000_000
        max_product = max(1, max_materialised // max(1, n_rows))

        # --- Diffusion parameters ---
        diffusion_type = trial.suggest_categorical("diffusion_type", ["flow", "vp"])
        n_t = trial.suggest_int("n_t", 10, min(50, max(10, max_product)))
        duplicate_K = trial.suggest_int("duplicate_K", 1, max(1, max_product // n_t))
        n_batch = trial.suggest_int("n_batch", 1024, 8192, step=1024)

        # --- XGBoost parameters ---
        max_depth = trial.suggest_int("max_depth", 4, 8)
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 200])
        eta = trial.suggest_float("eta", 0.1, 0.3, step=0.1)

        X_train = np.hstack([train_num_np, train_cat_np]) if cont_dim > 0 else train_cat_np

        model = None

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            model = ForestDiffusionModel(
                X=X_train, model='xgboost', diffusion_type=diffusion_type,
                n_t=n_t, duplicate_K=duplicate_K,
                max_depth=max_depth, n_estimators=n_estimators,
                eta=eta, reg_lambda=0.0, reg_alpha=0.0,
                cat_indexes=cat_indexes, int_indexes=int_indexes, seed=seed + trial.number,
                n_jobs=1 if device == "cuda" else -1,
                gpu_hist=(device == "cuda"),
                n_batch=n_batch
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fit_time_s = time.perf_counter() - t_start

            n_val = val_num_np.shape[0] if val_num_np.shape[1] > 0 else val_cat_np.shape[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_eval_start = time.perf_counter()

            synth_X = model.generate(batch_size=n_val)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            eval_time_s = time.perf_counter() - t_eval_start

            trial.set_user_attr("fit_time_s", fit_time_s)
            trial.set_user_attr("eval_time_s", eval_time_s)
            trial.set_user_attr("total_time_s", fit_time_s + eval_time_s)

        except Exception as e:
            Metrics.log_trial_error(trial, str(e),
                                    extra={"seed": seed + trial.number, "dataset": ds_name, "device": device})
            raise optuna.exceptions.TrialPruned()
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        num_dim = val_num_np.shape[1]
        gen_num = synth_X[:, :num_dim] if num_dim > 0 else np.empty((synth_X.shape[0], 0))
        gen_cat = synth_X[:, num_dim:].astype(int) if train_cat_np.shape[1] > 0 else np.empty((synth_X.shape[0], 0),
                                                                                              dtype=int)

        wd_loss = Metrics.compute_wasserstein_optuna(val_num_np, gen_num) if num_dim > 0 else 0.0
        js_loss = Metrics.compute_jensenshannon_optuna(
            val_cat_np, gen_cat, cardinalities, trial, ds_name, "cpu", seed
        ) if train_cat_np.shape[1] > 0 else 0.0

        if np.isnan(wd_loss) or np.isnan(js_loss):
            Metrics.log_trial_error(trial, "Loss is NaN",
                                    extra={"seed": seed + trial.number, "dataset": ds_name, "device": device})
            raise optuna.exceptions.TrialPruned("Loss is NaN")

        trial.set_user_attr("wasserstein", wd_loss)
        trial.set_user_attr("jensen_shannon", js_loss)

        return wd_loss + js_loss

    return objective


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    seed = 5
    seed_everything(seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default=str(DATASETS_PATH))
    ap.add_argument("--datasets", type=str, default="all")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--outdir", type=str, default="forestdiff_optuna_results_after_review")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data = pickle.load(f)

    my_data_norm = {KEY_NORMALIZER(k): v for k, v in my_data.items()}

    dataset_keys = DATASET_KEYS if args.datasets.lower() == "all" else \
        [k.strip() for k in args.datasets.split(",")]

    sampler = optuna.samplers.TPESampler(seed=seed)
    for ds_name in dataset_keys:
        ds_key = KEY_NORMALIZER(ds_name)
        print(f"Dataset: {ds_name}")

        if ds_key not in my_data_norm:
            print(f"  Skipping {ds_name}: dataset not found in pickle")
            continue

        df_raw = my_data_norm[ds_key]

        target_col = df_raw.attrs.get('target_variable')
        task_type = df_raw.attrs.get('task_type', 'classification')

        schema = TabularSchema.infer_from_dataframe(df_raw, target_col=target_col)
        dm = TabularDataModule(df=df_raw, schema=schema,
                               transforms=TransformPipeline(transforms=[DropMissingRows(), DropDataCols()]))
        dm.prepare_holdout(SplitConfigHoldout(val_size=args.test_size, shuffle=True, random_state=seed))

        train_df = dm.get_holdout().train.copy()
        val_df = dm.get_holdout().val.copy()

        cat_cols = list(schema.categorical_cols)
        discrete_cols = list(schema.discrete_cols)
        num_cols = list(schema.continuous_cols)

        if target_col:
            target_col_type = classify_feature_type(train_df[target_col])
            if target_col_type == "continuous":
                num_cols.append(target_col)
            elif target_col_type == "discrete":
                discrete_cols.append(target_col)
            elif target_col_type == "categorical":
                cat_cols.append(target_col)
            else:
                raise ValueError("Target column type not recognized")

        bad_cols = [c for c in train_df.columns if train_df[c].nunique() <= 1]
        if bad_cols:
            train_df = train_df.drop(columns=bad_cols)
            val_df = val_df.drop(columns=bad_cols)
            cat_cols = [c for c in cat_cols if c not in bad_cols]
            discrete_cols = [c for c in discrete_cols if c not in bad_cols]
            num_cols = [c for c in num_cols if c not in bad_cols]

        true_train_eval_df, true_val_eval_df = train_df.copy(), val_df.copy()

        num_disc_cols = num_cols + discrete_cols

        scaler = StandardScaler()
        if num_disc_cols:
            train_num_np = scaler.fit_transform(train_df[num_disc_cols])
            val_num_np = scaler.transform(val_df[num_disc_cols])
        else:
            train_num_np, val_num_np = np.empty((len(train_df), 0)), np.empty((len(val_df), 0))

        if cat_cols:
            global_encoder = OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1)
            global_encoder.fit(train_df[cat_cols])

            train_cat_np = global_encoder.transform(train_df[cat_cols])
            val_cat_np = global_encoder.transform(val_df[cat_cols])

            train_cat_np = np.clip(train_cat_np, 0, None)  # Remove -1 for train
            val_cat_np = np.clip(val_cat_np, 0, None)  # Remove -1 for val metrics

            cat_categories = {col: list(global_encoder.categories_[i]) for i, col in enumerate(cat_cols)}
            cardinalities = [len(cat_categories[col]) for col in cat_cols]
        else:
            cat_categories, train_cat_np, val_cat_np, cardinalities = {}, np.empty((len(train_df), 0),
                                                                                   dtype=int), np.empty(
                (len(val_df), 0), dtype=int), []

        num_dim = train_num_np.shape[1]
        cat_indexes = list(range(num_dim, num_dim + train_cat_np.shape[1])) if cat_cols else []
        int_indexes = [num_cols.index(c) for c in discrete_cols if c in num_cols] if discrete_cols else []

        study = optuna.create_study(study_name=f"forestdiff_study_{ds_key}",
                                    storage=f"sqlite:///{outdir}/forestdiff_optuna.db",
                                    load_if_exists=True, direction="minimize", sampler=sampler)

        study.optimize(
            make_forestdiff_objective(train_num_np, train_cat_np, val_num_np, val_cat_np, cardinalities,
                                      cat_indexes, int_indexes, seed, args.device, ds_key),
            n_trials=args.n_trials, gc_after_trial=True, show_progress_bar=True
        )

        all_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        all_trials.sort(key=lambda t: t.value)

        successful_generation = False
        for trial_idx, trial in enumerate(all_trials):
            bp, best_trial = trial.params, trial
            print(f"\nTrying trial ranking {trial_idx + 1} (Trial {best_trial.number}) with value: {best_trial.value}")

            X_train_full = np.hstack([train_num_np, train_cat_np]) if train_num_np.shape[1] > 0 else train_cat_np

            use_gpu = (args.device == "cuda")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            final_model = ForestDiffusionModel(
                X=X_train_full, model='xgboost', diffusion_type=bp["diffusion_type"], n_t=bp["n_t"],
                duplicate_K=bp["duplicate_K"], max_depth=bp["max_depth"], n_estimators=bp["n_estimators"],
                eta=bp["eta"], reg_lambda=0.0, reg_alpha=0.0, cat_indexes=cat_indexes, int_indexes=int_indexes,
                seed=seed + best_trial.number,
                n_jobs=1 if use_gpu else -1,
                gpu_hist=use_gpu,
                n_batch=bp["n_batch"]
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_sec = time.perf_counter() - start_time

            synth_X_val = final_model.generate(batch_size=len(val_df))
            synth_X_train = final_model.generate(batch_size=len(train_df))

            num_dim = train_num_np.shape[1]
            gen_num_val = synth_X_val[:, :num_dim] if num_dim > 0 else np.empty((synth_X_val.shape[0], 0))
            gen_cat_val = synth_X_val[:, num_dim:].astype(int) if train_cat_np.shape[1] > 0 else np.empty(
                (synth_X_val.shape[0], 0), dtype=int)

            gen_num_train = synth_X_train[:, :num_dim] if num_dim > 0 else np.empty((synth_X_train.shape[0], 0))
            gen_cat_train = synth_X_train[:, num_dim:].astype(int) if train_cat_np.shape[1] > 0 else np.empty(
                (synth_X_train.shape[0], 0), dtype=int)

            has_nans = np.isnan(gen_num_val).any() or np.isnan(gen_cat_val).any() or \
                       np.isnan(gen_num_train).any() or np.isnan(gen_cat_train).any()

            if not has_nans:
                print("-> Success! Generated data contains no NaNs.")
                successful_generation = True
                break
            print(f"-> Warning: Trial {trial.number} produced NaNs. Skipping to next best trial")

            del final_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not successful_generation:
            raise RuntimeError("All completed Optuna trials produced NaN values during generation.")


        def tensors_to_dataframe(g_num, g_cat, n_samples):
            """Inverse transforms tensors back into a Pandas DataFrame."""
            s_df = pd.DataFrame(index=range(n_samples))
            if num_disc_cols and g_num is not None:
                s_num = scaler.inverse_transform(g_num)
                for i, col in enumerate(num_disc_cols):
                    s_df[col] = s_num[:, i]
                for col in discrete_cols:
                    if col in s_df.columns:
                        s_df[col] = s_df[col].round().astype(int)
            if cat_cols and g_cat is not None:
                g_cat_np = g_cat
                for i, col in enumerate(cat_cols):
                    cats = cat_categories[col]
                    col_data = g_cat_np[:, i].astype(int)
                    out_range = (col_data < 0) | (col_data >= len(cats))

                    is_numeric = len(cats) > 0 and isinstance(cats[0], (int, float, np.integer, np.floating))
                    ext_cats = np.array(cats + ([-999] if is_numeric else ["UNKNOWN_GEN"]),
                                        dtype=object if not is_numeric else None)

                    col_data[out_range] = len(cats)
                    s_df[col] = ext_cats[col_data]
                    if is_numeric:
                        s_df[col] = s_df[col].astype(type(cats[0]))
            return s_df[train_df.columns]


        ml_eff = Metrics.evaluate_ml_efficacy(
            train_real=true_train_eval_df, test_real=true_val_eval_df,
            train_synth=tensors_to_dataframe(gen_num_train, gen_cat_train, len(train_df)),
            target_col=target_col, task_type=task_type
        )

        discrete_present = [c for c in discrete_cols if c in true_val_eval_df.columns and c not in bad_cols]
        pure_cat_cols = [c for c in cat_cols if c in true_val_eval_df.columns and c not in bad_cols]
        num_present = [c for c in num_cols if c in true_val_eval_df.columns and c not in bad_cols]

        synth_val_df = tensors_to_dataframe(gen_num_val, gen_cat_val, len(val_df))
        all_cat_discrete_eval = [c for c in (discrete_present + pure_cat_cols) if c in true_val_eval_df.columns]

        results = {
            "dataset": ds_key,
            "best_trial": best_trial.number,
            "fit_elapsed_sec": elapsed_sec,
            "best_params": bp,
            "Best_Tuning_Loss_Combined": best_trial.value,
            "Best_Tuning_WD_Scaled": best_trial.user_attrs.get("wasserstein"),
            "Best_Tuning_JS_Cat": best_trial.user_attrs.get("jensen_shannon"),
            "Continuous_Mean_KL_50bins": Metrics.compute_kl_histogram_continuous(
                true_val_eval_df, synth_val_df, num_present, bins=50) if num_present else None,
            "Continuous_Corr_Pearson": Metrics.compute_corr_distance_for_columns(
                true_val_eval_df, synth_val_df, num_present, method='pearson') if num_present else None,
            "Discrete_Mean_KL": Metrics.average_kl_discrete(
                true_val_eval_df, synth_val_df, discrete_present) if discrete_present else None,
            "Discrete_Corr_Spearman": Metrics.compute_corr_distance_for_columns(
                true_val_eval_df, synth_val_df, discrete_present, method='spearman') if discrete_present else None,
            "Categorical_Mean_KL": Metrics.average_kl_discrete(
                true_val_eval_df, synth_val_df, pure_cat_cols) if pure_cat_cols else None,
            "Discrete_Cat_Mean_KL": Metrics.average_kl_discrete(
                true_val_eval_df, synth_val_df, all_cat_discrete_eval) if all_cat_discrete_eval else None,
            "Categorical_NMI": Metrics.compute_nmi_distance_matrix(
                true_val_eval_df, synth_val_df, pure_cat_cols) if pure_cat_cols else None,
            "MMD": Metrics.compute_mmd_numpy(val_num_np, gen_num_val, seed=seed) if num_cols else None,
            **ml_eff
        }

        print(json.dumps(results, indent=4, cls=NumpyEncoder))
        (outdir / f"{ds_key}_final_metrics.json").write_text(json.dumps(results, indent=4, cls=NumpyEncoder))

        with open(outdir / f"{ds_key}_train.pkl", "wb") as f:
            pickle.dump(true_train_eval_df, f)
        with open(outdir / f"{ds_key}.test.pkl", "wb") as f:
            pickle.dump(true_val_eval_df, f)

        del final_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()