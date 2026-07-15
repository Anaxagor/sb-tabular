from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema, classify_feature_type
from sbtab.data.splits import SplitConfigHoldout
from sbtab.evaluation.metrics import Metrics
from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver
from sbtab.transforms.drop_cols import DropDataCols
from sbtab.transforms.missing import DropMissingRows
from sbtab.transforms.pipeline import TransformPipeline

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


def evaluate_ml_efficacy(
        train_real: pd.DataFrame,
        test_real: pd.DataFrame,
        train_synth: pd.DataFrame,
        target_col: str,
        task_type: str,
        cat_features: Optional[List[str]] = None,
        thread_count: int = -1
) -> Dict[str, float]:
    """Evaluates utility of synthetic data using the TSTR framework with CatBoost."""

    if target_col is None or target_col not in train_real.columns:
        if task_type == "classification":
            return {"F1_real": np.nan, "F1_synth": np.nan, "delta_F1_abs": np.nan, "delta_F1_pct": np.nan}
        return {"R2_real": np.nan, "R2_synth": np.nan, "delta_R2_abs": np.nan, "delta_R2_pct": np.nan}

    X_real, y_real = align_x_y(train_real.drop(columns=[target_col]), train_real[target_col])
    X_synth, y_synth = align_x_y(train_synth.drop(columns=[target_col]), train_synth[target_col])
    X_test, y_test = align_x_y(test_real.drop(columns=[target_col]), test_real[target_col])

    if task_type == "classification":
        y_real, y_synth, y_test = y_real.astype(str), y_synth.astype(str), y_test.astype(str)
    else:
        y_real, y_synth, y_test = y_real.astype(float), y_synth.astype(float), y_test.astype(float)

    if cat_features is None:
        cat_features = X_real.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    if cat_features:
        for df in [X_real, X_synth, X_test]:
            df[cat_features] = df[cat_features].astype(str)

    cb_params = {"random_seed": 42, "verbose": 0, "thread_count": thread_count}

    if task_type == "classification":
        model_real = CatBoostClassifier(**cb_params, cat_features=cat_features).fit(X_real, y_real)
        model_synth = CatBoostClassifier(**cb_params, cat_features=cat_features).fit(X_synth, y_synth)

        score_real = f1_score(y_test, model_real.predict(X_test), average='macro')
        score_synth = f1_score(y_test, model_synth.predict(X_test), average='macro')

        pct_diff = ((score_real - score_synth) / max(abs(score_real), 1e-9)) * 100
        return {
            "F1_real": score_real,
            "F1_synth": score_synth,
            "delta_F1_abs": score_real - score_synth,
            "delta_F1_pct": pct_diff
        }

    else:
        model_real = CatBoostRegressor(**cb_params, cat_features=cat_features).fit(X_real, y_real)
        model_synth = CatBoostRegressor(**cb_params, cat_features=cat_features).fit(X_synth, y_synth)

        score_real = r2_score(y_test, model_real.predict(X_test))
        score_synth = r2_score(y_test, model_synth.predict(X_test))

        pct_diff = ((score_real - score_synth) / max(abs(score_real), 1e-9)) * 100
        return {
            "R2_real": score_real,
            "R2_synth": score_synth,
            "delta_R2_abs": score_real - score_synth,
            "delta_R2_pct": pct_diff
        }

def align_x_y(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Drops NaNs synchronously from features and target."""
    combined = pd.concat([X, y.to_frame('__target__')], axis=1).dropna()
    return combined.drop('__target__', axis=1), combined['__target__']

def make_msbm_objective(train_num_t, train_cat_t, val_num_np, val_cat_np, cardinalities, is_ordered, seed, device,
                        ds_name):
    """Optuna objective builder for MSBM."""

    def objective(trial: optuna.trial.Trial):
        seed_everything(seed + trial.number)

        # MSBM Hyperparameters
        cat_emb_dim = trial.suggest_int("cat_emb_dim", 8, 32)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        time_dim = trial.suggest_int("time_dim", 32, 128, step=32)
        n_layers = trial.suggest_int("n_layers", 2, 6)

        steps = trial.suggest_int("num_steps", 20, 100, step=10)
        sigma = trial.suggest_float("sigma", 0.01, 1.0)
        alpha = trial.suggest_float("alpha", 0.01, 1.0)
        lambda_num = trial.suggest_float("lambda_num", 0.1, 1.0)
        lambda_cat = trial.suggest_float("lambda_cat", 0.1, 1.0)
        imf_len = trial.suggest_int("imf_len", 3, 9, step=2)
        fb_sequence = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        epochs = trial.suggest_int("epochs_per_direction", 5, 20)
        grad_clip = trial.suggest_float("grad_clip", 0.1, 1.0)
        dropout = trial.suggest_float("dropout", 0.01, 0.3)

        cfg = MixedSBMConfig(
            cat_emb_dim=cat_emb_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            n_layers=n_layers,
            dropout=dropout,
            num_steps=steps,
            sigma=sigma,
            alpha=alpha,
            lambda_num=lambda_num,
            lambda_cat=lambda_cat,
            lr=lr,
            batch_size=batch_size,
            epochs_per_direction=epochs,
            device=device,
            seed=seed,
            fb_sequence=fb_sequence,
            grad_clip=grad_clip
        )

        cont_dim = train_num_t.shape[1] if train_num_t.numel() > 0 else 0
        solver = MixedSBMSolver(continuous_dim=cont_dim, cardinalities=cardinalities, is_ordered=is_ordered, cfg=cfg)

        try:
            solver.fit(train_num_t, train_cat_t)
            gen_num_t, gen_cat_t = solver.sample(n_samples=val_num_np.shape[0] if cont_dim > 0 else val_cat_np.shape[0],
                                                 seed=seed)
        except Exception as e:
            Metrics.log_trial_error(trial, str(e), extra={"seed": seed, "dataset": ds_name, "device": device})
            raise optuna.exceptions.TrialPruned()

        finally:
            del solver
            gc.collect()
            torch.cuda.empty_cache()

        wd_loss = Metrics.compute_wasserstein_optuna(val_num_np, gen_num_t.detach().cpu().numpy())
        js_loss = Metrics.compute_jensenshannon_optuna(
            val_cat_np,
            gen_cat_t.detach().cpu().numpy(),
            cardinalities,
            trial,
            ds_name,
            device,
            seed
        )

        if np.isnan(wd_loss) or np.isnan(js_loss):
            msg = "Loss is NaN"
            Metrics.log_trial_error(trial, msg, extra={"seed": seed, "dataset": ds_name, "device": device})
            raise optuna.exceptions.TrialPruned(msg)

        trial.set_user_attr("wasserstein", wd_loss)
        trial.set_user_attr("jensen_shannon", js_loss)

        return wd_loss + js_loss

    return objective

class NumpyEncoder(json.JSONEncoder):
    """Encodes NumPy types for JSON serialization."""

    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

if __name__ == "__main__":
    seed = 5
    seed_everything(seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="../../data/datasets/datasets_continuous_only.pkl")
    ap.add_argument("--datasets", type=str, default="all")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-trials", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="msbm_optuna_results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data = pickle.load(f)
    dataset_keys = list(my_data.keys()) if args.datasets.lower() == "all" else [k.strip() for k in
                                                                                args.datasets.split(",")]

    sampler = optuna.samplers.TPESampler(seed=seed)
    for ds_name in dataset_keys:
        print(f"Dataset: {ds_name}")
        df_raw = my_data[ds_name]

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
        scaler = StandardScaler()
        if num_cols:
            train_num_np = scaler.fit_transform(train_df[num_cols])
            val_num_np = scaler.transform(val_df[num_cols])
        else:
            train_num_np, val_num_np = np.empty((len(train_df), 0)), np.empty((len(val_df), 0))

        all_cat_discrete_cols = cat_cols + discrete_cols
        if all_cat_discrete_cols:
            global_encoder = OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1)
            global_encoder.fit(train_df[all_cat_discrete_cols])

            train_cat_np = global_encoder.transform(train_df[all_cat_discrete_cols])
            val_cat_np = global_encoder.transform(val_df[all_cat_discrete_cols])

            train_df[all_cat_discrete_cols] = train_cat_np
            val_df[all_cat_discrete_cols] = val_cat_np

            cat_categories = {col: list(global_encoder.categories_[i]) for i, col in enumerate(all_cat_discrete_cols)}
            cardinalities = [len(cat_categories[col]) for col in all_cat_discrete_cols]
        else:
            cat_categories, train_cat_np, val_cat_np, cardinalities = ({},
                                                                       np.empty((len(train_df), 0), dtype=int),
                                                                       np.empty((len(val_df), 0), dtype=int),
                                                                       []
                                                                       )

        order_dict = {
            "Adult": ['education', 'education-num'],
            "Online Shoppers Purchasing Intention Dataset": ['Month'],
            "Eucalyptus": ['Utility', 'Year', 'Frosts', 'Rainfall', 'Altitude', 'Latitude'],
            "Forest Fires": ['X', 'Y', 'month', 'day'],
            "Insurance": ['children'],
            "House Sales": ['bedrooms', 'bathrooms', 'floors', 'view'],
            "Cardiovascular Disease": ['Cholesterol', 'Glucose'],
            "Churn Modelling": ['Tenure', 'NumOfProducts'],
            "Auto MPG": ['cylinders', 'model_year'],
            "Diamonds": ['cut', 'color', 'clarity'],
            "Real Estate": ['X4 number of convenience stores'],
            "Mushroom": ['ring-number', 'gill-spacing', 'gill-size'],
            "Car Evaluation": ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
            "Student Perf": ['age', 'failures', 'absences', 'G1', 'G2', 'G3', 'Medu', 'Fedu',
                             'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'],
            "Lymphography": ['lym_nodes_enlar', 'no_of_nodes_in', 'lym_nodes_dimin'],
            "Breast cancer": ['age', 'tumor_size', 'inv_nodes', 'deg-malig']
        }
        ordered_cols_ds = order_dict.get(ds_name, [])
        order_mask = torch.tensor([c in ordered_cols_ds for c in cat_cols] + [True] * len(discrete_cols),
                                  dtype=torch.bool)

        train_num_t = torch.tensor(train_num_np, dtype=torch.float32, device=args.device)
        train_cat_t = torch.tensor(train_cat_np, dtype=torch.long, device=args.device)

        study = optuna.create_study(study_name=f"msbm_study_{ds_name}", storage=f"sqlite:///{outdir}/msbm_optuna.db",
                                    load_if_exists=True, direction="minimize", sampler=sampler)

        study.optimize(
            make_msbm_objective(train_num_t, train_cat_t, val_num_np, val_cat_np, cardinalities, order_mask, seed,
                                args.device, ds_name), n_trials=args.n_trials, gc_after_trial=True,
            show_progress_bar=True)

        all_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        all_trials.sort(key=lambda t: t.value)

        successful_generation = False
        for trial_idx, trial in enumerate(all_trials):
            # covertype in continuous data is very unstable so we choose suboptimal trial
            if ds_name == 'covertype' and trial_idx < 2:
                continue
            bp, best_trial = trial.params, trial
            print(f"\nTrying trial ranking {trial_idx + 1} (Trial {best_trial.number}) with value: {best_trial.value}")

            final_cfg = MixedSBMConfig(
                cat_emb_dim=bp["cat_emb_dim"], hidden_dim=bp["hidden_dim"], time_dim=bp["time_dim"],
                n_layers=bp["n_layers"], num_steps=bp["num_steps"], sigma=bp["sigma"], alpha=bp["alpha"],
                grad_clip=bp["grad_clip"], fb_sequence=tuple("b" if i % 2 == 0 else "f" for i in range(bp["imf_len"])),
                dropout=bp["dropout"], lambda_num=bp["lambda_num"], lambda_cat=bp["lambda_cat"],
                lr=bp["lr"], batch_size=bp["batch_size"], epochs_per_direction=bp["epochs_per_direction"],
                device=args.device, seed=seed + best_trial.number
            )

            final_solver = MixedSBMSolver(continuous_dim=train_num_t.shape[1], cardinalities=cardinalities,
                                          is_ordered=order_mask, cfg=final_cfg)
            start_time = time.time()
            final_solver.fit(train_num_t, train_cat_t)
            elapsed_sec = time.time() - start_time

            gen_num_val, gen_cat_val = final_solver.sample(n_samples=len(val_df), seed=seed + 1)
            gen_num_train, gen_cat_train = final_solver.sample(n_samples=len(train_df), seed=seed + 2)

            has_nans = gen_num_val.isnan().any() or gen_cat_val.isnan().any() or \
                       gen_num_train.isnan().any() or gen_cat_train.isnan().any()

            if not has_nans:
                print(f"-> Success! Generated data contains no NaNs.")
                successful_generation = True
                break
            else:
                print(f"-> Warning: Trial {trial.number} produced NaNs. Skipping to next best trial")

        if not successful_generation:
            raise RuntimeError(
                "All completed Optuna trials produced NaN values during generation. Check your model stability/hyperparameters.")

        def tensors_to_dataframe(g_num, g_cat, n_samples):
            """Inverse transforms tensors back into a Pandas DataFrame."""
            s_df = pd.DataFrame(index=range(n_samples))
            if num_cols and g_num is not None:
                s_num = scaler.inverse_transform(g_num.cpu().numpy())
                for i, col in enumerate(num_cols): s_df[col] = s_num[:, i]
            if all_cat_discrete_cols and g_cat is not None:
                g_cat_np = g_cat.cpu().numpy()
                for i, col in enumerate(all_cat_discrete_cols):
                    cats = cat_categories[col]
                    col_data = g_cat_np[:, i].astype(int)
                    out_range = (col_data < 0) | (col_data >= len(cats))

                    is_numeric = len(cats) > 0 and isinstance(cats[0], (int, float, np.integer, np.floating))
                    ext_cats = np.array(cats + ([-999] if is_numeric else ["UNKNOWN_GEN"]),
                                        dtype=object if not is_numeric else None)

                    col_data[out_range] = len(cats)
                    s_df[col] = ext_cats[col_data]
                    if is_numeric: s_df[col] = s_df[col].astype(type(cats[0]))
            return s_df[train_df.columns]


        ml_eff = evaluate_ml_efficacy(
            train_real=true_train_eval_df, test_real=true_val_eval_df,
            train_synth=tensors_to_dataframe(gen_num_train, gen_cat_train, len(train_df)),
            target_col=target_col, task_type=task_type
        )

        discrete_present = [c for c in discrete_cols if c in true_val_eval_df.columns and c not in bad_cols]
        pure_cat_cols = [c for c in cat_cols if c in true_val_eval_df.columns and c not in bad_cols]
        num_present = [c for c in num_cols if c in true_val_eval_df.columns and c not in bad_cols]
        synth_val_df = tensors_to_dataframe(gen_num_val, gen_cat_val, len(val_df))

        results = {
            "dataset": ds_name,
            "best_trial": best_trial.number,
            "fit_elapsed_sec": elapsed_sec,
            "best_params": bp,
            "Best_Tuning_Loss_Combined": best_trial.value,
            "Best_Tuning_WD_Scaled": best_trial.user_attrs.get("wasserstein"),
            "Best_Tuning_JS_Cat": best_trial.user_attrs.get("jensen_shannon"),
            "Continuous_Mean_KL_50bins": Metrics.compute_kl_histogram_continuous(true_val_eval_df, synth_val_df, num_present,
                                                                         bins=50) if num_present else None,
            "Continuous_Corr_Pearson": Metrics.compute_corr_distance_for_columns(true_val_eval_df, synth_val_df, num_present,
                                                                         method='pearson') if num_present else None,
            "Discrete_Mean_KL": Metrics.average_kl_discrete(true_val_eval_df, synth_val_df,
                                                    discrete_present) if discrete_present else None,
            "Discrete_Corr_Spearman": Metrics.compute_corr_distance_for_columns(true_val_eval_df, synth_val_df,
                                                                        discrete_present,
                                                                        method='spearman') if discrete_present else None,
            "Categorical_Mean_KL": Metrics.average_kl_discrete(true_val_eval_df, synth_val_df,
                                                       pure_cat_cols) if pure_cat_cols else None,
            "Categorical_NMI": Metrics.compute_nmi_distance_matrix(true_val_eval_df, synth_val_df,
                                                           pure_cat_cols) if pure_cat_cols else None,
            "MMD": Metrics.compute_mmd_numpy(val_num_np, gen_num_val.cpu().numpy(), seed=seed) if num_cols else None,
            **ml_eff
        }

        print(json.dumps(results, indent=4, cls=NumpyEncoder))
        (outdir / f"{ds_name}_final_metrics.json").write_text(json.dumps(results, indent=4, cls=NumpyEncoder))
