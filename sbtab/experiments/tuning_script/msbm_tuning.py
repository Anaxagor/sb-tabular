from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import optuna
import torch
from numpy import floating

from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, f1_score
from sklearn.preprocessing import StandardScaler

from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema
from sbtab.data.splits import SplitConfigHoldout
from sbtab.solvers.msbm import MixedSBMSolver, MixedSBMConfig
from sbtab.transforms.missing import DropMissingRows
from sbtab.transforms.pipeline import TransformPipeline


def get_distributions(real: pd.Series, synth: pd.Series):
    """Aligns categories and returns probability distributions."""
    cats = list(set(real.dropna().unique()) | set(synth.dropna().unique()))
    p = real.value_counts(normalize=True).reindex(cats, fill_value=1e-9).values
    q = synth.value_counts(normalize=True).reindex(cats, fill_value=1e-9).values
    return p, q

def average_kl(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    if not cols: return 0.0
    kls = []
    for c in cols:
        p, q = get_distributions(real[c], synth[c])
        kls.append(float(entropy(p, q)))
    return float(np.mean(kls))

def average_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Average 1D Wasserstein distance across columns."""
    wds = []
    for c in cols:
        wds.append(float(wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy())))
    return float(np.mean(wds))

def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    corr_real = real.corr(numeric_only=True).fillna(0).to_numpy()
    corr_synth = synth.corr(numeric_only=True).fillna(0).to_numpy()
    return float(np.linalg.norm(corr_real - corr_synth, ord='fro'))

def evaluate_ml_efficacy(train_real: pd.DataFrame,
                         test_real: pd.DataFrame,
                         train_synth: pd.DataFrame,
                         target_col: str,
                         task_type: str) -> dict:
    """Trains a model on Real vs Synth and evaluates on Real Test."""
    X_real = train_real.drop(columns=[target_col]).fillna(0)
    y_real = train_real[target_col].fillna(0)
    X_synth = train_synth.drop(columns=[target_col]).fillna(0)
    y_synth = train_synth[target_col].fillna(0)
    X_test = test_real.drop(columns=[target_col]).fillna(0)
    y_test = test_real[target_col].fillna(0)

    if task_type == "classification":
        model_real = RandomForestClassifier(random_state=42).fit(X_real, y_real)
        model_synth = RandomForestClassifier(random_state=42).fit(X_synth, y_synth)
        score_real = f1_score(y_test, model_real.predict(X_test), average='weighted')
        score_synth = f1_score(y_test, model_synth.predict(X_test), average='weighted')
        return {"F1_real": score_real, "F1_synth": score_synth, "diff": score_real - score_synth}
    else:
        model_real = RandomForestRegressor(random_state=42).fit(X_real, y_real)
        model_synth = RandomForestRegressor(random_state=42).fit(X_synth, y_synth)
        score_real = r2_score(y_test, model_real.predict(X_test))
        score_synth = r2_score(y_test, model_synth.predict(X_test))
        return {"R2_real": score_real, "R2_synth": score_synth, "diff": score_real - score_synth}

def mean_js(real: pd.DataFrame, synth: pd.DataFrame, cardinalities: list[int]) -> floating[Any]:
    js = np.zeros(shape=real.shape[1])
    for idx, col in enumerate(real.columns):
        card = cardinalities[idx]

        p = np.bincount(real[col].astype(int), minlength=card)
        q = np.bincount(synth[col].astype(int), minlength=card)

        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q

        js[idx] = jensenshannon(p, q)

    return np.mean(js)

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

def create_noise_dataset(num_samples, cardinalities, device):
    noise_data = []
    for card in cardinalities:
        noise_data.append(torch.randint(0, card, (num_samples,), device=device))
    return torch.stack(noise_data, dim=1)

def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, pandas, PyTorch (CPU/GPU),
    and provide a deterministic worker initializer for DataLoader shuffling.
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    elif hasattr(torch, "set_deterministic_debug_mode"):
        torch.set_deterministic_debug_mode(1)


def compute_wasserstein(real_num: np.ndarray, synth_num: np.ndarray) -> float:
    if real_num.shape[1] == 0: return 0.0
    return float(np.mean([
        wasserstein_distance(real_num[:, i], synth_num[:, i])
        for i in range(real_num.shape[1])
    ]))

def compute_jensenshannon(real_cat: np.ndarray, synth_cat: np.ndarray, cardinalities: list) -> float:
    if real_cat.shape[1] == 0: return 0.0
    js_list = []
    for i, c in enumerate(cardinalities):
        p = np.bincount(real_cat[:, i].astype(int), minlength=c)
        q = np.bincount(synth_cat[:, i].astype(int), minlength=c)
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)
        js_list.append(jensenshannon(p, q))
    return float(np.mean(js_list))

def make_msbm_objective(train_num_t, train_cat_t, val_num_np, val_cat_np,
                        cardinalities, is_ordered, seed, device):
    seed_everything(seed)

    def objective(trial: optuna.trial.Trial):
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
        imf_len = trial.suggest_int("imf_len", 3, 7, step=2)
        fb_sequence = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        epochs = trial.suggest_int("epochs_per_direction", 5, 20)
        grad_clip = trial.suggest_float("grad_clip", 0.1, 1.0)

        cfg = MixedSBMConfig(
            cat_emb_dim=cat_emb_dim, hidden_dim=hidden_dim, time_dim=time_dim,
            n_layers=n_layers, num_steps=steps, sigma=sigma, alpha=alpha,
            lambda_num=lambda_num, lambda_cat=lambda_cat,
            lr=lr, batch_size=batch_size, epochs_per_direction=epochs,
            device=device, seed=seed, fb_sequence=fb_sequence, grad_clip=grad_clip
        )

        cont_dim = train_num_t.shape[1] if train_num_t.numel() > 0 else 0

        solver = MixedSBMSolver(
            continuous_dim=cont_dim,
            cardinalities=cardinalities,
            is_ordered=is_ordered,
            cfg=cfg
        )

        try:
            solver.fit(train_num_t, train_cat_t)

            n_samples = val_num_np.shape[0] if cont_dim > 0 else val_cat_np.shape[0]
            gen_num_t, gen_cat_t = solver.sample(n_samples=n_samples, seed=seed)

        except (ValueError, RuntimeError) as e:
            print(f"Trial failed/pruned due to instability: {e}")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        except Exception as e:
            print(f"Trial pruned due to error: {e}")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        gen_num_np = gen_num_t.cpu().numpy()
        gen_cat_np = gen_cat_t.cpu().numpy()

        wd_loss = compute_wasserstein(val_num_np, gen_num_np)
        js_loss = compute_jensenshannon(val_cat_np, gen_cat_np, cardinalities)

        trial.set_user_attr("wasserstein", wd_loss)
        trial.set_user_attr("jensen_shannon", js_loss)

        return wd_loss + js_loss

    return objective

if __name__ == "__main__":
    seed = 5
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="../../data/datasets/datasets_mixed.pkl")
    ap.add_argument("--datasets", type=str, default="all")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--outdir", type=str, default="msbm_optuna_results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data = pickle.load(f)
    dataset_keys = list(my_data.keys()) if args.datasets.lower() == "all" else [k.strip() for k in
                                                                                args.datasets.split(",")]

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=7, n_warmup_steps=2)

    for ds_name in dataset_keys:
        print(f"\n{'=' * 80}\nDataset: {ds_name}\n{'=' * 80}")
        df_raw = my_data[ds_name].copy()
        target_col = df_raw.attrs.get('target_variable')
        task_type = df_raw.attrs.get('task_type', 'classification')

        schema = TabularSchema.infer_from_dataframe(df_raw, target_col=target_col)
        dm = TabularDataModule(df=df_raw, schema=schema, transforms=TransformPipeline(transforms=[DropMissingRows()]))
        dm.prepare_holdout(SplitConfigHoldout(val_size=args.test_size, shuffle=True, random_state=seed))
        holdout = dm.get_holdout()

        train_df = holdout.train.copy()
        val_df = holdout.val.copy()

        cat_cols = list(schema.categorical_cols) + list(schema.discrete_cols)
        num_cols = list(schema.continuous_cols)

        if schema.target_col and schema.target_col not in (num_cols + cat_cols):
            cat_cols.append(schema.target_col)

        bad_cols = [c for c in train_df.columns if train_df[c].nunique() <= 1]
        if bad_cols:
            train_df = train_df.drop(columns=bad_cols)
            val_df = val_df.drop(columns=bad_cols)
            cat_cols = [c for c in cat_cols if c not in bad_cols]
            num_cols = [c for c in num_cols if c not in bad_cols]

        scaler = StandardScaler()
        if num_cols:
            train_num_np = scaler.fit_transform(train_df[num_cols].fillna(0))
            val_num_np = scaler.transform(val_df[num_cols].fillna(0))
        else:
            train_num_np = np.empty((len(train_df), 0))
            val_num_np = np.empty((len(val_df), 0))

        if cat_cols:
            for c in cat_cols:
                train_df[c], uniques = pd.factorize(train_df[c])
                val_mapper = {val: i for i, val in enumerate(uniques)}
                val_df[c] = val_df[c].map(val_mapper).fillna(0).astype(int)

            train_cat_np = train_df[cat_cols].values
            val_cat_np = val_df[cat_cols].values
            cardinalities = [int(train_df[c].max() + 1) for c in cat_cols]
        else:
            train_cat_np = np.empty((len(train_df), 0), dtype=int)
            val_cat_np = np.empty((len(val_df), 0), dtype=int)
            cardinalities = []

        order_dict = {
            "Adult": ['education', 'education-num'],
            "Credit Approval": [],
            "Online Shoppers Purchasing Intention Dataset": ['Month'],
            "Eucalyptus": ['Utility', 'Year', 'Frosts', 'Rainfall', 'Altitude', 'Latitude'],
            "Forest Fires": ['X', 'Y', 'month', 'day']
        }
        ordered_cols_ds = order_dict.get(ds_name, [])
        order_mask = torch.tensor([c in ordered_cols_ds for c in cat_cols], dtype=torch.bool)

        train_num_t = torch.tensor(train_num_np, dtype=torch.float32, device=args.device)
        train_cat_t = torch.tensor(train_cat_np, dtype=torch.long, device=args.device)

        study_name = f"msbm_study_{ds_name}"
        storage_name = f"sqlite:///{outdir}/msbm_optuna.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner
        )
        objective = make_msbm_objective(
            train_num_t, train_cat_t, val_num_np, val_cat_np,
            cardinalities, order_mask, seed, args.device
        )

        start_time = time.time()
        try:
            study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True, show_progress_bar=True)
        except Exception as e:
            print(f"Error during study: {e}")
            continue
        elapsed_sec = time.time() - start_time

        best = study.best_trial
        bp = best.params

        print(f"\n--- Best trial results ({ds_name}) ---")
        print(f"Best Loss (WD + JS): {best.value}")

        print(f"\n--- Computing final experiments for {ds_name} ---")

        final_cfg = MixedSBMConfig(
            cat_emb_dim=bp["cat_emb_dim"], hidden_dim=bp["hidden_dim"], time_dim=bp["time_dim"],
            n_layers=bp["n_layers"], num_steps=bp["num_steps"], sigma=bp["sigma"],
            lambda_num=bp["lambda_num"], lambda_cat=bp["lambda_cat"],
            lr=bp["lr"], batch_size=bp["batch_size"], epochs_per_direction=bp["epochs_per_direction"],
            device=args.device, seed=seed
        )

        final_solver = MixedSBMSolver(
            continuous_dim=train_num_t.shape[1],
            cardinalities=cardinalities,
            is_ordered=order_mask,
            cfg=final_cfg
        )

        final_solver.fit(train_num_t, train_cat_t)

        def generate_and_reconstruct(n_samples):
            gen_num_t, gen_cat_t = final_solver.sample(n_samples=n_samples, seed=seed)
            synth_df = pd.DataFrame(index=range(n_samples))

            if num_cols:
                synth_num_np = scaler.inverse_transform(gen_num_t.cpu().numpy())
                for i, col in enumerate(num_cols):
                    synth_df[col] = synth_num_np[:, i]

            if cat_cols:
                gen_cat_np = gen_cat_t.cpu().numpy()
                for i, col in enumerate(cat_cols):
                    synth_df[col] = gen_cat_np[:, i]

            return synth_df[train_df.columns]


        synth_val_df = generate_and_reconstruct(len(val_df))
        synth_train_df = generate_and_reconstruct(len(train_df))

        val_df_reconstructed = val_df.copy()
        if num_cols:
            val_df_reconstructed[num_cols] = scaler.inverse_transform(val_num_np)

        final_kl = average_kl(val_df_reconstructed, synth_val_df, list(synth_val_df.columns))
        corr_dist = correlation_distance(val_df_reconstructed, synth_val_df)

        train_df_reconstructed = train_df.copy()
        if num_cols:
            train_df_reconstructed[num_cols] = scaler.inverse_transform(train_num_np)

        ml_eff = evaluate_ml_efficacy(train_df_reconstructed, val_df_reconstructed, synth_train_df, target_col,
                                      task_type)

        results = {
            "dataset": ds_name,
            "best_trial": best.number,
            "n_trials": len(study.trials),
            "elapsed_sec": elapsed_sec,
            "best_params": bp,
            "Best_Tuning_Loss_Combined": best.value,
            "Final_Mean_KL": final_kl,
            "Corr_Distance": corr_dist,
            **ml_eff
        }

        print(json.dumps(results, indent=4))
        (outdir / f"{ds_name}_final_metrics.json").write_text(json.dumps(results, indent=4))