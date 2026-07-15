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
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from sbtab.bridge.losses import CSBMLoss
from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema
from sbtab.data.splits import SplitConfigHoldout
from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
from sbtab.solvers.csbm import CSBMUpdater, CSBMSolver
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

def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    if real.empty or synth.empty:
        return 0.0

    corr_real_df = real.astype(float).corr(method="spearman").fillna(0)
    corr_real_arr = corr_real_df.values.copy()
    np.fill_diagonal(corr_real_arr, 1.0)
    corr_real = corr_real_arr

    corr_synth_df = synth.astype(float).corr(method="spearman").fillna(0)
    corr_synth_arr = corr_synth_df.values.copy()
    np.fill_diagonal(corr_synth_arr, 1.0)
    corr_synth = corr_synth_arr

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

def make_csbm_objective(train_df, val_df, ds_name, seed, device):
    seed_everything(seed)

    cols = list(train_df.columns)
    cardinalities = [int(train_df[c].max() + 1) for c in cols]

    order_dict = {
        "Mushroom": ['ring-number', 'gill-spacing', 'gill-size'],
        "Car Evaluation": ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
        "Student Perf": ['age', 'failures', 'absences', 'G1', 'G2', 'G3', 'Medu', 'Fedu', 'traveltime', 'studytime',
                         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'],
        "Lymphography": ['lym_nodes_enlar', 'no_of_nodes_in', 'lym_nodes_dimin'],
        "Breast cancer": ['age', 'tumor_size', 'inv_nodes', 'deg-malig']
    }

    ordered_cols_for_ds = order_dict.get(ds_name, [])

    order_mask = torch.tensor([c in ordered_cols_for_ds for c in cols], dtype=torch.bool)

    train_tensor = torch.tensor(train_df.values, dtype=torch.long, device=device)

    def objective(trial: optuna.trial.Trial) -> floating[Any] | float:
        # Hyperparams

        # Model params
        emb_dim = trial.suggest_int("emb_dim", 32, 1024, log=True)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 1024, log=True)
        time_dim = trial.suggest_int("time_dim", 16, 128, step=16)

        # Solver's params
        steps = trial.suggest_int("steps", 20, 100, step=10)
        num_outer_iterations = trial.suggest_int("num_outer_iterations", 5, 70)
        epochs = trial.suggest_int("epochs", 5, 30)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        # Optimization
        fw_lr = trial.suggest_float("forward_lr", 1e-4, 5e-3, log=True)
        bw_lr = trial.suggest_float("backward_lr", 1e-4, 5e-3, log=True)

        # Regularization
        fw_decay = trial.suggest_float("forward_weight_decay", 1e-6, 1e-3, log=True)
        bw_decay = trial.suggest_float("backward_weight_decay", 1e-6, 1e-3, log=True)

        # CSBM Specific
        loss_lmbda = trial.suggest_float("loss_lambda", 0.01, 1.0, step=0.01)
        alpha = trial.suggest_float("alpha", 0.001, 0.1, step=0.001)

        # Determined params
        timegrid = TimeGrid(num_steps=steps)

        total_number_of_q_powers = steps

        max_card = max(cardinalities)
        padded_cardinalities = [max_card] * len(cardinalities)

        fw_model = CSBMTableMLP(
            cardinalities=padded_cardinalities,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim
        ).to(device)

        bw_model = CSBMTableMLP(
            cardinalities=padded_cardinalities,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim
        ).to(device)

        fw_opt = torch.optim.Adam(fw_model.parameters(), lr=fw_lr, weight_decay=fw_decay)
        bw_opt = torch.optim.Adam(bw_model.parameters(), lr=bw_lr, weight_decay=bw_decay)

        # Class initializations
        process = CategoricalReference(
            cardinalities,
            is_ordered=order_mask,
            total_number_of_q_powers=total_number_of_q_powers,
            alpha=alpha,
            device=device
        )
        loss = CSBMLoss(process, lmbda=loss_lmbda)

        updater = CSBMUpdater(
            forward_model=fw_model,
            backward_model=bw_model,
            forward_opt=fw_opt,
            backward_opt=bw_opt,
            ref_process=process,
            loss_fn=loss
        )
        sampler = DiscretePathSampler(timegrid=timegrid, reference=process)

        solver = CSBMSolver(
            updater=updater,
            sampler=sampler,
            num_outer_iterations=num_outer_iterations,
            epochs=epochs,
            batch_size=batch_size
        )

        try:
            g = torch.Generator()
            g.manual_seed(seed)

            p1_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True, generator=g)
            x_noise = create_noise_dataset(len(train_df), cardinalities, device)
            p0_loader = DataLoader(TensorDataset(x_noise), batch_size=batch_size, shuffle=True, generator=g)

            solver.fit(p1_loader, p0_loader)

            num_gen = len(train_df)
            z_noise = create_noise_dataset(num_gen, cardinalities, device)

            synth_data, _ = sampler.simulate(
                x_init=z_noise,
                model=fw_model,
                direction="forward"
            )

            return mean_js(val_df, pd.DataFrame(synth_data.cpu().numpy(), columns=val_df.columns), cardinalities)

        except Exception as e:
            print(e)
            return float("inf")

    return objective

if __name__ == "__main__":
    seed = 5
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="../../data/datasets/datasets_categorical.pkl")
    ap.add_argument("--datasets", type=str, default="all")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--outdir", type=str, default="csbm_optuna_results")
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

        for col in df_raw.columns:
            dtype_str = str(df_raw[col].dtype).lower()
            if 'object' in dtype_str or 'category' in dtype_str or 'str' in dtype_str:
                df_raw[col] = df_raw[col].astype(str).astype('category')
            else:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

        schema = TabularSchema.infer_from_dataframe(df_raw, target_col=target_col)
        dm = TabularDataModule(df=df_raw, schema=schema, transforms=TransformPipeline(transforms=[DropMissingRows()]))

        dm.prepare_holdout(SplitConfigHoldout(val_size=args.test_size, shuffle=True, random_state=seed))
        holdout = dm.get_holdout()

        train_df = holdout.train.copy()
        val_df = holdout.val.copy()

        cols_to_factorize = list(schema.categorical_cols)
        if schema.target_col and train_df[schema.target_col].dtype in ['object', 'category', 'str']:
            cols_to_factorize.append(schema.target_col)

        for col in cols_to_factorize:
            train_df[col], uniques = pd.factorize(train_df[col])
            val_mapper = {val: i for i, val in enumerate(uniques)}
            val_df[col] = val_df[col].map(val_mapper).fillna(0)

        train_df = train_df.astype(int)
        val_df = val_df.astype(int)

        bad_cols = [c for c in train_df.columns if train_df[c].nunique() <= 1]
        if bad_cols:
            print(f"Dropping constant columns: {bad_cols}")
            print(train_df[bad_cols].head(), val_df[bad_cols].head())
            train_df = train_df.drop(columns=bad_cols)
            val_df = val_df.drop(columns=bad_cols)

        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        objective = make_csbm_objective(train_df, val_df, ds_name, seed, args.device)

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
        print(f"Best Loss (Mean JS): {best.value}")

        print(f"\n--- Computing final experiments for {ds_name} ---")

        cols = list(train_df.columns)
        cardinalities = [int(train_df[c].max() + 1) for c in cols]

        order_dict = {
            "Mushroom": ['ring-number', 'gill-spacing', 'gill-size'],
            "Car Evaluation": ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
            "Student Perf": ['age', 'failures', 'absences', 'G1', 'G2', 'G3', 'Medu', 'Fedu',
                             'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'],
            "Lymphography": ['lym_nodes_enlar', 'no_of_nodes_in', 'lym_nodes_dimin'],
            "Breast cancer": ['age', 'tumor_size', 'inv_nodes', 'deg-malig']
        }

        ordered_cols_final = order_dict.get(ds_name, [])
        order_mask = torch.tensor([c in ordered_cols_final for c in cols], dtype=torch.bool)

        timegrid = TimeGrid(num_steps=bp["steps"])

        max_card = max(cardinalities)
        padded_cardinalities = [max_card] * len(cardinalities)

        fw_model = CSBMTableMLP(
            cardinalities=padded_cardinalities,
            emb_dim=bp["emb_dim"],
            hidden_dim=bp["hidden_dim"],
            time_dim=bp["time_dim"]
        ).to(args.device)

        bw_model = CSBMTableMLP(
            cardinalities=padded_cardinalities,
            emb_dim=bp["emb_dim"],
            hidden_dim=bp["hidden_dim"],
            time_dim=bp["time_dim"]
        ).to(args.device)

        fw_opt = torch.optim.Adam(fw_model.parameters(), lr=bp["forward_lr"], weight_decay=bp["forward_weight_decay"])
        bw_opt = torch.optim.Adam(bw_model.parameters(), lr=bp["backward_lr"], weight_decay=bp["backward_weight_decay"])

        process = CategoricalReference(
            cardinalities,
            is_ordered=order_mask,
            total_number_of_q_powers=bp["steps"],
            alpha=bp["alpha"],
            device=args.device
        )
        loss_fn = CSBMLoss(process, lmbda=bp["loss_lambda"])

        updater = CSBMUpdater(
            forward_model=fw_model, backward_model=bw_model,
            forward_opt=fw_opt, backward_opt=bw_opt,
            ref_process=process, loss_fn=loss_fn
        )
        sampler_ds = DiscretePathSampler(timegrid=timegrid, reference=process)

        final_solver = CSBMSolver(
            updater=updater, sampler=sampler_ds,
            num_outer_iterations=bp["num_outer_iterations"],
            epochs=bp["epochs"], batch_size=bp["batch_size"]
        )

        train_tensor = torch.tensor(train_df.values, dtype=torch.long, device=args.device)
        final_p1_loader = DataLoader(TensorDataset(train_tensor), batch_size=bp["batch_size"], shuffle=True,
                                     generator=g)

        final_x_noise = create_noise_dataset(len(train_df), cardinalities, args.device)
        final_p0_loader = DataLoader(TensorDataset(final_x_noise), batch_size=bp["batch_size"], shuffle=True,
                                     generator=g)

        final_solver.fit(final_p1_loader, final_p0_loader)

        # To avoid bias in synthetic distribution generate the number of samples equal to val_df len.
        num_gen = len(val_df)
        noise_final = create_noise_dataset(num_gen, cardinalities, args.device)

        x_synth_tensor, _ = sampler_ds.simulate(
            x_init=noise_final,
            model=fw_model,
            direction="forward"
        )

        synth_df = pd.DataFrame(x_synth_tensor.cpu().numpy(), columns=train_df.columns)

        final_kl = average_kl(val_df, synth_df, list(synth_df.columns))
        corr_dist = correlation_distance(val_df, synth_df)

        # Here generate new synthetic for evaluate_ml_efficiency function so that
        # RF could learn on a proper amount of data that's why we choose len(train_df) number of samples.
        noise_for_ml_eff = create_noise_dataset(len(train_df), cardinalities, args.device)

        x_synth_tensor, _ = sampler_ds.simulate(
            x_init=noise_for_ml_eff,
            model=fw_model,
            direction="forward"
        )

        synth_df = pd.DataFrame(x_synth_tensor.cpu().numpy(), columns=train_df.columns)

        ml_eff = evaluate_ml_efficacy(train_df, val_df, synth_df, target_col, task_type)

        results = {
            "dataset": ds_name,
            "bad_cols (<=1 unique vals)": bad_cols,
            "best_trial": best.number,
            "n_trials": len(study.trials),
            "elapsed_sec": elapsed_sec,
            "best_params": bp,
            "Best_Tuning_Loss_JS": best.value,
            "Final_Mean_KL": final_kl,
            "Corr_Distance": corr_dist,
            **ml_eff
        }

        print(json.dumps(results, indent=4))
        (outdir / f"{ds_name}_final_metrics.json").write_text(json.dumps(results, indent=4))
