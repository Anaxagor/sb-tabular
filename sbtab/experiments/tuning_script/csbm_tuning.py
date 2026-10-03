from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna
import torch
from numpy import floating
from sklearn.preprocessing import OrdinalEncoder

from torch.utils.data import DataLoader, TensorDataset

from sbtab.bridge.losses import CSBMLoss
from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema, classify_feature_type
from sbtab.data.splits import SplitConfigHoldout
from sbtab.evaluation import Metrics
from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
from sbtab.solvers.csbm import CSBMUpdater, CSBMSolver
from sbtab.transforms.drop_cols import DropDataCols
from sbtab.transforms.missing import DropMissingRows
from sbtab.transforms.pipeline import TransformPipeline

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

def make_csbm_objective(train_cat_t, val_cat_np, cardinalities, order_mask, ds_name, seed, device):
    seed_everything(seed)
    train_tensor = train_cat_t

    def objective(trial: optuna.trial.Trial) -> floating[Any] | float:
        # Hyperparams
        emb_dim = trial.suggest_int("emb_dim", 8, 32)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        time_dim = trial.suggest_int("time_dim", 32, 128, step=32)
        n_layers = trial.suggest_int("n_layers", 2, 6)
        dropout = trial.suggest_float("dropout", 0.01, 0.3)

        # Solver Params
        steps = trial.suggest_int("num_steps", 200, 2000, log=True)
        num_outer_iterations = trial.suggest_int("num_outer_iterations", 2, 10)
        inner_iters = trial.suggest_int("inner_iters", 500, 4000, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        epochs = int(inner_iters * batch_size / train_cat_t)

        # Optimization
        fw_lr = trial.suggest_float("forward_lr", 1e-4, 2e-3, log=True)
        bw_lr = trial.suggest_float("backward_lr", 1e-4, 2e-3, log=True)
        fw_decay = trial.suggest_float("forward_weight_decay", 1e-6, 1e-3, log=True)
        bw_decay = trial.suggest_float("backward_weight_decay", 1e-6, 1e-3, log=True)

        # CSBM Parameters
        loss_lmbda = trial.suggest_float("loss_lambda", 0.01, 1.0)
        alpha = trial.suggest_float("alpha", 0.01, 1.0)

        # Determined params
        timegrid = TimeGrid(num_steps=steps)

        fw_model = CSBMTableMLP(
            cardinalities=cardinalities,
            n_layers=n_layers,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            dropout=dropout
        ).to(device)

        bw_model = CSBMTableMLP(
            cardinalities=cardinalities,
            n_layers=n_layers,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            dropout=dropout
        ).to(device)

        fw_opt = torch.optim.AdamW(fw_model.parameters(), lr=fw_lr, weight_decay=fw_decay)
        bw_opt = torch.optim.AdamW(bw_model.parameters(), lr=bw_lr, weight_decay=bw_decay)

        # Class initializations
        process = CategoricalReference(
            cardinalities,
            is_ordered=order_mask,
            total_number_of_q_powers=steps,
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
            x_noise = create_noise_dataset(len(train_cat_t), cardinalities, device)
            p0_loader = DataLoader(TensorDataset(x_noise), batch_size=batch_size, shuffle=True, generator=g)

            solver.fit(p1_loader, p0_loader)

            num_gen = val_cat_np.shape[0]
            z_noise = create_noise_dataset(num_gen, cardinalities, device)

            synth_data, _ = sampler.simulate(
                x_init=z_noise,
                model=fw_model,
                direction="forward"
            )

            return Metrics.compute_jensenshannon_optuna(val_cat_np, synth_data.cpu().numpy(), cardinalities, trial, seed, ds_name, device)

        except Exception as e:
            print(e)
            return float("inf")

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
    g = torch.Generator()
    g.manual_seed(seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=str, default="../../data/datasets/datasets_categorical.pkl")
    ap.add_argument("--datasets", type=str, default="all")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="csbm_optuna_results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        my_data = pickle.load(f)
    dataset_keys = list(my_data.keys()) if args.datasets.lower() == "all" else [k.strip() for k in
                                                                                args.datasets.split(",")]

    sampler = optuna.samplers.TPESampler(seed=seed)
    for ds_name in dataset_keys:
        print(f"\n{'=' * 80}\nDataset: {ds_name}\n{'=' * 80}")
        if ds_name == "Student Perf":
            continue
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

        if target_col:
            target_col_type = classify_feature_type(train_df[target_col])
            if target_col_type == "discrete":
                discrete_cols.append(target_col)
            elif target_col_type == "categorical":
                cat_cols.append(target_col)
            else:
                raise ValueError("Target column type not recognized for purely categorical/discrete CSBM setup")

        bad_cols = [c for c in train_df.columns if train_df[c].nunique() <= 1]
        if bad_cols:
            print(f"Dropping constant columns: {bad_cols}")
            train_df = train_df.drop(columns=bad_cols)
            val_df = val_df.drop(columns=bad_cols)
            cat_cols = [c for c in cat_cols if c not in bad_cols]
            discrete_cols = [c for c in discrete_cols if c not in bad_cols]

        all_cat_discrete_cols = cat_cols + discrete_cols
        true_train_eval_df = train_df[all_cat_discrete_cols].copy()
        true_val_eval_df = val_df[all_cat_discrete_cols].copy()

        if all_cat_discrete_cols:
            global_encoder = OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1)
            global_encoder.fit(train_df[all_cat_discrete_cols])

            train_cat_np = global_encoder.transform(train_df[all_cat_discrete_cols])
            val_cat_np = global_encoder.transform(val_df[all_cat_discrete_cols])

            cat_categories = {col: list(global_encoder.categories_[i]) for i, col in enumerate(all_cat_discrete_cols)}
            cardinalities = [len(cat_categories[col]) for col in all_cat_discrete_cols]
        else:
            raise ValueError("No categorical or discrete columns found for CSBM.")

        order_dict = {
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

        train_cat_t = torch.tensor(train_cat_np, dtype=torch.long, device=args.device)

        study = optuna.create_study(
            study_name=f"csbm_study_{ds_name}",
            storage=f"sqlite:///{outdir}/csbm_optuna.db",
            load_if_exists=True,
            direction="minimize",
            sampler=sampler
        )

        start_time = time.time()
        try:
            study.optimize(
                make_csbm_objective(train_cat_t, val_cat_np, cardinalities, order_mask, ds_name, seed, args.device),
                n_trials=args.n_trials,
                gc_after_trial=True,
                show_progress_bar=True
            )
        except Exception as e:
            print(f"Error during study: {e}")
            continue

        elapsed_sec = time.time() - start_time
        export_trials_csv(study, outdir / f"{ds_name}_trials.csv")

        best = study.best_trial
        bp = best.params

        print(f"\n--- Best trial results ({ds_name}) ---")
        print(f"Best Loss (Mean JS): {best.value}")

        print(f"\n--- Computing final experiments for {ds_name} ---")

        timegrid = TimeGrid(num_steps=bp["steps"])

        fw_model = CSBMTableMLP(
            cardinalities=cardinalities,
            emb_dim=bp["emb_dim"],
            hidden_dim=bp["hidden_dim"],
            time_dim=bp["time_dim"],
            n_layers=bp["n_layers"],
            dropout=bp["dropout"]
        ).to(args.device)

        bw_model = CSBMTableMLP(
            cardinalities=cardinalities,
            emb_dim=bp["emb_dim"],
            hidden_dim=bp["hidden_dim"],
            time_dim=bp["time_dim"],
            n_layers=bp["n_layers"],
            dropout=bp["dropout"]
        ).to(args.device)

        fw_opt = torch.optim.AdamW(fw_model.parameters(), lr=bp["forward_lr"], weight_decay=bp["forward_weight_decay"])
        bw_opt = torch.optim.AdamW(bw_model.parameters(), lr=bp["backward_lr"], weight_decay=bp["backward_weight_decay"])

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

        final_p1_loader = DataLoader(TensorDataset(train_cat_t), batch_size=bp["batch_size"], shuffle=True, generator=g)
        final_x_noise = create_noise_dataset(len(train_df), cardinalities, args.device)
        final_p0_loader = DataLoader(TensorDataset(final_x_noise), batch_size=bp["batch_size"], shuffle=True,
                                     generator=g)

        final_solver.fit(final_p1_loader, final_p0_loader)


        def tensors_to_dataframe(g_cat, n_samples):
            """Inverse transforms tensors back into a Pandas DataFrame."""
            s_df = pd.DataFrame(index=range(n_samples))
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

                    original_dtype = true_train_eval_df[col].dtype
                    try:
                        s_df[col] = s_df[col].astype(original_dtype)
                    except (ValueError, TypeError):
                        pass

            return s_df[all_cat_discrete_cols]


        # Generate validation size synthetic data for evaluation metrics
        noise_val = create_noise_dataset(len(val_df), cardinalities, args.device)
        x_synth_val_tensor, _ = sampler_ds.simulate(
            x_init=noise_val,
            model=fw_model,
            direction="forward"
        )
        synth_val_df = tensors_to_dataframe(x_synth_val_tensor, len(val_df))

        # Generate train size synthetic data for ML efficacy evaluation
        noise_train = create_noise_dataset(len(train_df), cardinalities, args.device)
        x_synth_train_tensor, _ = sampler_ds.simulate(
            x_init=noise_train,
            model=fw_model,
            direction="forward"
        )
        synth_train_df = tensors_to_dataframe(x_synth_train_tensor, len(train_df))

        discrete_present = [c for c in discrete_cols if c in true_val_eval_df.columns and c not in bad_cols]
        pure_cat_cols = [c for c in cat_cols if c in true_val_eval_df.columns and c not in bad_cols]

        final_kl = Metrics.average_kl_discrete(true_val_eval_df, synth_val_df, pure_cat_cols) if pure_cat_cols else None

        corr_dist = Metrics.compute_corr_distance_for_columns(
            true_val_eval_df, synth_val_df, discrete_present, method="spearman"
        ) if discrete_present else None

        ml_eff = Metrics.evaluate_ml_efficacy(
            train_real=true_train_eval_df, test_real=true_val_eval_df,
            train_synth=synth_train_df,
            target_col=target_col, task_type=task_type
        )

        results = {
            "dataset": ds_name,
            "bad_cols (<=1 unique vals)": bad_cols,
            "best_trial": best.number,
            "n_trials": len(study.trials),
            "elapsed_sec": elapsed_sec,
            "best_params": bp,
            "Best_Tuning_Loss_JS": best.value,
            "Final_Mean_KL": final_kl,
            "Corr_Distance_Discrete_Spearman": corr_dist,
            **ml_eff
        }

        print(json.dumps(results, indent=4, cls=NumpyEncoder))
        (outdir / f"{ds_name}_final_metrics.json").write_text(json.dumps(results, indent=4, cls=NumpyEncoder))

        train_pkl_path = outdir / f"{ds_name}_train.pkl"
        test_pkl_path = outdir / f"{ds_name}.test.pkl"

        with open(train_pkl_path, "wb") as f:
            pickle.dump(true_train_eval_df, f)

        with open(test_pkl_path, "wb") as f:
            pickle.dump(true_val_eval_df, f)