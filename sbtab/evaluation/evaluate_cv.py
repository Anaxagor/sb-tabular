from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    f1_score,
)
from scipy.stats import entropy
from scipy.stats import wasserstein_distance

import torch
from catboost import CatBoostRegressor, CatBoostClassifier

from sbtab.bridge.losses import CSBMLoss
from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
from sbtab.solvers.csbm import CSBMUpdater, CSBMSolver
from sbtab.solvers.msbm import MixedSBMSolver, MixedSBMConfig
from sbtab.data.schema import TabularSchema

CONFIG = {
    "model": "msbm",
    "pickle": "../data/datasets/datasets_mixed.pkl",
    "results_dir": "../experiments/tuning_script/msbm_optuna_results",
    "output": "cv_msbm_results.csv",
    "datasets": "all",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def seed_everything(seed: int = 42) -> None:
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_distributions(real: pd.Series, synth: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    cats = list(set(real.dropna().unique()) | set(synth.dropna().unique()))
    p = real.value_counts(normalize=True).reindex(cats, fill_value=1e-9).values
    q = synth.value_counts(normalize=True).reindex(cats, fill_value=1e-9).values
    return p, q

def average_kl(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    if not cols:
        return 0.0
    kls = []
    for c in cols:
        p, q = get_distributions(real[c], synth[c])
        kls.append(float(entropy(p, q)))
    return float(np.mean(kls))

def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    corr_real = real.corr(numeric_only=True).fillna(0).to_numpy()
    corr_synth = synth.corr(numeric_only=True).fillna(0).to_numpy()
    return float(np.linalg.norm(corr_real - corr_synth, ord='fro'))

def compute_mmd(x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    if sigma is None:
        xx = np.sum(x**2, axis=1, keepdims=True) + np.sum(x**2, axis=1) - 2 * x @ x.T
        dists = xx[np.triu_indices_from(xx, k=1)]
        sigma = np.median(np.sqrt(np.maximum(dists, 0))) if len(dists) > 0 else 1.0
        sigma = max(sigma, 1e-5)
    gamma = 1.0 / (2 * sigma ** 2)
    K_xx = np.exp(-gamma * (np.sum(x**2, 1).reshape(-1,1) + np.sum(x**2, 1) - 2 * x @ x.T))
    K_yy = np.exp(-gamma * (np.sum(y**2, 1).reshape(-1,1) + np.sum(y**2, 1) - 2 * y @ y.T))
    K_xy = np.exp(-gamma * (np.sum(x**2, 1).reshape(-1,1) + np.sum(y**2, 1) - 2 * x @ y.T))
    mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
    return float(np.sqrt(np.maximum(mmd, 0)))

def create_noise_dataset(num_samples, cardinalities, device):
    noise_data = [torch.randint(0, card, (num_samples,), device=device) for card in cardinalities]
    return torch.stack(noise_data, dim=1)

class CrossValidator:
    def __init__(
        self,
        model_type: str,
        best_params: dict,
        ds_name: str,
        target_col: str,
        task_type: str,
        val_size: float = 0.2,
        device: str = "cuda",
        seed: int = 42,
        k_folds: int = 5
    ):
        self.model_type = model_type
        self.best_params = best_params
        self.ds_name = ds_name
        self.target_col = target_col
        self.task_type = task_type
        self.device = device
        self.seed = seed
        self.k_folds = k_folds

        seed_everything(seed)

    def _prepare_data_for_fold(self, train_df, test_df, cat_cols, num_cols):
        all_cols = cat_cols + num_cols
        train_df = train_df.dropna(subset=all_cols)
        test_df = test_df.dropna(subset=all_cols)

        cat_mappings = {}
        for c in cat_cols:
            train_cat = pd.Categorical(train_df[c])
            train_df[c] = train_cat.codes
            cat_mappings[c] = {val: code for code, val in enumerate(train_cat.categories)}

            test_codes = test_df[c].map(cat_mappings[c])
            test_codes = test_codes.fillna(0).astype(int)
            test_df[c] = test_codes

        train_cat_np = train_df[cat_cols].values if cat_cols else np.empty((len(train_df), 0), dtype=int)
        test_cat_np = test_df[cat_cols].values if cat_cols else np.empty((len(test_df), 0), dtype=int)

        if cat_cols:
            keep_idx = []
            for i, c in enumerate(cat_cols):
                if train_df[c].nunique() > 1:
                    keep_idx.append(i)
            if len(keep_idx) < len(cat_cols):
                dropped = [cat_cols[i] for i in range(len(cat_cols)) if i not in keep_idx]
                print(f"  Dropping constant categorical columns: {dropped}")
                train_cat_np = train_cat_np[:, keep_idx]
                test_cat_np = test_cat_np[:, keep_idx]
                cat_cols = [cat_cols[i] for i in keep_idx]

        scaler = StandardScaler()
        if num_cols:
            train_num_scaled = scaler.fit_transform(train_df[num_cols].fillna(0))
            test_num_scaled = scaler.transform(test_df[num_cols].fillna(0))
        else:
            train_num_scaled = np.empty((len(train_df), 0))
            test_num_scaled = np.empty((len(test_df), 0))

        train_num_orig = train_df[num_cols].values if num_cols else np.empty((len(train_df), 0))
        test_num_orig = test_df[num_cols].values if num_cols else np.empty((len(test_df), 0))

        train_num_t = torch.tensor(train_num_scaled, dtype=torch.float32, device=self.device)
        train_cat_t = torch.tensor(train_cat_np, dtype=torch.long, device=self.device)

        return {
            'train_num_t': train_num_t,
            'train_cat_t': train_cat_t,
            'train_num_scaled': train_num_scaled,
            'test_num_scaled': test_num_scaled,
            'train_num_orig': train_num_orig,
            'test_num_orig': test_num_orig,
            'train_cat_np': train_cat_np,
            'test_cat_np': test_cat_np,
            'scaler': scaler,
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'train_real_df': self._build_real_df(train_df, train_num_orig, cat_cols, num_cols),
            'test_real_df': self._build_real_df(test_df, test_num_orig, cat_cols, num_cols),
        }

    def _build_real_df(self, df_template, num_orig, cat_cols, num_cols):
        parts = []
        if num_cols:
            parts.append(pd.DataFrame(num_orig, columns=num_cols, index=df_template.index))
        if cat_cols:
            parts.append(df_template[cat_cols].astype(int))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df_template.index)

    def _train_and_sample_csbm(self, train_tensor, cardinalities, num_samples):
        bp = self.best_params
        cols = [f"col_{i}" for i in range(train_tensor.shape[1])]

        order_dict = {
            "Mushroom": ['ring-number', 'gill-spacing', 'gill-size'],
            "Car Evaluation": ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
            "Student Perf": ['age', 'failures', 'absences', 'G1', 'G2', 'G3',
                             'Medu', 'Fedu', 'traveltime', 'studytime',
                             'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'],
            "Lymphography": ['lym_nodes_enlar', 'no_of_nodes_in', 'lym_nodes_dimin'],
            "Breast cancer": ['age', 'tumor_size', 'inv_nodes', 'deg-malig']
        }
        ordered_cols = order_dict.get(self.ds_name, [])
        order_mask = torch.tensor([c in ordered_cols for c in cols], dtype=torch.bool)

        timegrid = TimeGrid(num_steps=bp["steps"])
        max_card = max(cardinalities)
        padded_cardinalities = [max_card] * len(cardinalities)

        fw_model = CSBMTableMLP(
            cardinalities=padded_cardinalities,
            emb_dim=bp["emb_dim"],
            hidden_dim=bp["hidden_dim"],
            time_dim=bp["time_dim"]
        ).to(self.device)
        bw_model = CSBMTableMLP(
            cardinalities=padded_cardinalities,
            emb_dim=bp["emb_dim"],
            hidden_dim=bp["hidden_dim"],
            time_dim=bp["time_dim"]
        ).to(self.device)

        fw_opt = torch.optim.Adam(fw_model.parameters(), lr=bp["forward_lr"], weight_decay=bp["forward_weight_decay"])
        bw_opt = torch.optim.Adam(bw_model.parameters(), lr=bp["backward_lr"], weight_decay=bp["backward_weight_decay"])

        process = CategoricalReference(
            cardinalities,
            is_ordered=order_mask,
            total_number_of_q_powers=bp["steps"],
            alpha=bp["alpha"],
            device=self.device
        )
        loss_fn = CSBMLoss(process, lmbda=bp["loss_lambda"])

        updater = CSBMUpdater(
            forward_model=fw_model, backward_model=bw_model,
            forward_opt=fw_opt, backward_opt=bw_opt,
            ref_process=process, loss_fn=loss_fn
        )
        sampler = DiscretePathSampler(timegrid=timegrid, reference=process)

        solver = CSBMSolver(
            updater=updater, sampler=sampler,
            num_outer_iterations=bp["num_outer_iterations"],
            epochs=bp["epochs"], batch_size=bp["batch_size"]
        )

        g = torch.Generator(device='cpu')
        g.manual_seed(self.seed)
        p1_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor),
            batch_size=bp["batch_size"], shuffle=True, generator=g
        )
        x_noise = create_noise_dataset(len(train_tensor), cardinalities, self.device)
        p0_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_noise), batch_size=bp["batch_size"], shuffle=True, generator=g
        )
        solver.fit(p1_loader, p0_loader)

        z_noise = create_noise_dataset(num_samples, cardinalities, self.device)
        synth_tensor, _ = sampler.simulate(
            x_init=z_noise, model=fw_model, direction="forward"
        )
        synth_np = synth_tensor.cpu().numpy()
        return pd.DataFrame(synth_np, columns=cols)

    def _train_and_sample_msbm(self, train_num_t, train_cat_t, cardinalities, is_ordered, num_samples):
        bp = self.best_params
        cont_dim = train_num_t.shape[1] if train_num_t.numel() > 0 else 0

        cfg = MixedSBMConfig(
            cat_emb_dim=bp["cat_emb_dim"], hidden_dim=bp["hidden_dim"], time_dim=bp["time_dim"],
            n_layers=bp["n_layers"], num_steps=bp["num_steps"], sigma=bp["sigma"],
            alpha=bp["alpha"], lambda_num=bp["lambda_num"], lambda_cat=bp["lambda_cat"],
            lr=bp["lr"], batch_size=bp["batch_size"], epochs_per_direction=bp["epochs_per_direction"],
            device=self.device, seed=self.seed
        )

        solver = MixedSBMSolver(
            continuous_dim=cont_dim,
            cardinalities=cardinalities,
            is_ordered=is_ordered,
            cfg=cfg
        )
        solver.fit(train_num_t, train_cat_t)
        gen_num_t, gen_cat_t = solver.sample(n_samples=num_samples, seed=self.seed)
        return gen_num_t.cpu().numpy(), gen_cat_t.cpu().numpy()

    def _compute_metrics(self, fold_data, synth_num_scaled, synth_cat_np, synth_num_orig):
        """Вычисляет все метрики для одного фолда."""
        metrics = {}
        if synth_num_scaled.shape[1] > 0:
            wds = [wasserstein_distance(fold_data['test_num_scaled'][:, i], synth_num_scaled[:, i])
                   for i in range(synth_num_scaled.shape[1])]
            metrics['wasserstein'] = np.mean(wds)
            metrics['mmd'] = compute_mmd(fold_data['test_num_scaled'], synth_num_scaled)
        else:
            metrics['wasserstein'] = np.nan
            metrics['mmd'] = np.nan

        cat_cols = fold_data['cat_cols']
        if cat_cols:
            real_cat_df = pd.DataFrame(fold_data['test_cat_np'], columns=cat_cols)
            synth_cat_df = pd.DataFrame(synth_cat_np, columns=cat_cols)
            metrics['mean_kl'] = average_kl(real_cat_df, synth_cat_df, cat_cols)
        else:
            metrics['mean_kl'] = 0.0

        real_orig_parts = []
        synth_orig_parts = []
        if synth_num_orig.shape[1] > 0:
            real_orig_parts.append(pd.DataFrame(fold_data['test_num_orig'], columns=fold_data['num_cols']))
            synth_orig_parts.append(pd.DataFrame(synth_num_orig, columns=fold_data['num_cols']))
        if cat_cols:
            real_orig_parts.append(pd.DataFrame(fold_data['test_cat_np'], columns=cat_cols))
            synth_orig_parts.append(pd.DataFrame(synth_cat_np, columns=cat_cols))
        real_orig = pd.concat(real_orig_parts, axis=1) if real_orig_parts else pd.DataFrame()
        synth_orig = pd.concat(synth_orig_parts, axis=1) if synth_orig_parts else pd.DataFrame()
        metrics['corr_dist'] = correlation_distance(real_orig, synth_orig)

        return metrics

    def _tstr_evaluate(self, train_real: pd.DataFrame, test_real: pd.DataFrame,
                       train_synth: pd.DataFrame, target_col: str) -> dict:
        X_train_real = train_real.drop(columns=[target_col]).fillna(0)
        y_train_real = train_real[target_col].fillna(0)
        X_train_synth = train_synth.drop(columns=[target_col]).fillna(0)
        y_train_synth = train_synth[target_col].fillna(0)
        X_test = test_real.drop(columns=[target_col]).fillna(0)
        y_test = test_real[target_col].fillna(0)

        if self.task_type == 'classification':
            model_real = CatBoostClassifier(random_seed=42, verbose=0)
            model_synth = CatBoostClassifier(random_seed=42, verbose=0)
            model_real.fit(X_train_real, y_train_real)
            model_synth.fit(X_train_synth, y_train_synth)
            f1_real = f1_score(y_test, model_real.predict(X_test), average='weighted')
            f1_synth = f1_score(y_test, model_synth.predict(X_test), average='weighted')
            dev = (f1_real - f1_synth) / (f1_real + 1e-9) * 100
            return {
                'tstr_f1_real': f1_real,
                'tstr_f1_synth': f1_synth,
                'tstr_f1_deviation_%': dev,
                'tstr_f1_diff_raw': f1_real - f1_synth
            }
        else:
            model_real = CatBoostRegressor(random_seed=42, verbose=0)
            model_synth = CatBoostRegressor(random_seed=42, verbose=0)
            model_real.fit(X_train_real, y_train_real)
            model_synth.fit(X_train_synth, y_train_synth)
            pred_real = model_real.predict(X_test)
            pred_synth = model_synth.predict(X_test)
            r2_real = r2_score(y_test, pred_real)
            r2_synth = r2_score(y_test, pred_synth)
            r2_dev = (r2_real - r2_synth) / (abs(r2_real) + 1e-9) * 100 if r2_real != 0 else 0.0
            return {
                'tstr_r2_real': r2_real,
                'tstr_r2_synth': r2_synth,
                'tstr_r2_deviation_%': r2_dev,
            }

    def run(self, df_raw: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        all_fold_results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(df_raw)):
            print(f"  Fold {fold + 1}/{self.k_folds} ...")
            train_df = df_raw.iloc[train_idx].copy()
            test_df = df_raw.iloc[test_idx].copy()

            fold_data = self._prepare_data_for_fold(train_df, test_df, cat_cols, num_cols)

            cat_cols_clean = fold_data['cat_cols']
            num_cols_clean = fold_data['num_cols']

            if cat_cols_clean:
                cardinalities = [
                    int(fold_data['train_cat_np'][:, i].max() + 1)
                    for i in range(fold_data['train_cat_np'].shape[1])
                ]
                is_ordered_mask = torch.tensor(
                    [c in self._ordered_cols() for c in cat_cols_clean], dtype=torch.bool
                )
            else:
                cardinalities = []
                is_ordered_mask = torch.tensor([], dtype=torch.bool)

            num_synth = fold_data['train_num_t'].shape[0]

            if self.model_type == 'csbm':
                train_tensor = torch.tensor(fold_data['train_cat_np'], dtype=torch.long, device=self.device)
                synth_df = self._train_and_sample_csbm(train_tensor, cardinalities, num_synth)
                synth_cat_np = synth_df.values
                synth_num_scaled = np.empty((num_synth, 0))
                synth_num_orig = np.empty((num_synth, 0))
            else:
                gen_num, gen_cat = self._train_and_sample_msbm(
                    fold_data['train_num_t'], fold_data['train_cat_t'],
                    cardinalities, is_ordered_mask, num_synth
                )
                synth_num_scaled = gen_num
                synth_cat_np = gen_cat
                if num_cols_clean:
                    synth_num_orig = fold_data['scaler'].inverse_transform(gen_num)
                else:
                    synth_num_orig = np.empty((num_synth, 0))

            fold_metrics = self._compute_metrics(fold_data, synth_num_scaled, synth_cat_np, synth_num_orig)

            train_coded_parts = []
            if num_cols_clean:
                train_coded_parts.append(pd.DataFrame(fold_data['train_num_orig'], columns=num_cols_clean))
            if cat_cols_clean:
                train_coded_parts.append(pd.DataFrame(fold_data['train_cat_np'], columns=cat_cols_clean))
            train_coded_df = pd.concat(train_coded_parts, axis=1)

            if self.model_type == 'csbm':
                synth_tstr_df = pd.DataFrame(synth_cat_np, columns=cat_cols_clean)
            else:
                synth_tstr_parts = []
                if num_cols_clean:
                    synth_tstr_parts.append(pd.DataFrame(synth_num_orig, columns=num_cols_clean))
                if cat_cols_clean:
                    synth_tstr_parts.append(pd.DataFrame(synth_cat_np, columns=cat_cols_clean))
                synth_tstr_df = pd.concat(synth_tstr_parts, axis=1) if synth_tstr_parts else pd.DataFrame()

            synth_tstr_df[self.target_col] = train_coded_df[self.target_col].values

            tstr_metrics = self._tstr_evaluate(
                fold_data['train_real_df'], fold_data['test_real_df'],
                synth_tstr_df, self.target_col
            )

            all_metrics = {**fold_metrics, **tstr_metrics, 'fold': fold}
            all_fold_results.append(all_metrics)

        df_results = pd.DataFrame(all_fold_results)
        mean_row = df_results.mean(numeric_only=True).to_dict()
        mean_row['fold'] = 'mean'
        df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)
        df_results.insert(0, 'dataset', self.ds_name)
        df_results.insert(1, 'model', self.model_type)

        return df_results

    def _ordered_cols(self) -> List[str]:
        order_dict = {
            "Mushroom": ['ring-number', 'gill-spacing', 'gill-size'],
            "Car Evaluation": ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
            "Student Perf": ['age', 'failures', 'absences', 'G1', 'G2', 'G3',
                             'Medu', 'Fedu', 'traveltime', 'studytime',
                             'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'],
            "Lymphography": ['lym_nodes_enlar', 'no_of_nodes_in', 'lym_nodes_dimin'],
            "Breast cancer": ['age', 'tumor_size', 'inv_nodes', 'deg-malig'],
            "Adult": ['education', 'education-num'],
            "Credit Approval": [],
            "Online Shoppers Purchasing Intention Dataset": ['Month'],
            "Eucalyptus": ['Utility', 'Year', 'Frosts', 'Rainfall', 'Altitude', 'Latitude'],
            "Forest Fires": ['X', 'Y', 'month', 'day']
        }
        return order_dict.get(self.ds_name, [])

def summarise_cv(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    df_folds = df[df['fold'] != 'mean'].copy()

    df_folds['fold'] = pd.to_numeric(df_folds['fold'])

    ignore_cols = {'dataset', 'model', 'fold'}
    metric_cols = [c for c in df_folds.columns if c not in ignore_cols]

    grouped = df_folds.groupby(['dataset', 'model'])

    summary_rows = []
    for (dataset, model), group in grouped:
        stats = {'dataset': dataset, 'model': model}
        for col in metric_cols:
            mean_val = group[col].mean()
            std_val  = group[col].std()
            stats[f'{col}_mean'] = mean_val
            stats[f'{col}_std']  = std_val
            stats[f'{col}_summary'] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)

    print(summary_df.to_string(index=False))

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSummary saved to {output_csv}")

    return summary_df

def main():
    model = CONFIG["model"]
    pickle_path = CONFIG["pickle"]
    results_dir = CONFIG["results_dir"]
    output_csv = CONFIG["output"]
    datasets_str = CONFIG["datasets"]
    device = CONFIG["device"]

    outpath = Path(output_csv)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "rb") as f:
        all_data = pickle.load(f)
    dataset_keys = list(all_data.keys()) if datasets_str.lower() == "all" else [k.strip() for k in datasets_str.split(",")]
    print(dataset_keys)
    all_cv_results = []

    for ds_name in dataset_keys:
        print(f"\n{'='*60}\nEvaluating {model} on {ds_name}\n{'='*60}")
        df_raw = all_data[ds_name].copy()
        target_col = df_raw.attrs.get('target_variable')
        task_type = df_raw.attrs.get('task_type', 'classification')

        param_file = Path(results_dir) / f"{ds_name}_final_metrics.json"
        if not param_file.exists():
            print(f"  Skipping {ds_name}: no best params file at {param_file}")
            continue
        with open(param_file) as f:
            tuning_info = json.load(f)
        best_params = tuning_info["best_params"]

        schema = TabularSchema.infer_from_dataframe(df_raw, target_col=target_col)
        cat_cols = list(schema.categorical_cols) + list(schema.discrete_cols)
        num_cols = [c for c in schema.continuous_cols if pd.api.types.is_numeric_dtype(df_raw[c])]

        if target_col and target_col not in (cat_cols + num_cols):
            if df_raw[target_col].dtype == 'object' or 'category' or 'str' in str(df_raw[target_col].dtype):
                cat_cols.append(target_col)
            else:
                num_cols.append(target_col)

        if model == "csbm":
            num_cols_without_target = [c for c in num_cols if c != target_col]
            cat_cols.extend(num_cols_without_target)
            num_cols = []

        cat_cols = list(dict.fromkeys(cat_cols))
        num_cols = list(dict.fromkeys(num_cols))

        cv = CrossValidator(
            model_type=model,
            best_params=best_params,
            ds_name=ds_name,
            target_col=target_col,
            task_type=task_type,
            device=device,
            seed=42,
            k_folds=5
        )

        df_fold_results = cv.run(df_raw, cat_cols, num_cols)
        all_cv_results.append(df_fold_results)

    if all_cv_results:
        final_df = pd.concat(all_cv_results, ignore_index=True)
        final_df.to_csv(outpath, index=False)
        print(f"\nAll results saved to {outpath}")
    else:
        print("No results computed.")

    summarise_cv(str(outpath), "mean_cv_сsbm_results.csv")


if __name__ == "__main__":
    main()