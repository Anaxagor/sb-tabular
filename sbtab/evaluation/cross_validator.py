from __future__ import annotations

import random
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score, f1_score
from scipy.stats import wasserstein_distance

import torch
from catboost import CatBoostRegressor, CatBoostClassifier

from sbtab.bridge.losses import CSBMLoss
from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid
from sbtab.data.datamodule import TabularDataModule
from sbtab.evaluation.metrics import Metrics
from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
from sbtab.solvers.csbm import CSBMUpdater, CSBMSolver
from sbtab.solvers.msbm import MixedSBMSolver, MixedSBMConfig


class CrossValidator:
    def __init__(
            self,
            model_type: str,
            best_params: dict,
            ds_name: str,
            target_col: str,
            task_type: str,
            device: str = "cuda",
            seed: int = 5,
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
        self.seed_everything(seed)

    def run(self, datamodule: TabularDataModule, pure_cat_cols: List[str], discrete_cols: List[str],
            num_cols: List[str]) -> pd.DataFrame:
        all_fold_results = []
        folds_dict = datamodule.get_all_folds()

        for fold_id, fold_data in folds_dict.items():
            print(f"  Fold {fold_id + 1}/{self.k_folds}")

            self.seed_everything(self.seed)

            train_df = fold_data.train.copy()
            test_df = fold_data.test.copy()

            clean_num = [c for c in num_cols if c in train_df.columns and train_df[c].std() > 1e-5]
            clean_cat = [c for c in pure_cat_cols if c in train_df.columns and train_df[c].nunique() > 1]
            clean_disc = [c for c in discrete_cols if c in train_df.columns and train_df[c].nunique() > 1]
            all_cat_discrete = clean_cat + clean_disc

            train_df = train_df.dropna(subset=clean_num + all_cat_discrete).reset_index(drop=True)
            test_df = test_df.dropna(subset=clean_num + all_cat_discrete).reset_index(drop=True)

            if all_cat_discrete:
                fold_encoder = OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1)
                train_cat_np = fold_encoder.fit_transform(train_df[all_cat_discrete])
                fold_cardinalities = [len(cats) for cats in fold_encoder.categories_]
                cat_categories = {col: list(fold_encoder.categories_[i]) for i, col in enumerate(all_cat_discrete)}
            else:
                train_cat_np = np.empty((len(train_df), 0), dtype=int)
                fold_cardinalities = []
                cat_categories = {}

            scaler = StandardScaler()
            if clean_num:
                train_num_scaled = scaler.fit_transform(train_df[clean_num].to_numpy())
                test_num_scaled = scaler.transform(test_df[clean_num].to_numpy())
            else:
                train_num_scaled = np.empty((len(train_df), 0))
                test_num_scaled = np.empty((len(test_df), 0))

            train_num_t = torch.tensor(train_num_scaled, dtype=torch.float32, device=self.device)
            train_cat_t = torch.tensor(train_cat_np, dtype=torch.long, device=self.device)

            train_real_final = pd.concat([train_df[clean_num], train_df[all_cat_discrete]], axis=1).reset_index(
                drop=True)
            test_real_final = pd.concat([test_df[clean_num], test_df[all_cat_discrete]], axis=1).reset_index(drop=True)
            num_synth = len(train_df)

            if self.model_type == 'csbm':
                synth_df_temp, elapsed_time = self._train_and_sample_csbm(
                    train_cat_t, fold_cardinalities, num_synth, all_cat_discrete, self.seed
                )
                synth_cat_np = synth_df_temp.values
                synth_num_scaled = np.zeros((num_synth, len(clean_num)))
            else:
                ordered_cols = self._ordered_cols()
                is_ordered_mask = torch.tensor([c in ordered_cols for c in all_cat_discrete], dtype=torch.bool)
                gen_num, gen_cat, elapsed_time = self._train_and_sample_msbm(
                    train_num_t, train_cat_t, fold_cardinalities, is_ordered_mask, num_synth, self.seed
                )
                synth_num_scaled = gen_num
                synth_cat_np = gen_cat

            synth_df = pd.DataFrame(index=range(num_synth))

            if clean_num:
                synth_num_orig = scaler.inverse_transform(synth_num_scaled)
                for i, col in enumerate(clean_num):
                    synth_df[col] = synth_num_orig[:, i]
            else:
                synth_num_orig = np.empty((num_synth, 0))

            if all_cat_discrete:
                for i, col in enumerate(all_cat_discrete):
                    cats = cat_categories[col]
                    col_data = synth_cat_np[:, i].astype(int)
                    out_range = (col_data < 0) | (col_data >= len(cats))

                    is_numeric = len(cats) > 0 and isinstance(cats[0], (int, float, np.integer, np.floating))
                    ext_cats = np.array(cats + ([-999] if is_numeric else ["UNKNOWN_GEN"]),
                                        dtype=object if not is_numeric else None)

                    col_data[out_range] = len(cats)
                    synth_df[col] = ext_cats[col_data]
                    if is_numeric:
                        synth_df[col] = synth_df[col].astype(type(cats[0]))

            synth_df = synth_df[train_real_final.columns]  # Восстанавливаем порядок
            synth_cat_orig = synth_df[all_cat_discrete].values if all_cat_discrete else np.empty((num_synth, 0),
                                                                                                 dtype=object)

            fold_data_dict = {
                'test_num_scaled': test_num_scaled,
                'clean_num_cols': clean_num,
                'clean_discrete_cols': clean_disc,
                'clean_pure_cat_cols': clean_cat,
                'all_cat_discrete_cols': all_cat_discrete,
                'test_real_df': test_real_final,
            }

            fold_metrics = self._compute_metrics(fold_data_dict, synth_num_scaled, synth_cat_orig, synth_num_orig)
            fold_metrics['fit_time'] = elapsed_time

            tstr_metrics = self._tstr_evaluate(
                train_real_final, test_real_final, synth_df, all_cat_discrete, self.seed
            )

            all_metrics = {**fold_metrics, **tstr_metrics, 'fold': fold_id}
            all_fold_results.append(all_metrics)

        df_results = pd.DataFrame(all_fold_results)
        mean_row = df_results.mean(numeric_only=True).to_dict()
        mean_row['fold'] = 'mean'
        df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)
        df_results.insert(0, 'dataset', self.ds_name)
        df_results.insert(1, 'model', self.model_type)

        return df_results

    def _train_and_sample_csbm(self, train_tensor, cardinalities, num_samples, all_cat_discrete, fold_seed):
        bp = self.best_params
        ordered_cols = self._ordered_cols()
        order_mask = torch.tensor([c in ordered_cols for c in all_cat_discrete], dtype=torch.bool)

        timegrid = TimeGrid(num_steps=bp["steps"])
        max_card = max(cardinalities) if cardinalities else 0
        padded_cardinalities = [max_card] * len(cardinalities)

        fw_model = CSBMTableMLP(padded_cardinalities, bp["emb_dim"], bp["hidden_dim"], bp["time_dim"]).to(self.device)
        bw_model = CSBMTableMLP(padded_cardinalities, bp["emb_dim"], bp["hidden_dim"], bp["time_dim"]).to(self.device)

        fw_opt = torch.optim.Adam(fw_model.parameters(), lr=bp["forward_lr"], weight_decay=bp["forward_weight_decay"])
        bw_opt = torch.optim.Adam(bw_model.parameters(), lr=bp["backward_lr"], weight_decay=bp["backward_weight_decay"])

        process = CategoricalReference(padded_cardinalities, is_ordered=order_mask,
                                       total_number_of_q_powers=bp["steps"], alpha=bp["alpha"],
                                       device=torch.device(self.device))
        loss_fn = CSBMLoss(process, lmbda=bp["loss_lambda"])

        updater = CSBMUpdater(fw_model, bw_model, fw_opt, bw_opt, process, loss_fn)
        sampler = DiscretePathSampler(timegrid, process)
        solver = CSBMSolver(updater, sampler, bp["num_outer_iterations"], bp["epochs"], bp["batch_size"])

        g = torch.Generator(device=self.device).manual_seed(fold_seed)
        p1_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_tensor),
                                                batch_size=bp["batch_size"], shuffle=True, generator=g)
        x_noise = self.create_noise_dataset(len(train_tensor), padded_cardinalities, self.device)
        p0_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_noise), batch_size=bp["batch_size"],
                                                shuffle=True, generator=g)

        start = time.time()
        solver.fit(p1_loader, p0_loader)
        elapsed = time.time() - start

        z_noise = self.create_noise_dataset(num_samples, padded_cardinalities, self.device)
        synth_tensor, _ = sampler.simulate(x_init=z_noise, model=fw_model, direction="forward")
        synth_np = synth_tensor.cpu().numpy()

        return pd.DataFrame(synth_np, columns=all_cat_discrete), elapsed

    def _train_and_sample_msbm(self, train_num_t, train_cat_t, cardinalities, is_ordered, num_samples, fold_seed):
        bp = self.best_params
        cont_dim = train_num_t.shape[1] if train_num_t.numel() > 0 else 0

        cfg = MixedSBMConfig(
            fb_sequence=tuple("b" if i % 2 == 0 else "f" for i in range(bp["imf_len"])),
            cat_emb_dim=bp["cat_emb_dim"], hidden_dim=bp["hidden_dim"], time_dim=bp["time_dim"],
            n_layers=bp["n_layers"], num_steps=bp["num_steps"], sigma=bp["sigma"],
            alpha=bp["alpha"], lambda_num=bp["lambda_num"], lambda_cat=bp["lambda_cat"],
            lr=bp["lr"], batch_size=bp["batch_size"], epochs_per_direction=bp["epochs_per_direction"],
            grad_clip=bp["grad_clip"], device=self.device, seed=fold_seed
        )

        solver = MixedSBMSolver(continuous_dim=cont_dim, cardinalities=cardinalities, is_ordered=is_ordered, cfg=cfg)
        start = time.time()
        solver.fit(train_num_t, train_cat_t)
        elapsed = time.time() - start

        gen_num_t, gen_cat_t = solver.sample(n_samples=num_samples, seed=fold_seed)
        return gen_num_t.cpu().numpy(), gen_cat_t.cpu().numpy(), elapsed

    def _compute_metrics(self, fold_data, synth_num_scaled, synth_cat_np, synth_num_orig):
        metrics = {}
        clean_num = fold_data['clean_num_cols']
        clean_disc = fold_data['clean_discrete_cols']
        clean_cat = fold_data['clean_pure_cat_cols']
        all_cat_discrete = fold_data['all_cat_discrete_cols']

        test_real_df = fold_data['test_real_df']
        synth_parts = []
        if clean_num:
            synth_parts.append(pd.DataFrame(synth_num_orig, columns=clean_num, index=range(len(synth_num_orig))))
        if all_cat_discrete:
            synth_parts.append(pd.DataFrame(synth_cat_np, columns=all_cat_discrete, index=range(len(synth_cat_np))))
        synth_df = pd.concat(synth_parts, axis=1)

        if clean_num:
            wds = [wasserstein_distance(fold_data['test_num_scaled'][:, i], synth_num_scaled[:, i]) for i in
                   range(synth_num_scaled.shape[1])]
            metrics['Continuous_Mean_WD'] = np.mean(wds)
            metrics['Continuous_Mean_KL_50bins'] = Metrics.compute_kl_histogram_continuous(test_real_df, synth_df,
                                                                                           clean_num, bins=50)
            metrics['Continuous_Corr_Pearson'] = Metrics.compute_corr_distance_for_columns(test_real_df, synth_df,
                                                                                           clean_num, method='pearson')
            metrics['mmd'] = Metrics.compute_mmd_numpy(fold_data['test_num_scaled'], synth_num_scaled)
        else:
            metrics['Continuous_Mean_WD'] = metrics['Continuous_Mean_KL_50bins'] = metrics['Continuous_Corr_Pearson'] = \
            metrics['mmd'] = np.nan

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

    def _align_x_y(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Drops NaNs synchronously from features and target."""
        combined = pd.concat([X, y.to_frame('__target__')], axis=1).dropna()
        return combined.drop('__target__', axis=1), combined['__target__']

    def _tstr_evaluate(self, train_real: pd.DataFrame, test_real: pd.DataFrame, train_synth: pd.DataFrame,
                       all_cat_discrete: list, fold_seed: int) -> dict:
        """Evaluates utility of synthetic data using the TSTR framework with CatBoost (Aligned with Tuning)."""

        if self.target_col is None or self.target_col not in train_real.columns:
            if self.task_type == "classification":
                return {'tstr_f1_real': np.nan, 'tstr_f1_synth': np.nan, 'tstr_f1_deviation_%': np.nan,
                        'tstr_f1_diff_raw': np.nan}
            return {'tstr_r2_real': np.nan, 'tstr_r2_synth': np.nan, 'tstr_r2_deviation_%': np.nan,
                    'tstr_r2_diff_raw': np.nan}

        X_train_real, y_train_real = self._align_x_y(train_real.drop(columns=[self.target_col]),
                                                     train_real[self.target_col])
        X_train_synth, y_train_synth = self._align_x_y(train_synth.drop(columns=[self.target_col]),
                                                       train_synth[self.target_col])
        X_test, y_test = self._align_x_y(test_real.drop(columns=[self.target_col]), test_real[self.target_col])

        if self.task_type == "classification":
            y_train_real, y_train_synth, y_test = y_train_real.astype(str), y_train_synth.astype(str), y_test.astype(
                str)
        else:
            y_train_real, y_train_synth, y_test = y_train_real.astype(float), y_train_synth.astype(
                float), y_test.astype(float)

        cat_features = [c for c in all_cat_discrete if c != self.target_col and c in X_train_real.columns]

        if cat_features:
            for df in [X_train_real, X_train_synth, X_test]:
                df[cat_features] = df[cat_features].dropna().astype(str)

        cb_params = {"random_seed": fold_seed, "verbose": 0, "thread_count": -1}
        metrics = {}

        if self.task_type == 'classification':
            model_real = CatBoostClassifier(**cb_params, cat_features=cat_features).fit(X_train_real, y_train_real)
            model_synth = CatBoostClassifier(**cb_params, cat_features=cat_features).fit(X_train_synth, y_train_synth)

            f1_real = f1_score(y_test, model_real.predict(X_test), average='macro')
            f1_synth = f1_score(y_test, model_synth.predict(X_test), average='macro')
            dev = ((f1_real - f1_synth) / max(abs(f1_real), 1e-9)) * 100

            metrics.update({
                'tstr_f1_real': f1_real,
                'tstr_f1_synth': f1_synth,
                'tstr_f1_deviation_%': dev,
                'tstr_f1_diff_raw': f1_real - f1_synth
            })
        else:
            model_real = CatBoostRegressor(**cb_params, cat_features=cat_features).fit(X_train_real, y_train_real)
            model_synth = CatBoostRegressor(**cb_params, cat_features=cat_features).fit(X_train_synth, y_train_synth)

            r2_real = r2_score(y_test, model_real.predict(X_test))
            r2_synth = r2_score(y_test, model_synth.predict(X_test))
            r2_dev = ((r2_real - r2_synth) / max(abs(r2_real), 1e-9)) * 100

            metrics.update({
                'tstr_r2_real': r2_real,
                'tstr_r2_synth': r2_synth,
                'tstr_r2_deviation_%': r2_dev,
                'tstr_r2_diff_raw': r2_real - r2_synth
            })
        return metrics

    def _ordered_cols(self) -> List[str]:
        return {
            "Mushroom": ['ring-number', 'gill-spacing', 'gill-size'],
            "Car Evaluation": ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
            "Student Perf": ['age', 'failures', 'absences', 'G1', 'G2', 'G3', 'Medu', 'Fedu', 'traveltime', 'studytime',
                             'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'],
            "Lymphography": ['lym_nodes_enlar', 'no_of_nodes_in', 'lym_nodes_dimin'],
            "Breast cancer": ['age', 'tumor_size', 'inv_nodes', 'deg-malig'],
            "Adult": ['education', 'education-num'],
            "Online Shoppers Purchasing Intention Dataset": ['Month'],
            "Eucalyptus": ['Utility', 'Year', 'Frosts', 'Rainfall', 'Altitude', 'Latitude'],
            "Forest Fires": ['X', 'Y', 'month', 'day'],
            "Insurance": ['children'],
            "House Sales in King County": ['bedrooms', 'bathrooms', 'floors', 'view'],
            "Cardiovascular Disease": ['Cholesterol', 'Glucose'],
            "Churn Modelling": ['Tenure', 'NumOfProducts'],
            "Auto MPG": ['cylinders', 'model_year'],
            "Diamonds": ['cut', 'color', 'clarity'],
            "Real Estate": ['X4 number of convenience stores']
        }.get(self.ds_name, [])

    @staticmethod
    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def create_noise_dataset(num_samples, cardinalities, device):
        noise_data = [torch.randint(0, card, (num_samples,), device=device) for card in cardinalities]
        return torch.stack(noise_data, dim=1)

    @staticmethod
    def summarise_cv(input_csv: str, output_csv: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(input_csv)
        df_folds = df[df['fold'] != 'mean'].copy()
        df_folds['fold'] = pd.to_numeric(df_folds['fold'])

        metric_cols = [c for c in df_folds.columns if c not in {'dataset', 'model', 'fold'}]
        grouped = df_folds.groupby(['dataset', 'model'])

        all_rows = []
        for (dataset, model), group in grouped:
            stats = {'dataset': dataset, 'model': model}
            for col in metric_cols:
                mean_val = group[col].mean()
                std_val = group[col].std()
                stats[f'{col}_mean'] = mean_val
                stats[f'{col}_std'] = std_val
                stats[f'{col}_summary'] = f"{mean_val:.4f} ± {std_val:.4f}"
            all_rows.append(stats)

        df = pd.DataFrame(all_rows)

        summary_cols = ['dataset', 'model'] + [col for col in df.columns if col.endswith('_summary')]
        only_summary_df = df[summary_cols]

        if output_csv:
            out_path = Path(output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            summary_path = out_path.parent / f"Summary_only_{out_path.name}"

            df.to_csv(out_path, index=False)
            only_summary_df.to_csv(summary_path, index=False)

        return df, only_summary_df
