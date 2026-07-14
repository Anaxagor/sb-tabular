from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import optuna
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel

class Metrics:
    """
    This class contains all custom metrics which are used in the pipeline.
    """
    @staticmethod
    def average_kl_discrete(real: pd.DataFrame, synth: pd.DataFrame, cat_cols: List[str], eps: float = 1e-12) -> float:
        """Calculates the average KL divergence for discrete/categorical columns."""
        if not cat_cols:
            return 0.0
        kls = []
        for c in cat_cols:
            all_cats = sorted(set(real[c].dropna().unique()) | set(synth[c].dropna().unique()))
            p_counts = real[c].value_counts().reindex(all_cats, fill_value=0).values.astype(np.float64)
            q_counts = synth[c].value_counts().reindex(all_cats, fill_value=0).values.astype(np.float64)

            p = p_counts / p_counts.sum()
            q = q_counts / q_counts.sum()

            p = (p + eps) / (1 + eps * len(p))
            q = (q + eps) / (1 + eps * len(q))

            kls.append(float(entropy(p, q)))
        return float(np.mean(kls))

    @staticmethod
    def compute_mmd_numpy(real: np.ndarray, synth: np.ndarray, max_samples: int = 5000, seed: int = 5) -> float:
        """Computes Maximum Mean Discrepancy (MMD) using an RBF kernel."""
        if real.shape[0] == 0 or synth.shape[0] == 0:
            return 0.0

        X, Y = real, synth
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        rng = np.random.default_rng(seed=seed)
        if X.shape[0] > max_samples:
            X = X[rng.choice(X.shape[0], max_samples, replace=False)]
        if Y.shape[0] > max_samples:
            Y = Y[rng.choice(Y.shape[0], max_samples, replace=False)]

        XX = rbf_kernel(X, X)
        YY = rbf_kernel(Y, Y)
        XY = rbf_kernel(X, Y)
        return float(XX.mean() + YY.mean() - 2 * XY.mean())

    @staticmethod
    def compute_kl_histogram_continuous(real_df: pd.DataFrame, synth_df: pd.DataFrame, num_cols: List[str],
                                        bins: int = 50) -> float:
        """Computes the average KL divergence for continuous variables using histograms."""
        if not num_cols:
            return 0.0
        kls = []
        eps = 1e-12
        for col in num_cols:
            real_vals = real_df[col].dropna().values
            synth_vals = synth_df[col].dropna().values
            if len(real_vals) == 0 or len(synth_vals) == 0:
                kls.append(0.0)
                continue

            min_val = min(real_vals.min(), synth_vals.min())
            max_val = max(real_vals.max(), synth_vals.max())
            if min_val == max_val:
                kls.append(0.0)
                continue

            edges = np.linspace(min_val, max_val, bins + 1)
            p_hist, _ = np.histogram(real_vals, bins=edges, density=True)
            q_hist, _ = np.histogram(synth_vals, bins=edges, density=True)

            p_hist = (p_hist + eps) / (p_hist + eps).sum()
            q_hist = (q_hist + eps) / (q_hist + eps).sum()

            kls.append(float(entropy(p_hist, q_hist)))
        return float(np.mean(kls))

    @staticmethod
    def compute_corr_distance_for_columns(real_df: pd.DataFrame, synth_df: pd.DataFrame, columns: List[str],
                                          method: str = "pearson") -> float:
        """Computes the Frobenius norm of the difference between correlation matrices."""
        if not columns or real_df.empty or synth_df.empty:
            return 0.0

        corr_real_arr = real_df[columns].astype(float).corr(method=method).fillna(0).values.copy()
        corr_synth_arr = synth_df[columns].astype(float).corr(method=method).fillna(0).values.copy()

        np.fill_diagonal(corr_real_arr, 0.0)
        np.fill_diagonal(corr_synth_arr, 0.0)

        return float(np.linalg.norm(corr_real_arr - corr_synth_arr, ord='fro'))

    @staticmethod
    def compute_nmi_distance_matrix(real_df: pd.DataFrame, synth_df: pd.DataFrame, cat_cols: List[str]) -> float:
        """Computes the Frobenius norm difference of pairwise NMI matrices."""
        if not cat_cols or len(cat_cols) < 2:
            return 0.0

        r_encoded = real_df[cat_cols].apply(lambda x: pd.factorize(x)[0])
        s_encoded = synth_df[cat_cols].apply(lambda x: pd.factorize(x)[0])

        n = len(cat_cols)
        real_nmi = np.zeros((n, n))
        synth_nmi = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                r_score = normalized_mutual_info_score(r_encoded.iloc[:, i], r_encoded.iloc[:, j],
                                                       average_method='arithmetic')
                s_score = normalized_mutual_info_score(s_encoded.iloc[:, i], s_encoded.iloc[:, j],
                                                       average_method='arithmetic')

                real_nmi[i, j] = real_nmi[j, i] = r_score
                synth_nmi[i, j] = synth_nmi[j, i] = s_score

        np.fill_diagonal(real_nmi, 0.0)
        np.fill_diagonal(synth_nmi, 0.0)

        return float(np.linalg.norm(real_nmi - synth_nmi, ord='fro'))

    @staticmethod
    def compute_wasserstein_optuna(real_num: np.ndarray, synth_num: np.ndarray) -> float:
        """Mean Wasserstein distance across all continuous dimensions."""
        if real_num.shape[1] == 0: return 0.0
        return float(np.mean([wasserstein_distance(real_num[:, i], synth_num[:, i]) for i in range(real_num.shape[1])]))

    @staticmethod
    def compute_jensenshannon_optuna(
            real_cat: np.ndarray, synth_cat: np.ndarray, cardinalities: List[int],
            trial: Optional[optuna.Trial] = None, seed: int = 5, ds_name: str = "",
            device: str = "cpu", penalty_weight: float = 1.0
    ) -> float:
        """Mean Jensen-Shannon divergence across discrete dimensions with out-of-bounds penalty."""
        if real_cat.shape[1] == 0:
            return 0.0

        js_list = []
        total_penalty = 0.0

        for i, c in enumerate(cardinalities):
            real_col = real_cat[:, i].astype(int)
            synth_col = synth_cat[:, i].astype(int)

            real_col_valid = real_col[real_col >= 0]

            out_of_bounds = (synth_col < 0) | (synth_col >= c)
            invalid_count = out_of_bounds.sum()

            if invalid_count > 0:
                out_of_bounds_ratio = invalid_count / len(synth_col)

                total_penalty += out_of_bounds_ratio * penalty_weight

                synth_col = np.clip(synth_col, 0, c - 1)

                if out_of_bounds_ratio > 0.05:
                    msg = f"Col {i}: {out_of_bounds_ratio:.1%} values out of range [0, {c - 1}]. Clipped."
                    Metrics.log_trial_error(trial, error_msg=msg, extra={"seed": seed, "dataset": ds_name, "device": device})

            if len(real_col_valid) > 0:
                p = np.bincount(real_col_valid, minlength=c)
            else:
                p = np.zeros(c)

            q = np.bincount(synth_col, minlength=c)
            p = p / (p.sum() + 1e-12)
            q = q / (q.sum() + 1e-12)
            js_list.append(jensenshannon(p, q))

        return float(np.mean(js_list) + total_penalty)

    @staticmethod
    def log_trial_error(trial: optuna.trial.Trial, error_msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Logs pruned trial configurations to a CSV file."""
        params = dict(trial.params)
        if 'fb_sequence' not in params and (imf_len := params.get('imf_len')) is not None:
            params['fb_sequence'] = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

        if extra:
            params.update(extra)
        params['error'] = error_msg

        file_exists = Path("pruned_trials_params.csv").is_file()
        pd.DataFrame([params]).to_csv("pruned_trials_params.csv", mode='a', header=not file_exists, index=False)
