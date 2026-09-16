from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import normalized_mutual_info_score, r2_score, f1_score
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

    @staticmethod
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

        X_real, y_real = Metrics.align_x_y(train_real.drop(columns=[target_col]), train_real[target_col])
        X_synth, y_synth = Metrics.align_x_y(train_synth.drop(columns=[target_col]), train_synth[target_col])
        X_test, y_test = Metrics.align_x_y(test_real.drop(columns=[target_col]), test_real[target_col])

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

    @staticmethod
    def align_x_y(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Drops NaNs synchronously from features and target."""
        combined = pd.concat([X, y.to_frame('__target__')], axis=1).dropna()
        return combined.drop('__target__', axis=1), combined['__target__']

    @staticmethod
    def _mi_pair(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> float:
        """Plug-in mutual information (nats) between two integer code arrays.

        The joint support is fixed to kx * ky cells (bincount with minlength),
        so the plug-in bias is identical for any two samples evaluated on the
        same grid — and therefore cancels in real-vs-synth differences.

        Args:
            x: Integer codes of the first variable, values in [0, kx).
            y: Integer codes of the second variable, values in [0, ky).
            kx: Support size of the first variable (e.g. its cardinality).
            ky: Support size of the second variable (e.g. its cardinality).

        Returns:
            Mutual information estimate in nats.
        """
        flat = x.astype(np.int64) * ky + y.astype(np.int64)
        pxy = np.bincount(flat, minlength=kx * ky).astype(np.float64).reshape(kx, ky)
        pxy = pxy / pxy.sum()
        px, py = pxy.sum(axis=1, keepdims=True), pxy.sum(axis=0, keepdims=True)
        mask = pxy > 0
        return float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py)[mask])))

    @staticmethod
    def pairwise_mi_error_codes(
            real: np.ndarray,
            synth: np.ndarray,
            cardinalities: List[int]
    ) -> float:
        """Relative pairwise-MI error between real and synthetic discrete samples.

        For every pair of discrete columns, computes the absolute difference of
        plug-in mutual information (nats) between real and synthetic data, and
        normalizes it by the mean real MI over all pairs:

            sum_pairs |MI_real - MI_synth| / (mean_pairs MI_real + eps)

        Scale-free (comparable across datasets) and insensitive to pairs that
        carry almost no dependence. Codes outside [0, cardinality) are excluded
        per pair. Since both real and synth are evaluated on the same
        cardinality-fixed grid with (approximately) equal n, the plug-in bias
        cancels in the difference.

        Args:
            real: Real discrete codes, shape [N, D]; -1 marks unseen categories.
            synth: Synthetic discrete codes, shape [M, D]; -1 marks unseen categories.
            cardinalities: Support sizes per column, length D. Fixes the joint grid
                for both samples; do not infer it from the data, otherwise real and
                synth may be evaluated on different grids and become incomparable.

        Returns:
            Relative pairwise-MI error (lower is better, 0 = perfect joint match).
            Returns 0.0 when there are fewer than 2 columns or too few valid rows.
        """
        real = np.asarray(real, dtype=np.int64)
        synth = np.asarray(synth, dtype=np.int64)
        if real.ndim != 2 or synth.ndim != 2 or real.shape[1] != synth.shape[1]:
            return 0.0
        if real.shape[1] < 2 or len(real) < 10 or len(synth) < 10:
            return 0.0

        abs_errs, real_mis = [], []
        for i in range(real.shape[1]):
            for j in range(i + 1, real.shape[1]):
                ci, cj = int(cardinalities[i]), int(cardinalities[j])
                mr = (real[:, i] >= 0) & (real[:, i] < ci) & (real[:, j] >= 0) & (real[:, j] < cj)
                ms = (synth[:, i] >= 0) & (synth[:, i] < ci) & (synth[:, j] >= 0) & (synth[:, j] < cj)
                if mr.sum() < 10 or ms.sum() < 10:
                    continue
                mi_r = Metrics._mi_pair(real[mr, i], real[mr, j], ci, cj)
                mi_s = Metrics._mi_pair(synth[ms, i], synth[ms, j], ci, cj)
                abs_errs.append(abs(mi_r - mi_s))
                real_mis.append(mi_r)

        if not abs_errs:
            return 0.0
        return float(np.sum(abs_errs) / (np.mean(real_mis) + 1e-8))

    @staticmethod
    def pairwise_mi_error(real: pd.DataFrame, synth: pd.DataFrame, cat_cols: List[str]) -> float:
        """Relative pairwise-MI error for DataFrames (see pairwise_mi_error_codes).

        Category supports are factorized over the union of real+synth values per
        column, so both samples are embedded into one common grid per pair.

        Args:
            real: Real DataFrame.
            synth: Synthetic DataFrame.
            cat_cols: Discrete/categorical columns present in both frames.

        Returns:
            Relative pairwise-MI error (lower is better). Returns 0.0 when fewer
            than 2 valid columns or too few valid rows per pair.
        """
        cat_cols = [c for c in cat_cols if c in real.columns and c in synth.columns]
        if len(cat_cols) < 2:
            return 0.0

        abs_errs, real_mis = [], []
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                a_r, b_r = real[cat_cols[i]], real[cat_cols[j]]
                a_s, b_s = synth[cat_cols[i]], synth[cat_cols[j]]
                vr, vs = a_r.notna() & b_r.notna(), a_s.notna() & b_s.notna()
                a_r, b_r, a_s, b_s = a_r[vr], b_r[vr], a_s[vs], b_s[vs]
                if len(a_r) < 10 or len(a_s) < 10:
                    continue
                ka = pd.factorize(pd.concat([a_r, a_s], ignore_index=True).astype(str))[0]
                kb = pd.factorize(pd.concat([b_r, b_s], ignore_index=True).astype(str))[0]
                k = max(ka.max() + 1, kb.max() + 1)
                n_r = len(a_r)
                mi_r = Metrics._mi_pair(ka[:n_r], kb[:n_r], k, k)
                mi_s = Metrics._mi_pair(ka[n_r:], kb[n_r:], k, k)
                abs_errs.append(abs(mi_r - mi_s))
                real_mis.append(mi_r)

        if not abs_errs:
            return 0.0
        return float(np.sum(abs_errs) / (np.mean(real_mis) + 1e-8))