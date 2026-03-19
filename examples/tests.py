# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import chi2_contingency
# from sklearn.preprocessing import LabelEncoder
# import torch
# from torch.utils.data import DataLoader, TensorDataset
#
# from sbtab.bridge.losses import EfficientCSBMLoss
# from sbtab.bridge.pathsampler import DiscretePathSampler
# from sbtab.bridge.reference import CategoricalReference
# from sbtab.bridge.timegrid import TimeGrid
# from sbtab.solvers.csbm import CSBMSolver, CSBMUpdater
# from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
#
# from scipy.stats import entropy, wasserstein_distance
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
#
#
# def calculate_advanced_metrics(real_df, synth_df):
#     metrics = {}
#     eps = 1e-10
#
#     # 1. Mean KL Divergence
#     kl_values = []
#     for col in real_df.columns:
#         r_counts = real_df[col].value_counts(normalize=True).sort_index()
#         s_counts = synth_df[col].value_counts(normalize=True).sort_index()
#
#         all_cats = sorted(set(r_counts.index) | set(s_counts.index))
#         p = np.array([r_counts.get(c, 0) for c in all_cats]) + eps
#         q = np.array([s_counts.get(c, 0) for c in all_cats]) + eps
#
#         p /= p.sum()
#         q /= q.sum()
#
#         kl_values.append(entropy(p, q))
#
#     metrics['Mean KL'] = np.mean(kl_values)
#
#     wd_values = [wasserstein_distance(real_df[col], synth_df[col]) for col in real_df.columns]
#     metrics['Mean WD'] = np.mean(wd_values)
#
#     real_corr = calculate_association_matrix(real_df)
#     synth_corr = calculate_association_matrix(synth_df)
#     metrics['Corr distance'] = np.linalg.norm(real_corr - synth_corr)
#
#     target_idx = 0
#     X_r, y_r = real_df.drop(columns=[real_df.columns[target_idx]]), real_df.iloc[:, target_idx]
#     X_s, y_s = synth_df.drop(columns=[synth_df.columns[target_idx]]), synth_df.iloc[:, target_idx]
#
#     rf_real = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42).fit(X_r, y_r)
#     r2_real = r2_score(y_r, rf_real.predict(X_r))
#
#     rf_synth = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42).fit(X_s, y_s)
#     r2_synth = r2_score(y_r, rf_synth.predict(X_r))
#
#     metrics['R2_real_raw'] = r2_real
#     metrics['R2_synth_raw'] = r2_synth
#     metrics['R2_abs_diff'] = abs(r2_real - r2_synth)
#
#     return metrics
#
# def cramers_v(x, y):
#     confusion_matrix = pd.crosstab(x, y)
#     if confusion_matrix.empty: return 0
#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     phi2 = chi2 / n
#     r, k = confusion_matrix.shape
#     phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
#     rcorr = r - ((r - 1) ** 2) / (n - 1)
#     kcorr = k - ((k - 1) ** 2) / (n - 1)
#     if min((kcorr - 1), (rcorr - 1)) <= 0: return 0
#     return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
#
#
# def calculate_association_matrix(df):
#     cols = df.columns
#     matrix = np.zeros((len(cols), len(cols)))
#     for i in range(len(cols)):
#         for j in range(i, len(cols)):
#             val = cramers_v(df.iloc[:, i], df.iloc[:, j])
#             matrix[i, j] = val
#             matrix[j, i] = val
#     return matrix
#
#
# def validate_results(real_df, synth_df, feature_names):
#     real_df = pd.DataFrame(real_df, columns=feature_names)
#     synth_df = pd.DataFrame(synth_df, columns=feature_names)
#
#     print("\n" + "=" * 30)
#     print("Validation")
#     print("=" * 30)
#
#     print("Calculating association matrices...")
#     real_corr = calculate_association_matrix(real_df)
#     synth_corr = calculate_association_matrix(synth_df)
#
#     frob_norm = np.linalg.norm(real_corr - synth_corr, ord='fro')
#     print(f"Frobenius Norm of Correlation Difference: {frob_norm:.4f}")
#
#     cols_to_plot = feature_names[:]
#     fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3 * len(cols_to_plot)))
#
#     for i, col in enumerate(cols_to_plot):
#         r_counts = real_df[col].value_counts(normalize=True).sort_index()
#         s_counts = synth_df[col].value_counts(normalize=True).sort_index()
#
#         combined = pd.DataFrame({
#             'Val': r_counts.index.tolist() + s_counts.index.tolist(),
#             'Prob': r_counts.values.tolist() + s_counts.values.tolist(),
#             'Type': ['Real'] * len(r_counts) + ['Synth'] * len(s_counts)
#         })
#         sns.barplot(data=combined, x='Val', y='Prob', hue='Type', ax=axes[i])
#         axes[i].set_title(f"Distribution: {col}")
#
#     plt.tight_layout()
#     plt.show()
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#     sns.heatmap(real_corr, ax=ax1, cmap='magma', vmin=0, vmax=1)
#     ax1.set_title("Real Associations")
#     sns.heatmap(synth_corr, ax=ax2, cmap='magma', vmin=0, vmax=1)
#     ax2.set_title("Synthetic Associations")
#     plt.show()
#
# def prepare_mushroom_data(path):
#     df = pd.read_csv(path)
#     if 'veil-type' in df.columns: df = df.drop(columns=['veil-type'])
#     target_col = 'class'
#     ordered_cols = ['ring-number', 'gill-spacing', 'gill-size']
#     feature_cols = [c for c in df.columns if c != target_col]
#
#     for col in feature_cols:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#
#     x_p0 = df[df[target_col] == 'e'].drop(columns=[target_col]).values
#     if len(x_p0) == 0:
#         x_p0 = df[df[target_col] == 0].drop(columns=[target_col]).values
#         x_p1 = df[df[target_col] == 1].drop(columns=[target_col]).values
#     else:
#         x_p1 = df[df[target_col] == 'p'].drop(columns=[target_col]).values
#
#     is_ordered_mask = torch.tensor([c in ordered_cols for c in feature_cols])
#     cardinalities = [df[c].nunique() for c in feature_cols]
#
#     return (torch.tensor(x_p0, dtype=torch.long),
#             torch.tensor(x_p1, dtype=torch.long),
#             cardinalities, is_ordered_mask, feature_cols)
#
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x_p0, x_p1, cardinalities, is_ordered, feature_names = prepare_mushroom_data("mushrooms.csv")
#
#     dl_p0 = DataLoader(TensorDataset(x_p0), batch_size=256, shuffle=True)
#     dl_p1 = DataLoader(TensorDataset(x_p1), batch_size=256, shuffle=True)
#
#     ref = CategoricalReference(cardinalities, is_ordered, alpha=0.02, device=device)
#     model_fwd = CSBMTableMLP(cardinalities).to(device)
#     model_bwd = CSBMTableMLP(cardinalities).to(device)
#
#     updater = CSBMUpdater(
#         forward_model=model_fwd, backward_model=model_bwd,
#         forward_opt=torch.optim.AdamW(model_fwd.parameters(), lr=1e-3),
#         backward_opt=torch.optim.AdamW(model_bwd.parameters(), lr=1e-3),
#         ref_process=ref, loss_fn=EfficientCSBMLoss(lmbda=0.01)
#     )
#
#     timegrid = TimeGrid(num_steps=20)
#     sampler = DiscretePathSampler(timegrid=timegrid, reference=ref)
#     solver = CSBMSolver(updater=updater, sampler=sampler, num_outer_iterations=20)
#
#     solver.fit(dl_p1, dl_p0)
#
#     print("\nStarting synthesis for validation...")
#     inference_sampler = DiscretePathSampler(forward_model=model_fwd, reference=ref, timegrid=timegrid)
#
#     x_start = x_p0[:1000].to(device)
#     with torch.no_grad():
#         x_synth = inference_sampler.sample(x_start)
#
#     real_sample = x_p1[:1000].cpu().numpy()
#     synth_sample = x_synth.cpu().numpy()
#
#     validate_results(real_sample, synth_sample, feature_names)
#     real_df = pd.DataFrame(real_sample, columns=feature_names)
#     synth_df = pd.DataFrame(synth_sample, columns=feature_names)
#
#     print("\n" + "=" * 30)
#     print("Metrics")
#     print("=" * 30)
#
#     adv_metrics = calculate_advanced_metrics(real_df, synth_df)
#
#     print(f"{'Metric':<20} | {'Value':<10}")
#     print("-" * 35)
#     for name, value in adv_metrics.items():
#         print(f"{name:<20} | {value:.4f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import chi2_contingency, entropy, wasserstein_distance
from sklearn.preprocessing import LabelEncoder

from sbtab.bridge.losses import EfficientCSBMLoss
from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
from sbtab.solvers.csbm import CSBMUpdater, CSBMSolver


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty: return 0
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    if min((kcorr - 1), (rcorr - 1)) <= 0: return 0
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def calculate_association_matrix(df):
    cols = df.columns
    matrix = np.zeros((len(cols), len(cols)))
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            val = cramers_v(df.iloc[:, i], df.iloc[:, j])
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix


def calculate_advanced_metrics(real_df, synth_df):
    metrics = {}
    eps = 1e-10
    kl_values = []
    for col in real_df.columns:
        r_counts = real_df[col].value_counts(normalize=True).sort_index()
        s_counts = synth_df[col].value_counts(normalize=True).sort_index()
        all_cats = sorted(set(r_counts.index) | set(s_counts.index))
        p = np.array([r_counts.get(c, 0) for c in all_cats]) + eps
        q = np.array([s_counts.get(c, 0) for c in all_cats]) + eps
        p /= p.sum()
        q /= q.sum()
        kl_values.append(entropy(p, q))
    metrics['Mean KL'] = np.mean(kl_values)
    metrics['Mean WD'] = np.mean([wasserstein_distance(real_df[col], synth_df[col]) for col in real_df.columns])
    real_corr = calculate_association_matrix(real_df)
    synth_corr = calculate_association_matrix(synth_df)
    metrics['Corr distance'] = np.linalg.norm(real_corr - synth_corr)
    return metrics


def validate_results(real_df, synth_df):
    real_corr = calculate_association_matrix(real_df)
    synth_corr = calculate_association_matrix(synth_df)
    print(f"Frobenius Norm of Correlation Difference: {np.linalg.norm(real_corr - synth_corr):.4f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(real_corr, ax=ax1, cmap='magma', vmin=0, vmax=1)
    ax1.set_title("Real Associations")
    sns.heatmap(synth_corr, ax=ax2, cmap='magma', vmin=0, vmax=1)
    ax2.set_title("Synthetic Associations")
    plt.show()


def prepare_mushroom_data(path):
    df = pd.read_csv(path)
    if 'veil-type' in df.columns: df = df.drop(columns=['veil-type'])
    feature_cols = [c for c in df.columns if c != 'class']
    ordered_cols = ['ring-number', 'gill-spacing', 'gill-size']
    for col in feature_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    is_ordered_mask = torch.tensor([c in ordered_cols for c in feature_cols])
    cardinalities = [df[c].nunique() for c in feature_cols]
    return torch.tensor(df[feature_cols].values, dtype=torch.long), cardinalities, is_ordered_mask, feature_cols


def create_noise_dataset(num_samples, cardinalities, device):
    noise_data = []
    for card in cardinalities:
        noise_data.append(torch.randint(0, card, (num_samples,), device=device))
    return torch.stack(noise_data, dim=1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_real, cardinalities, is_ordered, feature_names = prepare_mushroom_data("mushrooms.csv")
    dl_p1 = DataLoader(TensorDataset(x_real.to(device)), batch_size=256, shuffle=True)

    x_noise = create_noise_dataset(len(x_real), cardinalities, device)
    dl_p0 = DataLoader(TensorDataset(x_noise), batch_size=256, shuffle=True)

    ref = CategoricalReference(cardinalities, is_ordered, alpha=0.001, device=device)
    model_fwd = CSBMTableMLP(cardinalities).to(device)
    model_bwd = CSBMTableMLP(cardinalities).to(device)

    updater = CSBMUpdater(
        forward_model=model_fwd, backward_model=model_bwd,
        forward_opt=torch.optim.AdamW(model_fwd.parameters(), lr=1e-3),
        backward_opt=torch.optim.AdamW(model_bwd.parameters(), lr=1e-3),
        ref_process=ref, loss_fn=EfficientCSBMLoss(lmbda=0.2)
    )

    timegrid = TimeGrid(num_steps=30)
    sampler = DiscretePathSampler(timegrid=timegrid, reference=ref)
    solver = CSBMSolver(updater=updater, sampler=sampler, num_outer_iterations=15)

    solver.fit(dl_p1, dl_p0)

    print("\nGenerating synthetic data from pure noise...")
    num_gen = 2000
    z_noise = create_noise_dataset(num_gen, cardinalities, device)

    synth_data, _ = sampler.simulate(
        x_init=z_noise,
        model=model_fwd,
        direction="forward"
    )

    real_sample_np = x_real[:num_gen].cpu().numpy()
    synth_sample_np = synth_data.cpu().numpy()

    real_df = pd.DataFrame(real_sample_np, columns=feature_names)
    synth_df = pd.DataFrame(synth_sample_np, columns=feature_names)

    validate_results(real_df, synth_df, feature_names)

    print("\n" + "=" * 30)
    print("Metrics")
    print("=" * 30)
    adv_metrics = calculate_advanced_metrics(real_df, synth_df)
    for name, value in adv_metrics.items():
        print(f"{name:<20} | {value:.4f}")