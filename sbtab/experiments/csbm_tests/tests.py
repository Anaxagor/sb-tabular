import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import chi2_contingency, entropy, wasserstein_distance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sbtab.bridge.losses import CSBMLoss
from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
from sbtab.solvers.csbm import CSBMUpdater, CSBMSolver

import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    real_corr = real_df.corr(method='spearman').fillna(0).values
    synth_corr = synth_df.corr(method='spearman').fillna(0).values
    metrics['Corr spearman dist'] = np.linalg.norm(real_corr - synth_corr)
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


def plot_distribution_comparison(real_df, synth_df, feature_names):
    num_features = len(feature_names)
    cols = 4
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4.2 * rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_names):
        ax = axes[i]

        combined = pd.DataFrame({
            'Value': pd.concat([real_df[col], synth_df[col]]),
            'Type': ['Real'] * len(real_df) + ['Synthetic'] * len(synth_df)
        })

        sns.histplot(
            data=combined, x='Value', hue='Type',
            multiple='dodge', shrink=0.8, ax=ax,
            stat='probability', common_norm=False,
            palette=['#2874A6', '#E67E22'],
            legend=False,
            edgecolor='white',
            linewidth=0.5
        )

        ax.set_title(col, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('Prob' if i % cols == 0 else '', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.yaxis.grid(True, linestyle=':', alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    sns.despine()

    real_patch = mpatches.Patch(color='#2874A6', label='Real Data (Test)')
    synth_patch = mpatches.Patch(color='#E67E22', label='Synthetic Data')

    fig.legend(handles=[real_patch, synth_patch],
               loc='lower center',
               bbox_to_anchor=(0.5, 0.96),
               ncol=2, fontsize=15, frameon=False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    plt.subplots_adjust(hspace=0.6, wspace=0.25)

    plt.show()
if __name__ == "__main__":
    seed = 42
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    num_of_powers = 40
    alpha = 0.05873991654294587
    lr = 0.00541389036125277
    lmbda = 0.06782077036048736
    outer_iterations = 20
    epochs = 4
    emb_dim = 32
    time_dim = 256
    hidden_dim = 512
    scheduler_outer_iterations=4
    scheduler_alpha_multiplier=0.928013890561499
    weight_decay = 0.008775932695733348
    test_size=0.2

    x_all, cardinalities, is_ordered, feature_names = prepare_mushroom_data("mushrooms.csv")

    x_train, x_test = train_test_split(x_all.numpy(), test_size=test_size, random_state=seed)
    x_train = torch.tensor(x_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.long)

    dl_p1 = DataLoader(TensorDataset(x_train.to(device)), batch_size=batch_size, shuffle=True, generator=g)

    x_noise = create_noise_dataset(len(x_train), cardinalities, device)
    dl_p0 = DataLoader(TensorDataset(x_noise), batch_size=batch_size, shuffle=True, generator=g)

    ref = CategoricalReference(cardinalities, is_ordered, alpha=alpha, device=device, total_number_of_q_powers=num_of_powers)
    model_fwd = CSBMTableMLP(cardinalities, emb_dim, hidden_dim, time_dim).to(device)
    model_bwd = CSBMTableMLP(cardinalities, emb_dim, hidden_dim, time_dim).to(device)


    updater = CSBMUpdater(
        forward_model=model_fwd, backward_model=model_bwd,
        forward_opt=torch.optim.AdamW(model_fwd.parameters(), lr=lr, weight_decay=weight_decay),
        backward_opt=torch.optim.AdamW(model_bwd.parameters(), lr=lr, weight_decay=weight_decay),
        ref_process=ref, loss_fn=CSBMLoss(lmbda=lmbda, reference=ref)
    )

    timegrid = TimeGrid(num_steps=num_of_powers)
    sampler = DiscretePathSampler(timegrid=timegrid, reference=ref)
    solver = CSBMSolver(updater=updater, sampler=sampler, num_outer_iterations=outer_iterations, epochs=epochs, batch_size=batch_size)

    solver.fit(dl_p1, dl_p0, scheduler_outer_iterations=scheduler_outer_iterations, scheduler_alpha_multiplier=scheduler_alpha_multiplier)

    print("\nGenerating synthetic data from pure noise...")
    num_gen = len(x_test)
    z_noise = create_noise_dataset(num_gen, cardinalities, device)

    synth_data, _ = sampler.simulate(
        x_init=z_noise,
        model=model_fwd,
        direction="forward"
    )

    real_sample_np = x_test.cpu().numpy()
    synth_sample_np = synth_data.cpu().numpy()

    real_df = pd.DataFrame(real_sample_np, columns=feature_names)
    synth_df = pd.DataFrame(synth_sample_np, columns=feature_names)
    validate_results(real_df, synth_df)

    print("\nPlotting feature distributions...")
    plot_distribution_comparison(real_df, synth_df, feature_names)

    print("\n" + "=" * 30)
    print("Metrics")
    print("=" * 30)
    adv_metrics = calculate_advanced_metrics(real_df, synth_df)
    for name, value in adv_metrics.items():
        print(f"{name:<20} | {value:.4f}")
