import os
import random
import pandas as pd
import numpy as np
import torch
import optuna
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency, entropy, wasserstein_distance
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from sbtab.bridge.losses import CSBMLoss
from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
from sbtab.solvers.csbm import CSBMUpdater, CSBMSolver


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


def objective(trial):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_all, cardinalities, is_ordered, feature_names = prepare_mushroom_data("mushrooms.csv")
    x_train_np, x_test_np = train_test_split(x_all.numpy(), test_size=0.2, random_state=42)
    x_train = torch.tensor(x_train_np, dtype=torch.long).to(device)
    x_test = torch.tensor(x_test_np, dtype=torch.long).to(device)

    lr = trial.suggest_float("lr", 1e-3, 8e-3, log=True)
    lmbda = trial.suggest_float("lmbda", 0.01, 0.15, log=True)
    alpha = trial.suggest_float("alpha", 0.01, 0.1, log=True)
    alpha_mult = trial.suggest_float("alpha_mult", 0.8, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)

    batch_size = 256
    num_of_powers = 40
    outer_iterations = 20
    epochs = 4
    g = torch.Generator()
    g.manual_seed(42)

    # 3. Инициализация
    dl_p1 = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True, generator=g)
    x_noise_train = create_noise_dataset(len(x_train), cardinalities, device)
    dl_p0 = DataLoader(TensorDataset(x_noise_train), batch_size=batch_size, shuffle=True, generator=g)

    ref = CategoricalReference(cardinalities, is_ordered, alpha=alpha, device=device,
                               total_number_of_q_powers=num_of_powers)
    model_fwd = CSBMTableMLP(cardinalities, 32, 512, 256).to(device)
    model_bwd = CSBMTableMLP(cardinalities, 32, 512, 256).to(device)

    updater = CSBMUpdater(
        forward_model=model_fwd, backward_model=model_bwd,
        forward_opt=torch.optim.AdamW(model_fwd.parameters(), lr=lr, weight_decay=weight_decay),
        backward_opt=torch.optim.AdamW(model_bwd.parameters(), lr=lr, weight_decay=weight_decay),
        ref_process=ref, loss_fn=CSBMLoss(lmbda=lmbda, reference=ref)
    )

    timegrid = TimeGrid(num_steps=num_of_powers)
    sampler = DiscretePathSampler(timegrid=timegrid, reference=ref)

    solver = CSBMSolver(updater=updater, sampler=sampler, num_outer_iterations=outer_iterations, epochs=epochs, batch_size=batch_size)
    solver.fit(dl_p1, dl_p0, scheduler_alpha_multiplier=alpha_mult, scheduler_outer_iterations=4)
    try:
        with torch.no_grad():
            z_noise_val = create_noise_dataset(len(x_test), cardinalities, device)
            synth_data, _ = sampler.simulate(x_init=z_noise_val, model=model_fwd, direction="forward")

            val_df = pd.DataFrame(x_test.cpu().numpy(), columns=feature_names)
            synth_df = pd.DataFrame(synth_data.cpu().numpy(), columns=feature_names)

        final_metrics = calculate_advanced_metrics(val_df, synth_df)
        for k, v in final_metrics.items():
            trial.set_user_attr(k, v)

        return final_metrics['Corr distance']

    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


if __name__ == "__main__":
    os.makedirs("optuna_results", exist_ok=True)

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(objective, n_trials=100)

    print(f"\nЛучшая дистанция (валидация): {study.best_value:.4f}")
    print("Лучшие параметры:", study.best_params)