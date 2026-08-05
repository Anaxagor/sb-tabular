# sb-tabular

A research framework for **synthetic tabular data generation with Schrödinger Bridges (SB)**.

The repository implements several SB solver families under one data pipeline and compares them
against strong non-SB generative baselines (CTGAN, TabDDPM, STaSy, TabPFGen) with a unified
k-fold evaluation protocol.

---

## Research logic

The core question is how to best adapt Schrödinger Bridge / bridge-matching methods to tabular
data. All SB solvers transport samples between the data distribution and a simple reference
distribution (Gaussian for numeric columns, a discrete-diffusion bridge for categorical ones),
and generation runs the learned backward dynamics from the reference.

The main body of solvers is organized along **three orthogonal design axes**:

| Axis | Options |
|---|---|
| **Training algorithm** | **IPF-DSB** — Iterative Proportional Fitting with cache-based drift regression (Diffusion Schrödinger Bridge); **IMF-DSBM** — Iterative Markovian Fitting with bridge matching (DSBM) |
| **Time parameterization** | **Continuous-time** — one field model conditioned on scalar `t`; **Discrete-time** — a separate model per time-grid step |
| **Dependency structure** | **Joint** — one field over the full feature vector; **Feature-wise (structural)** — a DAG over features is learned with `pgmpy` (Hill-Climb + BIC), and per-feature scalar fields are conditioned on DAG parents (autoregressive generation) |

Each combination can be backed by either a **neural network** (time-conditioned MLP) or
**gradient boosting** (CatBoost) — one of the research goals is testing whether boosted models
can replace neural drift estimators on tabular data.

On top of that grid, three standalone solvers cover other points of the design space:

- **LightSB** — light Schrödinger Bridge with a Gaussian-mixture parameterization of the
  adjusted Schrödinger potential (fast, simulation-free training; continuous data).
- **CSBM** — Categorical Schrödinger Bridge Matching for purely categorical tables, built on a
  discrete-diffusion `CategoricalReference` bridge.
- **MixedSBM** — a mixed-type SBM: a single network predicts the continuous drift and
  per-categorical-column logits simultaneously, combining the Gaussian and categorical
  reference processes. The only solver that handles mixed tables natively end-to-end.

### Data pipeline

Every model (SB solver or baseline) sits behind the same pipeline:

1. `TabularSchema` — classifies columns as continuous / discrete / categorical
   (`infer_from_dataframe`) and validates the raw table.
2. `TransformPipeline` — missing-value handling (drop or type-aware impute), standard scaling
   for continuous columns, one-hot or integer encoding for categoricals. Global-safe steps run
   before splitting; everything stateful is **fit on the training subset of each split only**
   (no leakage), and pipelines are invertible so samples map back to the original scale.
3. `TabularDataModule` — provides holdout or k-fold splits with fold-wise refitted transforms.
4. The solver trains in the transformed numeric space and samples via the bridge primitives in
   `sbtab/bridge/` (time grid, reference processes, Euler–Maruyama integrator, path samplers).
5. Samples are inverse-transformed back to a raw-scale DataFrame and evaluated.

---

## Repository layout

```text
sb-tabular/
├── README.md
├── requirements.txt                   # Dependencies, grouped by purpose
├── examples/                          # Runnable end-to-end demos (see table below)
└── sbtab/
    ├── data/
    │   ├── schema.py                  # TabularSchema + feature-type inference
    │   ├── splits.py                  # K-fold and holdout split protocols
    │   ├── datamodule.py              # TabularDataModule (leakage-safe fold-wise transforms)
    │   ├── get_datasets.py            # Downloads + pickles benchmark dataset bundles
    │   └── datasets/                  # Pickled bundles: continuous / categorical / mixed
    │
    ├── transforms/
    │   ├── base.py                    # BaseTransform protocol + state (de)serialization
    │   ├── missing.py                 # DropMissingRows, TypeAwareImputer
    │   ├── continuous.py              # ContinuousStandardScaler
    │   ├── categorical.py             # One-hot / integer categorical representations
    │   └── pipeline.py                # TransformPipeline + default_* factory pipelines
    │
    ├── bridge/                        # Solver-agnostic SB primitives
    │   ├── timegrid.py                # TimeGrid (linear/geometric γ schedules)
    │   ├── reference.py               # GaussianReference, CategoricalReference
    │   ├── sde.py                     # Euler–Maruyama integrator
    │   ├── pathsampler.py             # PathSampler / DiscretePathSampler / MixedPathSampler
    │   └── losses.py                  # RegressionLoss, CSBMLoss, MixedSBMLoss
    │
    ├── models/
    │   ├── neural/                    # TimeConditionedMLP, per-step MLP fields (joint/scalar),
    │   │                              # CSBMTableMLP, MixedSbmMlp, NeuralTrainer, time embedding
    │   ├── boosted/                   # CatBoost fields: {continuous,discrete} × {joint,scalar}
    │   └── sb/                        # LightSBPotential (Gaussian-mixture potential)
    │
    ├── solvers/
    │   ├── continuous_time/
    │   │   ├── joint_distribution/{mlp,boosting}/{ipf_dsb,imf_dsbm}/
    │   │   └── feature_wise/boosting/ipf_dsb/
    │   ├── discrete_time/
    │   │   ├── joint_distribution/{mlp,boosting}/{ipf_dsb,imf_dsbm*}/
    │   │   └── feature_wise/boosting/{ipf_dsb,imf_dsbm_featurewise_boost}/
    │   ├── light_sb/                  # LightSBSolver
    │   ├── csbm/                      # CSBMSolver (categorical data)
    │   └── msbm/                      # MixedSBMSolver (mixed data)
    │
    ├── baselines/                     # Non-SB baselines under one fit/sample API
    │   ├── base.py                    # BaselineGenerativeModel ABC
    │   ├── ctgan/                     # SDV CTGANSynthesizer wrapper
    │   ├── tabddpm/                   # Vendored TabDDPM (Gaussian+multinomial diffusion)
    │   ├── stasy/                     # STaSy score-based SDE (self-paced training)
    │   └── tabpfn/                    # TabPFGen wrapper (SGLD-based generation)
    │
    ├── experiments/
    │   ├── *_metrics.py               # K-fold evaluation of the boosted joint/structural solvers
    │   ├── tuning_script/             # Optuna tuning (IPF-DSB, DSBM, LightSB, CTGAN, TabDDPM)
    │   ├── calculating_metrics/       # K-fold evaluation drivers per model + saved results
    │   └── visualization/             # Radar charts / average-rank comparison notebook
    │
    └── evaluation/                    # Metric utilities (being consolidated here)
```

## Solvers at a glance

| Solver | Time | Structure | Backend | Data types |
|---|---|---|---|---|
| `continuous_time/joint_distribution/mlp/ipf_dsb` — `IPFDSBSolver` | continuous | joint | MLP | continuous |
| `continuous_time/joint_distribution/mlp/imf_dsbm` — `IMFDSBMSolver` | continuous | joint | MLP | continuous |
| `continuous_time/joint_distribution/boosting/ipf_dsb` — `JointContinuousBoostedSolver` | continuous | joint | CatBoost | continuous |
| `continuous_time/joint_distribution/boosting/imf_dsbm` — `IMFDSBMContinuousJointCatBoostSolver` | continuous | joint | CatBoost | continuous |
| `continuous_time/feature_wise/boosting/ipf_dsb` — `StructuralContinuousBoostedSolver` | continuous | feature-wise (DAG) | CatBoost | continuous |
| `discrete_time/joint_distribution/mlp/ipf_dsb` — `IPFDSBSolver` | discrete | joint | MLP | continuous |
| `discrete_time/joint_distribution/mlp/imf_dsbm` — `IMFDSBMDiscreteJointMLPSolver` | discrete | joint | MLP | continuous |
| `discrete_time/joint_distribution/boosting/ipf_dsb` — `JointDiscreteBoostedSolver` | discrete | joint | CatBoost (per step) | continuous |
| `discrete_time/joint_distribution/boosting/imf_dsbm_boost` — `IMFDSBMBoostSolver` | discrete | joint | CatBoost (per step) | continuous |
| `discrete_time/feature_wise/boosting/ipf_dsb` — `StructuralDiscreteBoostedSolver` | discrete | feature-wise (DAG) | CatBoost | continuous |
| `discrete_time/feature_wise/boosting/imf_dsbm_featurewise_boost` — `FeaturewiseDSBMBoostSolver` | discrete | feature-wise (AR) | CatBoost | continuous |
| `light_sb` — `LightSBSolver` | — (simulation-free) | joint | Gaussian-mixture potential | continuous |
| `csbm` — `CSBMSolver` | discrete | joint | MLP (embeddings + masked logits) | categorical |
| `msbm` — `MixedSBMSolver` | discrete | joint | MLP (drift + logit heads) | mixed |

## Baselines

All baselines implement `BaselineGenerativeModel` (`fit(data)` / `sample(n, seed)`) from
`sbtab/baselines/base.py`:

- **CTGAN** — wrapper over SDV's `CTGANSynthesizer`, schema- and transform-aware.
- **TabDDPM** — vendored implementation (Gaussian + multinomial diffusion, EMA, LR annealing).
- **STaSy** — score-based SDE with self-paced training and predictor–corrector sampling.
- **TabPFGen** — TabPFN-based generation via SGLD (wraps `sebhaan/TabPFGen`).

## Datasets

`sbtab/data/get_datasets.py` downloads and pickles benchmark bundles into `sbtab/data/datasets/`:

- **Continuous** (`datasets_continuous_only.pkl`, 9 datasets): California Housing, Diabetes,
  Online News Popularity, King County Housing, Bank Loan, Bank Marketing, Online Shoppers,
  Covertype, German Credit. This is the bundle used by all tuning and k-fold evaluation scripts.
- **Categorical** (`datasets_categorical.pkl`): Student Performance, Lymphography,
  Breast Cancer, Car Evaluation, Mushroom.
- **Mixed** (`datasets_mixed.pkl`): Adult, Credit Approval, Online Shoppers, Eucalyptus,
  Forest Fires.

## Evaluation protocol

`sbtab/experiments/` runs the same 5-fold protocol for every model
(`calculating_metrics/*.py` for the MLP solvers, LightSB and baselines;
`joint_*`/`structural_*_metrics.py` for the boosted solvers). Per fold, a model is trained on
the train split and its samples are compared with the held-out test split via:

- **avg_wd** — mean per-column 1-D Wasserstein distance;
- **avg_kl_hist** — mean histogram-based KL divergence;
- **corr_frobenius** — Frobenius norm of the difference of correlation matrices;
- **swd** — sliced Wasserstein distance over the joint distribution;
- **utility_delta_r2_percent** — TSTR-style utility gap: R² of a CatBoost regressor trained on
  real vs synthetic data.

Results are written as `<dataset>_fold_metrics.csv` + `<dataset>_kfold_summary.json`
(committed examples under `calculating_metrics/dsbm_kfold_eval/` and `tabpfgen_kfold_eval/`).
Hyperparameters are tuned per dataset with **Optuna** (TPE + median pruning) in
`experiments/tuning_script/`; the cross-model comparison (average-rank radar charts) lives in
`experiments/visualization/`.

## Quickstart

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing

from sbtab.data.schema import TabularSchema
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.splits import SplitConfigHoldout
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import (
    IMFDSBMConfig,
    IMFDSBMSolver,
)

# 1) Data + schema + transforms (fit on train only, invertible)
df = fetch_california_housing(as_frame=True).frame
schema = TabularSchema.infer_from_dataframe(df)
dm = TabularDataModule(
    df=df,
    schema=schema,
    transforms=TransformPipeline.default_dropna_and_scale(),
)
dm.prepare_holdout(SplitConfigHoldout(val_size=0.2, shuffle=True, random_seed=42))
holdout = dm.get_holdout()  # .train / .val are transformed; .transforms is the fitted pipeline

# 2) Train an SB solver (continuous-time joint MLP, IMF-DSBM)
cfg = IMFDSBMConfig(
    fb_sequence=("b", "f", "b", "f", "b"),
    inner_iters=2000,
    num_steps=1000,
    sigma=0.1,
    first_coupling="ref",
    device="cpu",
    seed=42,
)
model = IMFDSBMSolver(dim=holdout.train.shape[1], cfg=cfg)
model.fit(holdout.train)

# 3) Sample and map back to the original scale
x_synth = model.sample(n=len(holdout.val), seed=123)
synth_scaled = pd.DataFrame(x_synth, columns=holdout.train.columns)
synth_df = holdout.transforms.inverse_transform(synth_scaled)
print(synth_df.head())
```

### Examples

| Script | Demonstrates |
|---|---|
| `examples/California_Housing_example.py` | Continuous-time joint MLP IMF-DSBM on California Housing |
| `examples/joint_discrete_time_mlp_example.py` | Discrete-time joint MLP IMF-DSBM |
| `examples/joint_continuous_time_boost_example.py` | Continuous-time joint CatBoost IMF-DSBM |
| `examples/boosted_dsbm_example.py` | Discrete-time joint CatBoost IMF-DSBM |
| `examples/feature_wise_discrete_time_boosting-example.py` | Feature-wise (autoregressive) CatBoost DSBM |
| `examples/light_sb_example.py` | LightSB solver |
| `examples/tabddpm_example.ipynb` | TabDDPM baseline |

## Installation

The package is not on PyPI yet; clone the repository and install the dependencies:

```bash
git clone https://github.com/ITMO-NSS-team/sb-tabular.git
cd sb-tabular
pip install -r requirements.txt
```

`requirements.txt` is grouped by purpose, so you can trim it to what you need:

- **Core** (always required): `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `tqdm` —
  enough for the data pipeline, bridge primitives, and all MLP-based solvers.
- **Boosted solvers**: `catboost`.
- **Feature-wise (structural) solvers**: `pgmpy`, `networkx` (DAG learning).
- **Tuning scripts**: `optuna`.
- **Baselines**: `sdv` (CTGAN), `tabpfgen` (TabPFGen); TabDDPM and STaSy are self-contained.
- **Dataset download**: `ucimlrepo`.
- **Optional** (commented out): `geotorch` — only for LightSB with a full covariance
  (`is_diagonal=False`); `matplotlib` — visualization notebooks.

Python ≥ 3.10 is assumed. There is no `pyproject.toml` yet, so run scripts from the repository
root (or add it to `PYTHONPATH`) so that `import sbtab` resolves.

## Unified benchmark contract

The initial model-independent data boundary lives in `sbtab/benchmark/`. It defines explicit
raw column semantics and the canonical table exchanged with future model adapters. See
[`docs/benchmark-contract.md`](docs/benchmark-contract.md) for its scope and invariants.
The first concrete declaration, UCI Online Shoppers 468, has a separate
[`source and semantics note`](docs/datasets/online-shoppers.md).

Run its focused tests from the repository root:

```bash
python -m unittest \
  tests.benchmark.test_contracts \
  tests.benchmark.test_import_boundaries \
  tests.benchmark.test_online_shoppers
```

## Status and known gaps

- The evaluation utilities are being consolidated into `sbtab/evaluation/`; the metric
  implementations currently live inside the `sbtab/experiments/*_metrics.py` scripts.
- CSBM (categorical) and MixedSBM (mixed) are the newest solvers and do not yet have example
  scripts or committed benchmark results.
- There is no `pyproject.toml` or repository-wide test suite yet. The unified benchmark
  contract has focused tests and CI; dependencies are tracked in `requirements.txt`.
