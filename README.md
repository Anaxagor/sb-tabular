# sb-tabular

A research framework for **synthetic tabular data generation with Schrödinger Bridges (SB)**.

The repository implements several SB solver families under one data pipeline and compares them
against non-SB generative baselines (CTGAN, TabDDPM, a simplified VE score-SDE, TabbyFlow and
ForestDiffusion) under one
versioned experimental protocol: separate, reproducible stages for splitting, tuning, cross-validation
and metric calculation, with every artifact saved locally.

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
  `CategoricalReference` Markov semigroup `Q(s) = exp(sR)` (uniform or ordered kernel).
- **MixedSBM** — a mixed-type SBM: a single network predicts the continuous drift and
  per-categorical-column logits simultaneously, combining the Gaussian and categorical
  reference processes. The SB solver that handles mixed tables natively end-to-end.

MixedSBM shares one network and optimizer across forward and backward training stages. Its
categorical reference uses exact transition powers and log-space bridge probabilities; continuous
and categorical paths have their own reference processes. CSBM uses a continuous-time categorical
Markov semigroup. MixedSBM checkpoints use `sbtab.mixedsbm/4`; earlier transition-law checkpoints
require retraining.

### Data pipeline

Every model (SB solver or baseline) sits behind the same pipeline:

1. **Explicit schema** — `configs/datasets/<name>.yaml` declares, per column, its role and type
   (`continuous` / `discrete` / `categorical`, optional ordinal order), the target, the task and the
   missing-value policy. Types are benchmark metadata, never inferred from a dtype or from the rows
   of a split. A classification target is categorical whatever its storage.
2. **Common preprocessing** (`sbtab/data/preprocessing.py`), fitted on the **current training rows
   only**: `StandardScaler` for continuous columns, no scaling or recoding for discrete columns, one
   label vocabulary per categorical column. A value outside a training vocabulary is an error — the
   vocabulary is never expanded by held-out data.
3. **Model adapter** (`sbtab/adapters/`) — `fit(train, schema, config, seed)`, `sample(n, seed)`,
   `save_checkpoint(path)`, `load_checkpoint(path)`. Continuous-only solvers see nominal columns
   one-hot and discrete columns standardised; native bridge solvers see finite states. Decoding is declared
   and measured (argmax; nearest *training-support* value with the rounding rate recorded). Continuous
   outputs receive no generic postprocessing that clips them to the training range. TabbyFlow's
   empirical quantile inverse is intrinsically bounded by its training range; this is part of its
   declared model representation. Configs are strict: an unknown key raises.
4. **Metrics** (`sbtab/evaluation/`) always read the common representation.

---

## Repository layout

```text
sb-tabular/
├── configs/
│   ├── protocols/                     # sbtab_8515_hpo100_cv5_v4 (default), sbtab_smoke_v4
│   ├── datasets/                      # explicit per-dataset schema metadata (28 datasets)
│   ├── search_spaces/                 # benchmark adapter profiles; smoke/ holds bounded variants
│   └── metrics/                       # metrics_v2.yaml (metric version sbtab.metrics/2)
├── docs/CLUSTER_EXPERIMENT.md         # deployment and run instructions
├── examples/                          # thin runnable demos built on the adapters
├── tests/                             # bridge / solvers / baselines / evaluation / experiments
└── sbtab/
    ├── data/                          # loading (cross-version bundles), registry, dataset_schema, preprocessing
    ├── bridge/                        # TimeGrid, Gaussian/Categorical references, SDE, path samplers, losses
    ├── models/                        # neural / boosted field models, LightSB potential
    ├── solvers/
    │   ├── registry.py                # solver_registry: stable ids, status, regimes
    │   ├── structure.py               # DAG learning shared by the structural solvers
    │   ├── continuous_time/ discrete_time/   # IPF-DSB and IMF-DSBM variants
    │   └── light_sb/ csbm/ msbm/ ForestDiffusion/
    ├── baselines/                     # ctgan, tabddpm, stasy, tabpfn, forest_diffusion, tabbyflow
    ├── adapters/                      # model adapters + reversible representations
    ├── evaluation/                    # THE metric implementation (tuning and evaluation share it)
    └── experiments/
        ├── experiment_common.py       # protocol loading, hashes/provenance, seed ledger, atomic I/O, timing
        ├── prepare_splits.py  tune.py  cross_validate.py  calculate_metrics.py  aggregate_results.py
        └── pipeline.py                # immutable plans, workers, locking and aggregation
```

`sbtab/data/{schema,splits,datamodule}.py` and `sbtab/transforms/` are the earlier pipeline; they are kept
as reusable compatibility APIs; the experiment stages use the explicit schema and common preprocessor.

## Registry

`sbtab.solvers.registry.solver_registry` is the source of truth; ids are stable.

| id | family | time | structure | backend | regimes (native → adapted) |
|---|---|---|---|---|---|
| `dsb_ct_joint_mlp` | IPF-DSB | time-conditioned | joint | MLP | continuous → discrete, mixed |
| `dsb_dt_joint_mlp` | IPF-DSB | per step | joint | MLP | continuous → discrete, mixed |
| `dsb_ct_joint_gbt` / `dsb_dt_joint_gbt` | IPF-DSB | time-conditioned / per step | joint | CatBoost | continuous → discrete, mixed |
| `dsb_ct_structural_gbt` / `dsb_dt_structural_gbt` | IPF-DSB | time-conditioned / per step | DAG learned on the fit rows | CatBoost + pgmpy | continuous → discrete, mixed |
| `dsbm_ct_joint_mlp` / `dsbm_ct_joint_gbt` | IMF-DSBM | time-conditioned | joint | MLP / CatBoost | continuous → discrete, mixed |
| `dsbm_dt_joint_mlp` / `dsbm_dt_joint_gbt` | IMF-DSBM | per step (state time) | joint | MLP / CatBoost | continuous → discrete, mixed |
| `dsbm_dt_structural_gbt` | IMF-DSBM | per step | autoregressive chain (optional map / learned DAG) | CatBoost | continuous → discrete, mixed |
| `lightsb` | LightSB | static potential | joint | Gaussian-mixture potential | continuous → discrete, mixed |
| `csbm` | CSBM / D-IMF | time-conditioned | joint, factorised head | MLP | discrete |
| `mixedsbm` | MixedSBM | time-conditioned | joint | MLP | continuous, discrete, mixed |
| `tabddpm` | baseline | diffusion steps | joint row (X, y) | torch | all |
| `ve_score_sde_simplified` | baseline | VE SDE | joint | torch | continuous → discrete, mixed |
| `ctgan` | baseline | – | joint | `sdv` | all |
| `tabbyflow` | baseline | flow matching | joint row (X, y) | torch | all |
| `forestdiffusion` | Forest-Flow / Forest-VP | field per time level | joint row (X, y) | XGBoost | continuous → discrete, mixed |

`csbm_annealed` is a registered **heuristic** (the reference is annealed between outer iterations); canonical
`csbm` keeps its reference fixed. Registered as **unavailable** — named in papers, result files or parameter
JSONs but without an executable benchmark adapter, and never silently substituted: `stasy` (the repository's "STaSy" is a simplified
VE score-SDE: no self-paced per-sample weights, fine-tuning stage, VP/sub-VP SDEs, probability-flow ODE sampler or
ncsnpp-tabular network), `lightsb_m` (the code is LightSB), and `tabsyn`.

`tabpfgen` is explicitly excluded from experiments: planning, tuning, CV and metric entrypoints reject
it, including explicit requests. New aggregates exclude its historical records with a recorded reason
and preserve the original saved results. Its optional standalone wrapper remains in `sbtab.baselines.tabpfn`.

CSBM and `csbm_annealed` accept `n_layers`, `dropout`, `forward_lr`, `backward_lr`,
`forward_weight_decay`, and `backward_weight_decay`. A direction-specific learning rate overrides `lr`;
when omitted, it inherits `lr`. Defaults remain two hidden layers, zero dropout, and weight decay `0.01`
in each direction, so existing `sbtab.csbm/2` checkpoints retain their architecture. The production search
spaces tune depth, dropout, and the two optimizers independently; all settings survive checkpoint reload.

TabbyFlow generates the joint row `(X, y)` with train-fitted uniform quantiles for numerical columns
and full one-hot encoding for nominal columns. Discrete numerical outputs are projected to their
training support. The production profile uses a Gaussian source, the OT conditional path and Euler
integration, with residual endpoint noise of `0.001`. Checkpoints include the fitted
representation and network for inference without retaining training rows. They do not resume the
optimizer or scheduler.

ForestDiffusion uses joint unconditional generation,
train-fitted z-scores and full one-hot encoding, with no continuous clipping. Its iterator avoids materializing
all time levels, but XGBoost's QuantileDMatrix retains quantized training data in memory. The default search
space uses Forest-Flow. The separate [Forest-VP production profile](configs/search_spaces/forest_vp/forestdiffusion.yaml)
keeps the same search ranges and fixes `diffusion_type: vp`. Both profiles default to CPU with four
threads, including automatic cluster routing. XGBoost >= 2.1 is required.

Create a Forest-VP plan through the existing pipeline:

```bash
python -m sbtab.experiments.pipeline plan \
    --output-root artifacts/forest-vp-production \
    --models forestdiffusion \
    --search-space-dir configs/search_spaces/forest_vp \
    --device cpu
```

For a bounded smoke run, add `--smoke`, use `--output-root artifacts/forest-vp-smoke`, and select
`--search-space-dir configs/search_spaces/smoke/forest_vp`. These directories contain only the
ForestDiffusion profile, so specify `--models forestdiffusion`. Run the resulting plan with the normal
`prepare`, `worker`, and `aggregate` stages. Flow and VP share the model ID `forestdiffusion`; use separate
output roots for them. Plans and resumed studies validate the search-space hash and reject a profile switch.
Direct `tune` calls can use `--search-space configs/search_spaces/forest_vp/forestdiffusion.yaml`.

All canonical SB entries sample **with** dynamics noise: a drift trained for the stochastic bridge is not a
probability-flow ODE, so `noise` is not a tunable option (a noiseless run reports `*_noiseless_heuristic`).

## Datasets

`configs/datasets/*.yaml` describes 28 datasets from the tracked bundles in `sbtab/data/datasets/`
(9 continuous, 5 categorical, 14 mixed). The bundles were pickled with numpy 2 / pandas 3;
`sbtab.data.loading.load_bundle` loads them under numpy 1.26 / pandas 2.2 as well, and `prepare_splits`
re-materialises every dataset it uses as Parquet + JSON schema with a value-based fingerprint.

**Category support.** The default protocol first removes rows containing categorical or discrete
values observed fewer than three times, iterating to a fixed point before splitting. This includes
rare target classes and can change the population or classification task. Original row IDs, removed
values and target changes are recorded in `eligibility_report.json`; filtering is never hidden.

The stratified 85/15 split is followed, when needed, by deterministic exchanges of whole rows between
T and V within the same target stratum. This retains all eligible rows, split sizes and stratum counts.
Five-fold KFold is then applied to the resulting sorted T pool with seed 42. Every finite-support value
in a held-out set must occur in its training set. The repair uses support and strata only, never model
scores, and records each exchange. Its bounded search can fail: unresolved datasets remain
`blocked_support` and are excluded before training. No encoder learns categories from held-out rows.

## Experimental protocol

Protocol `sbtab_8515_hpo100_cv5_v4` (`configs/protocols/`); every constant is part of the protocol hash and none
can be overridden on the command line.

| stage | behaviour |
|---|---|
| eligibility | **before any split**, rows whose value in a categorical / discrete column (incl. the classification target) occurs in fewer than **3** rows of the table are removed, iterated to a fixed point; recorded in `eligibility_report.json`; original row ids are kept |
| split | stratified `train_test_split(test_size=0.15, random_state=5)` → T (85 %) / V (15 %); regression targets use persisted quantile strata; deterministic same-stratum support repair if needed |
| tuning | 100 **allocated** Optuna trials (failures count, are kept, are never replaced), TPE seed 5, `n_jobs=1`, no pruning; fit on T, generate exactly len(V) rows, minimise the regime objective |
| CV | `KFold(5, shuffle=True, random_state=42)` on **T only**; fresh preprocessing + model per fold; hyperparameters only, never tuned weights; exactly len(T_k) rows |
| metrics | generated rows vs held-out E_k; everything a metric learns comes from T_k |

This is **fixed-hyperparameter CV after dataset-level tuning, not nested CV**: the tuning candidates were trained
on all of T, which contains every CV test fold.

```bash
python -m sbtab.experiments.prepare_splits --dataset insurance \
    --output-root artifacts/sbtab_8515_hpo100_cv5_v4
python -m sbtab.experiments.tune --dataset insurance --model mixedsbm \
    --splits artifacts/sbtab_8515_hpo100_cv5_v4/insurance/splits.json \
    --search-space configs/search_spaces/mixedsbm.yaml --resume
python -m sbtab.experiments.cross_validate --dataset insurance --model mixedsbm \
    --selected-config artifacts/sbtab_8515_hpo100_cv5_v4/insurance/mixedsbm/<run-id>/tuning/selected_config.json \
    --splits artifacts/sbtab_8515_hpo100_cv5_v4/insurance/splits.json \
    --output-root artifacts/sbtab_8515_hpo100_cv5_v4
python -m sbtab.experiments.calculate_metrics \
    --cv-run artifacts/sbtab_8515_hpo100_cv5_v4/insurance/mixedsbm/<run-id>/cv/cv_run_manifest.json \
    --metrics-config configs/metrics/metrics_v2.yaml
python -m sbtab.experiments.aggregate_results --output-root artifacts/sbtab_8515_hpo100_cv5_v4
```

Every stage has `--dry-run`. `--smoke` selects the **separate** protocol `sbtab_smoke_v4` (3 trials) together with
`configs/search_spaces/smoke/*.yaml` and its own artifact root; a smoke run is never evidence that the 100-trial
benchmark was completed. `tune --resume` allocates only the remaining budget and refuses to resume when the data,
split, search space, metric config, checkpoint format, protocol, dependency versions or implementation changed.
CV checks the same data/implementation compatibility and verifies saved artifact hashes before reusing a fold.
After changing the implementation, start a new run instead of extending an existing CV run.
`calculate_metrics` never fits a generator and never loads a checkpoint; a changed metric configuration writes to a
new `evaluation/<metric-version>-<hash>/` namespace.

### Complete pipeline and SLURM arrays

The [cluster run guide](docs/CLUSTER_EXPERIMENT.md) connects all stages with one array index per
compatible dataset/model pair. The submitter defaults to `SBTAB_DEVICE=auto`: ForestDiffusion and
boosted solvers run on CPU; neural generators, including TabbyFlow, use one V100 GPU. Generation
arrays perform tuning and fresh CV fits. Separate CPU metric arrays perform evaluation and TSTR,
each task depending on its matching successful generator through `aftercorr`. Aggregation waits for
all submitted arrays, including failures. No GPU is allocated to metric-only jobs.

`*.sbatch`, `scripts/slurm/cluster.local.sh`, and Slurm logs are ignored by Git. Copy
`cluster.example.sh` to `cluster.local.sh` and supply local `prepare.sbatch`, `experiment.sbatch`,
`metrics.sbatch`, and `aggregate.sbatch` before using the submitter. These ignored files must exist
on the cluster too. The sync helper uploads local deployment files while preserving cluster
environments, caches, `artifacts/` and logs.

Default plans include TabbyFlow and ForestDiffusion and select `dsb_ct_joint_mlp` and
`dsbm_ct_joint_mlp` for the two DSB families. Other
implemented variants can be selected explicitly with `--models` and use the same tuning/CV contract.
Unavailable adapters and missing optional dependencies are reported; models are never substituted.

```bash
python -m sbtab.experiments.pipeline plan --output-root artifacts/smoke \
  --smoke --datasets insurance --models tabbyflow --device cpu
python -m sbtab.experiments.pipeline prepare --plan artifacts/smoke/pipeline/plan.json
python -m sbtab.experiments.pipeline worker --plan artifacts/smoke/pipeline/plan.json --task-id 0 --stage generate
python -m sbtab.experiments.pipeline worker --plan artifacts/smoke/pipeline/plan.json --task-id 0 --stage metrics
python -m sbtab.experiments.pipeline aggregate --plan artifacts/smoke/pipeline/plan.json
```

### Metrics (`sbtab.metrics/2`, one implementation in `sbtab/evaluation/`)

- **Tuning objective** — continuous: mean 1-D Wasserstein in the training-standardised space; discrete: mean
  Jensen–Shannon *divergence* (natural log, ≤ log 2); mixed: mean WD + one combined discrete/categorical mean JS.
- **Marginal** — WD; `KL(real ‖ synthetic)` on 50 fixed bins (48 interior + under/overflow) whose edges come from the
  training rows; categorical/discrete KL on the training support + an unexpected-value bin; smoothing mass 1e-6.
- **Dependence** — compare within-table Pearson / Spearman / pairwise NMI matrices (Frobenius and normalised
  off-diagonal RMSE), η², cross-type Spearman.
- **Conditional** — per conditioning level: standardised WD and JS, macro and frequency-weighted, with eligible mass,
  missing-category mass and an explicit `incomplete_conditional_coverage` status. WD and JS are never pooled.
- **Joint** — signed unbiased product-kernel MMD² (RBF × Hamming), bandwidth from training rows, three seeded
  subsamples, matched real–real floor.
- **Utility (TSTR)** — `CatBoostClassifier` + macro-F1 or `CatBoostRegressor` + R²/MAE/RMSE/MAPE on raw target units;
  nominal predictors passed as `cat_features`. A fixed preset of documented CPU defaults is identical across
  folds and generators, with no parameter-resolution fit on another fold. Automatic learning-rate selection
  is disabled by explicitly setting default regularization (learning rate 0.03). MVS subsampling stays
  at 0.8 even below 100 training rows, where CatBoost would automatically choose 1. Real references are cached
  per dataset/fold. Percentage deviation is `100 * abs(synthetic - real) / abs(real)`; a zero real score
  leaves the relative deviation undefined. MAPE is undefined when any test target is zero.
- **Timing** — preprocessing, init, generator fit, checkpoint I/O, generation, inverse transform, metrics, utility.

Invalid generated data (non-finite values, unknown categories) is a status, never a perfect score; inapplicable
metrics are `null`, never 0; JSON output contains no NaN/Infinity tokens. Aggregates are means with **sample** standard
deviation (`ddof=1`) and `n_expected / n_valid / n_failed`.

## Quickstart

```python
from sbtab.data.preprocessing import CommonPreprocessor
from sbtab.experiments.experiment_common import load_protocol
from sbtab.experiments.prepare_splits import build_splits, load_eligible_dataset, require_ok
from sbtab.solvers.registry import get_adapter_class

protocol = load_protocol()
frame, schema, manifest, eligibility = load_eligible_dataset("insurance", protocol)
splits, support = build_splits(frame, schema, manifest, protocol)
require_ok(splits)
train_raw = frame.loc[splits["T_row_ids"]]

pre = CommonPreprocessor(schema).fit(train_raw)               # train-only scaler and vocabularies
adapter = get_adapter_class("mixedsbm")().fit(
    pre.transform(train_raw), schema,
    config=dict(n_stages=5, epochs_per_direction=40, num_steps=100, sigma=0.3), seed=0)

synthetic = adapter.sample(len(train_raw), seed=1)            # common representation, schema column order
print(pre.inverse_transform(synthetic).head())                # raw units and original labels
adapter.save_checkpoint("checkpoints/insurance_mixedsbm")     # reload without refitting: load_checkpoint(...)
```

### Examples

Each script takes `--quick` (a bounded run that says nothing about model quality).

| Script | Registry id | Dataset |
|---|---|---|
| `examples/California_Housing_example.py` | `dsb_ct_joint_mlp` | `california_housing` |
| `examples/joint_continuous_time_boost_example.py` | `dsbm_ct_joint_gbt` | `california_housing` |
| `examples/feature_wise_discrete_time_boosting-example.py` | `dsbm_dt_structural_gbt` (learned DAG) | `california_housing` |
| `examples/joint_continuous_time_boost_ipf_example.py` | `dsb_ct_joint_gbt` | `california_housing` |
| `examples/structural_discrete_time_boost_ipf_example.py` | `dsb_dt_structural_gbt` (learned DAG) | `california_housing` |
| `examples/joint_discrete_time_mlp_example.py` | `dsbm_dt_joint_mlp` | `california_housing` |
| `examples/boosted_dsbm_example.py` | `dsbm_dt_joint_gbt` | `california_housing` |
| `examples/light_sb_example.py` | `lightsb` | `california_housing` |
| `examples/mixedsbm_example.py` | `mixedsbm` | `insurance` (mixed) |
| `examples/csbm_example.py` | `csbm` | `car_evaluation` (discrete) |
| `examples/tabddpm_example.ipynb` | `tabddpm` | notebook walkthrough (tiny `steps` budget in the executed path) |

## Installation

```bash
git clone https://github.com/Anaxagor/sb-tabular.git
cd sb-tabular
pip install -r requirements.txt
python -m pytest tests            # from the repository root
```

`requirements.txt` is grouped by purpose. `sdv` (CTGAN) is optional: `import sbtab` works without it,
planning reports the missing package, and its integration tests **skip — a skip is not a validation
of the adapter**. The optional TabPFGen dependencies serve only the historical standalone wrapper;
the experiment pipeline neither uses them nor downloads pretrained weights. `geotorch` is only needed
for LightSB with a full covariance. Pin `catboost` for a benchmark run:
the utility evaluator uses the same declared defaults across datasets and folds for each task type. Python ≥ 3.10; run from the repository root.
For the GPU cluster experiment, use the pinned Python 3.11 environment and setup commands in the
[cluster run guide](docs/CLUSTER_EXPERIMENT.md).

## Reproducibility and limitations

Protocol v4 combines documented eligibility filtering, deterministic support repair, strict split and
selected-trial verification, and metric version 2. Earlier protocol files remain frozen for provenance;
start a fresh output root for current runs. Saved artifacts are checked against dataset values,
preprocessing, source/configuration hashes, dependencies, seeds and checkpoint formats. Completed
trials and folds can be reused only when their provenance matches. Mid-fit optimizer resume is not
supported; an interrupted allocated trial counts toward the trial budget.

IPF-DSB defaults to `horizon=2.0`, independently of step count. `horizon=None` retains the raw gamma
schedule, as configured for boosted variants. Finite time and Euler discretization introduce
approximation error; coarse grids must satisfy `alpha_ou * max(dt) < 1`.

Examples demonstrate adapters with small budgets and illustrative splits; use `sbtab.experiments`
for benchmark results. The production benchmark requires actual 100-trial runs and all five folds.
CPU tests and smoke runs do not validate execution on a cluster GPU. Optional-dependency and CUDA
checks report explicit skips when their prerequisites are unavailable.
