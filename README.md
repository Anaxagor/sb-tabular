# sb-tabular

A research framework for **synthetic tabular data generation with Schrödinger Bridges (SB)**.

The repository implements several SB solver families under one data pipeline and compares them
against non-SB generative baselines (CTGAN, TabDDPM, a simplified VE score-SDE, TabPFGen) under one
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
  reference processes. The only solver that handles mixed tables natively end-to-end.

After merging `feature/tuning` (`0b9f15f`), MixedSBM uses one network and optimizer
across forward/backward stages, the historical per-step `alpha` reference, and a
configurable step or epoch budget. Its bridge helpers are isolated under
`sbtab/solvers/msbm/`; CSBM keeps its semigroup reference. Production search-space
version 2 restores the tuning ranges while retaining the canonical protocol's
mandatory dynamics noise. New MSBM checkpoints use `sbtab.mixedsbm/3`; the former
two-network `/2` checkpoints and `cat_mixing_rate` configurations require the
pre-merge implementation. Existing tuning studies must start a new run.

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
   one-hot and discrete columns standardised; native solvers see finite states. Decoding is declared
   and measured (argmax; nearest *training-support* value with the rounding rate recorded). Continuous
   outputs are never clipped to the training range. Configs are strict: an unknown key raises.
4. **Metrics** (`sbtab/evaluation/`) always read the common representation.

---

## Repository layout

```text
sb-tabular/
├── configs/
│   ├── protocols/                     # sbtab_8515_hpo100_cv5_v2 (production, default), sbtab_smoke_v2; v1 files frozen
│   ├── datasets/                      # explicit per-dataset schema metadata (28 datasets)
│   ├── search_spaces/                 # one per registry id; smoke/ holds the bounded variants
│   └── metrics/                       # metrics_v1.yaml (metric version sbtab.metrics/1)
├── docs/IMPLEMENTATION_REPORT.md      # defects, contracts, protocol, feasibility, limitations
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
    │   └── light_sb/ csbm/ msbm/
    ├── baselines/                     # ctgan, tabddpm, stasy (= simplified VE score-SDE), tabpfn (= TabPFGen)
    ├── adapters/                      # model adapters + reversible representations
    ├── evaluation/                    # THE metric implementation (tuning and evaluation share it)
    └── experiments/
        ├── experiment_common.py       # protocol loading, hashes/provenance, seed ledger, atomic I/O, timing
        ├── prepare_splits.py  tune.py  cross_validate.py  calculate_metrics.py  aggregate_results.py
        └── legacy/                    # frozen pre-protocol scripts and historical result files
```

`sbtab/data/{schema,splits,datamodule}.py` and `sbtab/transforms/` are the earlier pipeline; they are kept
for the legacy scripts and are not used by the experiment stages.

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
| `tabpfgen` | baseline | – | SGLD + TabPFN | `tabpfgen`, `tabpfn` | all |
| `forestdiffusion` | Forest-Flow / Forest-VP | field per time level | joint row (X, y) | XGBoost | continuous → discrete, mixed |

`csbm_annealed` is a registered **heuristic** (the reference is annealed between outer iterations); canonical
`csbm` keeps its reference fixed. Registered as **unavailable** — named in papers, result files or parameter
JSONs but without an executable benchmark adapter, and never silently substituted: `stasy` (the repository's "STaSy" is a simplified
VE score-SDE: no self-paced per-sample weights, fine-tuning stage, VP/sub-VP SDEs, probability-flow ODE sampler or
ncsnpp-tabular network), `lightsb_m` (the code is LightSB), `tabbyflow` (tested standalone implementation;
benchmark adapter/checkpoints pending), and `tabsyn`.

ForestDiffusion was imported from `forest_diffusion` (`50635ca`) and corrected during the
[generative algorithm audit](docs/GENERATIVE_ALGORITHM_AUDIT.md). It uses joint unconditional generation,
train-fitted z-scores and full one-hot encoding, with no continuous clipping. Its iterator avoids materializing
all time levels, but XGBoost's QuantileDMatrix retains quantized training data in memory. The default search
space uses Forest-Flow; set `diffusion_type: vp` for Forest-VP. XGBoost >= 2.1 is required.

The audit also corrected MSBM's categorical reference; checkpoints now use `sbtab.mixedsbm/4`.
Older MSBM checkpoints must be retrained because their transition law differs.

All canonical SB entries sample **with** dynamics noise: a drift trained for the stochastic bridge is not a
probability-flow ODE, so `noise` is not a tunable option (a noiseless run reports `*_noiseless_heuristic`).

## Datasets

`configs/datasets/*.yaml` describes 28 datasets from the tracked bundles in `sbtab/data/datasets/`
(9 continuous, 5 categorical, 14 mixed). The bundles were pickled with numpy 2 / pandas 3;
`sbtab.data.loading.load_bundle` loads them under numpy 1.26 / pandas 2.2 as well, and `prepare_splits`
re-materialises every dataset it uses as Parquet + JSON schema with a value-based fingerprint.

**Category support.** Every value of a categorical / discrete column present in a held-out part must also be
present in the corresponding training part (V ⊆ T, and E_k ⊆ T_k for every fold). Without any row filtering
(protocol `v1`, frozen) only 18 of 28 datasets pass. The default protocol `v2` first removes rows carrying a value
seen fewer than 3 times (typically single-row artefacts such as `gender='Other'`): **25 of 28 pass**. Row loss is
≤ 1.2 % except for two small tables: `palmer_penguins` 5.5 % and `lymphography` 6.8 %. Two things to know:

- The rule also removes rare **target classes** — `lymphography` loses its 2-row class `normal`
  (`task_changed: true` in `eligibility_report.json`).
- A count threshold *reduces* but cannot *guarantee* coverage: `breast_cancer`, `house_sales` and `student_perf`
  stay blocked (in `breast_cancer` a value with exactly 3 rows has two of them in the same test fold). Measured
  passes by threshold: 1 → 18, 2 → 23, **3 → 25**, 4 → 26, 5 → 28, 15 → 26, 20 → 24 (large thresholds empty whole
  columns). Change `eligibility.min_value_count` in a **new** protocol file to use another value.

A dataset that still fails is stopped before tuning with a structured `support_report.json` — no alternative seed,
merged category or split-dependent row removal is used. See `docs/IMPLEMENTATION_REPORT.md` §4.

## Experimental protocol

Protocol `sbtab_8515_hpo100_cv5_v2` (`configs/protocols/`); every constant is part of the protocol hash and none
can be overridden on the command line.

| stage | behaviour |
|---|---|
| eligibility (v2) | **before any split**, rows whose value in a categorical / discrete column (incl. the classification target) occurs in fewer than **3** rows of the table are removed, iterated to a fixed point; recorded in `eligibility_report.json`; original row ids are kept |
| split | stratified `train_test_split(test_size=0.15, random_state=5)` → T (85 %) / V (15 %); regression targets use persisted quantile strata |
| tuning | 100 **allocated** Optuna trials (failures count, are kept, are never replaced), TPE seed 5, `n_jobs=1`, no pruning; fit on T, generate exactly len(V) rows, minimise the regime objective |
| CV | `KFold(5, shuffle=True, random_state=42)` on **T only**; fresh preprocessing + model per fold; hyperparameters only, never tuned weights; exactly len(T_k) rows |
| metrics | generated rows vs held-out E_k; everything a metric learns comes from T_k |

This is **fixed-hyperparameter CV after dataset-level tuning, not nested CV**: the tuning candidates were trained
on all of T, which contains every CV test fold.

```bash
python -m sbtab.experiments.prepare_splits --dataset insurance \
    --output-root artifacts/sbtab_8515_hpo100_cv5_v2
python -m sbtab.experiments.tune --dataset insurance --model mixedsbm \
    --splits artifacts/sbtab_8515_hpo100_cv5_v2/insurance/splits.json \
    --search-space configs/search_spaces/mixedsbm.yaml --resume
python -m sbtab.experiments.cross_validate --dataset insurance --model mixedsbm \
    --selected-config artifacts/sbtab_8515_hpo100_cv5_v2/insurance/mixedsbm/<run-id>/tuning/selected_config.json \
    --splits artifacts/sbtab_8515_hpo100_cv5_v2/insurance/splits.json \
    --output-root artifacts/sbtab_8515_hpo100_cv5_v2
python -m sbtab.experiments.calculate_metrics \
    --cv-run artifacts/sbtab_8515_hpo100_cv5_v2/insurance/mixedsbm/<run-id>/cv/cv_run_manifest.json \
    --metrics-config configs/metrics/metrics_v1.yaml
python -m sbtab.experiments.aggregate_results --output-root artifacts/sbtab_8515_hpo100_cv5_v2
```

Every stage has `--dry-run`. `--smoke` selects the **separate** protocol `sbtab_smoke_v2` (3 trials) together with
`configs/search_spaces/smoke/*.yaml` and its own artifact root; a smoke run is never evidence that the 100-trial
benchmark was completed. `tune --resume` allocates only the remaining budget and refuses to resume when the data,
split, search space, metric config, checkpoint format, protocol, dependency versions or implementation changed.
CV checks the same data/implementation compatibility and verifies saved artifact hashes before reusing a fold.
After changing the implementation, start a new run instead of extending an existing CV run.
`calculate_metrics` never fits a generator and never loads a checkpoint; a changed metric configuration writes to a
new `evaluation/<metric-version>-<hash>/` namespace.

### Complete pipeline and SLURM arrays

The [cluster run guide](docs/CLUSTER_EXPERIMENT.md) connects all stages into an array for partition
`rocky`, account `proj_1752`: one task per compatible dataset/model pair, 100 tuning trials, five
fresh CV fits, all test metrics and TSTR, followed by aggregation. Each task requests one GPU,
eight CPUs and two days; generator tuning, training and sampling use CUDA. The guide includes the
HSE login-02 environment setup, offline pretrained-weight preparation, a GPU smoke-check job,
resumable trials/folds and failure summaries. Install `requirements-cluster.txt` through
`scripts/setup_cluster_env.sh`, then edit the included `scripts/slurm/cluster.local.sh` locally.
Upload code, datasets and configuration with `bash scripts/sync_cluster.sh USER@HOST:/home/USER/sb-tabular`
(add `--dry-run` before the destination to preview). The helper verifies file contents and preserves
cluster environments, caches, `artifacts/` and `slurm_logs/`; no Git is needed on the cluster.
See the guide for the update workflow and dependency changes.

DSB and DSBM experiments use only the continuous-time joint MLP models `dsb_ct_joint_mlp` and
`dsbm_ct_joint_mlp`. Their discrete-time, boosted and structural variants are excluded from planning
and from standalone tuning/CV, including explicit `--models` selections. Other model families keep
their existing selection rules. Create a new plan/output root when switching from the earlier model set.

```bash
bash scripts/slurm/submit.sh --dry-run scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-production
# Remove --dry-run to submit the preparation job, experiment array and aggregation job.
```

### Metrics (`sbtab.metrics/1`, one implementation in `sbtab/evaluation/`)

- **Tuning objective** — continuous: mean 1-D Wasserstein in the training-standardised space; discrete: mean
  Jensen–Shannon *divergence* (natural log, ≤ log 2); mixed: mean WD + one combined discrete/categorical mean JS.
- **Marginal** — WD; `KL(real ‖ synthetic)` on 50 fixed bins (48 interior + under/overflow) whose edges come from the
  training rows; categorical/discrete KL on the training support + an unexpected-value bin; smoothing mass 1e-6.
- **Dependence** — Pearson / Spearman / NMI matrices (Frobenius and normalised off-diagonal RMSE), η², cross-type Spearman.
- **Conditional** — per conditioning level: standardised WD and JS, macro and frequency-weighted, with eligible mass,
  missing-category mass and an explicit `incomplete_conditional_coverage` status. WD and JS are never pooled.
- **Joint** — signed unbiased product-kernel MMD² (RBF × Hamming), bandwidth from training rows, three seeded
  subsamples, matched real–real floor.
- **Utility (TSTR)** — `CatBoostClassifier` + macro-F1 or `CatBoostRegressor` + R²/MAE/RMSE/MAPE on raw target units;
  nominal predictors passed as `cat_features`; defaults resolved once on the first real CV fold and frozen; the real
  reference is cached per dataset, not per generator. Positive gap = synthetic training is worse.
- **Timing** — preprocessing, init, generator fit, checkpoint I/O, generation, inverse transform, metrics, utility.

Invalid generated data (non-finite values, unknown categories) is a status, never a perfect score; inapplicable
metrics are `null`, never 0; JSON output contains no NaN/Infinity tokens. Aggregates are means with **sample** standard
deviation (`ddof=1`) and `n_expected / n_valid / n_failed`.

## Quickstart

```python
from sklearn.model_selection import train_test_split

from sbtab.data.preprocessing import CommonPreprocessor
from sbtab.data.registry import load_dataset
from sbtab.solvers.registry import get_adapter_class

frame, schema, _ = load_dataset("insurance")                 # explicit schema from configs/datasets/insurance.yaml
train_raw, held_raw = train_test_split(frame, test_size=0.15, random_state=5)   # illustrative split only

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
git clone https://github.com/ITMO-NSS-team/sb-tabular.git
cd sb-tabular
pip install -r requirements.txt
python -m pytest tests            # from the repository root
```

`requirements.txt` is grouped by purpose. `sdv` (CTGAN) and `tabpfgen` (TabPFGen) are optional: `import sbtab`
works without them, their stages report the missing package, and their tests **skip — a skip is not a validation
of the adapter**. `geotorch` is only needed for LightSB with a full covariance. Pin `catboost` for a benchmark run:
the utility evaluator freezes its resolved defaults per dataset. Python ≥ 3.10; run from the repository root.
For the GPU cluster experiment, use the pinned Python 3.11 environment and setup commands in the
[cluster run guide](docs/CLUSTER_EXPERIMENT.md#2-configure-the-hse-cluster-environment).
TabPFGen uses a reproducible training-only context subset when its training input exceeds
10,000 rows on GPU or 1,000 on CPU. Classification subsets retain all target classes; generated
sample sizes still follow the experiment protocol. Context selections are saved in checkpoints.
To repair rare-category coverage while keeping all eligible rows, select
`--protocol configs/protocols/sbtab_8515_hpo100_cv5_v3.yaml`. It records same-stratum train/validation
row swaps, retains the 85/15 sizes and ordinary five-fold KFold, and places every level of the
blocking columns in validation and in every fold's training set. V1/v2 retain their original behavior;
the default is still v2. Use a fresh output root for v3; see the cluster guide for the full command.

## Status and known gaps

The [follow-up review](docs/REVIEW_2026-09-23.md) records the refactor corrections. With the additional
[cluster orchestration checks](docs/CLUSTER_EXPERIMENT.md), the full suite on 2026-09-26 had **880 passed, 12 skipped**;
24 batch-script smoke pipelines spanning all ten selected generators completed 72 tuning trials and
120 fresh CV folds locally. Ten skipped tests require an allocated CUDA GPU; the other two concern
optional notebook validation and the missing-dependency branch for the installed `geotorch` package.
Evaluation outputs now use
an `-eval2` namespace suffix so corrected failure/status/aggregation records do not overwrite earlier evaluations.
See the [implementation report](docs/IMPLEMENTATION_REPORT.md) for the per-solver mathematical contracts,
dataset feasibility and earlier validation.

IPF-DSB solvers and adapters now default to `horizon=2.0`, independent of the step count. The gamma
schedule is rescaled to span that horizon; `horizon=None` explicitly restores the raw schedule.
Older checkpoints retain their original grids. The selected joint MLP DSB experiment still tunes
`horizon` over `[0.5, 3.0]`; retained boosted search spaces explicitly use `horizon: null` so their
`gamma_max` search keeps its original meaning. Boosted variants remain excluded from the experiments.
The longer default reduces the initial OU reference's mismatch with the Gaussian prior, but finite
time and Euler discretisation still introduce approximation error. Coarse grids must satisfy
`alpha_ou * max(dt) < 1`; increase the step count or reduce the horizon if this check fails.

Remaining limitations include:

- The production benchmark has **not** been run; only bounded smoke runs and tests were executed.
- The pinned environment includes real CTGAN and TabPFGen libraries, tested locally on CPU. CUDA execution
  must pass `scripts/slurm/check_gpu.sbatch` on an allocated cluster GPU before production submission.
- Checkpoints are inference-complete but not resumable mid-fit; the resume granularity is a trial or a fold.
- Historical results under `sbtab/experiments/legacy/` carry no run-to-commit provenance, use different metric
  definitions (`legacy/0`) and must not be mixed with `sbtab.metrics/1` results.
