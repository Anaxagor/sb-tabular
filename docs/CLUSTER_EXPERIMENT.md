# Running the standardized experiment

All launch paths use `sbtab.experiments`: prepare splits, tune each generator independently,
train fresh CV models, calculate metrics from saved synthetic data, and aggregate results.
The default protocol is `configs/protocols/sbtab_8515_hpo100_cv5_v4.yaml`; its smaller execution
check is `configs/protocols/sbtab_smoke_v4.yaml`. The [README](../README.md#experimental-protocol)
defines the preprocessing, splitting, objectives and metrics.

## Protocol and model selection

Before splitting, the protocol iteratively removes rows with categorical/discrete values occurring
fewer than three times, including rare classification targets. The eligibility artifact records all
removed rows and task changes. Stratified 85/15 splitting uses seed 5. Deterministic same-stratum
row exchanges repair support where possible without removing additional rows or changing sizes.
KFold uses five shuffled folds with seed 42 on the resulting sorted tuning-training pool only.
Unresolved support failures remain blocked; no held-out category is added to a fitted encoder.

Every dataset/model pair receives 100 allocated Optuna trials. Each trial fits on T and generates
len(V) rows. The minimum finite validation objective selects hyperparameters for five fresh CV fits;
each fold generates len(T_k) rows. Failed or interrupted trials consume their allocation. The smoke
protocol uses three trials with smaller training budgets and must have its own output directory.
CV follows dataset-level tuning; it is not an independent outer evaluation of the tuning procedure.

CatBoostClassifier evaluates classification with macro-F1; CatBoostRegressor evaluates regression
with R², MAE, RMSE and MAPE. Real and synthetic fits use the same documented default CPU preset,
seed and thread count across folds and generators. No utility hyperparameters are learned from
another fold. The evaluator reports absolute percentage deviations; a zero reference score makes
the relative deviation undefined. MAPE is undefined if any held-out target equals zero.
The preset uses learning rate 0.03 (the default with explicit default regularization) and freezes
MVS subsampling at 0.8 even for fewer than 100 rows, where the automatic value would be 1.

Default plans select the continuous-time joint MLP for each DSB family. Other implemented DSB
variants are available through explicit `--models` selections. Missing packages, unsupported regimes,
unavailable adapters and support failures are recorded instead of silently replacing models.
`--exclude-heuristic` omits `csbm_annealed`. Forest-Flow and Forest-VP share an adapter ID and use
separate search-space profiles and output roots; see the README for their commands.

## Environment and local cluster files

Follow the current [HSE cluster rules](https://hpc.hse.ru/llms.txt) when preparing local batch scripts.
Choose CPU or an appropriate GPU type for the selected models and configure the project/account,
partition, time and CPU requirements locally. Use arrays for independent dataset/model tasks.
Do not pass `--mem` on this cluster. Splitting, metrics and the CPU CatBoost utility evaluator do
not require a GPU. Install dependencies and any pretrained weights before running offline workers.

The repository tracks the submitter and a configuration template. It deliberately ignores
`*.sbatch`, `scripts/slurm/cluster.local.sh`, and `slurm_logs/`. A fresh clone therefore needs local
batch scripts before Slurm submission. Keep local copies of deployment files when migrating an older
checkout in which those files were tracked.

```bash
cp scripts/slurm/cluster.example.sh scripts/slurm/cluster.local.sh
# Edit cluster.local.sh for the local environment and selected model resources.
# Supply prepare.sbatch, experiment.sbatch, aggregate.sbatch in scripts/slurm/,
# or set SBTAB_BATCH_SCRIPT_DIR to their directory.
```

The submitter checks that all three scripts exist before planning or submitting any jobs. It passes:

| Local script | Positional arguments | Work to invoke |
|---|---|---|
| `prepare.sbatch` | repository root, configuration path, plan path | `pipeline prepare --plan PLAN` |
| `experiment.sbatch` | repository root, configuration path, plan path, stage, retry-failed-folds flag | `pipeline worker --plan PLAN --task-id "$SLURM_ARRAY_TASK_ID" --stage STAGE`; add `--retry-failed-folds` when enabled |
| `aggregate.sbatch` | repository root, configuration path, plan path | `pipeline aggregate --plan PLAN` |

Here `pipeline` means `python -m sbtab.experiments.pipeline` using the configured interpreter.
All jobs must access the same checkout and output paths. The output filesystem must support
POSIX `flock` and atomic rename. Source `scripts/slurm/common.sh` from local jobs if using its
shared environment setup. GPU jobs should verify CUDA before training; CPU checks do not establish
GPU availability or suitability of a requested card.

For the pinned environment, use `requirements-cluster.txt` through `scripts/setup_cluster_env.sh`.
The script supports a pre-created Python 3.11 environment via `SBTAB_BOOTSTRAP_PYTHON`, installs
the configured PyTorch wheel and dependencies, checks package compatibility and saves resolved
versions. Configure `SBTAB_PYTHON` in `cluster.local.sh` to point at the resulting interpreter.

```bash
export SBTAB_REPO_ROOT="$PWD"
source scripts/slurm/cluster.local.sh
"$SBTAB_PYTHON" -m sbtab.experiments.cluster_environment --download-tabpfn
```

TabPFGen requires cached classifier/regressor weights. Plans record their hashes and workers reject
missing or changed files. Its context is drawn without replacement from training rows only, retaining
each target class: up to 10,000 rows on GPU or 1,000 on CPU. The sample-size contract still applies
to the generated table. Context selections are saved in the model checkpoint.

## Upload and submit

The sync helper copies local deployment files, including ignored batch scripts and local
configuration, while preserving remote environments, caches, experiment outputs and logs. It does
not use `.gitignore` as a transfer filter. It verifies transferred contents and removes obsolete files
inside managed source directories. Stop jobs using this checkout before replacing its source files.

```bash
bash scripts/sync_cluster.sh --dry-run USER@HOST:/shared/path/sb-tabular
bash scripts/sync_cluster.sh USER@HOST:/shared/path/sb-tabular
```

Install changed dependencies on the cluster, then create the plan there. Git is not required on the
cluster: source/configuration content hashes are checked independently of Git metadata.

```bash
bash scripts/slurm/submit.sh --dry-run scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-production-v4
# Use the same command without --dry-run to submit.
```

For a bounded execution check, use a separate output root and explicit models/datasets:

```bash
bash scripts/slurm/submit.sh scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-smoke-v4 --smoke \
  --datasets diabetes car_evaluation insurance --models lightsb csbm mixedsbm
```

Submission creates preparation, an experiment array dependent on successful preparation, and
aggregation after all array tasks finish, including failures. One array task owns a dataset/model
pair; Optuna trials and folds are sequential within that task. Set `SBTAB_MAX_CONCURRENT` for array
concurrency. Stdout/stderr go to `slurm_logs/`; job IDs are saved beside the immutable plan.
The aggregate exits nonzero if any planned task lacks a complete evaluation, after saving results.

## Resume and inspect results

Re-run the same command to continue remaining trials and unfinished folds. Completed artifacts
are verified before reuse. Changing code, dependencies, dataset definitions, protocol, metrics or
search spaces requires a fresh output root. V1–v3 protocol files are retained for provenance;
current v4 results must not be mixed with their earlier metric definitions.

Use `SBTAB_TASK_IDS=0,7` for selected array indices and `SBTAB_STAGE` for `all`, `tune`, `cv`, or
`metrics`. Other planning arguments must match the original plan. Set
`SBTAB_RETRY_FAILED_FOLDS=1` to retry recorded failed folds. Mid-fit optimizer recovery is not
supported: interrupted CV fits restart; interrupted tuning trials remain allocated failures.
Concurrent workers and standalone tuning processes are locked to prevent duplicate allocation.

Metrics are recalculated from saved fold preprocessors and synthetic tables, without loading
or retraining a generator. Real-data utility references are shared across generators for the same
dataset, fold and evaluator configuration.

| Artifact | Path relative to the output root |
|---|---|
| Frozen plan, array indices and exclusions | `pipeline/plan.json` |
| Dataset preflight outcomes | `pipeline/preparation.json` |
| Task outcomes and failure details | `pipeline/tasks/00000.json`, etc. |
| Overall completion/failure summary | `pipeline/summary.json` |
| Eligibility, data and split memberships | `<dataset>/eligibility_report.json`, `data.parquet`, `splits.json` |
| Study and selected hyperparameters | `<dataset>/<model>/run-pipeline/tuning/` |
| Fresh models, synthetic tables and timing | `<dataset>/<model>/run-pipeline/cv/fold-<k>/` |
| Per-fold metrics and predictions | `<dataset>/<model>/run-pipeline/evaluation/` |
| Combined results | `aggregate/` |

Each fold records generator fit time separately from preprocessing, model initialization, sampling,
checkpoint I/O and inverse transformation. Timers synchronize CUDA at their boundaries. TabPFGen
stores context during fit and performs its SGLD/TabPFN computation during generation. Aggregates
include valid-fold counts and sample standard deviation (`ddof=1`), preserving undefined metrics
and failures instead of replacing them with zero.

The same plan/prepare/worker/aggregate commands can run locally without Slurm; the README contains
a complete CPU smoke example. Production results require completing all 100 trials and five folds.
