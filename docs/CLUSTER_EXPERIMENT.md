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
`--exclude-heuristic` omits `csbm_annealed`. TabbyFlow and ForestDiffusion are included by default
for all three regimes. TabbyFlow's production profile uses the OT path with Euler integration and
saves an inference checkpoint. Its numerical empirical quantile inverse is bounded by the training
range; the common pipeline adds no generic clipping of continuous outputs. ForestDiffusion defaults
to CPU with four threads. Forest-Flow and Forest-VP share an adapter ID and use separate search-space
profiles and output roots; see the README for their commands.

TabPFGen is excluded from every canonical experiment entrypoint, including explicit planning,
tuning, CV and metric requests. New aggregates exclude historical TabPFGen records with an explicit
reason and leave the saved source results intact. Its standalone wrapper remains optional; cluster
experiments require neither TabPFN dependencies nor pretrained weight downloads.

## Environment and local cluster files

Follow the current [HSE cluster rules](https://hpc.hse.ru/llms.txt) when preparing local batch scripts.
The supplied configuration uses the `rocky` partition and project `proj_1752`; set the account to
an authorized project for your run (1752, 1925 or 1825). Do not pass `--mem` or pin node numbers.
`SBTAB_DEVICE=auto` routes ForestDiffusion and CPU-only boosted solvers to `type_d`, and neural
generators, including TabbyFlow, to one V100 32 GB GPU on `type_a|type_b|type_c`, with four CPU cores.
Splitting, metrics, CPU CatBoost utility fits and aggregation use CPU allocations only. This keeps
metric calculation from holding a GPU idle. Install the pinned dependencies before launching jobs.

The repository tracks the submitter and a configuration template. It deliberately ignores
`*.sbatch`, `scripts/slurm/cluster.local.sh`, and `slurm_logs/`. A fresh clone therefore needs local
batch scripts before Slurm submission. Keep local copies of deployment files when migrating an older
checkout in which those files were tracked.

```bash
cp scripts/slurm/cluster.example.sh scripts/slurm/cluster.local.sh
# Edit cluster.local.sh for the local environment and selected model resources.
# Supply prepare.sbatch, experiment.sbatch, metrics.sbatch, aggregate.sbatch in scripts/slurm/,
# or set SBTAB_BATCH_SCRIPT_DIR to their directory.
```

The submitter checks that all four scripts exist before planning or submitting any jobs. It passes:

| Local script | Positional arguments | Work to invoke |
|---|---|---|
| `prepare.sbatch` | repository root, configuration path, plan path | `pipeline prepare --plan PLAN` |
| `experiment.sbatch` | repository root, configuration path, plan path, stage, retry-failed-folds flag | `pipeline worker --plan PLAN --task-id "$SLURM_ARRAY_TASK_ID" --stage STAGE`; add `--retry-failed-folds` when enabled |
| `metrics.sbatch` | repository root, configuration path, plan path | `pipeline worker --plan PLAN --task-id "$SLURM_ARRAY_TASK_ID" --stage metrics` |
| `aggregate.sbatch` | repository root, configuration path, plan path | `pipeline aggregate --plan PLAN` |

Here `pipeline` means `python -m sbtab.experiments.pipeline` using the configured interpreter.
All jobs must access the same checkout and output paths. The output filesystem must support
POSIX `flock` and atomic rename. Source `scripts/slurm/common.sh` from local jobs if using its
shared environment setup. GPU jobs should verify CUDA before training; CPU checks do not establish
GPU availability or suitability of a requested card.

Resource arrays in `cluster.local.sh` override the submitter's production defaults:

| Setting | Allocation | Time limit |
|---|---|---|
| `SBTAB_PREPARE_ARGS` | `type_d`, 4 CPU cores, 0 GPUs | 1 hour |
| `SBTAB_EXPERIMENT_ARGS` | `type_a\|type_b\|type_c`, 4 CPU cores, 1 V100 | 72 hours |
| `SBTAB_CPU_EXPERIMENT_ARGS` | `type_d`, 4 CPU cores, 0 GPUs | 72 hours |
| `SBTAB_METRICS_ARGS` | `type_d`, 4 CPU cores, 0 GPUs | 12 hours |
| `SBTAB_AGGREGATE_ARGS` | `type_d`, 1 CPU core, 0 GPUs | 1 hour |

For example, the CPU generation and metric settings in the template are:

```bash
SBTAB_CPU_EXPERIMENT_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=4 --time=3-00:00:00)
SBTAB_METRICS_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=4 --time=12:00:00)
SBTAB_DEVICE=auto
```

With `--smoke`, the submitter shortens both CPU/GPU generation and metric arrays to 30 minutes,
and preparation/aggregation to 15 minutes. Preparation and aggregation are singleton arrays;
generation and metric arrays use the dataset/model indices in the plan. `SBTAB_MAX_CONCURRENT`
defaults to 8 per submitted array, rather than a combined limit across CPU, GPU and metric arrays.

For the pinned environment, use `requirements-cluster.txt` through `scripts/setup_cluster_env.sh`.
The script supports a pre-created Python 3.11 environment via `SBTAB_BOOTSTRAP_PYTHON`, installs
the configured PyTorch wheel and dependencies, checks package compatibility and saves resolved
versions. Configure `SBTAB_PYTHON` in `cluster.local.sh` to point at the resulting interpreter.

```bash
export SBTAB_REPO_ROOT="$PWD"
source scripts/slurm/cluster.local.sh
"$SBTAB_PYTHON" -m sbtab.experiments.cluster_environment
```

Run CUDA validation on an allocated GPU. The optional ignored `check_gpu.sbatch` uses
`--array=0-10%2`, one GPU and four CPU cores per task, with a 30-minute limit. It performs independent
fit/sample/checkpoint checks for the 11 GPU-capable default models, including optional ForestDiffusion
CUDA execution; automatic production routing still runs ForestDiffusion on CPU. The script runs
`cluster_environment --require-cuda` before its selected test. Inspect skips as well as failures;
a skipped test does not verify that model on the GPU.

```bash
mkdir -p slurm_logs
sbatch scripts/slurm/check_gpu.sbatch "$PWD" "$PWD/scripts/slurm/cluster.local.sh"
```

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
  --datasets diabetes car_evaluation insurance --models tabbyflow forestdiffusion
```

With the default `SBTAB_STAGE=all`, submission creates a preparation array, then separate CPU and
GPU generation arrays after successful preparation. Each generation task performs tuning and CV
(`--stage generate`). A CPU metric array follows each generation array using `aftercorr`, so a metric
task can start as soon as its matching generator task succeeds. Failed dependencies cancel the
affected metric tasks. Aggregation uses `afterany` on preparation and every submitted generation
and metric array, so it also waits for generators if a dependent metric array is cancelled early.

One array index owns a dataset/model pair; Optuna trials and folds are sequential within that task.
The immutable plan records the resolved device for each model and task. Stdout/stderr go to
`slurm_logs/`; job IDs are saved in `pipeline/submissions.log` beside the plan.
The aggregate exits nonzero if any planned task lacks a complete evaluation, after saving results.

## Resume and inspect results

Re-run the same command to continue remaining trials and unfinished folds. Completed artifacts
are verified before reuse. Changing code, dependencies, dataset definitions, protocol, metrics,
search spaces, model selection or resolved devices requires a fresh output root. V1–v3 protocol files are retained for provenance;
current v4 results must not be mixed with their earlier metric definitions.

Use `SBTAB_TASK_IDS=0,7` for selected array indices and `SBTAB_STAGE` for `all`, `generate`, `tune`,
`cv`, or `metrics`. `generate` performs tuning and CV without evaluation; `metrics` submits only CPU
evaluation tasks after preparation. Other planning arguments must match the original plan. Set
`SBTAB_RETRY_FAILED_FOLDS=1` to retry recorded failed folds. Mid-fit optimizer recovery is not
supported: interrupted CV fits restart; interrupted tuning trials remain allocated failures.
Concurrent workers and standalone tuning processes are locked to prevent duplicate allocation.

Metrics are recalculated from saved fold preprocessors and synthetic tables, without loading
or retraining a generator. Real-data utility references are shared across generators for the same
dataset, fold and evaluator configuration.

| Artifact | Path relative to the output root |
|---|---|
| Frozen plan, model/task devices, array indices and exclusions | `pipeline/plan.json` |
| Dataset preflight outcomes | `pipeline/preparation.json` |
| Task outcomes and failure details | `pipeline/tasks/00000.json`, etc. |
| Overall completion/failure summary | `pipeline/summary.json` |
| Eligibility, data and split memberships | `<dataset>/eligibility_report.json`, `data.parquet`, `splits.json` |
| Study and selected hyperparameters | `<dataset>/<model>/run-pipeline/tuning/` |
| Fresh models, synthetic tables and timing | `<dataset>/<model>/run-pipeline/cv/fold-<k>/` |
| Per-fold metrics and predictions | `<dataset>/<model>/run-pipeline/evaluation/` |
| Combined results | `aggregate/` |

Each fold records generator fit time separately from preprocessing, model initialization, sampling,
checkpoint I/O and inverse transformation. GPU timers synchronize CUDA at their boundaries. Aggregates
include valid-fold counts and sample standard deviation (`ddof=1`), preserving undefined metrics
and failures instead of replacing them with zero.

Inspect trial `status.json` and fold `manifest.json` for `failure_kind` and structured
`failure.details`: numerical failures include the available solver direction, step, noise level
and chunk. `checkpoint_loaded` and `sampling_probe` distinguish a loading error from failed
generation after loading. `numerical_diagnostics` flags extreme finite outputs without changing
selection or clipping values. Reuse of a final selection requires the minimum-objective trial's
checkpoint and sampling probe to pass; a failed winner is never replaced silently.

Evaluation writes to a new `*-eval3` namespace and reports generation, fidelity, utility and
evaluation completeness separately. `complete_five_fold` requires successful evaluation in all
five folds. Overall metric ranks use complete five-fold datasets shared by every compared model.
Old per-run summaries remain intact; new aggregation recomputes their completion from fold evidence.
Use a fresh output root for runs after an implementation change.

The same plan/prepare/worker/aggregate commands can run locally without Slurm; the README contains
a complete CPU smoke example. Production results require completing all 100 trials and five folds.
