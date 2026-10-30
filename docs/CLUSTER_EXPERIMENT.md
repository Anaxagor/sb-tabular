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

## Environment and local deployment

The published interface is `python -m sbtab.experiments.pipeline`. The entire `scripts/slurm/`
directory, batch files, local cluster configuration and scheduler logs are excluded from the
repository. A fresh clone uses the Python CLI directly. Cluster users can supply local scheduler
wrappers with their own account, resource settings and paths; keep those details outside tracked
source and follow the cluster's current rules.

Splitting, metrics, CPU CatBoost utility fits and aggregation require CPU allocations only.
`pipeline plan --device auto` assigns ForestDiffusion and CPU-only boosted solvers to CPU and
neural generators to CUDA. `--device cpu` makes a bounded local execution check possible without
a GPU. The immutable plan records each model's resolved device; local scheduler wrappers must
request matching resources. Measure representative tasks before setting production time limits.

For the pinned Python 3.11 environment, use `requirements-cluster.txt` through the generic setup
helper. It accepts a pre-created Python 3.11 interpreter through `SBTAB_BOOTSTRAP_PYTHON`, installs
the configured PyTorch wheel and dependencies, checks package compatibility and saves resolved
versions. On Linux it installs a CUDA-enabled PyTorch build; this does not require reserving a GPU
for CPU-only pipeline stages.

```bash
bash scripts/setup_cluster_env.sh .venv-cluster
source .venv-cluster/bin/activate
python -m sbtab.experiments.cluster_environment
```

On an allocated GPU, additionally run `python -m sbtab.experiments.cluster_environment --require-cuda`
and `python -m pytest tests/experiments/test_gpu_launch.py`. Inspect skips as well as failures:
a skipped test does not verify that model on the GPU. CPU checks do not establish GPU availability.

All workers must use the same checkout, environment and output paths. The output filesystem must
support POSIX `flock` and atomic rename. Stop jobs using a checkout before replacing its source
files. Preserve the original source, environment and complete artifacts when an older experiment
must continue under its original provenance; never edit compatibility hashes to bypass validation.

## Synchronize source

The `scripts/sync_cluster.sh` helper requires a locally supplied, ignored
`scripts/slurm/cluster.local.sh`. It copies source and local deployment files to the chosen
machine, independently of `.gitignore`; those local files remain outside the published repository.
It preserves remote environments, caches, experiment outputs and logs, verifies transferred
contents, and removes obsolete files inside managed source directories. Review the preview before
uploading. A fresh clone can use the pipeline CLI without this optional synchronization step.

```bash
bash scripts/sync_cluster.sh --dry-run USER@HOST:/absolute/path/sb-tabular
bash scripts/sync_cluster.sh USER@HOST:/absolute/path/sb-tabular
```

Install changed dependencies on the execution machine, then create the plan at its final path.
Git is not required there: source and configuration content hashes are checked independently of
Git metadata. Plans contain local absolute paths and runtime metadata; keep generated artifacts
and deployment details outside version control.

## Plan and run

A bounded CPU example selects one dataset/model pair and uses the separate smoke protocol:

```bash
python -m sbtab.experiments.pipeline plan --output-root artifacts/smoke \
  --smoke --datasets insurance --models tabbyflow --device cpu
python -m sbtab.experiments.pipeline prepare --plan artifacts/smoke/pipeline/plan.json
python -m sbtab.experiments.pipeline worker --plan artifacts/smoke/pipeline/plan.json \
  --task-id 0 --stage generate
python -m sbtab.experiments.pipeline worker --plan artifacts/smoke/pipeline/plan.json \
  --task-id 0 --stage metrics
python -m sbtab.experiments.pipeline aggregate --plan artifacts/smoke/pipeline/plan.json
```

For production, create a new output root without `--smoke`:

```bash
python -m sbtab.experiments.pipeline plan --output-root artifacts/production --device auto
python -m sbtab.experiments.pipeline prepare --plan artifacts/production/pipeline/plan.json
```

Inspect `pipeline/plan.json` for task indices, devices and exclusions. Run `worker --stage generate`
for each task, then `worker --stage metrics` for its saved outputs; use `--plan` and `--task-id`
as in the smoke example. `worker --stage all` combines both stages on the worker's allocation.
Separating metrics lets GPU generation release its allocation before CPU evaluation begins.
Call `aggregate --plan artifacts/production/pipeline/plan.json` after all planned tasks finish.

For a local Slurm deployment, map array indices to the plan's task indices. The worker accepts
`SLURM_ARRAY_TASK_ID` when `--task-id` is omitted. Preparation must complete before workers start;
metric tasks must follow successful generation for the same index, and aggregation must wait for
all tasks, including failures. One task owns one dataset/model pair; tuning trials and folds run
sequentially within that task. Scheduler wrappers, submission records and logs remain local.

The aggregate saves results and exits nonzero if any planned task lacks a complete evaluation.
Planning itself does not fit a model or submit a scheduler job.

## Resume and inspect results

### Resume an unchanged current plan

Re-run the same command to continue remaining trials and unfinished folds. Completed artifacts
are verified before reuse. Changing code, dependencies, dataset definitions, protocol, metrics,
search spaces, model selection or resolved devices requires a fresh output root. V1–v3 protocol files are retained for provenance;
current v4 results must not be mixed with their earlier metric definitions.

Select a task with `worker --task-id` and a stage with `--stage all`, `generate`, `tune`, `cv`, or
`metrics`. `generate` performs tuning and CV without evaluation; `metrics` evaluates saved outputs
on CPU after preparation. Other planning arguments must match the original plan. Pass
`--retry-failed-folds` to retry recorded failed folds. Mid-fit optimizer recovery is not supported:
interrupted CV fits restart; interrupted tuning trials remain allocated failures. Concurrent
workers and standalone tuning processes are locked to prevent duplicate allocation.

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
