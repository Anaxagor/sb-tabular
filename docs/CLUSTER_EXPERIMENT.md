# Complete experiment pipeline on SLURM

The cluster scripts use **partition `rocky`** and **account `proj_1752`**. Each array task runs one
compatible dataset/model pair through tuning, five-fold training, and test evaluation. DSB and DSBM
are restricted to **`dsb_ct_joint_mlp`** and **`dsbm_ct_joint_mlp`**: one continuous-time joint MLP
per family. All other DSB/DSBM configurations are excluded, including explicit selections through
`--models` and standalone tuning/CV commands. Other families retain their existing selection rules,
including the explicitly labelled `csbm_annealed` heuristic. Missing dependencies, unavailable implementations,
incompatible regimes and dataset support failures are recorded.

The supplied launch configuration uses **one GPU, eight CPUs and two days per
array task**, with separate stdout/stderr logs and cluster-default memory. The array is indexed by dataset × model, not by
trial: each task tunes and then trains its own model. Up to eight tasks run concurrently by default.
GPU execution is mandatory for generator tuning, CV training and sampling. Preparation,
metrics and the fixed CatBoost utility evaluator use CPUs.

## 1. Experimental setup

1. Split the eligible dataset into tuning training pool **T (85%)** and validation set **V (15%)**,
   stratified by the classification target or regression target quantiles, with `random_state=5`.
2. Fit preprocessing exclusively on T during tuning. Standardize continuous columns, label-encode
   categorical columns, and leave discrete numerical values unchanged. The declared target remains
   part of the generated row. Missing-value handling follows the dataset configuration.
3. Run **100 allocated Optuna trials per dataset/model pair**, sequentially within that task. Fit each
   candidate on T, generate exactly `len(V)` rows, and minimize the validation objective below. Failed
   attempts count toward the budget. Select the minimum finite objective among completed trials.
4. Within T, construct **five shuffled KFold splits**, with `random_state=42`. V never enters these
   folds. Each split has training rows T_k and held-out test rows E_k.
5. Fit a fresh preprocessor and model on T_k using the selected hyperparameters. Generate exactly
   `len(T_k)` rows. Save the checkpoint, synthetic data, seeds, transformation state and timing.
6. Compare synthetic data against E_k. Compute TSTR on E_k with the same test rows as the real-data
   reference, then aggregate fold results with arithmetic means and sample standard deviations.

This follows the requested CV procedure: **the test sets are the held-out CV folds within T**,
not an additional fixed holdout. It is fixed-hyperparameter CV after dataset-level tuning, rather
than nested CV: the tuning candidates were fitted on all of T.

| Data regime | Validation objective | Test metrics from the supplied table |
|---|---|---|
| Continuous | Mean Wasserstein distance | Mean WD, mean KL (50 bins), Pearson association distance, utility ΔR² (%) |
| Discrete/categorical | Mean Jensen–Shannon divergence | Mean KL, Spearman distance for discrete columns, NMI association distance for categorical columns, utility ΔF1 (%) |
| Mixed | Mean WD on continuous columns + mean JS on discrete/categorical columns | Continuous WD/KL/Pearson, discrete KL/Spearman, categorical KL/NMI, task-appropriate utility gaps |

The existing evaluator also computes validity, conditional diagnostics, mixed-space MMD, MAE, RMSE
and MAPE where applicable. Metrics whose inputs are insufficient or undefined keep their status;
they are not replaced by zeros. The 50 continuous KL bins include underflow and overflow bins, with
edges fitted on T_k. Association distances compare real and synthetic association matrices.

For utility, use **CatBoostClassifier for classification (macro F1)** and **CatBoostRegressor for
regression (R², MAE, RMSE, MAPE)**. CatBoost's automatic defaults are resolved once on the first real
training fold of each dataset, then frozen across every generator and fold for that dataset. The
reference and synthetic fits use identical parameters, CPU, seed 0 and four threads; there is no
utility tuning or early stopping. Reported percentage gaps are positive when synthetic training is
worse; the denominator is `max(abs(reference_score), 1e-8)`. MAPE is undefined when a test target is zero.

The recommended launch commands below explicitly select `sbtab_8515_hpo100_cv5_v3`. It keeps v2's
eligibility rule (remove rows with finite-support values occurring fewer than three times, before
splitting) and repairs category coverage when the initial split fails. It identifies the blocking
columns, then exchanges whole rows between T and V within the same target class or regression
quantile stratum. This preserves all eligible rows, T/V sizes and stratum counts. Every value of a
blocking column must occur in V and in the training part of every CV fold; its T rows must therefore
span at least two test folds. A value with only three rows need not appear in all five test folds.
KFold itself is unchanged (`shuffle=True`, `random_state=42`) and is applied to the repaired T pool.

The repair is deterministic with seed 5 and uses only category support and target strata, never
model scores or fitted encoders. The bounded search can still fail; unresolved coverage remains
`blocked_support`. All swaps are recorded in `support_report.json` and `splits.json`, and their
deterministic reconstruction is verified when the artifacts are loaded.

V1/v2 remain available with their original behavior. Omitting `--protocol` still selects v2, under
which `breast_cancer`, `house_sales` and `student_perf` are blocked (25/28 datasets pass). V3 passes
the support checks on all 28 configured datasets without removing additional rows. For example,
`breast_cancer` retains 283 eligible rows (240 T / 43 V); two same-class swaps leave one `14-Dec`
row in V and distribute its two remaining rows across different CV test folds. All values of
`inv-nodes` occur in V. Use `--protocol configs/protocols/sbtab_8515_hpo100_cv5_v1.yaml` for the
version without eligibility filtering.

## 2. Configure the HSE cluster environment

### Edit locally and upload

Keep code, dataset definitions, search spaces, sbatch scripts and
`scripts/slurm/cluster.local.sh` in this local project. Edit the configuration locally, including
the account and any custom cluster Python path. `cluster.example.sh` is a reference template;
the supplied launch commands read `cluster.local.sh`.

Run this from the local project directory, replacing `CLUSTER_HOST` with your SSH host or alias:

```bash
# Preview the update without changing files on the cluster.
bash scripts/sync_cluster.sh --dry-run ideeva@CLUSTER_HOST:/home/ideeva/sb-tabular
# Upload and verify the copied files.
bash scripts/sync_cluster.sh ideeva@CLUSTER_HOST:/home/ideeva/sb-tabular
```

The helper needs `rsync` on both machines and SSH access; it does not use Git. The destination
parent directory must exist. Use an SSH alias for jump hosts or custom ports; use an absolute
remote project path without spaces. The helper finds the local project relative to its own path.

It uploads project files, including `cluster.local.sh`, new uncommitted files and the bundled
datasets, using content checksums rather than timestamps alone. It does not apply `.gitignore`
as a transfer filter. A second checksum pass verifies that the uploaded files match. The helper
removes obsolete files only inside `sbtab/`, `configs/`, `scripts/`, `tests/`, `docs/` and `examples/`.
Treat these directories as locally managed source/data. It preserves excluded paths on both ends:
`artifacts/`, `slurm_logs/`, `.venv*/`, `venv/`, `env/`, `.cache/`, `catboost_info/`, Git/editor
metadata, Python caches, root study databases and `.bak` backups. The historical `examples/sbtab`
shortcut with an absolute local path is also excluded. Other files found only at the
cluster project root are retained. Store experiment outputs under `artifacts/` or outside the
project's source directories. Custom environment directories inside the project should use one
of the excluded names; environments outside the project are unaffected.

Upload only after jobs using this project have finished or been cancelled, and submit new jobs
after upload verification succeeds. Runtime/config changes require a fresh output root; account,
resource and documentation changes alone do not. Uploading requirements updates the files, not
installed packages. If dependencies changed, run on the cluster before creating a new plan:

```bash
cd ~/sb-tabular
export SBTAB_REPO_ROOT="$PWD"
source scripts/slurm/cluster.local.sh
SBTAB_BOOTSTRAP_PYTHON="$SBTAB_PYTHON" \
  bash scripts/setup_cluster_env.sh "$(dirname "$(dirname "$SBTAB_PYTHON")")"
"$SBTAB_PYTHON" -m sbtab.experiments.cluster_environment --download-tabpfn
```

### Initial environment setup

Follow HSE's [Rocky Linux instructions](https://hpc.hse.ru/instructions/rocky) and
[Python environment instructions](https://hpc.hse.ru/instructions/python/anaconda): create a fresh
Python environment on **login-02**. Compute nodes have **no Internet access**, so install all
packages and cache pretrained weights before submitting work.

Use a shared checkout and output directory at the same absolute paths on login and compute nodes.
The output filesystem must support POSIX `flock` and atomic rename. Run these commands from the
updated repository checkout on login-02:

```bash
module purge
module load python
cd /shared/path/to/sb-tabular

conda create --prefix "$PWD/.venv-cluster" python=3.11 pip -y
SBTAB_BOOTSTRAP_PYTHON="$PWD/.venv-cluster/bin/python" bash scripts/setup_cluster_env.sh

export SBTAB_REPO_ROOT="$PWD"
source scripts/slurm/cluster.local.sh
"$SBTAB_PYTHON" -m sbtab.experiments.cluster_environment --download-tabpfn
```

The setup script installs **PyTorch 2.6.0 with CUDA 12.4**, then the pinned direct dependencies in
`requirements-cluster.txt`, runs `pip check`, and saves all resolved package versions to
`.venv-cluster/requirements-resolved.txt`. Both CTGAN and TabPFGen are included. TabPFGen 0.1.4 needs
SciPy >=1.15 and scikit-learn 1.5.2; the profile pins compatible versions and TabPFN 2.0.9.
The CUDA wheel carries its runtime; installing a separate CUDA toolkit is unnecessary for these
prebuilt packages. See the [official PyTorch 2.6 installation options](https://pytorch.org/get-started/previous-versions/).
`SBTAB_CUDA_WHEEL=cu118` or `cu126` can select another PyTorch 2.6 CUDA build during setup if needed.
Create a new plan after changing the environment.

The environment path can be changed by passing it to `scripts/setup_cluster_env.sh` and setting
`SBTAB_PYTHON` in `cluster.local.sh`. The included configuration points to `.venv-cluster/bin/python` and shares
TabPFN weights under `.cache/tabpfn`. The environment check downloads both default classifier and
regressor weights, verifies package imports/versions and prints weight SHA256 hashes. Plans freeze
those hashes and workers reject changed or missing weights. Batch jobs run with `HF_HUB_OFFLINE=1`.

Both the included `scripts/slurm/cluster.local.sh` and the reference template contain:

```bash
SBATCH_SITE_ARGS=(--partition=rocky --account=proj_1752)
SBTAB_PREPARE_ARGS=(--cpus-per-task=4 --time=01:00:00)
SBTAB_EXPERIMENT_ARGS=(--gpus=1 --cpus-per-task=8 --time=2-00:00:00)
SBTAB_AGGREGATE_ARGS=(--cpus-per-task=1 --time=01:00:00)
SBTAB_DEVICE=cuda
```

`scripts/slurm/cluster.local.sh` is included in the local project and is ready to upload; no
template-copy step is required. Its paths resolve relative to the project root on the cluster.
Include this file when copying updates, because it overrides the submitter's defaults; uploading
only `cluster.example.sh` does not change the active configuration. If the cluster uses a custom
Python environment path, put that path in your local configuration before uploading it.
To update only the account in an older cluster copy while preserving its other settings, run on
the cluster from the repository root:

```bash
sed -i.bak 's/proj_1825/proj_1752/g' scripts/slurm/cluster.local.sh
```

The backup is `cluster.local.sh.bak`. The submitter prints the absolute configuration path and
each exact `sbatch` command before submission (and in `--dry-run` mode); check its `--account`
option. Existing jobs retain the account used when they were submitted.

If `cluster.local.sh` was copied from the earlier template, remove its memory requests and the
submitter's old defaults on login-02 before retrying:

```bash
sed -i -E 's/ --mem=[^ )]+//g' \
  scripts/slurm/submit.sh scripts/slurm/cluster.example.sh scripts/slurm/cluster.local.sh
```

This only changes scheduler resources, so the existing experiment plan can be reused.

The submitter freezes effective search-space copies with `device: cuda` for neural generators and
`enable_gpu: true` for CTGAN. Hyperparameter ranges and budgets remain unchanged. Workers verify an
actual CUDA matrix multiplication before allocating trials; unavailable CUDA is an error, never a
CPU fallback. Memory is left to the cluster defaults, without explicit memory requests. Wall-time
settings are starting values, not measured production bounds. No GPU type constraint is imposed.

Run the compute-node check before the production array. It performs real GPU fit, sample and
checkpoint reload checks for all ten selected models, using tiny synthetic inputs:

```bash
mkdir -p slurm_logs
sbatch --wait \
  scripts/slurm/check_gpu.sbatch "$PWD" "$PWD/scripts/slurm/cluster.local.sh"
```

A successful check must report ten passing CUDA tests. Local CPU checks cannot establish this.
Keep code, configuration, weights and packages unchanged once plans have been created.

All configured datasets are read from the three tracked bundles under `sbtab/data/datasets/`;
batch preparation does not download datasets. Keep these files in the cluster checkout.

## 3. Preview and submit

Preview the full production array. This writes its immutable plan and prints the `sbatch` commands,
without submitting jobs, loading datasets or fitting models:

```bash
bash scripts/slurm/submit.sh --dry-run scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-production \
  --protocol configs/protocols/sbtab_8515_hpo100_cv5_v3.yaml
```

Submit with the same arguments, omitting `--dry-run`:

```bash
bash scripts/slurm/submit.sh scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-production \
  --protocol configs/protocols/sbtab_8515_hpo100_cv5_v3.yaml
```

Submission creates three jobs:

1. **Preparation:** load each selected dataset once, save immutable splits and support reports.
2. **Experiment array:** start after successful preparation. One task owns one dataset/model pair;
   at most eight tasks run simultaneously by default (`SBTAB_MAX_CONCURRENT`). Its Optuna trials
   and five folds execute sequentially, while different pairs run in parallel.
3. **Aggregation:** run after every array task finishes, including failures. The summary includes
   blocked, failed, interrupted and unstarted tasks as well as successful results.

The dependencies use SLURM's `afterok` and `afterany` semantics; logs use `%A_%a` for the array job
and task IDs. See the [official SLURM array documentation](https://slurm.schedmd.com/job_array.html).
All four batch scripts save separate stdout (`.out`) and stderr (`.err`) files in
`slurm_logs/` at the repository root, with prefixes `prepare`, `experiment`, `aggregate` and
`gpu-check`. The submitter creates this directory before submitting jobs and passes absolute log
paths. For direct `sbatch` calls, run from the repository root and create `slurm_logs/` first, as
shown for the GPU check above. Logs from earlier launches remain at their original paths.
If preparation fails, `--kill-on-invalid-dep=yes` prevents the array from waiting indefinitely.
The aggregator exits nonzero when any planned task lacks a successful full evaluation, after saving
the available results and failure summary. Dataset support failures therefore remain visible.

With the pinned environment, the production plan contains **232 candidate tasks across 28 dataset
configurations and ten models**. With protocol v3 all 232 pairs pass the dataset-support checks;
v2 blocks 26 pairs across three datasets and leaves 206 pairs.
TabPFGen now samples its conditioning context without replacement from the current training
subset, up to 10,000 rows on GPU or 1,000 on CPU. Classification sampling retains every target
class. The fit seed determines the subset; checkpoints retain its row positions and sampling
metadata. Codecs and label-prior estimates still use the full training subset. Validation and
test rows are excluded from context selection, and synthetic output sizes remain those required
by the protocol. The 500 encoded-feature and ten-class limits still apply.
The full-plan aggregate reports dataset support failures and exits nonzero even when all runnable
tasks succeed.

For an initial cluster check, use a separate smoke output root:

```bash
bash scripts/slurm/submit.sh scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-smoke --smoke \
  --protocol configs/protocols/sbtab_smoke_v3.yaml \
  --datasets diabetes car_evaluation insurance --models lightsb csbm mixedsbm
```

Smoke uses three trials and small training budgets on the allocated GPU. It validates the execution path; it does not
replace the production experiment. Omitting `--datasets` and `--models` selects all available pairs
within the model scope above. The two retained DSB/DSBM models keep their existing hyperparameter ranges; execution devices are fixed to CUDA.
Use `--exclude-heuristic` to omit `csbm_annealed`. Custom search-space directories can be selected
with `--search-space-dir`; the protocol and the search-space kind must match.

## 4. Resume and inspect results

Git is **not required on the cluster**. Copying the project files is supported; there is no need
to copy `.git`. Runtime compatibility is checked using file contents under `sbtab/` and `configs/`,
independently of Git availability, Git metadata and Python bytecode caches. Git information, when
available, is recorded only for reporting. Dataset values and split memberships have separate
integrity checks.

Plans created with the earlier Git-based implementation hash must be recreated in a new output
root after copying the corrected code. The UTF-8, TabPFGen context and coverage-repair fixes also
change the implementation hash. Select v3 explicitly and use a fresh output root after copying:

```bash
cd /shared/path/to/sb-tabular
bash scripts/slurm/submit.sh scripts/slurm/cluster.local.sh \
  --output-root "$PWD/artifacts/production-gpu-v4" \
  --protocol configs/protocols/sbtab_8515_hpo100_cv5_v3.yaml
```

Create the plan on the cluster after finishing the file copy. Do not overwrite the code or
configuration while jobs are queued or running. If preparation fails before the array starts,
an aggregate summary with `not_run` for every task means no generator training took place.
Experiment YAML/JSON files and saved preprocessing/adapter metadata are explicitly read and
written as UTF-8, including on workers with an ASCII locale. No locale flag is needed for these files.

Re-run the same submission command to continue remaining trials and unfinished folds. Completed
trials keep their budget allocations and successful folds are verified and reused. The plan is
immutable; changing models, datasets, code, dependencies or search spaces requires a new output root.
Plans created before the DSB/DSBM restriction must be regenerated in a new output root; they cannot
be resumed under the reduced model selection.

To resubmit only selected task IDs, set `SBTAB_TASK_IDS` to a comma-separated list. To recalculate
metrics without generator training, also select the metrics stage:

```bash
SBTAB_TASK_IDS=0,7 SBTAB_STAGE=metrics \
  bash scripts/slurm/submit.sh scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-production
```

Stage choices are `all`, `tune`, `cv`, and `metrics`. The data/model selection arguments must match
those used to create the plan. A recorded failed CV fold is retained by default; set
`SBTAB_RETRY_FAILED_FOLDS=1` to retrain only failed folds. Successful fold artifacts are retained.
Metrics are recalculated from saved synthetic tables and preprocessors, without loading generator
checkpoints. Shared real-data utility references are fitted once per dataset/fold/evaluator and reused.

Mid-fit training resume is still unsupported. On timeout/preemption, the in-progress tuning trial
is marked failed on the next resume and counts toward the 100-trial allocation. An unfinished CV
fit starts again from scratch. Choose wall time with this limitation in mind; resubmission does not
recover optimizer state from a partially trained model.

| Output | Location under the output root unless stated otherwise |
|---|---|
| Array indices, exclusions and frozen configuration | `pipeline/plan.json` |
| Dataset preflight outcomes | `pipeline/preparation.json` |
| Batch stdout/stderr logs | `<repository>/slurm_logs/` |
| Submitted job IDs | `pipeline/submissions.log` |
| Per-task stage outcomes and failure traces | `pipeline/tasks/00000.json`, etc. |
| Overall completion/failure counts | `pipeline/summary.json` |
| Data and train/validation/test memberships | `<dataset>/data.parquet`, `<dataset>/splits.json` |
| Study, trial checkpoints and selected hyperparameters | `<dataset>/<model>/run-pipeline/tuning/` |
| Fresh fold models and generated datasets | `<dataset>/<model>/run-pipeline/cv/fold-<k>/` |
| Training and generation times in seconds for each CV fold | `<dataset>/<model>/run-pipeline/cv/fold-<k>/timing.json` |
| Per-fold metrics, utility predictions, mean/std summaries | `<dataset>/<model>/run-pipeline/evaluation/*-eval2/` |
| Combined CSV/Parquet and JSON results | `aggregate/` |

Training times are recorded automatically for every model during CV; no SLURM option is needed.
Each fold's `timing.json` contains a `seconds` object with:

- `generator_fit_seconds`: generator fitting, including internal encodings and all IPF/IMF stages,
  coupling/cache refreshes and graph learning. Common preprocessing, model initialization,
  checkpoint output, generation and utility evaluation are excluded.
- `training_total_seconds`: common preprocessing + model initialization + generator fitting.
- `generation_seconds`: sampling time, measured separately from training.
- `inverse_transform_seconds`: decoding the model output to the common table representation.

Timers use wall-clock intervals with CUDA synchronization at their boundaries, so queued GPU work
finishes before the interval ends. Fold manifests also retain the timing values. Reusing a completed
fold preserves its original timings; caught failures retain elapsed times in the fold artifacts.
After evaluation, `summary.csv` / `summary.json` contain `generator.generator_fit_seconds` and
`generator.training_total_seconds`, including the mean, sample standard deviation (`ddof=1`) and
valid-fold count. `aggregate/per_fold_all.csv` contains the individual values for all evaluated
dataset/model/fold combinations. TabPFGen's `fit` stores a conditioning context; its main SGLD and
TabPFN work is performed during sampling and is therefore reported as generation time.

The same worker can be used locally for debugging:

```bash
python -m sbtab.experiments.pipeline plan --output-root artifacts/local-smoke \
  --smoke --datasets diabetes --models lightsb
python -m sbtab.experiments.pipeline prepare --plan artifacts/local-smoke/pipeline/plan.json
python -m sbtab.experiments.pipeline worker --plan artifacts/local-smoke/pipeline/plan.json --task-id 0
python -m sbtab.experiments.pipeline aggregate --plan artifacts/local-smoke/pipeline/plan.json
```

## 5. GPU launch preparation (2026-09-26)

The pinned Python 3.11 environment was installed from scratch and passed `pip check`. Both default
TabPFN v2 weight files were downloaded and verified. Real CTGAN and TabPFGen libraries are included
in the tests; optional-baseline validation no longer relies only on fakes.

The final suite passed **880 tests, with 12 skips**: ten need CUDA, one needs the optional notebook
validator `nbformat`, and one tests the absence of `geotorch`, which is installed in this environment.
Batch entry points completed **24 dataset/model smoke pipelines spanning all ten generators**,
with **72 tuning trials and 120 fresh CV fits**, checkpoint reload checks, test metrics, TSTR and
aggregation. This includes a fresh complete TabPFGen regression run after the checkpoint correction.
Two long TabPFGen classification batch checks on CPU were interrupted; its real-library classification
unit tests passed, and its GPU path remains part of the required compute-node check.

The launch review corrected SDV's version-dependent CUDA argument, TabPFGen's CUDA label-counting
bug, and a TabPFGen checkpoint layout mismatch that changed regression predictions. It also added
pre-trial TabPFN context-limit checks, immutable CUDA search configurations, weight hashes and the
compute-node smoke-check script. The production submission preview requests `--array=0-231%8`,
`--gpus=1`, eight CPUs, partition `rocky` and account `proj_1752`.

These checks run locally on CPU. No production benchmark or real SLURM submission was performed,
and the CUDA tests require the allocated-GPU check in Section 2. Local evidence is kept in
[the validation report](../artifacts/cluster_readiness_2026-09-26/validation.json),
[the test log](../artifacts/cluster_readiness_2026-09-26/pytest-final.log), and
[the corrected TabPFGen pipeline log](../artifacts/cluster_readiness_2026-09-26/tabpfgen-corrected.log)
(ignored by Git).

## 6. Earlier verification (2026-09-24)

The complete test suite passed: **798 passed, 9 skipped**. Eleven orchestration tests cover immutable
plans, the production budget, dependency failures, blocked datasets, interrupted tuning/CV, explicit
failed-fold retries, duplicate-worker locking, concurrent utility caches, and the sbatch arguments
and dependency chain. The nine skipped tests require unavailable optional dependencies.

All three batch entry points were also executed locally, including seven experiment tasks running
two at a time: the compatible pairs of `diabetes`, `car_evaluation`, `insurance` with `lightsb`,
`csbm`, and `mixedsbm`. All seven passed, completing **21 smoke tuning trials and 35 fresh CV fits**,
checkpoint reload verification, held-out metrics, TSTR and aggregation. The original local production
plan, before the DSB/DSBM restriction, contained 428 tasks across 28 datasets and 17 available models;
381 tasks passed dataset preflight.
The 47 tasks involving the three blocked datasets retain explicit support failures.

Evidence is in [pytest output](../artifacts/cluster_validation_2026-09-24/pytest.txt),
[batch validation](../artifacts/cluster_validation_2026-09-24/batch_validation.json), and
[the smoke summary](../artifacts/cluster_validation_2026-09-24/smoke/pipeline/summary.json).
These are ignored local artifacts. No production training or real SLURM submission was performed;
the cluster's installed dependencies and scheduler resource limits must be used when creating its plan.
