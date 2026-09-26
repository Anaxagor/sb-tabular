# Complete experiment pipeline on SLURM

The cluster scripts use **partition `rocky`** and **account `proj_1825`**. Each array task runs one
compatible dataset/model pair through tuning, five-fold training, and test evaluation. DSB and DSBM
are restricted to **`dsb_ct_joint_mlp`** and **`dsbm_ct_joint_mlp`**: one continuous-time joint MLP
per family. All other DSB/DSBM configurations are excluded, including explicit selections through
`--models` and standalone tuning/CV commands. Other families retain their existing selection rules,
including the explicitly labelled `csbm_annealed` heuristic. Missing dependencies, unavailable implementations,
incompatible regimes and dataset support failures are recorded.

With the dependencies currently installed locally, the reduced selection has **8 models and 176
dataset/model tasks** across the 28 configured datasets. Thirteen new model-selection checks plus
the eleven existing pipeline checks passed on 2026-09-26. Cluster counts also depend on which
optional model dependencies are installed.

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

The default protocol remains `sbtab_8515_hpo100_cv5_v2`: its existing eligibility rule removes rows
with finite-support values occurring fewer than three times, before splitting, and records the
removals. Use `--protocol configs/protocols/sbtab_8515_hpo100_cv5_v1.yaml` for the version without
this filtering. Target stratification alone cannot guarantee feature-category coverage. Both
protocols validate support in V and every E_k and stop a dataset that fails. The last full preflight
found 25/28 eligible configurations passing v2; `breast_cancer`, `house_sales`, and `student_perf`
remain blocked. No alternative seed or test-informed encoder is used to bypass these failures.

## 2. Configure the cluster environment

Use a shared checkout and a shared output directory accessible at the **same absolute paths** on
the login and compute nodes. The output filesystem must support POSIX `flock` and atomic rename.
Each dataset/model study has one writer. Locks protect duplicate workers and shared utility caches.
Generate the plan on the cluster, after setting up the same Python environment used by batch jobs.

```bash
cd /shared/path/to/sb-tabular
cp scripts/slurm/cluster.example.sh scripts/slurm/cluster.local.sh
```

Edit `cluster.local.sh` to activate the environment, load the required modules, or set
`SBTAB_PYTHON=/shared/path/to/environment/bin/python`. The template already contains:

```bash
SBATCH_SITE_ARGS=(--partition=rocky --account=proj_1825)
SBTAB_PREPARE_ARGS=(--cpus-per-task=4 --mem=16G --time=01:00:00)
SBTAB_EXPERIMENT_ARGS=(--cpus-per-task=4 --mem=32G --time=2-00:00:00)
SBTAB_AGGREGATE_ARGS=(--cpus-per-task=1 --mem=8G --time=01:00:00)
```

These memory and time limits are configurable starting values. The production search spaces run
on CPU and use four CatBoost threads. Request at least four CPUs for experiment tasks. GPU execution
requires a separate plan with appropriate model device settings in the search-space YAMLs and matching
SLURM GPU requests; allocating a GPU alone does not change a model's device.

Install the required dependencies in that environment before creating the plan. CTGAN requires SDV;
TabPFGen requires `tabpfgen` and `tabpfn`. By default, unavailable dependencies produce explicit
exclusions in the plan. An explicitly requested model with missing dependencies causes planning to
fail. The plan checks library versions, source provenance and configuration hashes on every worker;
keep the checkout and environment unchanged while jobs run.

## 3. Preview and submit

Preview the full production array. This writes its immutable plan and prints the `sbatch` commands,
without submitting jobs, loading datasets or fitting models:

```bash
bash scripts/slurm/submit.sh --dry-run scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-production
```

Submit with the same arguments, omitting `--dry-run`:

```bash
bash scripts/slurm/submit.sh scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-production
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
If preparation fails, `--kill-on-invalid-dep=yes` prevents the array from waiting indefinitely.
The aggregator exits nonzero when any planned task lacks a successful full evaluation, after saving
the available results and failure summary. Dataset support failures therefore remain visible.

For an initial cluster check, use a separate smoke output root:

```bash
bash scripts/slurm/submit.sh scripts/slurm/cluster.local.sh \
  --output-root /shared/results/sbtab-smoke --smoke \
  --datasets diabetes car_evaluation insurance --models lightsb csbm mixedsbm
```

Smoke uses three trials and small training budgets. It validates the execution path; it does not
replace the production experiment. Omitting `--datasets` and `--models` selects all available pairs
within the model scope above. The two retained DSB/DSBM models keep their existing search spaces.
Use `--exclude-heuristic` to omit `csbm_annealed`. Custom search-space directories can be selected
with `--search-space-dir`; the protocol and the search-space kind must match.

## 4. Resume and inspect results

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

| Output | Location under the output root |
|---|---|
| Array indices, exclusions and frozen configuration | `pipeline/plan.json` |
| Dataset preflight outcomes | `pipeline/preparation.json` |
| Batch logs and submitted job IDs | `pipeline/logs/`, `pipeline/submissions.log` |
| Per-task stage outcomes and failure traces | `pipeline/tasks/00000.json`, etc. |
| Overall completion/failure counts | `pipeline/summary.json` |
| Data and train/validation/test memberships | `<dataset>/data.parquet`, `<dataset>/splits.json` |
| Study, trial checkpoints and selected hyperparameters | `<dataset>/<model>/run-pipeline/tuning/` |
| Fresh fold models and generated datasets | `<dataset>/<model>/run-pipeline/cv/fold-<k>/` |
| Per-fold metrics, utility predictions, mean/std summaries | `<dataset>/<model>/run-pipeline/evaluation/*-eval2/` |
| Combined CSV/Parquet and JSON results | `aggregate/` |

The same worker can be used locally for debugging:

```bash
python -m sbtab.experiments.pipeline plan --output-root artifacts/local-smoke \
  --smoke --datasets diabetes --models lightsb
python -m sbtab.experiments.pipeline prepare --plan artifacts/local-smoke/pipeline/plan.json
python -m sbtab.experiments.pipeline worker --plan artifacts/local-smoke/pipeline/plan.json --task-id 0
python -m sbtab.experiments.pipeline aggregate --plan artifacts/local-smoke/pipeline/plan.json
```

## 5. Verification performed locally (2026-09-24)

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
