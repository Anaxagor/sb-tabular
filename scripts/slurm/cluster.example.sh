#!/usr/bin/env bash
# Copy to cluster.local.sh. Sourced by the submitter AND each batch job.
SBATCH_SITE_ARGS=(--partition=rocky --account=proj_1825)

# Either point at the environment's interpreter or activate it in this file.
# Example: source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate synth
# Example: module load Python/3.11
SBTAB_PYTHON="${SBTAB_PYTHON:-python}"

# CPU defaults match the repository's current search spaces. Tune requests to the
# cluster limits and dataset sizes; these are starting values, not measured bounds.
SBTAB_PREPARE_ARGS=(--cpus-per-task=4 --mem=16G --time=01:00:00)
SBTAB_EXPERIMENT_ARGS=(--cpus-per-task=4 --mem=32G --time=2-00:00:00)
SBTAB_AGGREGATE_ARGS=(--cpus-per-task=1 --mem=8G --time=01:00:00)
SBTAB_MAX_CONCURRENT="${SBTAB_MAX_CONCURRENT:-8}"

# Recovery controls. Reuse the same plan/output root to resume remaining trials
# and unfinished folds. Set SBTAB_TASK_IDS=0,7 to resubmit only those array tasks.
SBTAB_STAGE="${SBTAB_STAGE:-all}"       # all | tune | cv | metrics
SBTAB_TASK_IDS="${SBTAB_TASK_IDS:-}"
SBTAB_RETRY_FAILED_FOLDS="${SBTAB_RETRY_FAILED_FOLDS:-0}"
