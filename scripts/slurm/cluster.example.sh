#!/usr/bin/env bash
# Copy this template to the git-ignored cluster.local.sh and adapt it locally.
# Sourced by the submitter AND each batch job. When uploading configuration
# changes, include cluster.local.sh: copying only this template has no effect.
SBATCH_SITE_ARGS=(--partition=rocky --account=proj_1752)

# Either point at the environment's interpreter or activate it in this file.
# Example: source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate synth
# Example: module load Python/3.11
SBTAB_PYTHON="${SBTAB_PYTHON:-$SBTAB_REPO_ROOT/.venv-cluster/bin/python}"
# Pre-download TabPFN weights into this shared cache before submitting jobs.
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$SBTAB_REPO_ROOT/.cache/tabpfn}"
export HF_HOME="${HF_HOME:-$SBTAB_REPO_ROOT/.cache/huggingface}"

# GPU generation, with CPU preprocessing and the fixed CatBoost utility evaluator.
# Memory is left to the cluster defaults; do not pass explicit memory requests.
SBTAB_PREPARE_ARGS=(--cpus-per-task=4 --time=01:00:00)
SBTAB_EXPERIMENT_ARGS=(--gpus=1 --cpus-per-task=8 --time=2-00:00:00)
SBTAB_AGGREGATE_ARGS=(--cpus-per-task=1 --time=01:00:00)
SBTAB_MAX_CONCURRENT="${SBTAB_MAX_CONCURRENT:-8}"
SBTAB_DEVICE="${SBTAB_DEVICE:-cuda}"

# Recovery controls. Reuse the same plan/output root to resume remaining trials
# and unfinished folds. Set SBTAB_TASK_IDS=0,7 to resubmit only those array tasks.
SBTAB_STAGE="${SBTAB_STAGE:-all}"       # all | tune | cv | metrics
SBTAB_TASK_IDS="${SBTAB_TASK_IDS:-}"
SBTAB_RETRY_FAILED_FOLDS="${SBTAB_RETRY_FAILED_FOLDS:-0}"
