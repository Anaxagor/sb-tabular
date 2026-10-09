#!/usr/bin/env bash
# Cluster configuration included with the project for direct file uploads.
# Sourced by the submitter AND each batch job.
# Edit this file locally, then upload with scripts/sync_cluster.sh.
SBATCH_SITE_ARGS=(--partition=rocky --account=proj_1752)

# The environment and caches live on the cluster, relative to its project root.
SBTAB_PYTHON="${SBTAB_PYTHON:-$SBTAB_REPO_ROOT/.venv-cluster/bin/python}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$SBTAB_REPO_ROOT/.cache/tabpfn}"
export HF_HOME="${HF_HOME:-$SBTAB_REPO_ROOT/.cache/huggingface}"

# GPU tuning/training/sampling; CPU preprocessing and utility evaluation.
# Leave memory allocation to the cluster defaults.
SBTAB_PREPARE_ARGS=(--cpus-per-task=4 --time=01:00:00)
SBTAB_EXPERIMENT_ARGS=(--gpus=1 --cpus-per-task=8 --time=2-00:00:00)
SBTAB_AGGREGATE_ARGS=(--cpus-per-task=1 --time=01:00:00)
SBTAB_MAX_CONCURRENT="${SBTAB_MAX_CONCURRENT:-8}"
SBTAB_DEVICE="${SBTAB_DEVICE:-cuda}"

# Recovery controls for resuming an existing plan/output root.
SBTAB_STAGE="${SBTAB_STAGE:-all}"       # all | tune | cv | metrics
SBTAB_TASK_IDS="${SBTAB_TASK_IDS:-}"
SBTAB_RETRY_FAILED_FOLDS="${SBTAB_RETRY_FAILED_FOLDS:-0}"
