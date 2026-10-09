#!/usr/bin/env bash
# Arguments supplied explicitly by submit.sh; no scheduler-specific working-dir assumptions.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Expected REPO_ROOT CLUSTER_CONFIG PLAN" >&2
    exit 2
fi
SBTAB_REPO_ROOT=$1
SBTAB_CLUSTER_CONFIG=$2
SBTAB_PLAN=$3
cd "$SBTAB_REPO_ROOT"
source "$SBTAB_CLUSTER_CONFIG"
: "${SBTAB_PYTHON:=python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export TQDM_DISABLE=1
# sbatch --export=ALL is used; this also preserves that environment for any srun.
export SLURM_EXPORT_ENV=ALL
