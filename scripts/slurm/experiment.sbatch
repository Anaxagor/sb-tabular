#!/usr/bin/env bash
#SBATCH --job-name=sbtab-experiment
#SBATCH --output=slurm_logs/experiment-%A_%a.out
#SBATCH --error=slurm_logs/experiment-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=proj_1752
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
set -euo pipefail
source "$1/scripts/slurm/common.sh" "$@"
: "${SLURM_ARRAY_TASK_ID:?This script must be submitted as an array}"
SBTAB_WORKER_ARGS=(--plan "$SBTAB_PLAN" --task-id "$SLURM_ARRAY_TASK_ID" --stage "${4:-all}")
if [[ ${5:-0} == 1 ]]; then
    SBTAB_WORKER_ARGS+=(--retry-failed-folds)
fi
exec "$SBTAB_PYTHON" -u -m sbtab.experiments.pipeline worker "${SBTAB_WORKER_ARGS[@]}"
