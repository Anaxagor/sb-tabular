#!/usr/bin/env bash
#SBATCH --job-name=sbtab-prepare
#SBATCH --output=slurm_logs/prepare-%j.out
#SBATCH --error=slurm_logs/prepare-%j.err
#SBATCH --account=proj_1752
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
source "$1/scripts/slurm/common.sh" "$@"
exec "$SBTAB_PYTHON" -u -m sbtab.experiments.pipeline prepare --plan "$SBTAB_PLAN"
