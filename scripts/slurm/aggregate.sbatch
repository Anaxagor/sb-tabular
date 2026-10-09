#!/usr/bin/env bash
#SBATCH --job-name=sbtab-aggregate
#SBATCH --output=slurm_logs/aggregate-%j.out
#SBATCH --error=slurm_logs/aggregate-%j.err
#SBATCH --account=proj_1752
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
source "$1/scripts/slurm/common.sh" "$@"
exec "$SBTAB_PYTHON" -u -m sbtab.experiments.pipeline aggregate --plan "$SBTAB_PLAN"
