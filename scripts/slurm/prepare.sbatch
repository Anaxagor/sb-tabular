#!/usr/bin/env bash
#SBATCH --job-name=sbtab-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
source "$1/scripts/slurm/common.sh" "$@"
exec "$SBTAB_PYTHON" -u -m sbtab.experiments.pipeline prepare --plan "$SBTAB_PLAN"
