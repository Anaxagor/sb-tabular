#!/usr/bin/env bash
#SBATCH --job-name=sbtab-gpu-check
#SBATCH --output=slurm_logs/gpu-check-%j.out
#SBATCH --error=slurm_logs/gpu-check-%j.err
#SBATCH --account=proj_1752
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:30:00
set -euo pipefail
# No experiment plan is needed for this bounded fit/sample/reload check.
source "$1/scripts/slurm/common.sh" "$1" "$2" ""
"$SBTAB_PYTHON" -u -m sbtab.experiments.cluster_environment --require-cuda
exec "$SBTAB_PYTHON" -u -m pytest -o addopts='-p no:cacheprovider' -q -rs \
    tests/experiments/test_gpu_launch.py -k actual_cuda
