#!/usr/bin/env bash
# Copy to the ignored cluster.local.sh. Both submitter and batch workers source it.
# Resource rules: https://hpc.hse.ru/llms.txt and /hardware/hpc-cluster
SBATCH_SITE_ARGS=(--partition=rocky --account=proj_1752)
SBTAB_PYTHON="${SBTAB_PYTHON:-$SBTAB_REPO_ROOT/.venv-cluster/bin/python}"

# Automatic routing: ForestDiffusion/boosted solvers on CPU; neural models on V100.
# Types A/B/C all have V100 32GB. No node numbers or explicit memory requests.
SBTAB_PREPARE_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=4 --time=01:00:00)
SBTAB_EXPERIMENT_ARGS=(--constraint='type_a|type_b|type_c' --gpus=1 --cpus-per-task=4 --time=3-00:00:00)
SBTAB_CPU_EXPERIMENT_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=4 --time=3-00:00:00)
SBTAB_METRICS_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=4 --time=12:00:00)
SBTAB_AGGREGATE_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=1 --time=01:00:00)
# Limit per submitted array (CPU, GPU and corresponding metrics arrays separately).
SBTAB_MAX_CONCURRENT="${SBTAB_MAX_CONCURRENT:-8}"
SBTAB_DEVICE="${SBTAB_DEVICE:-auto}"

# Keep the same plan/output root only when code, dependencies and model set match.
SBTAB_STAGE="${SBTAB_STAGE:-all}"       # all | generate | tune | cv | metrics
SBTAB_TASK_IDS="${SBTAB_TASK_IDS:-}"     # optional comma-separated task indices
SBTAB_RETRY_FAILED_FOLDS="${SBTAB_RETRY_FAILED_FOLDS:-0}"
