#!/usr/bin/env bash
# Usage: bash scripts/slurm/submit.sh [--dry-run] CLUSTER_CONFIG --output-root ROOT [plan options]
set -euo pipefail
SBTAB_DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then
    SBTAB_DRY_RUN=1
    shift
fi
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--dry-run] CLUSTER_CONFIG --output-root ROOT [--datasets ...] [--models ...] [--smoke]" >&2
    exit 2
fi
SBTAB_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
SBTAB_CLUSTER_CONFIG=$(cd -- "$(dirname -- "$1")" && pwd -P)/$(basename -- "$1")
shift
cd "$SBTAB_REPO_ROOT"
SBATCH_SITE_ARGS=(--partition=rocky --account=proj_1825)
SBTAB_PREPARE_ARGS=(--cpus-per-task=4 --mem=16G --time=01:00:00)
SBTAB_EXPERIMENT_ARGS=(--cpus-per-task=4 --mem=32G --time=2-00:00:00)
SBTAB_AGGREGATE_ARGS=(--cpus-per-task=1 --mem=8G --time=01:00:00)
source "$SBTAB_CLUSTER_CONFIG"
: "${SBTAB_PYTHON:=python}"
: "${SBTAB_MAX_CONCURRENT:=8}"
: "${SBTAB_STAGE:=all}"
: "${SBTAB_TASK_IDS:=}"
: "${SBTAB_RETRY_FAILED_FOLDS:=0}"
if [[ ! $SBTAB_MAX_CONCURRENT =~ ^[1-9][0-9]*$ ]]; then
    echo "SBTAB_MAX_CONCURRENT must be a positive integer" >&2
    exit 2
fi
case "$SBTAB_STAGE" in all|tune|cv|metrics) ;; *) echo "Invalid SBTAB_STAGE" >&2; exit 2;; esac
if [[ $SBTAB_RETRY_FAILED_FOLDS != 0 && $SBTAB_RETRY_FAILED_FOLDS != 1 ]]; then
    echo "SBTAB_RETRY_FAILED_FOLDS must be 0 or 1" >&2
    exit 2
fi
if [[ $SBTAB_DRY_RUN == 0 ]]; then
    command -v sbatch >/dev/null || { echo "sbatch is not available; use --dry-run for a local preview" >&2; exit 2; }
fi
# Plan creation is lightweight: schemas/configs/dependencies only. Dataset loading
# and support checks happen in the preparation job on a compute node.
SBTAB_PLAN_RESULT=$("$SBTAB_PYTHON" -m sbtab.experiments.pipeline plan "$@")
printf '%s\n' "$SBTAB_PLAN_RESULT"
SBTAB_PLAN=$(printf '%s' "$SBTAB_PLAN_RESULT" | "$SBTAB_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["plan"])')
SBTAB_N_TASKS=$(printf '%s' "$SBTAB_PLAN_RESULT" | "$SBTAB_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["n_tasks"])')
if [[ -n $SBTAB_TASK_IDS ]]; then
    # Accept explicit comma-separated indices. Reject ranges here to keep validation exact.
    "$SBTAB_PYTHON" -c 'import sys; s=sys.argv[1].split(","); n=int(sys.argv[2]); assert all(x.isdecimal() and 0 <= int(x) < n for x in s), "invalid task indices"' "$SBTAB_TASK_IDS" "$SBTAB_N_TASKS"
    SBTAB_ARRAY="$SBTAB_TASK_IDS%$SBTAB_MAX_CONCURRENT"
else
    SBTAB_ARRAY="0-$((SBTAB_N_TASKS - 1))%$SBTAB_MAX_CONCURRENT"
fi
SBTAB_CONTROL_DIR=$(dirname -- "$SBTAB_PLAN")
mkdir -p "$SBTAB_CONTROL_DIR/logs"

submit_job() {
    if [[ $SBTAB_DRY_RUN == 1 ]]; then
        printf '%q ' sbatch "$@" >&2
        printf '\n' >&2
        printf 'DRY_RUN\n'
    else
        local result
        result=$(sbatch "$@")
        # --parsable may emit jobid;cluster for federated submissions.
        result=${result%%;*}
        [[ $result =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $result" >&2; return 1; }
        printf '%s\n' "$result"
    fi
}
SBTAB_COMMON_ARGS=(--parsable --export=ALL --chdir="$SBTAB_REPO_ROOT" "${SBATCH_SITE_ARGS[@]}")
SBTAB_PREPARE_JOB=$(submit_job "${SBTAB_COMMON_ARGS[@]}" "${SBTAB_PREPARE_ARGS[@]}" \
    --output="$SBTAB_CONTROL_DIR/logs/prepare-%j.out" \
    "$SBTAB_REPO_ROOT/scripts/slurm/prepare.sbatch" "$SBTAB_REPO_ROOT" "$SBTAB_CLUSTER_CONFIG" "$SBTAB_PLAN")
if [[ $SBTAB_DRY_RUN == 0 ]]; then printf 'prepare %s\n' "$SBTAB_PREPARE_JOB" >> "$SBTAB_CONTROL_DIR/submissions.log"; fi
SBTAB_ARRAY_JOB=$(submit_job "${SBTAB_COMMON_ARGS[@]}" "${SBTAB_EXPERIMENT_ARGS[@]}" \
    --dependency="afterok:$SBTAB_PREPARE_JOB" --kill-on-invalid-dep=yes --array="$SBTAB_ARRAY" \
    --output="$SBTAB_CONTROL_DIR/logs/experiment-%A_%a.out" \
    "$SBTAB_REPO_ROOT/scripts/slurm/experiment.sbatch" "$SBTAB_REPO_ROOT" "$SBTAB_CLUSTER_CONFIG" "$SBTAB_PLAN" \
    "$SBTAB_STAGE" "$SBTAB_RETRY_FAILED_FOLDS")
if [[ $SBTAB_DRY_RUN == 0 ]]; then printf 'array %s\n' "$SBTAB_ARRAY_JOB" >> "$SBTAB_CONTROL_DIR/submissions.log"; fi
SBTAB_AGGREGATE_JOB=$(submit_job "${SBTAB_COMMON_ARGS[@]}" "${SBTAB_AGGREGATE_ARGS[@]}" \
    --dependency="afterany:$SBTAB_ARRAY_JOB" \
    --output="$SBTAB_CONTROL_DIR/logs/aggregate-%j.out" \
    "$SBTAB_REPO_ROOT/scripts/slurm/aggregate.sbatch" "$SBTAB_REPO_ROOT" "$SBTAB_CLUSTER_CONFIG" "$SBTAB_PLAN")
if [[ $SBTAB_DRY_RUN == 0 ]]; then printf 'aggregate %s\n' "$SBTAB_AGGREGATE_JOB" >> "$SBTAB_CONTROL_DIR/submissions.log"; fi
printf 'prepare=%s array=%s aggregate=%s\n' "$SBTAB_PREPARE_JOB" "$SBTAB_ARRAY_JOB" "$SBTAB_AGGREGATE_JOB"
