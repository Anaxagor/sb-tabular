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
SBATCH_SITE_ARGS=(--partition=rocky --account=proj_1752)
# HSE types A/B/C have V100 32GB; D is CPU-only. No node IDs or memory requests.
SBTAB_PREPARE_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=4 --time=01:00:00)
SBTAB_EXPERIMENT_ARGS=(--constraint='type_a|type_b|type_c' --gpus=1 --cpus-per-task=4 --time=3-00:00:00)
SBTAB_CPU_EXPERIMENT_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=4 --time=3-00:00:00)
SBTAB_METRICS_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=4 --time=12:00:00)
SBTAB_AGGREGATE_ARGS=(--constraint=type_d --gpus=0 --cpus-per-task=1 --time=01:00:00)
source "$SBTAB_CLUSTER_CONFIG"
printf 'Cluster configuration: %s\n' "$SBTAB_CLUSTER_CONFIG" >&2
: "${SBTAB_PYTHON:=python}"
: "${SBTAB_MAX_CONCURRENT:=8}"
: "${SBTAB_STAGE:=all}"
: "${SBTAB_DEVICE:=auto}"
: "${SBTAB_TASK_IDS:=}"
: "${SBTAB_RETRY_FAILED_FOLDS:=0}"
: "${SBTAB_BATCH_SCRIPT_DIR:=$SBTAB_REPO_ROOT/scripts/slurm}"
# Ignored local scripts must exist before planning or submitting a partial chain.
for SBTAB_BATCH_NAME in prepare experiment metrics aggregate; do
    if [[ ! -f "$SBTAB_BATCH_SCRIPT_DIR/$SBTAB_BATCH_NAME.sbatch" ]]; then
        printf 'Missing local batch script: %s/%s.sbatch; provide site-specific scripts or set SBTAB_BATCH_SCRIPT_DIR.\n' \
            "$SBTAB_BATCH_SCRIPT_DIR" "$SBTAB_BATCH_NAME" >&2
        exit 2
    fi
done
SBTAB_BATCH_SCRIPT_DIR=$(cd -- "$SBTAB_BATCH_SCRIPT_DIR" && pwd -P)
if [[ ! $SBTAB_MAX_CONCURRENT =~ ^[1-9][0-9]*$ ]]; then
    echo "SBTAB_MAX_CONCURRENT must be a positive integer" >&2
    exit 2
fi
case "$SBTAB_STAGE" in all|generate|tune|cv|metrics) ;; *) echo "Invalid SBTAB_STAGE" >&2; exit 2;; esac
case "$SBTAB_DEVICE" in auto|cpu|cuda) ;; *) echo "SBTAB_DEVICE must be auto, cpu or cuda" >&2; exit 2;; esac
if [[ $SBTAB_RETRY_FAILED_FOLDS != 0 && $SBTAB_RETRY_FAILED_FOLDS != 1 ]]; then
    echo "SBTAB_RETRY_FAILED_FOLDS must be 0 or 1" >&2
    exit 2
fi
if [[ $SBTAB_DRY_RUN == 0 ]]; then
    command -v sbatch >/dev/null || { echo "sbatch is not available; use --dry-run for a local preview" >&2; exit 2; }
fi
# Planning reads schemas/configuration only; data preparation runs on a compute node.
SBTAB_PLAN_RESULT=$("$SBTAB_PYTHON" -m sbtab.experiments.pipeline plan "$@" --device "$SBTAB_DEVICE")
printf '%s\n' "$SBTAB_PLAN_RESULT"
SBTAB_PLAN=$(printf '%s' "$SBTAB_PLAN_RESULT" | "$SBTAB_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["plan"])')
SBTAB_SMOKE=$("$SBTAB_PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["smoke"]))' "$SBTAB_PLAN")
if [[ $SBTAB_SMOKE == 1 ]]; then
    # Short integration checks should release cluster resources promptly.
    SBTAB_PREPARE_ARGS+=(--time=00:15:00)
    SBTAB_EXPERIMENT_ARGS+=(--time=00:30:00)
    SBTAB_CPU_EXPERIMENT_ARGS+=(--time=00:30:00)
    SBTAB_METRICS_ARGS+=(--time=00:30:00)
    SBTAB_AGGREGATE_ARGS+=(--time=00:15:00)
fi
SBTAB_GROUPS=$("$SBTAB_PYTHON" - "$SBTAB_PLAN" "$SBTAB_TASK_IDS" "$SBTAB_STAGE" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    plan = json.load(f)
n = len(plan["tasks"])
parts = sys.argv[2].split(",") if sys.argv[2] else [str(i) for i in range(n)]
if not all(p.isdecimal() and 0 <= int(p) < n for p in parts):
    raise SystemExit("SBTAB_TASK_IDS must contain valid comma-separated array indices")
selected = sorted(set(map(int, parts)))
def compact(ids):
    chunks = []
    for value in ids:
        if chunks and value == chunks[-1][1] + 1:
            chunks[-1][1] = value
        else:
            chunks.append([value, value])
    return ",".join(f"{a}-{b}" for a, b in chunks)
if sys.argv[3] == "metrics":
    print("metrics=" + compact(selected))
else:
    for device in ("cpu", "cuda"):
        ids = [i for i in selected if plan["tasks"][i]["device"] == device]
        if ids:
            print(device + "=" + compact(ids))
PY
)
SBTAB_CONTROL_DIR=$(dirname -- "$SBTAB_PLAN")
SBTAB_LOG_DIR="$SBTAB_REPO_ROOT/slurm_logs"
mkdir -p "$SBTAB_LOG_DIR"
submit_job() {
    printf '%q ' sbatch "$@" >&2
    printf '\n' >&2
    if [[ $SBTAB_DRY_RUN == 1 ]]; then
        # Dummy numeric IDs preserve valid dependency syntax in a submission preview.
        printf '0\n'
    else
        local result
        result=$(sbatch "$@")
        result=${result%%;*}
        [[ $result =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $result" >&2; return 1; }
        printf '%s\n' "$result"
    fi
}
record_job() {
    if [[ $SBTAB_DRY_RUN == 0 ]]; then
        printf '%s %s\n' "$1" "$2" >> "$SBTAB_CONTROL_DIR/submissions.log"
    fi
}
SBTAB_COMMON_ARGS=(--parsable --export=ALL --chdir="$SBTAB_REPO_ROOT" "${SBATCH_SITE_ARGS[@]}")
SBTAB_PREPARE_JOB=$(submit_job "${SBTAB_COMMON_ARGS[@]}" "${SBTAB_PREPARE_ARGS[@]}" --array=0-0 \
    --output="$SBTAB_LOG_DIR/prepare-%A_%a.out" --error="$SBTAB_LOG_DIR/prepare-%A_%a.err" \
    "$SBTAB_BATCH_SCRIPT_DIR/prepare.sbatch" "$SBTAB_REPO_ROOT" "$SBTAB_CLUSTER_CONFIG" "$SBTAB_PLAN")
record_job prepare "$SBTAB_PREPARE_JOB"
SBTAB_DEPENDENCY_JOBS=("$SBTAB_PREPARE_JOB")
while IFS='=' read -r SBTAB_GROUP SBTAB_INDICES; do
    [[ -n $SBTAB_INDICES ]] || continue
    SBTAB_ARRAY="$SBTAB_INDICES%$SBTAB_MAX_CONCURRENT"
    if [[ $SBTAB_GROUP == metrics ]]; then
        SBTAB_PARENT=$SBTAB_PREPARE_JOB
        SBTAB_DEPENDENCY="afterok:$SBTAB_PARENT"
    else
        SBTAB_RESOURCE_ARGS=("${SBTAB_CPU_EXPERIMENT_ARGS[@]}")
        if [[ $SBTAB_GROUP == cuda ]]; then
            SBTAB_RESOURCE_ARGS=("${SBTAB_EXPERIMENT_ARGS[@]}")
        fi
        SBTAB_GENERATION_STAGE=$SBTAB_STAGE
        [[ $SBTAB_STAGE != all ]] || SBTAB_GENERATION_STAGE=generate
        SBTAB_PARENT=$(submit_job "${SBTAB_COMMON_ARGS[@]}" "${SBTAB_RESOURCE_ARGS[@]}" \
            --dependency="afterok:$SBTAB_PREPARE_JOB" --kill-on-invalid-dep=yes --array="$SBTAB_ARRAY" \
            --output="$SBTAB_LOG_DIR/experiment-%A_%a.out" --error="$SBTAB_LOG_DIR/experiment-%A_%a.err" \
            "$SBTAB_BATCH_SCRIPT_DIR/experiment.sbatch" "$SBTAB_REPO_ROOT" "$SBTAB_CLUSTER_CONFIG" "$SBTAB_PLAN" \
            "$SBTAB_GENERATION_STAGE" "$SBTAB_RETRY_FAILED_FOLDS")
        record_job "generate-$SBTAB_GROUP" "$SBTAB_PARENT"
        SBTAB_DEPENDENCY_JOBS+=("$SBTAB_PARENT")
        # Each CPU evaluator waits only for its matching successful generator task.
        SBTAB_DEPENDENCY="aftercorr:$SBTAB_PARENT"
    fi
    if [[ $SBTAB_STAGE == all || $SBTAB_STAGE == metrics ]]; then
        SBTAB_METRICS_JOB=$(submit_job "${SBTAB_COMMON_ARGS[@]}" "${SBTAB_METRICS_ARGS[@]}" \
            --dependency="$SBTAB_DEPENDENCY" --kill-on-invalid-dep=yes --array="$SBTAB_ARRAY" \
            --output="$SBTAB_LOG_DIR/metrics-%A_%a.out" --error="$SBTAB_LOG_DIR/metrics-%A_%a.err" \
            "$SBTAB_BATCH_SCRIPT_DIR/metrics.sbatch" "$SBTAB_REPO_ROOT" "$SBTAB_CLUSTER_CONFIG" "$SBTAB_PLAN")
        record_job "metrics-$SBTAB_GROUP" "$SBTAB_METRICS_JOB"
        SBTAB_DEPENDENCY_JOBS+=("$SBTAB_METRICS_JOB")
    fi
done <<< "$SBTAB_GROUPS"
# Wait for generators too if someone cancels a dependent metrics array early.
SBTAB_DEPENDENCIES=$(IFS=:; printf '%s' "${SBTAB_DEPENDENCY_JOBS[*]}")
SBTAB_AGGREGATE_JOB=$(submit_job "${SBTAB_COMMON_ARGS[@]}" "${SBTAB_AGGREGATE_ARGS[@]}" \
    --dependency="afterany:$SBTAB_DEPENDENCIES" --array=0-0 \
    --output="$SBTAB_LOG_DIR/aggregate-%A_%a.out" --error="$SBTAB_LOG_DIR/aggregate-%A_%a.err" \
    "$SBTAB_BATCH_SCRIPT_DIR/aggregate.sbatch" "$SBTAB_REPO_ROOT" "$SBTAB_CLUSTER_CONFIG" "$SBTAB_PLAN")
record_job aggregate "$SBTAB_AGGREGATE_JOB"
printf 'prepare=%s dependency-arrays=%s aggregate=%s\n' "$SBTAB_PREPARE_JOB" "$SBTAB_DEPENDENCIES" "$SBTAB_AGGREGATE_JOB"
