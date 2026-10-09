#!/usr/bin/env bash
# Run locally. The destination is an SSH host/alias and an absolute project path.
set -euo pipefail

usage() {
    echo "Usage: bash scripts/sync_cluster.sh [--dry-run] USER@HOST:/absolute/path/sb-tabular"
}
SBTAB_SYNC_PREVIEW=0
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ ${1:-} == --dry-run ]]; then
    SBTAB_SYNC_PREVIEW=1
    shift
fi
if [[ $# != 1 ]]; then
    usage >&2
    exit 2
fi
SBTAB_SYNC_DEST=${1%/}
# Keep the remote-shell destination unambiguous on older macOS rsync versions.
# Use an SSH config alias for custom ports, jump hosts or IPv6 addresses.
if [[ ! $SBTAB_SYNC_DEST =~ ^[[:alnum:]_][[:alnum:]_.@-]*:/[[:alnum:]_./-]+$ ]]; then
    echo "Use USER@HOST:/absolute/project/path (no spaces), or an SSH alias instead of USER@HOST." >&2
    exit 2
fi
SBTAB_SYNC_PATH=${SBTAB_SYNC_DEST#*:}
case "$SBTAB_SYNC_PATH/" in
    */../*|*/./*|*//*) echo "Use a normalized absolute project path without . or .. components." >&2; exit 2;;
esac
SBTAB_SYNC_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
command -v rsync >/dev/null || { echo "rsync is required on the local machine and cluster." >&2; exit 2; }
[[ -f "$SBTAB_SYNC_ROOT/scripts/slurm/cluster.local.sh" ]] || {
    echo "Missing scripts/slurm/cluster.local.sh: create the cluster configuration locally before uploading." >&2
    exit 2
}

# Do not use .gitignore as a transfer filter: it can hide datasets and local
# configuration. Exclusions protect both the sender and receiver, including
# during deletion. Receiver-only protection keeps other cluster-root files.
SBTAB_SYNC_ARGS=(
    -rlt --checksum --itemize-changes --safe-links --delete
    --exclude=/.git --exclude=/.idea --exclude=/.vscode
    --exclude=/.codex --exclude=/.agents --exclude=/.claude --exclude=/.aws
    --exclude=/.env '--exclude=/.env.*'
    --exclude=/artifacts --exclude=/slurm_logs --exclude=/catboost_info
    '--exclude=/.venv*' --exclude=/venv --exclude=/env --exclude=/.cache
    --exclude=__pycache__/ '--exclude=*.pyc' --exclude=.DS_Store
    --exclude=.pytest_cache/ --exclude=.mypy_cache/ --exclude=.ruff_cache/
    '--exclude=/*.db' '--exclude=/*.sqlite3' '--exclude=/*.sqlite3-*'
    '--exclude=*.bak'
    # Historical local-only symlink with an absolute macOS checkout path.
    --exclude=/examples/sbtab
    '--filter=R /sbtab/***' '--filter=R /configs/***' '--filter=R /scripts/***'
    '--filter=R /tests/***' '--filter=R /docs/***' '--filter=R /examples/***'
    '--filter=P *'
)
printf 'Local project: %s\nCluster project: %s\n' "$SBTAB_SYNC_ROOT" "$SBTAB_SYNC_DEST"
if [[ $SBTAB_SYNC_PREVIEW == 1 ]]; then
    rsync "${SBTAB_SYNC_ARGS[@]}" --dry-run -- "$SBTAB_SYNC_ROOT/" "$SBTAB_SYNC_DEST/"
    echo "Preview only; no files updated. Run without --dry-run to upload."
else
    rsync "${SBTAB_SYNC_ARGS[@]}" -- "$SBTAB_SYNC_ROOT/" "$SBTAB_SYNC_DEST/"
    # A second checksum pass catches a partial copy or edits made during upload.
    SBTAB_SYNC_PENDING=$(rsync "${SBTAB_SYNC_ARGS[@]}" --dry-run -- "$SBTAB_SYNC_ROOT/" "$SBTAB_SYNC_DEST/")
    if [[ -n $SBTAB_SYNC_PENDING ]]; then
        printf 'Files still differ after upload; rerun before submitting jobs:\n%s\n' "$SBTAB_SYNC_PENDING" >&2
        exit 1
    fi
    echo "Upload verified. cluster.local.sh and project files match the local copy."
fi
