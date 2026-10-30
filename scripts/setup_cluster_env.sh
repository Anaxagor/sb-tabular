#!/usr/bin/env bash
# Run from any directory; the environment must be on a shared filesystem.
set -euo pipefail
SBTAB_SETUP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
SBTAB_ENV_DIR=${1:-"$SBTAB_SETUP_ROOT/.venv-cluster"}
SBTAB_BOOTSTRAP_PYTHON=${SBTAB_BOOTSTRAP_PYTHON:-python3.11}
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [environment-directory] (SBTAB_BOOTSTRAP_PYTHON defaults to python3.11)" >&2
    exit 2
fi
"$SBTAB_BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
if [[ ! -e $SBTAB_ENV_DIR ]]; then
    "$SBTAB_BOOTSTRAP_PYTHON" -m venv "$SBTAB_ENV_DIR"
fi
SBTAB_ENV_PYTHON="$SBTAB_ENV_DIR/bin/python"
"$SBTAB_ENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Existing environment must use Python 3.11"'
"$SBTAB_ENV_PYTHON" -m pip install 'pip==25.1.1'
if [[ $(uname -s) == Linux ]]; then
    SBTAB_CUDA_WHEEL=${SBTAB_CUDA_WHEEL:-cu124}
    case "$SBTAB_CUDA_WHEEL" in cu118|cu124|cu126) ;; *) echo "Use cu118, cu124, or cu126 for PyTorch 2.6" >&2; exit 2;; esac
    "$SBTAB_ENV_PYTHON" -m pip install "torch==2.6.0+$SBTAB_CUDA_WHEEL" \
        --index-url "https://download.pytorch.org/whl/$SBTAB_CUDA_WHEEL"
    "$SBTAB_ENV_PYTHON" -c 'import torch; assert torch.version.cuda is not None, "CUDA PyTorch is required"'
fi
"$SBTAB_ENV_PYTHON" -m pip install -r "$SBTAB_SETUP_ROOT/requirements-cluster.txt"
"$SBTAB_ENV_PYTHON" -m pip check
"$SBTAB_ENV_PYTHON" -m pip freeze > "$SBTAB_ENV_DIR/requirements-resolved.txt"
printf 'Environment ready: %s\nResolved dependencies: %s\n' "$SBTAB_ENV_PYTHON" "$SBTAB_ENV_DIR/requirements-resolved.txt"
