#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="${PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"

if [[ -n "${CANN_TOOLKIT_ROOT:-}" && -f "$CANN_TOOLKIT_ROOT/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$CANN_TOOLKIT_ROOT/set_env.sh"
elif [[ -f "$HOME/Ascend/ascend-toolkit/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/Ascend/ascend-toolkit/set_env.sh"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

if [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib/x86_64:${LD_LIBRARY_PATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$ROOT_DIR/infer_om.py" --model "$ROOT_DIR/weights/best.om" "$@"
