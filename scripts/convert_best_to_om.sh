#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_MODEL="${1:-"$ROOT_DIR/weights/best.onnx"}"
OUTPUT_DIR="${2:-"$ROOT_DIR/weights"}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
OUTPUT_PREFIX="$OUTPUT_DIR/best"

CANN_TOOLKIT_ROOT="${CANN_TOOLKIT_ROOT:-}"

export PATH="${PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"

if [[ -n "$CANN_TOOLKIT_ROOT" && -f "$CANN_TOOLKIT_ROOT/set_env.sh" ]]; then
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

if ! command -v atc >/dev/null 2>&1; then
  echo "atc was not found. Install/activate Ascend CANN first, then rerun this script." >&2
  exit 1
fi

if [[ ! -f "$ONNX_MODEL" ]]; then
  echo "ONNX model not found: $ONNX_MODEL" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

atc \
  --model="$ONNX_MODEL" \
  --framework=5 \
  --output="$OUTPUT_PREFIX" \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --soc_version="$SOC_VERSION" \
  --log=info

echo "OM model written to: $OUTPUT_PREFIX.om"
