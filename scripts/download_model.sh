#!/usr/bin/env bash
# Download one model package into backend/agent/models/<pkg>/.
# One model = one directory, never mixed, never committed (see .gitignore).
# Usage: ./download_model.sh [HF_MODEL_ID] [PKG_DIR]
#   e.g. ./download_model.sh Qwen/Qwen3-ASR-0.6B qwen3-asr-0.6b
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: ./download_model.sh [HF_MODEL_ID] [PKG_DIR] [OPTIONS]"
  echo "  e.g. ./download_model.sh Qwen/Qwen3-ASR-0.6B qwen3-asr-0.6b"
  echo ""
  echo "Environment variables:"
  echo "  HF_ENDPOINT    Custom endpoint or mirror (default: https://huggingface.co, e.g. https://hf-mirror.com)"
  echo "  HF_TOKEN       Hugging Face token for gated models"
  echo "  DOWNLOAD_TOOL  'auto' (default, uses aria2c if available), 'aria2c', or 'python'"
  exit 0
fi

MODEL_ID="${1:-Qwen/Qwen3-ASR-0.6B}"
PKG_DIR="${2:-qwen3-asr-0.6b}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/models/$PKG_DIR"

shift 2 2>/dev/null || shift $# 2>/dev/null || true

PYTHON_BIN="python3"
if ! command -v python3 &>/dev/null; then
  if command -v python &>/dev/null; then
    PYTHON_BIN="python"
  elif command -v uv &>/dev/null; then
    PYTHON_BIN="uv run python"
  else
    echo "[Error] python3 or uv is required to run download_model." >&2
    exit 1
  fi
fi

mkdir -p "$DEST"
$PYTHON_BIN "$ROOT/scripts/download_model.py" --model-id "$MODEL_ID" --dest "$DEST" "$@"
echo "Saved $MODEL_ID -> $DEST"
