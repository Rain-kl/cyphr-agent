#!/usr/bin/env bash
# Download one model package into backend/agent/models/<pkg>/.
# One model = one directory, never mixed, never committed (see .gitignore).
# Usage: ./download_model.sh [HF_MODEL_ID] [PKG_DIR]
#   e.g. ./download_model.sh Qwen/Qwen3-ASR-0.6B qwen3-asr-0.6b
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-ASR-0.6B}"
PKG_DIR="${2:-qwen3-asr-0.6b}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/models/$PKG_DIR"

mkdir -p "$DEST"
uvx --from "huggingface_hub[cli]" hf download "$MODEL_ID" --local-dir "$DEST"
echo "Saved $MODEL_ID -> $DEST"
