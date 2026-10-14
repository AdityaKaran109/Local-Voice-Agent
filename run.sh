#!/usr/bin/env bash
# One-command setup + start. Re-running is safe.
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8765}"

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

mkdir -p models
BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
[ -f models/kokoro-v1.0.onnx ] || curl -L -o models/kokoro-v1.0.onnx "$BASE/kokoro-v1.0.onnx"
[ -f models/voices-v1.0.bin ] || curl -L -o models/voices-v1.0.bin "$BASE/voices-v1.0.bin"

if command -v ollama >/dev/null; then
  curl -sf localhost:11434/api/tags >/dev/null || (ollama serve >/dev/null 2>&1 &) && sleep 2
  MODEL=$(grep -E '^OLLAMA_MODEL=' .env 2>/dev/null | cut -d= -f2); MODEL=${MODEL:-ministral-3:latest}
  ollama list | grep -q "${MODEL%%:*}" || ollama pull "$MODEL"
  ollama list | grep -q nomic-embed-text || ollama pull nomic-embed-text
fi

echo "→ Voice Agent Lab on http://localhost:$PORT"
exec .venv/bin/uvicorn backend.main:app --port "$PORT"
