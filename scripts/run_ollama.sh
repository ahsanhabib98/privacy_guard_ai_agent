#!/usr/bin/env bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Please install ollama"
  exit 1
fi

echo "Starting Ollama on http://127.0.0.1:11434"
exec ollama serve
