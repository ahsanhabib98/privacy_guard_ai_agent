#!/usr/bin/env bash
set -euo pipefail
MODEL="${OLLAMA_MODEL:-qwen3.5:4b}"
ollama pull "$MODEL"
