#!/bin/bash
# Humanizer launcher for macOS / Linux. First run installs everything (free); later runs start instantly.
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is required (free): https://www.python.org/downloads/"
  read -r -p "Press Enter to exit"; exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up (this takes a few minutes)…"
  "$PY" -m venv .venv || exit 1
  .venv/bin/python -m pip install --upgrade pip -q
fi
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [ "$(uname)" = "Linux" ]; then
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130 -q
  fi
  .venv/bin/python -m pip install -r requirements.txt -q && touch .venv/.installed || exit 1
fi

# Start Ollama in the background if it's installed but not running
if command -v ollama >/dev/null 2>&1 && ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  (ollama serve >/dev/null 2>&1 &)
  sleep 3
fi

.venv/bin/python setup_models.py
.venv/bin/python app.py
