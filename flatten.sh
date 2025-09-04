#!/bin/bash
# Wrapper to call flatten.py with your project's .venv and auto-load .env
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

# Activate venv if present
if [ -d "$VENV_PATH" ]; then
  source "$VENV_PATH/bin/activate"
fi

# Auto-load .env into environment for child processes
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
elif [ -f "$SCRIPT_DIR/../.env" ]; then
  set -a
  source "$SCRIPT_DIR/../.env"
  set +a
fi

python3 "$SCRIPT_DIR/flatten.py" "$@"
