
#!/usr/bin/env bash
#
# run_bot.sh — wrapper for app.py
# Usage:
#   ./run_bot.sh 5m AAPL 0.1 2 1 --poll 60 --strategy ema_cross --trade-mode paper --broker alpaca
#
# This wrapper prefers ./.venv/bin/python if it exists, so you don't need to "activate" the venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  VENV_PY="${VENV_PY_OVERRIDE:-python3}"
fi

exec "$VENV_PY" "$SCRIPT_DIR/app.py" "$@"
