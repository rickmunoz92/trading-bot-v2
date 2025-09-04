
#!/usr/bin/env bash
# Usage:
#   ./positions.sh [SYMBOL] [--broker alpaca|ibkr|local] [--trade-mode paper|live] [--equity 100000]
# Examples:
#   ./positions.sh AAPL --broker alpaca --trade-mode paper
#   ./positions.sh --broker local --trade-mode paper --equity 100000
#
# This wrapper prefers ./.venv/bin/python if it exists, so you don't need to "activate" the venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  # Fallback to user-specified python or system python3
  VENV_PY="${VENV_PY_OVERRIDE:-python3}"
fi

exec "$VENV_PY" "$SCRIPT_DIR/positions_report.py" "$@"
