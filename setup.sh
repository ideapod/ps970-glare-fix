#!/usr/bin/env bash
# One-time setup: creates a private Python environment in .venv next to this script
# and installs the requirements into it. Safe to run again (it just updates).
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. On a Mac: install it from https://www.python.org/downloads/ (or 'brew install python')." >&2
  exit 1
fi
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt
echo
echo "Ready. Run it with:"
echo "  ./glare-fix \"/path/to/scans\" \"/path/to/output\" --preview"
