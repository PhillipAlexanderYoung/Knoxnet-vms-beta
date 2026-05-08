#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run ./install.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python start_desktop.py
