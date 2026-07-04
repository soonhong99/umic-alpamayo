#!/usr/bin/env bash
# One-time / per-boot setup on Jetson AGX Thor.
#   bash scripts/setup_thor.sh
# Does three things: lock clocks (asks sudo), activate the alpamayo venv
# if present, run the environment check.

set -u
cd "$(dirname "$0")/.."

echo "== [1/3] jetson_clocks (DVFS lock — mandatory for measurement) =="
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks && echo "clocks locked."
else
    echo "jetson_clocks not found (non-Jetson host?) — skipping."
fi

echo "== [2/3] python environment =="
VENV="$HOME/alpamayo1.5/a1_5_venv"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "activated: $VENV"
else
    echo "alpamayo venv not found at $VENV — using current python."
    echo "(kernel smoke still works; run_pipeline.py needs the venv)"
fi
python3 -m pip install -q -e . 2>/dev/null || \
    echo "editable install failed — scripts still work via their own sys.path"

echo "== [3/3] environment check =="
python3 scripts/check_env.py
