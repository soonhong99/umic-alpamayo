#!/usr/bin/env bash
# ONE command from a fresh boot to the full benchmark:
#
#   bash scripts/run_all.sh                       # eager vs UMIC A/B (default)
#   bash scripts/run_all.sh --mode umic --runs 3  # args pass through to run_pipeline.py
#
# Steps: [1] lock clocks (sudo, asked once)  [2] activate alpamayo venv
#        [3] environment + kernel check      [4] benchmark (warmup 5 + runs 3)
# Aborts before the benchmark if the environment check fails.

set -u
cd "$(dirname "$0")/.."

echo "== [1/4] jetson_clocks (DVFS lock — mandatory for measurement) =="
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks && echo "clocks locked."
else
    echo "jetson_clocks not found (non-Jetson host?) — skipping."
fi

echo "== [2/4] python environment =="
VENV="$HOME/alpamayo1.5/a1_5_venv"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "activated: $VENV"
else
    echo "alpamayo venv not found at $VENV — using current python."
fi

echo "== [3/4] environment + kernel check =="
if ! PYTHONPATH=src python3 scripts/check_env.py; then
    echo "environment check FAILED — fix the [FAIL] items above, then re-run."
    exit 1
fi

echo "== [4/4] benchmark (model load ~3-4 min, then warmup 5 + measured runs) =="
PYTHONPATH=src exec python3 scripts/run_pipeline.py --mode both --runs 3 "$@"
