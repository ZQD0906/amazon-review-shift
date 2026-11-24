#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m review_shift manifest
python -m review_shift process --config configs/full.json
python -m review_shift analyze --config configs/full.json
python -m review_shift backtest --config configs/full.json
python scripts/run_experiments.py --config configs/full.json --mode blocks
python scripts/run_experiments.py --config configs/full.json --mode users
python scripts/run_experiments.py --config configs/full.json --mode scale
