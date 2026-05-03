#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "venv/bin/activate" ]]; then
    source "venv/bin/activate"
  elif [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
FRED_START="${FRED_START:-1993-01-01}"
FRED_END="${FRED_END:-2026-04-19}"
COVERAGE="${COVERAGE:-overlap}"

"$PYTHON_BIN" scripts/download_yfinance.py \
  --symbols SPY \
  --period max \
  --interval 1d \
  --output-dir data/raw/yfinance \
  --format jsonl

"$PYTHON_BIN" scripts/download_fred.py \
  --start "$FRED_START" \
  --end "$FRED_END" \
  --output-dir data/raw/fred \
  --format json

"$PYTHON_BIN" scripts/build_dataset.py \
  --market-file data/raw/yfinance/yfinance_spy.jsonl \
  --fred-dir data/raw/fred \
  --output data/processed/merged_market_macro.csv \
  --coverage "$COVERAGE"

"$PYTHON_BIN" scripts/build_m1_dataset.py \
  --input data/processed/merged_market_macro.csv \
  --output data/processed/m1/m1_dataset.csv
