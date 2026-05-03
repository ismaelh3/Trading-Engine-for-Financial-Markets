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
INPUT="${INPUT:-data/processed/m1/m1_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/m1_classification}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
TEST_START_DATE="${TEST_START_DATE:-2022-01-03}"
CLASSIFICATION_SOURCE_COLUMN="${CLASSIFICATION_SOURCE_COLUMN:-target_future_vol_20d}"
TUNING_BACKEND="${TUNING_BACKEND:-auto}"
TUNING_MODE="${TUNING_MODE:-default}"
TUNING_METRIC="${TUNING_METRIC:-macro_f1}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-30}"
OPTUNA_PRUNER="${OPTUNA_PRUNER:-median}"
TORCH_DEVICE="${TORCH_DEVICE:-auto}"
TORCH_EPOCHS="${TORCH_EPOCHS:-40}"
TUNING_TORCH_EPOCHS="${TUNING_TORCH_EPOCHS:-10}"

EXTRA_ARGS=()
if [[ -n "${OPTUNA_TIMEOUT_SECONDS:-}" ]]; then
  EXTRA_ARGS+=(--optuna-timeout-seconds "$OPTUNA_TIMEOUT_SECONDS")
fi
if [[ "${NO_PROGRESS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-progress)
fi

"$PYTHON_BIN" scripts/run_m1_experiment.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --results-root "$RESULTS_ROOT" \
  --models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --task-type classification \
  --classification-source-column "$CLASSIFICATION_SOURCE_COLUMN" \
  --test-start-date "$TEST_START_DATE" \
  --tuning-mode "$TUNING_MODE" \
  --tuning-backend "$TUNING_BACKEND" \
  --tuning-metric "$TUNING_METRIC" \
  --tuning-torch-epochs "$TUNING_TORCH_EPOCHS" \
  --optuna-trials "$OPTUNA_TRIALS" \
  --optuna-pruner "$OPTUNA_PRUNER" \
  --torch-device "$TORCH_DEVICE" \
  --torch-epochs "$TORCH_EPOCHS" \
  ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
