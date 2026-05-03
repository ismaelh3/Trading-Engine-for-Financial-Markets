# Trading-Engine-for-Financial-Markets

![Project preview](https://cdn.prod.website-files.com/64ab95bb2efb9f363d9145c3/6980493f8133325c59fa8c0b_688933990e97e74a2fab8a55-1753824092415.webp)


This repo builds a daily SPY + FRED dataset, engineers Milestone 1 features, and runs Milestone 2 walk-forward experiments for:

- regression: forecast future 20-day volatility
- classification: predict low / medium / high volatility regimes

Implemented model families:

- `naive`
- `elastic_net`
- `xgboost`
- `lstm`
- `cnn`
- `cnn_lstm`
- `ctts`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you run `elastic_net` or `xgboost`, make sure `scikit-learn` and `xgboost` are installed in the same interpreter you use to launch the scripts.

## Reproducibility

The project is reproducible across machines if the same Python environment, data snapshot, CLI flags, and torch device are used. Exact deep-learning results can still differ slightly between CPU, CUDA, and Apple MPS because PyTorch kernels are not always bit-for-bit identical across backends.

Recommended baseline:

- Use Python 3.11.
- Run from the repository root.
- Install dependencies in one virtual environment: `python -m pip install -r requirements.txt`.
- Keep `FRED_API_KEY` in `.env` or in your shell environment; do not commit it.
- Use the fixed data window in the bash script: `FRED_START=1993-01-01`, `FRED_END=2026-04-19`.
- For strict comparisons, reuse the same `data/raw/` snapshot instead of redownloading, because yfinance and FRED can revise historical observations.
- Use `TORCH_DEVICE=cpu` when you need the closest cross-machine reproducibility. Use `TORCH_DEVICE=mps` or `cuda` for speed, accepting small numerical differences.
- Use the same `OPTUNA_TRIALS`, `OPTUNA_PRUNER`, and `--optuna-seed` values when comparing Optuna-tuned sequence models.

`requirements.txt` is not fully version-pinned, so future dependency releases can change results or behavior. For strict archival reproducibility, create and keep a lock file from a known-good environment:

```bash
python -m pip freeze > requirements-lock.txt
```

Then another machine can recreate that environment with:

```bash
python -m pip install -r requirements-lock.txt
```

## Bash Workflows

The `bash/` directory contains runnable end-to-end commands:

```bash
bash/01_download_and_build_data.sh
bash/02_run_regression_models.sh
bash/03_run_classification_models.sh
```

Useful overrides:

```bash
TORCH_DEVICE=cpu OPTUNA_TRIALS=30 bash/02_run_regression_models.sh
TORCH_DEVICE=cpu OPTUNA_TRIALS=30 bash/03_run_classification_models.sh
```

## Refresh Data

Download SPY history:

```bash
python scripts/download_yfinance.py --symbols SPY --format jsonl
```

Download FRED series:

```bash
python scripts/download_fred.py \
  --start 1993-01-01 \
  --end 2026-04-19 \
  --format json
```

`scripts/download_fred.py` expects `FRED_API_KEY` in the environment or in `.env`.

Build the merged market + macro dataset:

```bash
python scripts/build_dataset.py \
  --market-file data/raw/yfinance/yfinance_spy.jsonl
```

By default the merge keeps only the overlapping SPY/FRED range. If you want the full market series even when some macro series start later, use:

```bash
python scripts/build_dataset.py \
  --market-file data/raw/yfinance/yfinance_spy.jsonl \
  --coverage all
```

Build the engineered Milestone 1 dataset:

```bash
python scripts/build_m1_dataset.py \
  --input data/processed/merged_market_macro.csv \
  --output data/processed/m1/m1_dataset.csv
```

## Run Regression

Recommended full regression run:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1 \
  --results-root results/regression \
  --models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --task-type regression \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-backend auto \
  --tuning-metric qlike \
  --torch-loss qlike
```

Run only the sequence models and print epoch losses:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1 \
  --results-root results/regression \
  --models lstm cnn cnn_lstm ctts \
  --task-type regression \
  --torch-loss qlike \
  --torch-log-epochs
```

## Run Classification

Recommended full classification run:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1_classification \
  --results-root results/classification \
  --models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --task-type classification \
  --classification-source-column target_future_vol_20d \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-backend auto \
  --tuning-metric macro_f1
```

`--tuning-backend auto` uses exhaustive grid search for `naive`, `elastic_net`, and `xgboost`, and Optuna for the torch sequence models (`lstm`, `cnn`, `cnn_lstm`, `ctts`). Use `--tuning-backend grid` to force the previous grid-search behavior for every model, or tune Optuna runs with:

```bash
--optuna-trials 30 \
--optuna-timeout-seconds 1800 \
--optuna-pruner median
```

Sequence-model classification run with epoch logs:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1_classification \
  --results-root results/classification \
  --models lstm cnn cnn_lstm ctts \
  --task-type classification \
  --classification-source-column target_future_vol_20d \
  --tuning-metric macro_f1 \
  --torch-log-epochs
```

## MPS / Device Selection

Only the torch sequence models can use GPU-style acceleration in this repo:

- `lstm`
- `cnn`
- `cnn_lstm`
- `ctts`

`naive`, `elastic_net`, and `xgboost` remain CPU-only.

Device selection is controlled by `--torch-device`:

- `auto`: prefer `cuda`, then `mps`, then `cpu`
- `mps`: force Apple Metal
- `cuda`: force CUDA
- `cpu`: force CPU

Example on Apple Silicon:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1 \
  --results-root results/regression \
  --models lstm cnn cnn_lstm ctts \
  --task-type regression \
  --torch-device mps \
  --torch-log-epochs
```

## Output Layout

Each experiment writes:

- CSV artifacts under a task-scoped output directory
- per-model JSON summaries under a task-scoped results directory

If the path you pass does not already include `regression` or `classification`, the runner appends the task type automatically. For example:

- `--output-dir artifacts/m1 --task-type regression` writes to `artifacts/m1/regression`
- `--output-dir artifacts/m1 --task-type classification` writes to `artifacts/m1/classification`
- `--results-root results --task-type regression` writes to `results/regression`
- `--results-root results --task-type classification` writes to `results/classification`

Typical artifact files:

- `predictions_<model>.csv`
- `backtest_<model>.csv`
- `tuning_summary_<model>.csv`
- `selected_params_<model>.json`
- `training_history_<model>.csv` for sequence models
- `metrics_summary.csv`
- `backtest_summary.csv`

Classification prediction files contain:

- `predicted_class`
- `actual_class`
- `prob_class_0`
- `prob_class_1`
- `prob_class_2`

## Important Recommendation

The runner now namespaces both `--output-dir` and `--results-root` by task type automatically when needed. You can still pass explicit task-scoped paths yourself if you want full control:

- `results/regression`
- `results/classification`
- `artifacts/m1/regression`
- `artifacts/m1/classification`

## Plot Regression Results

The regression plotter creates:

- one predicted-vs-realized figure for `--forecast-model`
- one training-curve figure for `--forecast-model` if `training_history_<model>.csv` exists
- one multi-model equity curve

Example for CTTS:

```bash
python scripts/plot_m1_results.py \
  --results-root results/regression \
  --artifacts-dir artifacts/m1 \
  --forecast-model ctts \
  --equity-models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --output-dir plots/regression
```

If you also want the CNN forecast/training figures, run it again with `--forecast-model cnn`:

```bash
python scripts/plot_m1_results.py \
  --results-root results/regression \
  --artifacts-dir artifacts/m1 \
  --forecast-model cnn \
  --equity-models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --output-dir plots/regression
```

## Plot Classification Results

The classification plotter creates:

- one class-timeline figure for `--classification-model`
- one confusion matrix for `--classification-model`
- one training-curve figure for `--classification-model` if `training_history_<model>.csv` exists
- one multi-model equity curve

Example for CTTS:

```bash
python scripts/plot_m1_classification_results.py \
  --results-root results/classification \
  --artifacts-dir artifacts/m1_classification \
  --classification-model ctts \
  --equity-models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --output-dir plots/classification
```

If you also want the CNN classification plots, run it again with `--classification-model cnn`:

```bash
python scripts/plot_m1_classification_results.py \
  --results-root results/classification \
  --artifacts-dir artifacts/m1_classification \
  --classification-model cnn \
  --equity-models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --output-dir plots/classification
```

## Tuning Notes

Supported tuning modes:

- `--tuning-mode off`
- `--tuning-mode default`
- `--tuning-mode full`

Regression tuning metrics:

- `qlike`
- `mae`
- `rmse`

Classification tuning metrics:

- `accuracy`
- `balanced_accuracy`
- `macro_f1`

## Current Behavior

- Regression and classification are separate runs.
- Sequence models are retrained from scratch for each walk-forward block.
- Sequence-model training history is saved for the first representative block only.
- The backtest is a post-prediction trading simulation, not model training.
- Worker-mode subprocesses are used automatically when you run more than one model at once.
