# Trading Engine for Financial Markets

![Project preview](https://cdn.prod.website-files.com/64ab95bb2efb9f363d9145c3/6980493f8133325c59fa8c0b_688933990e97e74a2fab8a55-1753824092415.webp)

This project studies whether market, macroeconomic, and sequence-model features can improve daily volatility forecasting for SPY. It builds a reproducible research pipeline that downloads market and FRED macro data, constructs a daily modeling dataset, trains multiple model families under walk-forward validation, and evaluates both forecast quality and trading-simulation outcomes.

The analysis is organized around two related prediction tasks:

- **Volatility regression:** forecast future 20-day realized volatility.
- **Volatility-regime classification:** classify future volatility into low, medium, and high regimes.

The intent is not to present a deployable trading system. The repository is a research environment for comparing modeling approaches under time-aware splits, with enough structure to rerun the full experiment on another machine.

## Research Design

The experiment uses a walk-forward setup to avoid random train/test leakage in time-series data. For each model, the pipeline:

1. Builds a daily SPY and macro feature set.
2. Creates forward-looking volatility targets from realized future returns.
3. Tunes model hyperparameters on a pre-test validation window.
4. Freezes the selected hyperparameters.
5. Retrains through walk-forward blocks and evaluates out-of-sample predictions.
6. Runs a simple post-prediction allocation backtest for comparison across models.

The backtest is used as a diagnostic layer, not as proof of tradability. Forecast metrics and classification metrics remain the primary model-comparison outputs.

## Data

The dataset combines:

- SPY daily OHLCV data from yfinance.
- FRED macro and market series, including CPI, unemployment, Treasury yields, VIX, and credit spread proxies.
- Engineered volatility, return, drawdown, moving-average, volume, macro-lag, and calendar features.

The default research window uses:

```text
FRED_START=1993-01-01
FRED_END=2026-04-19
```

FRED data requires a `FRED_API_KEY` in `.env` or in the shell environment.

## Models

The comparison includes both classical and deep sequence models:

- `naive`: realized-volatility baseline.
- `elastic_net`: regularized linear model for regression and elastic-net logistic regression for classification.
- `xgboost`: gradient-boosted trees.
- `lstm`: recurrent neural sequence model.
- `cnn`: temporal convolutional sequence model.
- `cnn_lstm`: convolutional feature extractor followed by an LSTM.
- `ctts`: convolutional tokenizer plus transformer encoder for time-series sequences.

Classical models are tuned with exhaustive grid search. Deep sequence models use Optuna by default when `--tuning-backend auto` is selected.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

All scripts should be run from the repository root.

## Reproducibility

The project is reproducible across machines when the same Python environment, data snapshot, CLI flags, and torch device are used. Exact deep-learning outputs can still differ slightly across CPU, CUDA, and Apple MPS because PyTorch kernels are not always bit-for-bit identical across hardware backends.

Recommended baseline:

- Use Python 3.11.
- Install dependencies into one virtual environment.
- Keep `FRED_API_KEY` outside version control.
- Reuse the same `data/raw/` snapshot for strict comparisons, because yfinance and FRED can revise historical observations.
- Use `TORCH_DEVICE=cpu` for the closest cross-machine reproducibility.
- Use the same `OPTUNA_TRIALS`, `OPTUNA_PRUNER`, and `--optuna-seed` values when comparing Optuna-tuned sequence models.

`requirements.txt` is not fully version-pinned. For archival reproducibility, create a lock file from a known-good environment:

```bash
python -m pip freeze > requirements-lock.txt
```

Another machine can then recreate that environment with:

```bash
python -m pip install -r requirements-lock.txt
```

## End-to-End Workflows

The `bash/` directory contains the standard experiment workflows:

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

The model-running scripts do not generate plots automatically. Plotting commands are listed below.

## Data Pipeline

Download SPY daily data:

```bash
python scripts/download_yfinance.py \
  --symbols SPY \
  --period max \
  --interval 1d \
  --format jsonl
```

Download FRED series:

```bash
python scripts/download_fred.py \
  --start 1993-01-01 \
  --end 2026-04-19 \
  --format json
```

Build the merged market and macro dataset:

```bash
python scripts/build_dataset.py \
  --market-file data/raw/yfinance/yfinance_spy.jsonl \
  --fred-dir data/raw/fred \
  --output data/processed/merged_market_macro.csv
```

Build the modeling dataset:

```bash
python scripts/build_m1_dataset.py \
  --input data/processed/merged_market_macro.csv \
  --output data/processed/m1/m1_dataset.csv
```

By default, the merge keeps only dates covered by both market and FRED data. To retain the full market history and forward-fill macro data where possible, use `--coverage all` in `scripts/build_dataset.py`.

## Regression Experiment

Run all regression models:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1 \
  --results-root results \
  --models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --task-type regression \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-backend auto \
  --tuning-metric qlike \
  --tuning-torch-epochs 10 \
  --optuna-trials 30 \
  --optuna-pruner median \
  --torch-loss qlike \
  --torch-device auto \
  --torch-epochs 40
```

Regression tuning can use `qlike`, `mae`, or `rmse`. The default research run uses `qlike`.

## Classification Experiment

Run all volatility-regime classification models:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1_classification \
  --results-root results \
  --models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --task-type classification \
  --classification-source-column target_future_vol_20d \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-backend auto \
  --tuning-metric macro_f1 \
  --tuning-torch-epochs 10 \
  --optuna-trials 30 \
  --optuna-pruner median \
  --torch-device auto \
  --torch-epochs 40
```

Classification tuning can use `accuracy`, `balanced_accuracy`, or `macro_f1`. The default research run uses `macro_f1` because class balance and minority-regime performance matter more than raw accuracy.

## Hyperparameter Tuning

The tuning backend is controlled by:

```bash
--tuning-backend auto
```

`auto` means:

- grid search for `naive`, `elastic_net`, and `xgboost`
- Optuna for `lstm`, `cnn`, `cnn_lstm`, and `ctts`

Use `--tuning-backend grid` to force exhaustive grid search for every model. Use `--tuning-mode off` to use fixed defaults without search.

Common Optuna controls:

```bash
--optuna-trials 30 \
--optuna-timeout-seconds 1800 \
--optuna-pruner median \
--optuna-seed 42
```

Optuna can improve deep-model selection, but it can also overfit a validation window if the search space is too flexible. For conservative research comparisons, prefer smaller search spaces, CPU runs for reproducibility, and repeated checks against fixed-parameter baselines.

## Device Selection

Only the torch sequence models use accelerator devices:

- `lstm`
- `cnn`
- `cnn_lstm`
- `ctts`

Device behavior:

- `auto`: prefer CUDA, then Apple MPS, then CPU.
- `cuda`: force CUDA.
- `mps`: force Apple Metal.
- `cpu`: force CPU.

Classical models remain CPU-only.

Example:

```bash
TORCH_DEVICE=mps OPTUNA_TRIALS=30 bash/02_run_regression_models.sh
```

## Outputs

Experiment outputs are task-scoped. If the path does not already include `regression` or `classification`, the runner appends the task name automatically.

Examples:

- `--output-dir artifacts/m1 --task-type regression` writes to `artifacts/m1/regression`
- `--output-dir artifacts/m1_classification --task-type classification` writes to `artifacts/m1_classification/classification`
- `--results-root results --task-type regression` writes to `results/regression`
- `--results-root results --task-type classification` writes to `results/classification`

Typical artifacts:

- `predictions_<model>.csv`
- `backtest_<model>.csv`
- `tuning_summary_<model>.csv`
- `selected_params_<model>.json`
- `training_history_<model>.csv` for sequence models
- `metrics_summary.csv`
- `backtest_summary.csv`

Per-model JSON summaries are written under `results/<task>/<model>/results.json`.

## Plotting

Regression plots:

```bash
python scripts/plot_m1_results.py \
  --results-root results/regression \
  --artifacts-dir artifacts/m1/regression \
  --forecast-model ctts \
  --equity-models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --output-dir plots/regression
```

Classification plots:

```bash
python scripts/plot_m1_classification_results.py \
  --results-root results/classification \
  --artifacts-dir artifacts/m1_classification/classification \
  --classification-model ctts \
  --equity-models naive elastic_net xgboost lstm cnn cnn_lstm ctts \
  --output-dir plots/classification
```

The plotters generate forecast/regime timelines, training curves where available, confusion matrices for classification, and multi-model equity curves.

## Repository Structure

```text
bash/                         End-to-end shell workflows
data/                         Raw and processed datasets
pipeline/                     Data, model, tuning, evaluation, and backtest utilities
scripts/                      Download, dataset-building, experiment, and plotting entrypoints
artifacts/                    Experiment CSV outputs
results/                      Per-model JSON result summaries
plots/                        Generated figures
```

## Interpretation Notes

- Regression and classification are separate experiments.
- Sequence models are retrained from scratch for each walk-forward block.
- Hyperparameters are selected before the walk-forward test and then frozen.
- Training history is saved for the first representative sequence-model block.
- The allocation backtest is a diagnostic transformation of predictions, not the training objective.
- Deep models may overfit on this data; validation design and search-space discipline are part of the research question.
