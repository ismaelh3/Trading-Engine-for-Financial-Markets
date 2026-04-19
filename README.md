# Trading-Engine-for-Financial-Markets

This repo builds a daily SPY + FRED dataset, engineers Milestone 1 features, and runs Milestone 2 walk-forward experiments for:

- regression: forecast future 20-day volatility
- classification: predict low / medium / high volatility regimes

Implemented model families:

- `naive`
- `elastic_net`
- `xgboost`
- `lstm`
- `cnn`
- `ctts`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you run `elastic_net` or `xgboost`, make sure `scikit-learn` and `xgboost` are installed in the same interpreter you use to launch the scripts.

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
  --models naive elastic_net xgboost lstm cnn ctts \
  --task-type regression \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-metric qlike \
  --torch-loss qlike
```

Run only the sequence models and print epoch losses:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1 \
  --results-root results/regression \
  --models lstm cnn ctts \
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
  --models naive elastic_net xgboost lstm cnn ctts \
  --task-type classification \
  --classification-source-column target_future_vol_20d \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-metric macro_f1
```

Sequence-model classification run with epoch logs:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1_classification \
  --results-root results/classification \
  --models lstm cnn ctts \
  --task-type classification \
  --classification-source-column target_future_vol_20d \
  --tuning-metric macro_f1 \
  --torch-log-epochs
```

## MPS / Device Selection

Only the torch sequence models can use GPU-style acceleration in this repo:

- `lstm`
- `cnn`
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
  --models lstm cnn ctts \
  --task-type regression \
  --torch-device mps \
  --torch-log-epochs
```

## Output Layout

Each experiment writes:

- CSV artifacts under `--output-dir`
- per-model JSON summaries under `--results-root/<model>/results.json`

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

Use separate results roots for regression and classification:

- `results/regression`
- `results/classification`

If you reuse the same `results/` folder for both tasks, the latest run for a model can overwrite `results/<model>/results.json`, which can confuse the plotting scripts.

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
  --equity-models naive elastic_net xgboost lstm cnn ctts \
  --output-dir plots/regression
```

If you also want the CNN forecast/training figures, run it again with `--forecast-model cnn`:

```bash
python scripts/plot_m1_results.py \
  --results-root results/regression \
  --artifacts-dir artifacts/m1 \
  --forecast-model cnn \
  --equity-models naive elastic_net xgboost lstm cnn ctts \
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
  --equity-models naive elastic_net xgboost lstm cnn ctts \
  --output-dir plots/classification
```

If you also want the CNN classification plots, run it again with `--classification-model cnn`:

```bash
python scripts/plot_m1_classification_results.py \
  --results-root results/classification \
  --artifacts-dir artifacts/m1_classification \
  --classification-model cnn \
  --equity-models naive elastic_net xgboost lstm cnn ctts \
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

## Repository

https://github.com/ismaelh3/Trading-Engine-for-Financial-Markets
