# Trading-Engine-for-Financial-Markets

Daily SPY market data comes from `yfinance`, and macro-financial series come from FRED.

## Setup

Create and activate your environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you plan to run `elastic_net` and `xgboost`, make sure those packages are installed in the same interpreter you use to launch the experiment.

## Refresh Data

Download raw SPY history:

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

Build the merged market + FRED dataset:

```bash
python scripts/build_dataset.py \
  --market-file data/raw/yfinance/yfinance_spy.jsonl
```

By default, the builder keeps only the overlapping date range covered by both market data and FRED data. Use `--coverage all` if you want to keep the full market series with partial macro coverage.

Build the engineered Milestone 1 dataset:

```bash
python scripts/build_m1_dataset.py \
  --input data/processed/merged_market_macro.csv \
  --output data/processed/m1/m1_dataset.csv
```

## Run Regression Experiments

Run the full regression ladder:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1 \
  --models naive elastic_net xgboost lstm cnn ctts \
  --task-type regression \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-metric qlike \
  --torch-loss qlike
```

If you want epoch-level loss prints for sequence models:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1 \
  --models cnn ctts \
  --task-type regression \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-metric qlike \
  --torch-loss qlike \
  --torch-log-epochs
```

## Run Classification Experiments

Run the full classification ladder:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1_classification \
  --models naive elastic_net xgboost lstm cnn ctts \
  --task-type classification \
  --classification-source-column target_future_vol_20d \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-metric macro_f1
```

## Output Structure

Each run writes:

- CSV artifacts under `--output-dir`
- per-model JSON summaries under `results/<model>/results.json`

Typical regression artifacts:

- `predictions_<model>.csv`
- `backtest_<model>.csv`
- `tuning_summary_<model>.csv`
- `selected_params_<model>.json`
- `training_history_<model>.csv` for sequence models
- `metrics_summary.csv`
- `backtest_summary.csv`

Typical classification artifacts are similar, but prediction files contain:

- `predicted_class`
- `actual_class`
- `prob_class_0`, `prob_class_1`, `prob_class_2`

## Plot Regression Results

Plot forecast, training curve, and equity curve:

```bash
python scripts/plot_m1_results.py \
  --results-root results \
  --artifacts-dir artifacts/m1 \
  --forecast-model ctts \
  --equity-models naive elastic_net xgboost lstm cnn ctts \
  --output-dir plots/regression
```

If `results/<model>/results.json` points to older incompatible artifacts, force artifact-only resolution with:

```bash
python scripts/plot_m1_results.py \
  --results-root /tmp/no_results_here \
  --artifacts-dir artifacts/m1 \
  --forecast-model ctts \
  --equity-models naive elastic_net xgboost lstm cnn ctts \
  --output-dir plots/regression
```

## Plot Classification Results

Plot classification timeline, confusion matrix, training curve, and equity curve:

```bash
python scripts/plot_m1_classification_results.py \
  --results-root results \
  --artifacts-dir artifacts/m1_classification \
  --classification-model elastic_net \
  --equity-models naive elastic_net xgboost lstm cnn ctts \
  --output-dir plots/classification
```

## Tuning Notes

Useful tuning modes:

- `--tuning-mode off`: use the hard-coded defaults with no pre-test tuning
- `--tuning-mode default`: run a modest pre-test grid search
- `--tuning-mode full`: run a larger grid search

Regression tuning metrics:

- `qlike`
- `mae`
- `rmse`

Classification tuning metrics:

- `accuracy`
- `balanced_accuracy`
- `macro_f1`

## Important Behavior

- Regression and classification are separate runs. Use `--task-type regression` or `--task-type classification`.
- Sequence models are trained from scratch for each walk-forward block.
- Sequence-model training history is saved for the first representative block only.
- The backtest is a post-prediction strategy simulation, not model training.

## Repository

https://github.com/ismaelh3/Trading-Engine-for-Financial-Markets
