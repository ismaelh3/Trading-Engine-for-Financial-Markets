# Trading-Engine-for-Financial-Markets

Daily SPY market data now comes from `yfinance` instead of Alpha Vantage free tier.

Download raw SPY history:

```bash
python scripts/download_yfinance.py --symbols SPY --format jsonl
```

Build the merged market + FRED dataset:

```bash
python scripts/build_dataset.py --market-file data/raw/yfinance/yfinance_spy.jsonl
```

`scripts/build_dataset.py` still accepts `--alpha-file` as a backward-compatible alias for `--market-file`.

By default, the builder keeps only the overlapping date range covered by both market data and FRED data. Use `--coverage all` if you want to keep the full market series with partial macro coverage.

## Milestone 1 pipeline

Build the engineered Milestone 1 volatility dataset:

```bash
python scripts/build_m1_dataset.py \
  --input data/processed/merged_market_macro.csv \
  --output data/processed/m1/m1_dataset.csv
```

Run the walk-forward experiment with the recommended model ladder:

```bash
python scripts/run_m1_experiment.py \
  --input data/processed/m1/m1_dataset.csv \
  --output-dir artifacts/m1 \
  --models naive elastic_net xgboost lstm cnn \
  --test-start-date 2022-01-03 \
  --tuning-mode default \
  --tuning-metric qlike
```

The experiment writes per-model prediction files, backtest files, a metrics summary, a backtest summary, and any available explainability outputs.
It now shows progress bars by default while the walk-forward blocks are running. Use `--no-progress` if you want plain summary logging only.

Useful tuning modes:

- `--tuning-mode off`: use the hard-coded defaults with no pre-test tuning.
- `--tuning-mode default`: run a modest pre-test grid search for each model.
- `--tuning-mode full`: run a larger grid search, which is slower but more exhaustive.

The runner tunes each model once on the first pre-test train/validation split, freezes the selected hyperparameters, and then evaluates the final walk-forward test out of sample.
