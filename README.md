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
