#!/usr/bin/env python3
"""Build the Milestone 1 feature and target dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m1_pipeline.data import build_m1_dataset, load_merged_market_macro_dataset, save_dataset


DEFAULT_INPUT = "data/processed/merged_market_macro.csv"
DEFAULT_OUTPUT = "data/processed/m1/m1_dataset.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Milestone 1 volatility forecasting dataset.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Merged market and macro CSV built by scripts/build_dataset.py.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path for the engineered Milestone 1 dataset.",
    )
    parser.add_argument(
        "--macro-lag-days",
        type=int,
        default=21,
        help="Conservative trading-day lag applied to monthly macro series such as CPI and UNRATE.",
    )
    args = parser.parse_args()

    merged_df = load_merged_market_macro_dataset(args.input)
    m1_df = build_m1_dataset(
        merged_df=merged_df,
        macro_lag_days=args.macro_lag_days,
    )
    save_dataset(m1_df, args.output)

    print(f"saved M1 dataset -> {args.output}")
    print(f"rows={len(m1_df)} cols={len(m1_df.columns)}")
    print(f"date range: {m1_df['date'].min().date()} to {m1_df['date'].max().date()}")


if __name__ == "__main__":
    main()
