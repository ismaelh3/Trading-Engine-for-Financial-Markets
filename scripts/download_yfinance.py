#!/usr/bin/env python3
"""Download daily market data from yfinance and save it as JSON or JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def to_float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def to_int_or_none(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(float(value))


def fetch_daily_series(
    symbol: str,
    period: str,
    interval: str,
) -> list[dict[str, object]]:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: yfinance. Install it with `pip install -r requirements.txt`.",
        ) from exc

    history = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=True,
    )
    if history.empty:
        raise ValueError(f"No market data returned for {symbol}.")

    fetched_at = datetime.now(timezone.utc).isoformat()
    index_tz = getattr(history.index, "tz", None)
    timezone_name = str(index_tz) if index_tz is not None else None
    records: list[dict[str, object]] = []

    for timestamp, row in history.sort_index().iterrows():
        records.append(
            {
                "source": "yfinance",
                "symbol": symbol,
                "date": timestamp.date().isoformat(),
                "open": to_float_or_none(row.get("Open")),
                "high": to_float_or_none(row.get("High")),
                "low": to_float_or_none(row.get("Low")),
                "close": to_float_or_none(row.get("Close")),
                "adjusted_close": to_float_or_none(row.get("Adj Close")),
                "volume": to_int_or_none(row.get("Volume")),
                "dividend_amount": to_float_or_none(row.get("Dividends")),
                "split_coefficient": to_float_or_none(row.get("Stock Splits")),
                "timezone": timezone_name,
                "fetched_at_utc": fetched_at,
            }
        )

    return records


def write_records(
    output_path: Path,
    records: list[dict[str, object]],
    file_format: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "json":
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_output_path(output_dir: Path, symbol: str, file_format: str) -> Path:
    safe_symbol = symbol.lower().replace(" ", "_")
    return output_dir / f"yfinance_{safe_symbol}.{file_format}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download daily market data from yfinance as JSON or JSONL.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY"],
        help="Symbols to download.",
    )
    parser.add_argument(
        "--period",
        default="max",
        help="History period passed to yfinance. Use max for full available history.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="History interval passed to yfinance. Use 1d for daily data.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/yfinance",
        help="Directory where output files will be written.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="jsonl",
        help="Output format.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for symbol in args.symbols:
        records = fetch_daily_series(
            symbol=symbol,
            period=args.period,
            interval=args.interval,
        )
        output_path = build_output_path(output_dir, symbol, args.format)
        write_records(output_path, records, args.format)
        print(f"saved {symbol} -> {output_path}")


if __name__ == "__main__":
    main()
