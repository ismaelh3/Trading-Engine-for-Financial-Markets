#!/usr/bin/env python3
"""Download Alpha Vantage daily adjusted data and save it as JSON or JSONL."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

load_dotenv()


def parse_alpha_vantage_response(
    payload: dict[str, object],
    symbol: str,
) -> tuple[dict[str, str], dict[str, dict[str, str]], bool]:
    if "Error Message" in payload:
        raise ValueError(f"Alpha Vantage error for {symbol}: {payload['Error Message']}")
    if "Note" in payload:
        raise ValueError(f"Alpha Vantage rate limit hit for {symbol}: {payload['Note']}")
    if "Information" in payload:
        raise ValueError(f"Alpha Vantage info for {symbol}: {payload['Information']}")

    metadata = payload.get("Meta Data")
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing metadata for {symbol}.")

    if "Time Series (Daily)" in payload:
        series = payload["Time Series (Daily)"]
        if not isinstance(series, dict):
            raise ValueError(f"Malformed daily series for {symbol}.")

        sample_row = next(iter(series.values()), {})
        has_adjusted_fields = isinstance(sample_row, dict) and "5. adjusted close" in sample_row
        return metadata, series, has_adjusted_fields

    raise ValueError(
        f"No supported daily series returned for {symbol}. Payload keys: {list(payload.keys())}",
    )


def fetch_daily_series(
    api_key: str,
    symbol: str,
    outputsize: str,
    function_name: str,
) -> list[dict[str, object]]:
    params = {
        "function": function_name,
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    metadata, series, has_adjusted_fields = parse_alpha_vantage_response(payload, symbol)
    fetched_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, object]] = []

    for date_str, row in sorted(series.items()):
        if has_adjusted_fields:
            record = {
                "source": "alpha_vantage",
                "function": function_name,
                "symbol": symbol,
                "date": date_str,
                "open": float(row["1. open"]),
                "high": float(row["2. high"]),
                "low": float(row["3. low"]),
                "close": float(row["4. close"]),
                "adjusted_close": float(row["5. adjusted close"]),
                "volume": int(float(row["6. volume"])),
                "dividend_amount": float(row["7. dividend amount"]),
                "split_coefficient": float(row["8. split coefficient"]),
                "timezone": metadata.get("5. Time Zone"),
                "fetched_at_utc": fetched_at,
            }
        else:
            record = {
                "source": "alpha_vantage",
                "function": function_name,
                "symbol": symbol,
                "date": date_str,
                "open": float(row["1. open"]),
                "high": float(row["2. high"]),
                "low": float(row["3. low"]),
                "close": float(row["4. close"]),
                "adjusted_close": None,
                "volume": int(float(row["5. volume"])),
                "dividend_amount": None,
                "split_coefficient": None,
                "timezone": metadata.get("5. Time Zone"),
                "fetched_at_utc": fetched_at,
            }
        records.append(record)
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
    return output_dir / f"alpha_vantage_{safe_symbol}.{file_format}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Alpha Vantage daily adjusted data as JSON or JSONL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ALPHA_VANTAGE_API_KEY"),
        help="Alpha Vantage API key. Defaults to env var ALPHA_VANTAGE_API_KEY.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY"],
        help="Symbols to download.",
    )
    parser.add_argument(
        "--outputsize",
        choices=("compact", "full"),
        default="full",
        help="Use compact for the latest ~100 rows or full for full history.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/alpha_vantage",
        help="Directory where output files will be written.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--function",
        choices=("TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED"),
        default="TIME_SERIES_DAILY",
        help=(
            "Alpha Vantage endpoint to call. "
            "TIME_SERIES_DAILY is the safer default for free keys."
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=12.5,
        help="Delay between symbol requests to stay under free-tier rate limits.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing Alpha Vantage API key. Pass --api-key or set ALPHA_VANTAGE_API_KEY.",
        )

    output_dir = Path(args.output_dir)

    for index, symbol in enumerate(args.symbols):
        try:
            records = fetch_daily_series(
                api_key=args.api_key,
                symbol=symbol,
                outputsize=args.outputsize,
                function_name=args.function,
            )
        except ValueError as exc:
            if args.function == "TIME_SERIES_DAILY_ADJUSTED":
                print(
                    f"{exc} Falling back to TIME_SERIES_DAILY for {symbol}.",
                )
                records = fetch_daily_series(
                    api_key=args.api_key,
                    symbol=symbol,
                    outputsize=args.outputsize,
                    function_name="TIME_SERIES_DAILY",
                )
            else:
                raise
        output_path = build_output_path(output_dir, symbol, args.format)
        write_records(output_path, records, args.format)
        print(f"saved {symbol} -> {output_path}")

        if index < len(args.symbols) - 1:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
