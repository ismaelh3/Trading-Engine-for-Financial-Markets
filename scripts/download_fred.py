#!/usr/bin/env python3
"""Download FRED series and save them as JSON or JSONL."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


DEFAULT_SERIES = {
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "DGS10": "DGS10",
    "DGS2": "DGS2",
    "VIX": "VIXCLS",
    "CREDIT_SPREAD": "BAA10Y",
}

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

load_dotenv()


def parse_series_args(series_args: list[str] | None) -> dict[str, str]:
    if not series_args:
        return DEFAULT_SERIES.copy()

    parsed: dict[str, str] = {}
    for item in series_args:
        if "=" in item:
            alias, series_id = item.split("=", 1)
            parsed[alias.strip()] = series_id.strip()
        else:
            series_id = item.strip()
            parsed[series_id] = series_id
    return parsed


def fetch_fred_series(
    api_key: str,
    series_id: str,
    start: str,
    end: str,
) -> list[dict[str, object]]:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    observations = payload.get("observations", [])
    fetched_at = datetime.now(timezone.utc).isoformat()

    records: list[dict[str, object]] = []
    for row in observations:
        value = row.get("value")
        if value == ".":
            numeric_value = None
        else:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = None

        records.append(
            {
                "source": "fred",
                "series_id": series_id,
                "date": row.get("date"),
                "value": numeric_value,
                "raw_value": value,
                "realtime_start": row.get("realtime_start"),
                "realtime_end": row.get("realtime_end"),
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


def build_output_path(output_dir: Path, alias: str, file_format: str) -> Path:
    safe_alias = alias.lower().replace(" ", "_")
    return output_dir / f"fred_{safe_alias}.{file_format}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FRED series and save each one as JSON or JSONL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("FRED_API_KEY"),
        help="FRED API key. Defaults to env var FRED_API_KEY.",
    )
    parser.add_argument(
        "--start",
        default="2014-01-01",
        help="Observation start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default="2025-12-31",
        help="Observation end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/fred",
        help="Directory where output files will be written.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--series",
        nargs="*",
        help=(
            "Series to download. Use SERIES_ID or ALIAS=SERIES_ID. "
            "If omitted, downloads the default macro/market set."
        ),
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing FRED API key. Pass --api-key or set FRED_API_KEY.",
        )

    series_map = parse_series_args(args.series)
    output_dir = Path(args.output_dir)

    for alias, series_id in series_map.items():
        records = fetch_fred_series(
            api_key=args.api_key,
            series_id=series_id,
            start=args.start,
            end=args.end,
        )
        output_path = build_output_path(output_dir, alias, args.format)
        write_records(output_path, records, args.format)
        print(f"saved {series_id} -> {output_path}")


if __name__ == "__main__":
    main()
