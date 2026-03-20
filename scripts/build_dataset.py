#!/usr/bin/env python3
"""Build a merged daily market and macro dataset from raw JSON/JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ALPHA_FILE = "data/raw/alpha_vantage/alpha_vantage_spy.jsonl"
DEFAULT_FRED_DIR = "data/raw/fred"
DEFAULT_OUTPUT = "data/processed/merged_market_macro.csv"

FRED_COLUMN_MAP = {
    "CPI": "cpi",
    "CPIAUCSL": "cpi",
    "UNRATE": "unrate",
    "DGS10": "dgs10",
    "DGS2": "dgs2",
    "VIX": "vix",
    "VIXCLS": "vix",
    "CREDIT_SPREAD": "credit_spread",
    "BAA10Y": "credit_spread",
}


def read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_alpha_vantage_market_data(path: Path) -> pd.DataFrame:
    records = read_json_or_jsonl(path)
    if not records:
        raise ValueError(f"No market records found in {path}")

    market_df = pd.DataFrame(records).copy()
    market_df["date"] = pd.to_datetime(market_df["date"])
    market_df = market_df.sort_values("date").drop_duplicates(subset=["date"])

    market_df = market_df.rename(
        columns={
            "symbol": "ticker",
            "open": "spy_open",
            "high": "spy_high",
            "low": "spy_low",
            "close": "spy_close",
            "adjusted_close": "spy_adjusted_close",
            "volume": "spy_volume",
        }
    )

    keep_columns = [
        "date",
        "ticker",
        "spy_open",
        "spy_high",
        "spy_low",
        "spy_close",
        "spy_adjusted_close",
        "spy_volume",
    ]
    return market_df[keep_columns]


def load_single_fred_file(path: Path) -> pd.DataFrame:
    records = read_json_or_jsonl(path)
    if not records:
        raise ValueError(f"No FRED records found in {path}")

    fred_df = pd.DataFrame(records).copy()
    fred_df["date"] = pd.to_datetime(fred_df["date"])
    fred_df = fred_df.sort_values("date").drop_duplicates(subset=["date"])

    series_id = fred_df["series_id"].iloc[0]
    value_column = FRED_COLUMN_MAP.get(series_id, series_id.lower())

    fred_df = fred_df.rename(columns={"value": value_column})
    return fred_df[["date", value_column]]


def load_fred_dataset(raw_dir: Path) -> pd.DataFrame:
    fred_files = sorted(raw_dir.glob("fred_*.*"))
    if not fred_files:
        raise ValueError(f"No FRED files found in {raw_dir}")

    merged_fred: pd.DataFrame | None = None
    for path in fred_files:
        single_df = load_single_fred_file(path)
        if merged_fred is None:
            merged_fred = single_df
        else:
            merged_fred = merged_fred.merge(single_df, on="date", how="outer")

    if merged_fred is None:
        raise ValueError(f"Failed to build FRED dataset from {raw_dir}")

    merged_fred = merged_fred.sort_values("date").reset_index(drop=True)
    value_columns = [column for column in merged_fred.columns if column != "date"]
    merged_fred[value_columns] = merged_fred[value_columns].ffill()
    return merged_fred


def merge_market_and_macro(
    market_df: pd.DataFrame,
    fred_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = pd.merge_asof(
        market_df.sort_values("date"),
        fred_df.sort_values("date"),
        on="date",
        direction="backward",
    )
    return merged_df.sort_values("date").reset_index(drop=True)


def write_output(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        df.to_csv(output_path, index=False)
        return

    records = df.to_dict(orient="records")
    if output_format == "json":
        output_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")


def infer_output_format(output_path: Path, explicit_format: str | None) -> str:
    if explicit_format:
        return explicit_format

    suffix = output_path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    return "csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a merged daily dataset from Alpha Vantage and FRED raw files.",
    )
    parser.add_argument(
        "--alpha-file",
        default=DEFAULT_ALPHA_FILE,
        help="Path to raw Alpha Vantage JSON or JSONL file.",
    )
    parser.add_argument(
        "--fred-dir",
        default=DEFAULT_FRED_DIR,
        help="Directory containing raw FRED JSON or JSONL files.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output file path. Format inferred from extension unless --format is given.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json", "jsonl"),
        help="Optional explicit output format override.",
    )
    args = parser.parse_args()

    alpha_path = Path(args.alpha_file)
    fred_dir = Path(args.fred_dir)
    output_path = Path(args.output)
    output_format = infer_output_format(output_path, args.format)

    market_df = load_alpha_vantage_market_data(alpha_path)
    fred_df = load_fred_dataset(fred_dir)
    merged_df = merge_market_and_macro(market_df, fred_df)

    write_output(merged_df, output_path, output_format)
    print(f"saved merged dataset -> {output_path}")
    print(f"rows={len(merged_df)} cols={len(merged_df.columns)}")
    print(f"date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")


if __name__ == "__main__":
    main()
