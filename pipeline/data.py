"""Dataset engineering and time-aware split utilities for Milestone 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "feat_return_1d",
    "feat_return_5d",
    "feat_return_20d",
    "feat_realized_vol_5d",
    "feat_realized_vol_20d",
    "feat_realized_vol_60d",
    "feat_downside_vol_20d",
    "feat_drawdown_60d",
    "feat_ma_ratio_20d",
    "feat_ma_ratio_60d",
    "feat_range_pct_1d",
    "feat_range_pct_5d",
    "feat_volume_zscore_20d",
    "feat_vix_level",
    "feat_vix_change_5d",
    "feat_yield_10y",
    "feat_yield_2y",
    "feat_term_spread",
    "feat_credit_spread",
    "feat_cpi_yoy",
    "feat_unrate_level",
    "feat_unrate_change_21d",
    "feat_month_sin",
    "feat_month_cos",
]


@dataclass(frozen=True)
class WalkForwardBlock:
    block_id: int
    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def load_merged_market_macro_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _annualized_downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    downside = returns.clip(upper=0.0) ** 2
    return np.sqrt(252.0 * downside.rolling(window).mean())


def _conservative_macro_lag(series: pd.Series, lag_days: int) -> pd.Series:
    return series.shift(lag_days)


def build_m1_dataset(
    merged_df: pd.DataFrame,
    macro_lag_days: int = 21,
    target_horizons: tuple[int, ...] = (5, 20),
) -> pd.DataFrame:
    df = merged_df.sort_values("date").reset_index(drop=True).copy()
    df["spy_adjusted_close"] = df["spy_adjusted_close"].astype(float)
    df["spy_volume"] = df["spy_volume"].astype(float)

    df["cpi_lagged"] = _conservative_macro_lag(df["cpi"], macro_lag_days)
    df["unrate_lagged"] = _conservative_macro_lag(df["unrate"], macro_lag_days)

    log_price = np.log(df["spy_adjusted_close"])
    df["asset_log_return_1d"] = log_price.diff()
    df["asset_return_1d"] = df["spy_adjusted_close"].pct_change()

    df["feat_return_1d"] = df["asset_return_1d"]
    df["feat_return_5d"] = df["spy_adjusted_close"].pct_change(5)
    df["feat_return_20d"] = df["spy_adjusted_close"].pct_change(20)
    df["feat_realized_vol_5d"] = df["asset_log_return_1d"].rolling(5).std() * np.sqrt(252.0)
    df["feat_realized_vol_20d"] = df["asset_log_return_1d"].rolling(20).std() * np.sqrt(252.0)
    df["feat_realized_vol_60d"] = df["asset_log_return_1d"].rolling(60).std() * np.sqrt(252.0)
    df["feat_downside_vol_20d"] = _annualized_downside_volatility(df["asset_log_return_1d"], 20)
    df["feat_drawdown_60d"] = df["spy_adjusted_close"] / df["spy_adjusted_close"].rolling(60).max() - 1.0
    df["feat_ma_ratio_20d"] = df["spy_adjusted_close"] / df["spy_adjusted_close"].rolling(20).mean() - 1.0
    df["feat_ma_ratio_60d"] = df["spy_adjusted_close"] / df["spy_adjusted_close"].rolling(60).mean() - 1.0
    df["feat_range_pct_1d"] = (df["spy_high"] - df["spy_low"]) / df["spy_adjusted_close"]
    df["feat_range_pct_5d"] = df["feat_range_pct_1d"].rolling(5).mean()

    log_volume = np.log(df["spy_volume"].replace(0, np.nan))
    rolling_volume_mean = log_volume.rolling(20).mean()
    rolling_volume_std = log_volume.rolling(20).std()
    df["feat_volume_zscore_20d"] = (log_volume - rolling_volume_mean) / rolling_volume_std

    df["feat_vix_level"] = df["vix"]
    df["feat_vix_change_5d"] = df["vix"].diff(5)
    df["feat_yield_10y"] = df["dgs10"]
    df["feat_yield_2y"] = df["dgs2"]
    df["feat_term_spread"] = df["dgs10"] - df["dgs2"]
    df["feat_credit_spread"] = df["credit_spread"]
    df["feat_cpi_yoy"] = df["cpi_lagged"].pct_change(252)
    df["feat_unrate_level"] = df["unrate_lagged"]
    df["feat_unrate_change_21d"] = df["unrate_lagged"].diff(21)

    month = df["date"].dt.month
    df["feat_month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["feat_month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    epsilon = 1e-8
    squared_log_returns = df["asset_log_return_1d"].pow(2)
    for horizon in target_horizons:
        future_var = squared_log_returns.rolling(horizon).sum().shift(-horizon)
        future_vol = np.sqrt((252.0 / horizon) * future_var)
        df[f"target_future_vol_{horizon}d"] = future_vol
        df[f"target_log_future_vol_{horizon}d"] = np.log(future_vol + epsilon)

    keep_columns = [
        "date",
        "ticker",
        "spy_adjusted_close",
        "asset_return_1d",
        "asset_log_return_1d",
        *DEFAULT_FEATURE_COLUMNS,
        *[f"target_future_vol_{horizon}d" for horizon in target_horizons],
        *[f"target_log_future_vol_{horizon}d" for horizon in target_horizons],
    ]
    output_df = df[keep_columns].dropna().reset_index(drop=True)
    return output_df


def save_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def feature_columns_from_dataset(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("feat_")]


def generate_walk_forward_blocks(
    df: pd.DataFrame,
    test_start_date: str,
    min_train_days: int = 756,
    validation_days: int = 252,
    embargo_days: int = 20,
    retrain_every_days: int = 21,
) -> list[WalkForwardBlock]:
    if retrain_every_days <= 0:
        raise ValueError("retrain_every_days must be positive.")

    dates = df["date"]
    test_start_timestamp = pd.Timestamp(test_start_date)
    test_start_positions = np.flatnonzero(dates >= test_start_timestamp)
    if len(test_start_positions) == 0:
        raise ValueError(f"No rows found at or after test start date {test_start_date}.")

    test_start_index = int(test_start_positions[0])
    blocks: list[WalkForwardBlock] = []
    block_id = 0

    for block_start in range(test_start_index, len(df), retrain_every_days):
        train_end_exclusive = block_start - embargo_days
        if train_end_exclusive <= validation_days:
            continue

        validation_start = train_end_exclusive - validation_days
        if validation_start < min_train_days:
            continue

        train_index = np.arange(0, validation_start, dtype=int)
        validation_index = np.arange(validation_start, train_end_exclusive, dtype=int)
        block_end = min(block_start + retrain_every_days, len(df))
        test_index = np.arange(block_start, block_end, dtype=int)
        if len(test_index) == 0:
            continue

        blocks.append(
            WalkForwardBlock(
                block_id=block_id,
                train_index=train_index,
                validation_index=validation_index,
                test_index=test_index,
                train_end_date=dates.iloc[train_end_exclusive - 1],
                test_start_date=dates.iloc[test_index[0]],
                test_end_date=dates.iloc[test_index[-1]],
            )
        )
        block_id += 1

    if not blocks:
        raise ValueError("No valid walk-forward blocks were generated. Check your split settings.")
    return blocks

