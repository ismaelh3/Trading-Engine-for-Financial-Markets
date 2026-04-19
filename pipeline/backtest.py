"""Simple allocation rules and backtest helpers for Milestone 1."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def apply_allocation_rule(
    predicted_vol: float,
    low_threshold: float,
    high_threshold: float,
) -> float:
    if predicted_vol <= low_threshold:
        return 1.0
    if predicted_vol <= high_threshold:
        return 0.5
    return 0.0


def build_strategy_weights(
    prediction_df: pd.DataFrame,
    rebalance_every_days: int,
) -> pd.DataFrame:
    strategy_df = prediction_df.sort_values("date").reset_index(drop=True).copy()
    strategy_df["signal_weight"] = np.nan

    for index in range(len(strategy_df)):
        if index % rebalance_every_days != 0:
            continue
        strategy_df.loc[index, "signal_weight"] = apply_allocation_rule(
            predicted_vol=float(strategy_df.loc[index, "predicted_vol"]),
            low_threshold=float(strategy_df.loc[index, "train_low_vol_threshold"]),
            high_threshold=float(strategy_df.loc[index, "train_high_vol_threshold"]),
        )

    strategy_df["signal_weight"] = strategy_df["signal_weight"].ffill().fillna(0.0)
    strategy_df["applied_weight"] = strategy_df["signal_weight"].shift(1).fillna(0.0)
    return strategy_df


def run_backtest(
    prediction_df: pd.DataFrame,
    rebalance_every_days: int = 5,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    strategy_df = build_strategy_weights(
        prediction_df=prediction_df,
        rebalance_every_days=rebalance_every_days,
    )
    cost_rate = transaction_cost_bps / 10_000.0

    strategy_df["turnover"] = strategy_df["signal_weight"].diff().abs().fillna(0.0)
    strategy_df["transaction_cost"] = strategy_df["turnover"] * cost_rate
    strategy_df["strategy_return"] = (
        strategy_df["applied_weight"] * strategy_df["asset_return_1d"]
        - strategy_df["transaction_cost"]
    )
    strategy_df["benchmark_return"] = strategy_df["asset_return_1d"]
    strategy_df["cash_return"] = 0.0

    strategy_df["strategy_equity"] = (1.0 + strategy_df["strategy_return"]).cumprod()
    strategy_df["benchmark_equity"] = (1.0 + strategy_df["benchmark_return"]).cumprod()
    return strategy_df


def apply_classification_allocation_rule(predicted_class: int) -> float:
    if predicted_class <= 0:
        return 1.0
    if predicted_class == 1:
        return 0.5
    return 0.0


def build_classification_strategy_weights(
    prediction_df: pd.DataFrame,
    rebalance_every_days: int,
) -> pd.DataFrame:
    strategy_df = prediction_df.sort_values("date").reset_index(drop=True).copy()
    strategy_df["signal_weight"] = np.nan

    for index in range(len(strategy_df)):
        if index % rebalance_every_days != 0:
            continue
        strategy_df.loc[index, "signal_weight"] = apply_classification_allocation_rule(
            predicted_class=int(strategy_df.loc[index, "predicted_class"]),
        )

    strategy_df["signal_weight"] = strategy_df["signal_weight"].ffill().fillna(0.0)
    strategy_df["applied_weight"] = strategy_df["signal_weight"].shift(1).fillna(0.0)
    return strategy_df


def run_classification_backtest(
    prediction_df: pd.DataFrame,
    rebalance_every_days: int = 5,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    strategy_df = build_classification_strategy_weights(
        prediction_df=prediction_df,
        rebalance_every_days=rebalance_every_days,
    )
    cost_rate = transaction_cost_bps / 10_000.0

    strategy_df["turnover"] = strategy_df["signal_weight"].diff().abs().fillna(0.0)
    strategy_df["transaction_cost"] = strategy_df["turnover"] * cost_rate
    strategy_df["strategy_return"] = (
        strategy_df["applied_weight"] * strategy_df["asset_return_1d"]
        - strategy_df["transaction_cost"]
    )
    strategy_df["benchmark_return"] = strategy_df["asset_return_1d"]
    strategy_df["cash_return"] = 0.0
    strategy_df["strategy_equity"] = (1.0 + strategy_df["strategy_return"]).cumprod()
    strategy_df["benchmark_equity"] = (1.0 + strategy_df["benchmark_return"]).cumprod()
    return strategy_df


def annualized_return(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return math.nan
    equity_curve = (1.0 + daily_returns).cumprod()
    total_return = float(equity_curve.iloc[-1])
    periods = len(daily_returns)
    if periods == 0 or total_return <= 0:
        return math.nan
    return total_return ** (252.0 / periods) - 1.0


def annualized_volatility(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return math.nan
    return float(daily_returns.std(ddof=0) * math.sqrt(252.0))


def sharpe_ratio(daily_returns: pd.Series) -> float:
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol == 0 or math.isnan(ann_vol):
        return math.nan
    return annualized_return(daily_returns) / ann_vol


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return math.nan
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def summarize_backtest(strategy_df: pd.DataFrame) -> dict[str, float]:
    strategy_returns = strategy_df["strategy_return"]
    benchmark_returns = strategy_df["benchmark_return"]

    return {
        "strategy_annualized_return": annualized_return(strategy_returns),
        "strategy_annualized_volatility": annualized_volatility(strategy_returns),
        "strategy_sharpe_ratio": sharpe_ratio(strategy_returns),
        "strategy_max_drawdown": max_drawdown(strategy_df["strategy_equity"]),
        "strategy_average_weight": float(strategy_df["applied_weight"].mean()),
        "strategy_turnover_total": float(strategy_df["turnover"].sum()),
        "benchmark_annualized_return": annualized_return(benchmark_returns),
        "benchmark_annualized_volatility": annualized_volatility(benchmark_returns),
        "benchmark_sharpe_ratio": sharpe_ratio(benchmark_returns),
        "benchmark_max_drawdown": max_drawdown(strategy_df["benchmark_equity"]),
        "observations": float(len(strategy_df)),
    }
