#!/usr/bin/env python3
"""Create report-ready plots from Milestone 1 experiment artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _prepare_matplotlib() -> tuple:
    mpl_dir = ROOT / ".mplconfig"
    cache_dir = ROOT / ".cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    return plt, mdates


def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def plot_predicted_vs_realized(
    predictions_path: str | Path,
    output_path: str | Path,
    model_label: str,
) -> None:
    plt, mdates = _prepare_matplotlib()
    df = _load_csv(predictions_path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        df["date"],
        df["actual_vol"],
        label="Realized 20-day volatility",
        linewidth=2.2,
        color="#1f2937",
    )
    ax.plot(
        df["date"],
        df["predicted_vol"],
        label=f"Predicted volatility ({model_label})",
        linewidth=1.8,
        color="#0f766e",
        alpha=0.95,
    )

    ax.set_title(f"Predicted vs. Realized SPY Volatility: {model_label}")
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("Date")
    ax.legend(frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(
    backtest_paths: dict[str, str | Path],
    output_path: str | Path,
) -> None:
    plt, mdates = _prepare_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    benchmark_plotted = False
    palette = {
        "Elastic Net": "#1d4ed8",
        "XGBoost": "#b45309",
        "LSTM": "#059669",
        "Buy-and-Hold SPY": "#111827",
    }

    for label, path in backtest_paths.items():
        df = _load_csv(path)
        ax.plot(
            df["date"],
            df["strategy_equity"],
            label=label,
            linewidth=2.0,
            color=palette.get(label),
        )
        if not benchmark_plotted:
            ax.plot(
                df["date"],
                df["benchmark_equity"],
                label="Buy-and-Hold SPY",
                linewidth=2.4,
                color=palette["Buy-and-Hold SPY"],
                linestyle="--",
            )
            benchmark_plotted = True

    ax.set_title("Cumulative Equity Curves: Milestone 1 Out-of-Sample Test")
    ax.set_ylabel("Cumulative wealth")
    ax.set_xlabel("Date")
    ax.legend(frameon=True, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create report-ready plots from a completed Milestone 1 artifact directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/m1_tuned_full",
        help="Directory containing predictions_*.csv and backtest_*.csv files.",
    )
    parser.add_argument(
        "--forecast-model",
        default="elastic_net",
        choices=("naive", "elastic_net", "xgboost", "lstm", "cnn"),
        help="Model to use for the predicted-vs-realized volatility figure.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated figure files. Defaults to the artifact directory.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir) if args.output_dir else artifacts_dir

    forecast_labels = {
        "naive": "Naive",
        "elastic_net": "Elastic Net",
        "xgboost": "XGBoost",
        "lstm": "LSTM",
        "cnn": "1D CNN",
    }

    plot_predicted_vs_realized(
        predictions_path=artifacts_dir / f"predictions_{args.forecast_model}.csv",
        output_path=output_dir / f"forecast_vs_realized_{args.forecast_model}.png",
        model_label=forecast_labels[args.forecast_model],
    )

    plot_equity_curves(
        backtest_paths={
            "Elastic Net": artifacts_dir / "backtest_elastic_net.csv",
            "XGBoost": artifacts_dir / "backtest_xgboost.csv",
            "LSTM": artifacts_dir / "backtest_lstm.csv",
        },
        output_path=output_dir / "equity_curves_m1.png",
    )

    print(f"saved forecast plot -> {output_dir / f'forecast_vs_realized_{args.forecast_model}.png'}")
    print(f"saved equity plot -> {output_dir / 'equity_curves_m1.png'}")


if __name__ == "__main__":
    main()
