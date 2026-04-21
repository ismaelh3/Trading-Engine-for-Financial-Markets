#!/usr/bin/env python3
"""Create report-ready plots from Milestone 1 experiment artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_LABELS = {
    "naive": "Naive",
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "cnn": "1D CNN",
    "cnn_lstm": "CNN+LSTM",
    "ctts": "CTTS",
}

MODEL_ORDER = ["naive", "elastic_net", "xgboost", "lstm", "cnn", "cnn_lstm", "ctts"]


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


def _load_model_results(results_root: str | Path, model_name: str) -> dict[str, object]:
    results_path = Path(results_root) / model_name / "results.json"
    return json.loads(results_path.read_text(encoding="utf-8"))


def _resolve_artifact_path(
    model_name: str,
    artifact_key: str,
    results_root: str | Path | None,
    artifacts_dir: str | Path | None,
    fallback_filename: str,
) -> Path:
    if results_root is not None:
        results_path = Path(results_root) / model_name / "results.json"
        if results_path.exists():
            payload = _load_model_results(results_root, model_name)
            if payload.get("task_type") == "regression":
                artifact_path = payload.get("artifacts", {}).get(artifact_key)
                if artifact_path:
                    return Path(str(artifact_path))

    if artifacts_dir is None:
        raise FileNotFoundError(
            f"Could not resolve `{artifact_key}` for model `{model_name}` from results or artifacts dir.",
        )
    return Path(artifacts_dir) / fallback_filename


def _discover_equity_models(
    results_root: str | Path | None,
    artifacts_dir: str | Path | None,
) -> list[str]:
    discovered: set[str] = set()

    if results_root is not None:
        root_path = Path(results_root)
        if root_path.exists():
            for model_name in MODEL_ORDER:
                results_path = root_path / model_name / "results.json"
                if not results_path.exists():
                    continue
                payload = json.loads(results_path.read_text(encoding="utf-8"))
                if payload.get("task_type") == "regression":
                    discovered.add(model_name)

    if artifacts_dir is not None:
        artifacts_path = Path(artifacts_dir)
        if artifacts_path.exists():
            for path in artifacts_path.glob("backtest_*.csv"):
                model_name = path.stem.removeprefix("backtest_")
                if model_name in MODEL_LABELS:
                    discovered.add(model_name)

    return [model_name for model_name in MODEL_ORDER if model_name in discovered]


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


def plot_training_curve(
    training_history_path: str | Path,
    output_path: str | Path,
    model_label: str,
) -> None:
    plt, _ = _prepare_matplotlib()
    df = pd.read_csv(training_history_path)
    if "block_id" not in df.columns:
        df["block_id"] = 0
    if "train_batch_loss" not in df.columns:
        df["train_batch_loss"] = df["train_loss"]
    df = df.sort_values(["block_id", "epoch"]).reset_index(drop=True)

    stats = df.groupby("epoch")[["train_loss", "validation_loss", "train_batch_loss"]].agg(["mean", "std"]).reset_index()
    stats.columns = [
        "epoch",
        "train_loss_mean",
        "train_loss_std",
        "validation_loss_mean",
        "validation_loss_std",
        "train_batch_loss_mean",
        "train_batch_loss_std",
    ]
    block_count = int(df["block_id"].nunique())
    if "is_best_epoch" in df.columns:
        best_epochs = df.loc[df["is_best_epoch"] == 1, "epoch"].to_numpy(dtype=float)
    else:
        best_epochs = np.array([], dtype=float)
    loss_name = str(df["loss_name"].iloc[0]) if "loss_name" in df.columns else "loss"

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.plot(
        stats["epoch"],
        stats["train_loss_mean"],
        label="Train loss (eval mode)",
        linewidth=2.0,
        color="#1d4ed8",
    )
    ax.fill_between(
        stats["epoch"],
        stats["train_loss_mean"] - stats["train_loss_std"].fillna(0.0),
        stats["train_loss_mean"] + stats["train_loss_std"].fillna(0.0),
        color="#1d4ed8",
        alpha=0.12,
    )
    ax.plot(
        stats["epoch"],
        stats["validation_loss_mean"],
        label="Validation loss",
        linewidth=2.0,
        color="#b45309",
    )
    ax.fill_between(
        stats["epoch"],
        stats["validation_loss_mean"] - stats["validation_loss_std"].fillna(0.0),
        stats["validation_loss_mean"] + stats["validation_loss_std"].fillna(0.0),
        color="#b45309",
        alpha=0.12,
    )
    if "train_batch_loss" in df.columns:
        ax.plot(
            stats["epoch"],
            stats["train_batch_loss_mean"],
            label="Train batch loss",
            linewidth=1.4,
            color="#6b7280",
            linestyle=":",
        )
    if len(best_epochs) > 0:
        best_epoch_center = float(np.median(best_epochs))
        ax.axvline(
            best_epoch_center,
            color="#059669",
            linewidth=1.6,
            linestyle="--",
            label="Median best epoch",
        )
        if block_count > 1:
            ax.axvspan(
                float(np.min(best_epochs)),
                float(np.max(best_epochs)),
                color="#059669",
                alpha=0.08,
            )
    ylabel = "Loss"
    if loss_name == "qlike":
        ylabel = "QLIKE loss (lower is better)"
    elif loss_name == "huber":
        ylabel = "Huber loss"
    elif loss_name == "mse":
        ylabel = "MSE loss"
    ax.set_title(f"Training Curve: {model_label} ({block_count} block{'s' if block_count != 1 else ''})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    ax.text(
        0.99,
        0.02,
        "Shaded bands show +/-1 std across walk-forward blocks.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
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
        default="artifacts/m1",
        help="Fallback directory containing predictions_*.csv and backtest_*.csv files.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory containing per-model results.json files.",
    )
    parser.add_argument(
        "--forecast-model",
        default="elastic_net",
        choices=tuple(MODEL_ORDER),
        help="Model to use for the predicted-vs-realized volatility figure.",
    )
    parser.add_argument(
        "--equity-models",
        nargs="+",
        choices=tuple(MODEL_ORDER),
        default=None,
        help="Optional explicit list of models to include in the equity-curve plot.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated figure files. Defaults to the artifact directory.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None
    results_root = Path(args.results_root) if args.results_root else None
    output_dir = Path(args.output_dir) if args.output_dir else (
        artifacts_dir if artifacts_dir is not None else Path(".")
    )

    plot_predicted_vs_realized(
        predictions_path=_resolve_artifact_path(
            model_name=args.forecast_model,
            artifact_key="predictions_csv",
            results_root=results_root,
            artifacts_dir=artifacts_dir,
            fallback_filename=f"predictions_{args.forecast_model}.csv",
        ),
        output_path=output_dir / f"forecast_vs_realized_{args.forecast_model}.png",
        model_label=MODEL_LABELS[args.forecast_model],
    )
    training_history_path = _resolve_artifact_path(
        model_name=args.forecast_model,
        artifact_key="training_history_csv",
        results_root=results_root,
        artifacts_dir=artifacts_dir,
        fallback_filename=f"training_history_{args.forecast_model}.csv",
    )
    if training_history_path.exists():
        plot_training_curve(
            training_history_path=training_history_path,
            output_path=output_dir / f"training_curve_{args.forecast_model}.png",
            model_label=MODEL_LABELS[args.forecast_model],
        )

    equity_models = args.equity_models or _discover_equity_models(
        results_root=results_root,
        artifacts_dir=artifacts_dir,
    )
    if not equity_models:
        raise FileNotFoundError(
            "No equity models were found in the provided results or artifacts directories.",
        )

    plot_equity_curves(
        backtest_paths={
            MODEL_LABELS[model_name]: _resolve_artifact_path(
                model_name=model_name,
                artifact_key="backtest_csv",
                results_root=results_root,
                artifacts_dir=artifacts_dir,
                fallback_filename=f"backtest_{model_name}.csv",
            )
            for model_name in equity_models
        },
        output_path=output_dir / "equity_curves_m1.png",
    )

    print(f"saved forecast plot -> {output_dir / f'forecast_vs_realized_{args.forecast_model}.png'}")
    if training_history_path.exists():
        print(f"saved training curve -> {output_dir / f'training_curve_{args.forecast_model}.png'}")
    print(f"saved equity plot -> {output_dir / 'equity_curves_m1.png'}")


if __name__ == "__main__":
    main()
