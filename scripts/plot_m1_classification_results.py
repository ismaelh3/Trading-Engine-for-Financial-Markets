#!/usr/bin/env python3
"""Create report-ready plots from Milestone 1 classification experiment artifacts."""

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
CLASS_LABELS = {
    0: "Low Vol",
    1: "Mid Vol",
    2: "High Vol",
}


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
            if payload.get("task_type") == "classification":
                artifact_path = payload.get("artifacts", {}).get(artifact_key)
                if artifact_path:
                    return Path(str(artifact_path))

    if artifacts_dir is None:
        raise FileNotFoundError(
            f"Could not resolve `{artifact_key}` for model `{model_name}` from results or artifacts dir.",
        )
    return Path(artifacts_dir) / fallback_filename


def _discover_classification_models(
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
                if payload.get("task_type") == "classification":
                    discovered.add(model_name)

    if artifacts_dir is not None:
        artifacts_path = Path(artifacts_dir)
        if artifacts_path.exists():
            for path in artifacts_path.glob("predictions_*.csv"):
                model_name = path.stem.removeprefix("predictions_")
                if model_name in MODEL_LABELS:
                    discovered.add(model_name)

    return [model_name for model_name in MODEL_ORDER if model_name in discovered]


def plot_class_timeline(
    predictions_path: str | Path,
    output_path: str | Path,
    model_label: str,
) -> None:
    plt, mdates = _prepare_matplotlib()
    df = _load_csv(predictions_path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.step(
        df["date"],
        df["actual_class"],
        where="mid",
        linewidth=2.2,
        color="#1f2937",
        label="Actual class",
    )
    ax.step(
        df["date"],
        df["predicted_class"],
        where="mid",
        linewidth=1.8,
        color="#0f766e",
        alpha=0.95,
        label=f"Predicted class ({model_label})",
    )

    ax.set_title(f"Predicted vs. Actual Volatility Class: {model_label}")
    ax.set_ylabel("Volatility Regime")
    ax.set_xlabel("Date")
    ax.set_yticks(list(CLASS_LABELS.keys()))
    ax.set_yticklabels(list(CLASS_LABELS.values()))
    ax.legend(frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    predictions_path: str | Path,
    output_path: str | Path,
    model_label: str,
) -> None:
    plt, _ = _prepare_matplotlib()
    df = _load_csv(predictions_path)
    actual = df["actual_class"].to_numpy(dtype=int)
    predicted = df["predicted_class"].to_numpy(dtype=int)

    matrix = np.zeros((3, 3), dtype=int)
    for actual_class, predicted_class in zip(actual, predicted, strict=False):
        matrix[actual_class, predicted_class] += 1

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized share")

    tick_labels = [CLASS_LABELS[class_id] for class_id in range(3)]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion Matrix: {model_label}")

    for row in range(3):
        for col in range(3):
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{normalized[row, col]:.0%}",
                ha="center",
                va="center",
                color="#111827" if normalized[row, col] < 0.55 else "white",
                fontsize=10,
            )

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
        "Naive": "#6b7280",
        "Elastic Net": "#1d4ed8",
        "XGBoost": "#b45309",
        "LSTM": "#059669",
        "1D CNN": "#7c3aed",
        "CTTS": "#b91c1c",
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

    ax.set_title("Cumulative Equity Curves: Classification Walk-Forward Test")
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
    if "train_score" not in df.columns and "validation_score" in df.columns:
        df["train_score"] = np.nan
    df = df.sort_values(["block_id", "epoch"]).reset_index(drop=True)

    metric_label = "Validation score"
    if "selection_metric" in df.columns:
        selection_metric = str(df["selection_metric"].iloc[0])
        metric_label = selection_metric.replace("_", " ").title()

    aggregate_columns = ["train_loss", "validation_loss", "train_batch_loss"]
    if "validation_score" in df.columns:
        aggregate_columns.extend(["train_score", "validation_score"])
    stats = df.groupby("epoch")[aggregate_columns].agg(["mean", "std"]).reset_index()

    flattened_columns = ["epoch"]
    for column in aggregate_columns:
        flattened_columns.extend([f"{column}_mean", f"{column}_std"])
    stats.columns = flattened_columns

    block_count = int(df["block_id"].nunique())
    if "is_best_epoch" in df.columns:
        best_epochs = df.loc[df["is_best_epoch"] == 1, "epoch"].to_numpy(dtype=float)
    else:
        best_epochs = np.array([], dtype=float)

    fig, (ax_loss, ax_score) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    ax_loss.plot(
        stats["epoch"],
        stats["train_loss_mean"],
        label="Train loss (eval mode)",
        linewidth=2.0,
        color="#1d4ed8",
    )
    ax_loss.fill_between(
        stats["epoch"],
        stats["train_loss_mean"] - stats["train_loss_std"].fillna(0.0),
        stats["train_loss_mean"] + stats["train_loss_std"].fillna(0.0),
        color="#1d4ed8",
        alpha=0.12,
    )
    ax_loss.plot(
        stats["epoch"],
        stats["validation_loss_mean"],
        label="Validation loss",
        linewidth=2.0,
        color="#b45309",
    )
    ax_loss.fill_between(
        stats["epoch"],
        stats["validation_loss_mean"] - stats["validation_loss_std"].fillna(0.0),
        stats["validation_loss_mean"] + stats["validation_loss_std"].fillna(0.0),
        color="#b45309",
        alpha=0.12,
    )
    ax_loss.plot(
        stats["epoch"],
        stats["train_batch_loss_mean"],
        label="Train batch loss",
        linewidth=1.4,
        color="#6b7280",
        linestyle=":",
    )
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend(frameon=True, loc="upper right")

    if "validation_score" in df.columns:
        if "train_score_mean" in stats.columns:
            ax_score.plot(
                stats["epoch"],
                stats["train_score_mean"],
                label=f"Train {metric_label}",
                linewidth=1.6,
                color="#94a3b8",
            )
        ax_score.plot(
            stats["epoch"],
            stats["validation_score_mean"],
            label=f"Validation {metric_label}",
            linewidth=2.0,
            color="#059669",
        )
        ax_score.fill_between(
            stats["epoch"],
            stats["validation_score_mean"] - stats["validation_score_std"].fillna(0.0),
            stats["validation_score_mean"] + stats["validation_score_std"].fillna(0.0),
            color="#059669",
            alpha=0.12,
        )
        ax_score.set_ylabel(metric_label)
        ax_score.legend(frameon=True, loc="best")

    if len(best_epochs) > 0:
        best_epoch_center = float(np.median(best_epochs))
        for axis in (ax_loss, ax_score):
            axis.axvline(
                best_epoch_center,
                color="#7c3aed",
                linewidth=1.6,
                linestyle="--",
            )
            if block_count > 1:
                axis.axvspan(
                    float(np.min(best_epochs)),
                    float(np.max(best_epochs)),
                    color="#7c3aed",
                    alpha=0.08,
                )

    ax_loss.set_title(f"Training Curve: {model_label} ({block_count} block{'s' if block_count != 1 else ''})")
    ax_score.set_xlabel("Epoch")
    ax_score.text(
        0.99,
        0.02,
        "Early stopping uses validation score; bands show +/-1 std across blocks.",
        transform=ax_score.transAxes,
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
        description="Create report-ready plots from a completed Milestone 1 classification artifact directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/m1_classification",
        help="Fallback directory containing classification predictions_*.csv and backtest_*.csv files.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory containing per-model results.json files.",
    )
    parser.add_argument(
        "--classification-model",
        default="elastic_net",
        choices=tuple(MODEL_ORDER),
        help="Model to use for the class timeline and confusion matrix figures.",
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
        help="Directory for generated figure files. Defaults to the artifact directory.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None
    results_root = Path(args.results_root) if args.results_root else None
    output_dir = Path(args.output_dir) if args.output_dir else (
        artifacts_dir if artifacts_dir is not None else Path(".")
    )

    predictions_path = _resolve_artifact_path(
        model_name=args.classification_model,
        artifact_key="predictions_csv",
        results_root=results_root,
        artifacts_dir=artifacts_dir,
        fallback_filename=f"predictions_{args.classification_model}.csv",
    )

    plot_class_timeline(
        predictions_path=predictions_path,
        output_path=output_dir / f"class_timeline_{args.classification_model}.png",
        model_label=MODEL_LABELS[args.classification_model],
    )
    plot_confusion_matrix(
        predictions_path=predictions_path,
        output_path=output_dir / f"confusion_matrix_{args.classification_model}.png",
        model_label=MODEL_LABELS[args.classification_model],
    )
    training_history_path = _resolve_artifact_path(
        model_name=args.classification_model,
        artifact_key="training_history_csv",
        results_root=results_root,
        artifacts_dir=artifacts_dir,
        fallback_filename=f"training_history_{args.classification_model}.csv",
    )
    if training_history_path.exists():
        plot_training_curve(
            training_history_path=training_history_path,
            output_path=output_dir / f"training_curve_{args.classification_model}.png",
            model_label=MODEL_LABELS[args.classification_model],
        )

    equity_models = args.equity_models or _discover_classification_models(
        results_root=results_root,
        artifacts_dir=artifacts_dir,
    )
    if not equity_models:
        raise FileNotFoundError(
            "No classification models were found in the provided results or artifacts directories.",
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
        output_path=output_dir / "equity_curves_m1_classification.png",
    )

    print(f"saved class timeline -> {output_dir / f'class_timeline_{args.classification_model}.png'}")
    print(f"saved confusion matrix -> {output_dir / f'confusion_matrix_{args.classification_model}.png'}")
    if training_history_path.exists():
        print(f"saved training curve -> {output_dir / f'training_curve_{args.classification_model}.png'}")
    print(f"saved equity plot -> {output_dir / 'equity_curves_m1_classification.png'}")


if __name__ == "__main__":
    main()
