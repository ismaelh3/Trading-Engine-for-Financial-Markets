#!/usr/bin/env python3
"""Run the Milestone 1 walk-forward volatility forecasting experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.backtest import run_backtest, run_classification_backtest, summarize_backtest
from pipeline.data import feature_columns_from_dataset, generate_walk_forward_blocks
from pipeline.evaluation import summarize_classification_predictions, summarize_predictions
from pipeline.models import (
    run_elastic_net_block,
    run_elastic_net_classification_block,
    run_naive_block,
    run_naive_classification_block,
    run_torch_sequence_block,
    run_torch_sequence_classification_block,
    run_xgboost_block,
    run_xgboost_classification_block,
    tune_elastic_net_model,
    tune_elastic_net_classification_model,
    tune_naive_model,
    tune_naive_classification_model,
    tune_torch_sequence_model,
    tune_torch_sequence_classification_model,
    tune_xgboost_model,
    tune_xgboost_classification_model,
)


DEFAULT_INPUT = "data/processed/m1/m1_dataset.csv"
DEFAULT_OUTPUT_DIR = "artifacts/m1"
SUPPORTED_MODELS = ("naive", "elastic_net", "xgboost", "lstm", "cnn", "ctts")


def _load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _write_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _log_message(message: str, progress_enabled: bool) -> None:
    if progress_enabled:
        tqdm.write(message)
    else:
        print(message)


def _aggregate_explainability(
    explainability_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    if not explainability_frames:
        return pd.DataFrame(columns=["feature", "importance"])

    stacked = pd.concat(explainability_frames, ignore_index=True)
    aggregated = (
        stacked.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", key=lambda s: s.abs(), ascending=False)
    )
    return aggregated


def _copy_if_exists(source_path: Path, destination_path: Path) -> None:
    if source_path.exists():
        shutil.copy2(source_path, destination_path)


def _model_results_dir(results_root: Path, model_name: str) -> Path:
    model_dir = results_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _find_missing_dependencies(model_names: list[str]) -> dict[str, list[str]]:
    package_by_model = {
        "elastic_net": ["sklearn"],
        "xgboost": ["xgboost"],
        "lstm": ["torch"],
        "cnn": ["torch"],
        "ctts": ["torch"],
    }
    missing_by_model: dict[str, list[str]] = {}
    for model_name in model_names:
        required_packages = package_by_model.get(model_name, [])
        missing = [
            package_name
            for package_name in required_packages
            if importlib.util.find_spec(package_name) is None
        ]
        if missing:
            missing_by_model[model_name] = missing
    return missing_by_model


def _raise_for_missing_dependencies(model_names: list[str]) -> None:
    missing_by_model = _find_missing_dependencies(model_names)
    if not missing_by_model:
        return

    package_aliases = {
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "torch": "torch",
    }
    missing_packages = sorted(
        {
            package_aliases.get(package_name, package_name)
            for package_list in missing_by_model.values()
            for package_name in package_list
        }
    )
    model_descriptions = ", ".join(
        f"{model_name} ({', '.join(package_aliases.get(pkg, pkg) for pkg in packages)})"
        for model_name, packages in sorted(missing_by_model.items())
    )
    install_command = f"{sys.executable} -m pip install {' '.join(missing_packages)}"
    raise SystemExit(
        "Missing required dependencies for selected models: "
        f"{model_descriptions}. Install them with `{install_command}` "
        "or rerun with only the models whose dependencies are available.",
    )


def _unique_preserve_order(values: list[object]) -> list[object]:
    return list(dict.fromkeys(values))


def _build_parameter_grid(
    model_name: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    if model_name == "naive":
        if args.tuning_mode == "off":
            return [{"current_vol_feature": args.current_vol_feature}]
        candidate_features = _unique_preserve_order(
            [
                args.current_vol_feature,
                "feat_realized_vol_5d",
                "feat_realized_vol_20d",
                "feat_realized_vol_60d",
            ]
        )
        return [{"current_vol_feature": feature_name} for feature_name in candidate_features]

    if model_name == "elastic_net":
        if args.tuning_mode == "off":
            return [{"alpha": 0.001, "l1_ratio": 0.2}]
        if args.tuning_mode == "full":
            alphas = [0.0001, 0.0003, 0.001, 0.003, 0.01]
            l1_ratios = [0.1, 0.3, 0.5, 0.8]
        else:
            alphas = [0.0003, 0.001, 0.003]
            l1_ratios = [0.1, 0.3, 0.6]
        return [
            {"alpha": alpha, "l1_ratio": l1_ratio}
            for alpha, l1_ratio in product(alphas, l1_ratios)
        ]

    if model_name == "xgboost":
        if args.tuning_mode == "off":
            return [
                {
                    "n_estimators": 300,
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "min_child_weight": 1.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                }
            ]
        if args.tuning_mode == "full":
            n_estimators = [200, 400, 600]
            max_depth = [2, 3, 4]
            learning_rates = [0.03, 0.05]
            min_child_weight = [1.0, 5.0]
        else:
            n_estimators = [200, 400]
            max_depth = [2, 3]
            learning_rates = [0.03, 0.05]
            min_child_weight = [1.0]
        return [
            {
                "n_estimators": estimator_count,
                "max_depth": depth,
                "learning_rate": learning_rate,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": child_weight,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            }
            for estimator_count, depth, learning_rate, child_weight in product(
                n_estimators,
                max_depth,
                learning_rates,
                min_child_weight,
            )
        ]

    if model_name == "lstm":
        default_params = {
            "lookback_window": args.lookback_window,
            "hidden_size": 32,
            "dense_size": 16,
            "dropout": 0.0,
            "learning_rate": args.torch_learning_rate,
        }
        if args.tuning_mode == "off":
            return [default_params]
        if args.tuning_mode == "full":
            hidden_sizes = [16, 32, 48]
            dropouts = [0.0, 0.1]
            learning_rates = _unique_preserve_order([5e-4, args.torch_learning_rate, 2e-3])
            lookbacks = _unique_preserve_order([40, args.lookback_window, 80])
        else:
            hidden_sizes = [16, 32]
            dropouts = [0.0, 0.1]
            learning_rates = _unique_preserve_order([5e-4, args.torch_learning_rate])
            lookbacks = [args.lookback_window]
        return [
            {
                "lookback_window": lookback_window,
                "hidden_size": hidden_size,
                "dense_size": 16,
                "dropout": dropout,
                "learning_rate": learning_rate,
            }
            for lookback_window, hidden_size, dropout, learning_rate in product(
                lookbacks,
                hidden_sizes,
                dropouts,
                learning_rates,
            )
        ]

    if model_name == "cnn":
        default_params = {
            "lookback_window": args.lookback_window,
            "channels": 32,
            "kernel_size": 3,
            "dense_size": 16,
            "dropout": 0.0,
            "learning_rate": args.torch_learning_rate,
        }
        if args.tuning_mode == "off":
            return [default_params]
        if args.tuning_mode == "full":
            channels = [16, 32, 48]
            kernel_sizes = [3, 5]
            dropouts = [0.0, 0.1]
            learning_rates = _unique_preserve_order([5e-4, args.torch_learning_rate, 2e-3])
            lookbacks = _unique_preserve_order([40, args.lookback_window, 80])
        else:
            channels = [16, 32]
            kernel_sizes = [3, 5]
            dropouts = [0.0, 0.1]
            learning_rates = _unique_preserve_order([5e-4, args.torch_learning_rate])
            lookbacks = [args.lookback_window]
        return [
            {
                "lookback_window": lookback_window,
                "channels": channel_count,
                "kernel_size": kernel_size,
                "dense_size": 16,
                "dropout": dropout,
                "learning_rate": learning_rate,
            }
            for lookback_window, channel_count, kernel_size, dropout, learning_rate in product(
                lookbacks,
                channels,
                kernel_sizes,
                dropouts,
                learning_rates,
            )
        ]

    if model_name == "ctts":
        default_params = {
            "lookback_window": args.lookback_window,
            "embedding_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "ff_multiplier": 4,
            "kernel_size": 3,
            "conv_stride": 1,
            "dense_size": 32,
            "dropout": 0.1,
            "learning_rate": args.torch_learning_rate,
            "weight_decay": 1e-4,
        }
        if args.tuning_mode == "off":
            return [default_params]
        if args.tuning_mode == "full":
            embedding_dims = [32, 64]
            num_layers_list = [1, 2, 3]
            kernel_sizes = [3, 5]
            conv_strides = [1]
            dropouts = [0.0, 0.1]
            learning_rates = _unique_preserve_order([5e-4, args.torch_learning_rate, 2e-3])
            lookbacks = _unique_preserve_order([40, args.lookback_window, 80])
            weight_decays = [1e-5, 1e-4]
        else:
            embedding_dims = [32, 64]
            num_layers_list = [1, 2]
            kernel_sizes = [3, 5]
            conv_strides = [1]
            dropouts = [0.0, 0.1]
            learning_rates = _unique_preserve_order([5e-4, args.torch_learning_rate])
            lookbacks = [args.lookback_window]
            weight_decays = [1e-4]
        return [
            {
                "lookback_window": lookback_window,
                "embedding_dim": embedding_dim,
                "num_heads": 4,
                "num_layers": num_layers,
                "ff_multiplier": 4,
                "kernel_size": kernel_size,
                "conv_stride": conv_stride,
                "dense_size": 32,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }
            for lookback_window, embedding_dim, num_layers, kernel_size, conv_stride, dropout, learning_rate, weight_decay in product(
                lookbacks,
                embedding_dims,
                num_layers_list,
                kernel_sizes,
                conv_strides,
                dropouts,
                learning_rates,
                weight_decays,
            )
        ]

    raise ValueError(f"Unsupported model {model_name}.")


def _tune_model(
    model_name: str,
    df: pd.DataFrame,
    blocks: list,
    feature_columns: list[str],
    target_column: str,
    args: argparse.Namespace,
    progress_enabled: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    tuning_block = blocks[0]
    parameter_grid = _build_parameter_grid(model_name, args)

    if args.tuning_mode == "off":
        summary = pd.DataFrame([{**parameter_grid[0], "validation_score": np.nan}])
        return parameter_grid[0], summary

    _log_message(
        (
            f"tuning {model_name} on pre-test window "
            f"{tuning_block.test_start_date.date()} with {len(parameter_grid)} candidate(s)"
        ),
        progress_enabled=progress_enabled,
    )

    if args.task_type == "classification":
        if model_name == "naive":
            selected_params, summary = tune_naive_classification_model(
                df=df,
                train_index=tuning_block.train_index,
                validation_index=tuning_block.validation_index,
                source_column=args.classification_source_column,
                candidate_features=[params["current_vol_feature"] for params in parameter_grid],
                metric=args.tuning_metric,
            )
        elif model_name == "elastic_net":
            selected_params, summary = tune_elastic_net_classification_model(
                df=df,
                train_index=tuning_block.train_index,
                validation_index=tuning_block.validation_index,
                feature_columns=feature_columns,
                source_column=args.classification_source_column,
                parameter_grid=parameter_grid,
                metric=args.tuning_metric,
            )
        elif model_name == "xgboost":
            selected_params, summary = tune_xgboost_classification_model(
                df=df,
                train_index=tuning_block.train_index,
                validation_index=tuning_block.validation_index,
                feature_columns=feature_columns,
                source_column=args.classification_source_column,
                parameter_grid=parameter_grid,
                metric=args.tuning_metric,
            )
        elif model_name in {"lstm", "cnn", "ctts"}:
            selected_params, summary = tune_torch_sequence_classification_model(
                model_name=model_name,
                df=df,
                train_index=tuning_block.train_index,
                validation_index=tuning_block.validation_index,
                feature_columns=feature_columns,
                source_column=args.classification_source_column,
                parameter_grid=parameter_grid,
                metric=args.tuning_metric,
                lookback_window=args.lookback_window,
                epochs=args.tuning_torch_epochs,
                batch_size=args.torch_batch_size,
                learning_rate=args.torch_learning_rate,
                device_name=args.torch_device,
            )
        else:
            raise ValueError(f"Unsupported model {model_name}.")
    elif model_name == "naive":
        selected_params, summary = tune_naive_model(
            df=df,
            train_index=tuning_block.train_index,
            validation_index=tuning_block.validation_index,
            target_column=target_column,
            candidate_features=[params["current_vol_feature"] for params in parameter_grid],
            metric=args.tuning_metric,
        )
    elif model_name == "elastic_net":
        selected_params, summary = tune_elastic_net_model(
            df=df,
            train_index=tuning_block.train_index,
            validation_index=tuning_block.validation_index,
            feature_columns=feature_columns,
            target_column=target_column,
            parameter_grid=parameter_grid,
            metric=args.tuning_metric,
        )
    elif model_name == "xgboost":
        selected_params, summary = tune_xgboost_model(
            df=df,
            train_index=tuning_block.train_index,
            validation_index=tuning_block.validation_index,
            feature_columns=feature_columns,
            target_column=target_column,
            parameter_grid=parameter_grid,
            metric=args.tuning_metric,
        )
    elif model_name in {"lstm", "cnn", "ctts"}:
        selected_params, summary = tune_torch_sequence_model(
            model_name=model_name,
            df=df,
            train_index=tuning_block.train_index,
            validation_index=tuning_block.validation_index,
            feature_columns=feature_columns,
            target_column=target_column,
            parameter_grid=parameter_grid,
            metric=args.tuning_metric,
            lookback_window=args.lookback_window,
            epochs=args.tuning_torch_epochs,
            batch_size=args.torch_batch_size,
            learning_rate=args.torch_learning_rate,
            training_loss=args.torch_loss,
            device_name=args.torch_device,
        )
    else:
        raise ValueError(f"Unsupported model {model_name}.")

    _log_message(
        f"selected params for {model_name}: {selected_params}",
        progress_enabled=progress_enabled,
    )
    return selected_params, summary


def _append_optional_argument(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _build_worker_command(
    script_path: Path,
    args: argparse.Namespace,
    model_name: str,
    worker_output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path),
        "--worker-mode",
        "--input",
        args.input,
        "--output-dir",
        str(worker_output_dir),
        "--results-root",
        args.results_root,
        "--models",
        model_name,
        "--task-type",
        args.task_type,
        "--test-start-date",
        args.test_start_date,
        "--target-column",
        args.target_column,
        "--classification-source-column",
        args.classification_source_column,
        "--current-vol-feature",
        args.current_vol_feature,
        "--min-train-days",
        str(args.min_train_days),
        "--validation-days",
        str(args.validation_days),
        "--embargo-days",
        str(args.embargo_days),
        "--retrain-every-days",
        str(args.retrain_every_days),
        "--lookback-window",
        str(args.lookback_window),
        "--torch-epochs",
        str(args.torch_epochs),
        "--torch-batch-size",
        str(args.torch_batch_size),
        "--torch-learning-rate",
        str(args.torch_learning_rate),
        "--torch-loss",
        args.torch_loss,
        "--torch-device",
        args.torch_device,
        "--torch-log-epochs" if args.torch_log_epochs else "",
        "--rebalance-every-days",
        str(args.rebalance_every_days),
        "--transaction-cost-bps",
        str(args.transaction_cost_bps),
        "--tuning-mode",
        args.tuning_mode,
        "--tuning-metric",
        args.tuning_metric,
        "--tuning-torch-epochs",
        str(args.tuning_torch_epochs),
    ]
    command = [part for part in command if part]
    if args.no_progress:
        command.append("--no-progress")
    return command


def _aggregate_saved_outputs(
    output_dir: Path,
    model_names: list[str],
    task_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, object]] = []
    backtest_rows: list[dict[str, object]] = []

    for model_name in model_names:
        predictions_path = output_dir / f"predictions_{model_name}.csv"
        backtest_path = output_dir / f"backtest_{model_name}.csv"
        if not predictions_path.exists() or not backtest_path.exists():
            raise FileNotFoundError(
                f"Missing expected artifacts for {model_name} in {output_dir}.",
            )

        prediction_df = pd.read_csv(predictions_path)
        backtest_df = pd.read_csv(backtest_path)

        if task_type == "classification":
            metric_summary = summarize_classification_predictions(prediction_df)
        else:
            metric_summary = summarize_predictions(prediction_df)
        metric_summary["model"] = model_name
        metrics_rows.append(metric_summary)

        backtest_summary = summarize_backtest(backtest_df)
        backtest_summary["model"] = model_name
        backtest_rows.append(backtest_summary)

    metrics_df = pd.DataFrame(metrics_rows)
    if task_type == "classification":
        metrics_df = metrics_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    else:
        metrics_df = metrics_df.sort_values("mae").reset_index(drop=True)
    backtest_df = pd.DataFrame(backtest_rows).sort_values(
        "strategy_sharpe_ratio",
        ascending=False,
    ).reset_index(drop=True)
    return metrics_df, backtest_df


def _run_models_isolated(
    script_path: Path,
    args: argparse.Namespace,
    progress_enabled: bool,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_root = output_dir / "_model_runs"
    worker_root.mkdir(parents=True, exist_ok=True)

    _log_message(
        f"running models in isolated worker processes: {', '.join(args.models)}",
        progress_enabled=progress_enabled,
    )

    for model_position, model_name in enumerate(args.models, start=1):
        _log_message(
            f"launching worker {model_position}/{len(args.models)} for {model_name}",
            progress_enabled=progress_enabled,
        )
        worker_output_dir = worker_root / model_name
        worker_output_dir.mkdir(parents=True, exist_ok=True)
        command = _build_worker_command(
            script_path=script_path,
            args=args,
            model_name=model_name,
            worker_output_dir=worker_output_dir,
        )
        subprocess.run(command, check=True, cwd=str(ROOT))

        for artifact_name in [
            f"predictions_{model_name}.csv",
            f"backtest_{model_name}.csv",
            f"explainability_{model_name}.csv",
            f"training_history_{model_name}.csv",
            f"tuning_summary_{model_name}.csv",
            f"selected_params_{model_name}.json",
        ]:
            _copy_if_exists(worker_output_dir / artifact_name, output_dir / artifact_name)

        _copy_if_exists(
            worker_output_dir / "run_manifest.json",
            output_dir / f"run_manifest_{model_name}.json",
        )

    metrics_df, backtest_df = _aggregate_saved_outputs(output_dir, args.models, args.task_type)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    backtest_df.to_csv(output_dir / "backtest_summary.csv", index=False)

    manifest = {
        "input": args.input,
        "output_dir": args.output_dir,
        "models": args.models,
        "target_column": args.target_column,
        "test_start_date": args.test_start_date,
        "tuning_mode": args.tuning_mode,
        "tuning_metric": args.tuning_metric,
        "torch_loss": args.torch_loss,
        "torch_device": args.torch_device,
        "execution_mode": "isolated_workers",
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    _log_message(f"saved outputs -> {output_dir}", progress_enabled=progress_enabled)
    _log_message("metrics summary:", progress_enabled=progress_enabled)
    _log_message(metrics_df.to_string(index=False), progress_enabled=progress_enabled)
    _log_message("backtest summary:", progress_enabled=progress_enabled)
    _log_message(backtest_df.to_string(index=False), progress_enabled=progress_enabled)


def _run_single_model(
    model_name: str,
    df: pd.DataFrame,
    blocks: list,
    feature_columns: list[str],
    target_column: str,
    selected_params: dict[str, object],
    args: argparse.Namespace,
    progress_enabled: bool,
    model_position: int,
    model_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    explainability_frames: list[pd.DataFrame] = []
    training_history_output = pd.DataFrame()

    block_iterator = tqdm(
        blocks,
        total=len(blocks),
        desc=f"[{model_position}/{model_count}] {model_name}",
        leave=False,
        dynamic_ncols=True,
        disable=not progress_enabled,
    )

    for block in block_iterator:
        combined_train_index = np.concatenate([block.train_index, block.validation_index])

        if args.task_type == "classification" and model_name == "naive":
            block_prediction = run_naive_classification_block(
                df=df,
                train_index=combined_train_index,
                test_index=block.test_index,
                source_column=args.classification_source_column,
                current_vol_feature=str(selected_params["current_vol_feature"]),
            )
        elif args.task_type == "classification" and model_name == "elastic_net":
            block_prediction = run_elastic_net_classification_block(
                df=df,
                train_index=combined_train_index,
                test_index=block.test_index,
                feature_columns=feature_columns,
                source_column=args.classification_source_column,
                model_params=selected_params,
            )
        elif args.task_type == "classification" and model_name == "xgboost":
            block_prediction = run_xgboost_classification_block(
                df=df,
                train_index=combined_train_index,
                test_index=block.test_index,
                feature_columns=feature_columns,
                source_column=args.classification_source_column,
                model_params=selected_params,
            )
        elif args.task_type == "classification" and model_name in {"lstm", "cnn", "ctts"}:
            block_prediction = run_torch_sequence_classification_block(
                model_name=model_name,
                df=df,
                train_index=combined_train_index,
                test_index=block.test_index,
                feature_columns=feature_columns,
                source_column=args.classification_source_column,
                lookback_window=args.lookback_window,
                epochs=args.torch_epochs,
                batch_size=args.torch_batch_size,
                learning_rate=args.torch_learning_rate,
                metric=args.tuning_metric,
                device_name=args.torch_device,
                log_epoch_losses=args.torch_log_epochs and block.block_id == 0,
                model_params=selected_params,
            )
        elif model_name == "naive":
            block_prediction = run_naive_block(
                df=df,
                test_index=block.test_index,
                target_column=target_column,
                current_vol_feature=str(selected_params["current_vol_feature"]),
                train_target=df.loc[combined_train_index, target_column],
            )
        elif model_name == "elastic_net":
            block_prediction = run_elastic_net_block(
                df=df,
                train_index=combined_train_index,
                test_index=block.test_index,
                feature_columns=feature_columns,
                target_column=target_column,
                model_params=selected_params,
            )
        elif model_name == "xgboost":
            block_prediction = run_xgboost_block(
                df=df,
                train_index=combined_train_index,
                test_index=block.test_index,
                feature_columns=feature_columns,
                target_column=target_column,
                model_params=selected_params,
            )
        elif model_name in {"lstm", "cnn", "ctts"}:
            block_prediction = run_torch_sequence_block(
                model_name=model_name,
                df=df,
                train_index=combined_train_index,
                test_index=block.test_index,
                feature_columns=feature_columns,
                target_column=target_column,
                lookback_window=args.lookback_window,
                epochs=args.torch_epochs,
                batch_size=args.torch_batch_size,
                learning_rate=args.torch_learning_rate,
                training_loss=args.torch_loss,
                device_name=args.torch_device,
                log_epoch_losses=args.torch_log_epochs and block.block_id == 0,
                model_params=selected_params,
            )
        else:
            raise ValueError(f"Unsupported model {model_name}.")

        if args.task_type == "classification":
            block_df = pd.DataFrame(
                {
                    "date": block_prediction.dates.to_numpy(),
                    "model": model_name,
                    "predicted_class": block_prediction.predicted_class,
                    "actual_class": block_prediction.actual_class,
                    "train_low_vol_threshold": block_prediction.low_threshold,
                    "train_high_vol_threshold": block_prediction.high_threshold,
                }
            )
            if block_prediction.class_probabilities is not None:
                for class_id in range(block_prediction.class_probabilities.shape[1]):
                    block_df[f"prob_class_{class_id}"] = block_prediction.class_probabilities[:, class_id]
        else:
            block_df = pd.DataFrame(
                {
                    "date": block_prediction.dates.to_numpy(),
                    "model": model_name,
                    "predicted_log_vol": block_prediction.predicted_log_vol,
                    "actual_log_vol": block_prediction.actual_log_vol,
                    "predicted_vol": np.exp(block_prediction.predicted_log_vol),
                    "actual_vol": np.exp(block_prediction.actual_log_vol),
                    "train_low_vol_threshold": block_prediction.low_threshold,
                    "train_high_vol_threshold": block_prediction.high_threshold,
                }
            )

        block_with_market = block_df.merge(
            df.loc[:, ["date", "asset_return_1d"]],
            on="date",
            how="left",
        )
        prediction_frames.append(block_with_market)

        if block_prediction.explainability is not None:
            explainability = block_prediction.explainability.copy()
            explainability["block_id"] = block.block_id
            explainability_frames.append(explainability)
        if training_history_output.empty and block_prediction.training_history is not None:
            training_history_output = block_prediction.training_history.copy()
            training_history_output["block_id"] = block.block_id

        if progress_enabled:
            block_iterator.set_postfix_str(
                f"{block.test_start_date.date()}->{block.test_end_date.date()}",
            )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    explainability_output = _aggregate_explainability(explainability_frames)
    return predictions, explainability_output, training_history_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Milestone 1 walk-forward volatility forecasting experiment.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Engineered M1 dataset built by scripts/build_m1_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where predictions, metrics, and backtest summaries will be written.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory where per-model JSON results will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=SUPPORTED_MODELS,
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--task-type",
        choices=("regression", "classification"),
        default="regression",
        help="Whether to run the volatility forecast as regression or classification.",
    )
    parser.add_argument(
        "--test-start-date",
        default="2022-01-03",
        help="First date of the final walk-forward test period.",
    )
    parser.add_argument(
        "--target-column",
        default="target_log_future_vol_20d",
        help="Target column to forecast.",
    )
    parser.add_argument(
        "--classification-source-column",
        default="target_future_vol_20d",
        help="Continuous future-volatility column used to derive class labels in classification mode.",
    )
    parser.add_argument(
        "--current-vol-feature",
        default="feat_realized_vol_20d",
        help="Current realized volatility feature used by the naive baseline.",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=756,
        help="Minimum number of training rows required before the first block.",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=252,
        help="Validation rows reserved immediately before each embargo period.",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=20,
        help="Embargo between the training/validation data and the next test block.",
    )
    parser.add_argument(
        "--retrain-every-days",
        type=int,
        default=21,
        help="Number of trading days in each walk-forward prediction block.",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=60,
        help="Sequence length for the LSTM and CNN models.",
    )
    parser.add_argument(
        "--torch-epochs",
        type=int,
        default=30,
        help="Maximum training epochs for the sequence models.",
    )
    parser.add_argument(
        "--torch-batch-size",
        type=int,
        default=64,
        help="Mini-batch size for the sequence models.",
    )
    parser.add_argument(
        "--torch-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the sequence models.",
    )
    parser.add_argument(
        "--torch-loss",
        choices=("mse", "huber", "qlike"),
        default="qlike",
        help="Training loss used by the sequence models.",
    )
    parser.add_argument(
        "--torch-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Device used by the torch sequence models. Sklearn and XGBoost models remain on CPU.",
    )
    parser.add_argument(
        "--torch-log-epochs",
        action="store_true",
        help="Print epoch-level training and validation losses for the first sequence-model block.",
    )
    parser.add_argument(
        "--rebalance-every-days",
        type=int,
        default=5,
        help="Trading-day rebalance interval for the allocation layer.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="Round-trip transaction cost in basis points applied to turnover.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars and use only summary logging.",
    )
    parser.add_argument(
        "--tuning-mode",
        choices=("off", "default", "full"),
        default="default",
        help="Hyperparameter tuning scope. Default tunes on the pre-test split once and freezes parameters for the walk-forward test.",
    )
    parser.add_argument(
        "--tuning-metric",
        choices=("qlike", "mae", "rmse", "accuracy", "balanced_accuracy", "macro_f1"),
        default="qlike",
        help="Validation metric used to select tuned model parameters.",
    )
    parser.add_argument(
        "--tuning-torch-epochs",
        type=int,
        default=10,
        help="Training epochs used during the one-time LSTM/CNN tuning stage.",
    )
    parser.add_argument(
        "--worker-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    progress_enabled = not args.no_progress
    _raise_for_missing_dependencies(args.models)
    if args.task_type == "regression" and args.tuning_metric in {"accuracy", "balanced_accuracy", "macro_f1"}:
        raise SystemExit("Classification metrics are only valid with `--task-type classification`.")
    if args.task_type == "classification" and args.tuning_metric in {"qlike", "mae", "rmse"}:
        raise SystemExit("Regression metrics are only valid with `--task-type regression`.")

    if not args.worker_mode and len(args.models) > 1:
        _run_models_isolated(
            script_path=Path(__file__).resolve(),
            args=args,
            progress_enabled=progress_enabled,
        )
        return

    df = _load_dataset(args.input)
    feature_columns = feature_columns_from_dataset(df)
    blocks = generate_walk_forward_blocks(
        df=df,
        test_start_date=args.test_start_date,
        min_train_days=args.min_train_days,
        validation_days=args.validation_days,
        embargo_days=args.embargo_days,
        retrain_every_days=args.retrain_every_days,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    _log_message(
        (
            f"loaded dataset rows={len(df)} features={len(feature_columns)} "
            f"date range={df['date'].min().date()}->{df['date'].max().date()}"
        ),
        progress_enabled=progress_enabled,
    )
    _log_message(
        (
            f"running {len(args.models)} model(s) across {len(blocks)} walk-forward block(s); "
            f"test starts {args.test_start_date}; tuning={args.tuning_mode}/{args.tuning_metric}"
        ),
        progress_enabled=progress_enabled,
    )

    metrics_rows: list[dict[str, object]] = []
    backtest_rows: list[dict[str, object]] = []

    for model_position, model_name in enumerate(args.models, start=1):
        _log_message(
            f"starting model {model_position}/{len(args.models)}: {model_name}",
            progress_enabled=progress_enabled,
        )
        selected_params, tuning_summary = _tune_model(
            model_name=model_name,
            df=df,
            blocks=blocks,
            feature_columns=feature_columns,
            target_column=args.target_column,
            args=args,
            progress_enabled=progress_enabled,
        )
        predictions, explainability, training_history = _run_single_model(
            model_name=model_name,
            df=df,
            blocks=blocks,
            feature_columns=feature_columns,
            target_column=args.target_column,
            selected_params=selected_params,
            args=args,
            progress_enabled=progress_enabled,
            model_position=model_position,
            model_count=len(args.models),
        )
        predictions_path = output_dir / f"predictions_{model_name}.csv"
        predictions.to_csv(predictions_path, index=False)
        tuning_summary_path = output_dir / f"tuning_summary_{model_name}.csv"
        tuning_summary.to_csv(tuning_summary_path, index=False)
        _write_json(output_dir / f"selected_params_{model_name}.json", selected_params)
        training_history_path = None
        if not training_history.empty:
            training_history_path = output_dir / f"training_history_{model_name}.csv"
            training_history.to_csv(training_history_path, index=False)

        if args.task_type == "classification":
            metric_summary = summarize_classification_predictions(predictions)
        else:
            metric_summary = summarize_predictions(predictions)
        metric_summary["model"] = model_name
        metrics_rows.append(metric_summary)

        if args.task_type == "classification":
            strategy_df = run_classification_backtest(
                prediction_df=predictions,
                rebalance_every_days=args.rebalance_every_days,
                transaction_cost_bps=args.transaction_cost_bps,
            )
        else:
            strategy_df = run_backtest(
                prediction_df=predictions,
                rebalance_every_days=args.rebalance_every_days,
                transaction_cost_bps=args.transaction_cost_bps,
            )
        strategy_path = output_dir / f"backtest_{model_name}.csv"
        strategy_df.to_csv(strategy_path, index=False)

        backtest_summary = summarize_backtest(strategy_df)
        backtest_summary["model"] = model_name
        backtest_rows.append(backtest_summary)

        if not explainability.empty:
            explainability_path = output_dir / f"explainability_{model_name}.csv"
            explainability.to_csv(explainability_path, index=False)

        model_results_dir = _model_results_dir(results_root, model_name)
        result_payload = {
            "model": model_name,
            "input": args.input,
            "target_column": args.target_column,
            "task_type": args.task_type,
            "classification_source_column": args.classification_source_column,
            "test_start_date": args.test_start_date,
            "selected_params": selected_params,
            "torch_loss": args.torch_loss if model_name in {"lstm", "cnn", "ctts"} else None,
            "torch_device": args.torch_device if model_name in {"lstm", "cnn", "ctts"} else None,
            "metrics": metric_summary,
            "backtest": backtest_summary,
            "artifacts": {
                "predictions_csv": str(predictions_path),
                "backtest_csv": str(strategy_path),
                "tuning_summary_csv": str(tuning_summary_path),
                "selected_params_json": str(output_dir / f"selected_params_{model_name}.json"),
                "training_history_csv": str(training_history_path) if training_history_path is not None else None,
                "explainability_csv": str(output_dir / f"explainability_{model_name}.csv") if not explainability.empty else None,
            },
        }
        _write_json(model_results_dir / "results.json", result_payload)

        _log_message(
            (
                f"finished {model_name}: predictions={predictions_path.name} "
                + (
                    f"macro_f1={metric_summary['macro_f1']:.6f} "
                    f"accuracy={metric_summary['accuracy']:.6f} "
                    if args.task_type == "classification"
                    else f"mae={metric_summary['mae']:.6f} qlike={metric_summary['qlike']:.6f} "
                )
                + f"selected={selected_params}"
            ),
            progress_enabled=progress_enabled,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    if args.task_type == "classification":
        metrics_df = metrics_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    else:
        metrics_df = metrics_df.sort_values("mae").reset_index(drop=True)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    backtest_df = pd.DataFrame(backtest_rows).sort_values(
        "strategy_sharpe_ratio",
        ascending=False,
    ).reset_index(drop=True)
    backtest_df.to_csv(output_dir / "backtest_summary.csv", index=False)

    manifest = {
        "input": args.input,
        "output_dir": args.output_dir,
        "results_root": args.results_root,
        "models": args.models,
        "task_type": args.task_type,
        "target_column": args.target_column,
        "classification_source_column": args.classification_source_column,
        "test_start_date": args.test_start_date,
        "n_blocks": len(blocks),
        "feature_count": len(feature_columns),
        "tuning_mode": args.tuning_mode,
        "tuning_metric": args.tuning_metric,
        "torch_loss": args.torch_loss,
        "torch_device": args.torch_device,
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    _log_message(f"saved outputs -> {output_dir}", progress_enabled=progress_enabled)
    _log_message("metrics summary:", progress_enabled=progress_enabled)
    _log_message(metrics_df.to_string(index=False), progress_enabled=progress_enabled)
    _log_message("backtest summary:", progress_enabled=progress_enabled)
    _log_message(backtest_df.to_string(index=False), progress_enabled=progress_enabled)


if __name__ == "__main__":
    main()
