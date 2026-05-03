"""Unified tuning helpers for grid and Optuna-backed model selection.

1. Optuna is better to tune Deep Models since it uses Bayesian Opt instead of Grid Search like GridSearchCV.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .models import (
    _classification_score,
    _fit_torch_sequence_classifier_predict,
    _fit_torch_sequence_predict,
    _split_inner_validation_indices,
    _validation_score,
)


SEQUENCE_MODELS = {"lstm", "cnn", "cnn_lstm", "ctts"}
GRID_MODELS = {"naive", "elastic_net", "xgboost"}


@dataclass
class TuningResult:
    selected_params: dict[str, object]
    summary: pd.DataFrame
    backend: str
    metadata: dict[str, object]


def resolve_tuning_backend(model_name: str, requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    if model_name in SEQUENCE_MODELS:
        return "optuna"
    return "grid"


def _require_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: optuna. Install it with `pip install -r requirements.txt` "
            "or rerun with `--tuning-backend grid`.",
        ) from exc
    return optuna


def _build_pruner(optuna: Any, pruner_name: str) -> Any:
    if pruner_name == "none":
        return optuna.pruners.NopPruner()
    if pruner_name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=3)
    if pruner_name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner()
    raise ValueError(f"Unsupported Optuna pruner {pruner_name}.")


def _suggest_sequence_params(model_name: str, trial: Any) -> dict[str, object]:
    lookback_window = trial.suggest_categorical("lookback_window", [40, 60, 80])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)

    if model_name == "lstm":
        return {
            "lookback_window": lookback_window,
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "dense_size": trial.suggest_categorical("dense_size", [16, 32]),
            "dropout": dropout,
            "learning_rate": learning_rate,
        }

    if model_name == "cnn":
        return {
            "lookback_window": lookback_window,
            "channels": trial.suggest_categorical("channels", [16, 32, 64]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 10]),
            "dense_size": trial.suggest_categorical("dense_size", [16, 32]),
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    if model_name == "cnn_lstm":
        return {
            "lookback_window": lookback_window,
            "channels": trial.suggest_categorical("channels", [16, 32, 64]),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 10]),
            "dense_size": trial.suggest_categorical("dense_size", [16, 32]),
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    if model_name == "ctts":
        embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
        valid_heads = [head_count for head_count in [2, 4, 8] if embedding_dim % head_count == 0]
        return {
            "lookback_window": lookback_window,
            "embedding_dim": embedding_dim,
            "num_heads": trial.suggest_categorical("num_heads", valid_heads),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "ff_multiplier": 4,
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 10]),
            "conv_stride": trial.suggest_categorical("conv_stride", [1, 2, 5]),
            "dense_size": trial.suggest_categorical("dense_size", [32, 64]),
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    raise ValueError(f"Unsupported sequence model {model_name}.")


def tune_sequence_model_with_optuna(
    *,
    model_name: str,
    task_type: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    source_column: str,
    metric: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    training_loss: str,
    device_name: str,
    n_trials: int,
    timeout_seconds: int | None,
    pruner_name: str,
    random_seed: int,
) -> TuningResult:
    optuna = _require_optuna()
    direction = "maximize" if task_type == "classification" else "minimize"
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=_build_pruner(optuna, pruner_name),
    )

    def objective(trial: Any) -> float:
        params = _suggest_sequence_params(model_name, trial)
        trial.set_user_attr("full_params", params)
        candidate_lookback = int(params.get("lookback_window", lookback_window))
        try:
            inner_train_index, inner_validation_index = _split_inner_validation_indices(
                train_index=train_index,
                lookback_window=candidate_lookback,
            )
            if task_type == "classification":
                predicted_class, actual_class, _, _, _, _, _ = _fit_torch_sequence_classifier_predict(
                    model_name=model_name,
                    df=df,
                    train_index=inner_train_index,
                    validation_index=inner_validation_index,
                    evaluation_index=validation_index,
                    feature_columns=feature_columns,
                    source_column=source_column,
                    lookback_window=lookback_window,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    metric=metric,
                    device_name=device_name,
                    log_epoch_losses=False,
                    model_params=params,
                    optuna_trial=trial if pruner_name != "none" else None,
                )
                return float(_classification_score(actual_class, predicted_class, metric))

            predicted_log_vol, actual_log_vol, _, _ = _fit_torch_sequence_predict(
                model_name=model_name,
                df=df,
                train_index=inner_train_index,
                validation_index=inner_validation_index,
                evaluation_index=validation_index,
                feature_columns=feature_columns,
                target_column=target_column,
                lookback_window=lookback_window,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                training_loss=training_loss,
                device_name=device_name,
                log_epoch_losses=False,
                model_params=params,
                optuna_trial=trial if pruner_name != "none" else None,
            )
            return float(_validation_score(actual_log_vol, predicted_log_vol, metric))
        except RuntimeError as exc:
            if str(exc) == "OPTUNA_TRIAL_PRUNED":
                raise optuna.TrialPruned() from exc
            raise
        except ValueError as exc:
            raise optuna.TrialPruned() from exc

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    rows = []
    for trial in study.trials:
        rows.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "validation_score": trial.value,
                **trial.user_attrs.get("full_params", trial.params),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty and "validation_score" in summary.columns:
        summary = summary.sort_values(
            "validation_score",
            ascending=direction == "minimize",
            na_position="last",
        ).reset_index(drop=True)

    return TuningResult(
        selected_params=dict(study.best_trial.user_attrs.get("full_params", study.best_trial.params)),
        summary=summary,
        backend="optuna",
        metadata={
            "optuna_trials": len(study.trials),
            "optuna_best_value": study.best_value,
            "optuna_direction": direction,
            "optuna_pruner": pruner_name,
        },
    )
