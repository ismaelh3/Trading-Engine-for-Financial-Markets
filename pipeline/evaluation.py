"""Predictive metrics for Milestone 1 volatility experiments."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0.0:
        return math.nan
    numerator = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - numerator / denominator


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    clipped_true = np.clip(y_true, 1e-8, None)
    clipped_pred = np.clip(y_pred, 1e-8, None)
    variance_true = clipped_true**2
    variance_pred = clipped_pred**2
    return float(np.mean(np.log(variance_pred) + variance_true / variance_pred))


def summarize_predictions(prediction_df: pd.DataFrame) -> dict[str, float]:
    y_true = prediction_df["actual_vol"].to_numpy(dtype=float)
    y_pred = prediction_df["predicted_vol"].to_numpy(dtype=float)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "qlike": qlike_loss(y_true, y_pred),
        "prediction_correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "n_predictions": float(len(prediction_df)),
    }


def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) == 0:
        return math.nan
    recalls = []
    for class_id in classes:
        mask = y_true == class_id
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == class_id)))
    if not recalls:
        return math.nan
    return float(np.mean(recalls))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 0:
        return math.nan
    f1_scores: list[float] = []
    for class_id in classes:
        true_positive = float(np.sum((y_true == class_id) & (y_pred == class_id)))
        false_positive = float(np.sum((y_true != class_id) & (y_pred == class_id)))
        false_negative = float(np.sum((y_true == class_id) & (y_pred != class_id)))
        precision_denom = true_positive + false_positive
        recall_denom = true_positive + false_negative
        precision = true_positive / precision_denom if precision_denom > 0 else 0.0
        recall = true_positive / recall_denom if recall_denom > 0 else 0.0
        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def summarize_classification_predictions(prediction_df: pd.DataFrame) -> dict[str, float]:
    y_true = prediction_df["actual_class"].to_numpy(dtype=int)
    y_pred = prediction_df["predicted_class"].to_numpy(dtype=int)
    return {
        "accuracy": classification_accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1_score(y_true, y_pred),
        "n_predictions": float(len(prediction_df)),
    }
