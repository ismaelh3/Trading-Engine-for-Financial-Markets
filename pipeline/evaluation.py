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

