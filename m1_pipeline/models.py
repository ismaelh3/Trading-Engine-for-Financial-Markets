"""Model adapters and tuning helpers for Milestone 1 volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import mean_absolute_error, qlike_loss, root_mean_squared_error


SEQUENCE_MODELS = {"lstm", "cnn"}


def _require_sklearn() -> tuple:
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: scikit-learn. Install it with `pip install -r requirements.txt`.",
        ) from exc
    return ElasticNet, Pipeline, StandardScaler


def _require_xgboost() -> type:
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: xgboost. Install it with `pip install -r requirements.txt`.",
        ) from exc
    return XGBRegressor


def _require_torch() -> tuple:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: torch. Install it with `pip install -r requirements.txt`.",
        ) from exc
    return torch, nn, DataLoader, TensorDataset


@dataclass
class BlockPrediction:
    dates: pd.Series
    predicted_log_vol: np.ndarray
    actual_log_vol: np.ndarray
    low_threshold: float
    high_threshold: float
    explainability: pd.DataFrame | None = None


def _train_target_quantiles(y_train_log: pd.Series) -> tuple[float, float]:
    y_train_vol = np.exp(y_train_log.to_numpy(dtype=float))
    return float(np.quantile(y_train_vol, 0.50)), float(np.quantile(y_train_vol, 0.80))


def _validation_score(
    actual_log_vol: np.ndarray,
    predicted_log_vol: np.ndarray,
    metric: str,
) -> float:
    actual_vol = np.exp(actual_log_vol)
    predicted_vol = np.exp(predicted_log_vol)

    if metric == "qlike":
        return qlike_loss(actual_vol, predicted_vol)
    if metric == "mae":
        return mean_absolute_error(actual_vol, predicted_vol)
    if metric == "rmse":
        return root_mean_squared_error(actual_vol, predicted_vol)
    raise ValueError(f"Unsupported tuning metric {metric}.")


def run_naive_block(
    df: pd.DataFrame,
    test_index: np.ndarray,
    target_column: str,
    current_vol_feature: str,
    train_target: pd.Series,
) -> BlockPrediction:
    epsilon = 1e-8
    predicted_log_vol = np.log(df.loc[test_index, current_vol_feature].to_numpy(dtype=float) + epsilon)
    actual_log_vol = df.loc[test_index, target_column].to_numpy(dtype=float)
    low_threshold, high_threshold = _train_target_quantiles(train_target)
    return BlockPrediction(
        dates=df.loc[test_index, "date"],
        predicted_log_vol=predicted_log_vol,
        actual_log_vol=actual_log_vol,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def tune_naive_model(
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    target_column: str,
    candidate_features: list[str],
    metric: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    unique_features = [feature for feature in dict.fromkeys(candidate_features) if feature in df.columns]
    if not unique_features:
        raise ValueError("No valid naive candidate features were provided for tuning.")

    epsilon = 1e-8
    actual_log_vol = df.loc[validation_index, target_column].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    best_feature = unique_features[0]
    best_score = float("inf")

    for feature_name in unique_features:
        predicted_log_vol = np.log(
            df.loc[validation_index, feature_name].to_numpy(dtype=float) + epsilon,
        )
        validation_score = _validation_score(actual_log_vol, predicted_log_vol, metric)
        rows.append(
            {
                "current_vol_feature": feature_name,
                "validation_score": validation_score,
            }
        )
        if validation_score < best_score:
            best_score = validation_score
            best_feature = feature_name

    summary = pd.DataFrame(rows).sort_values("validation_score").reset_index(drop=True)
    return {"current_vol_feature": best_feature}, summary


def _fit_elastic_net_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    model_params: dict[str, object] | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    ElasticNet, Pipeline, StandardScaler = _require_sklearn()
    params = model_params or {}

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                ElasticNet(
                    alpha=float(params.get("alpha", 0.001)),
                    l1_ratio=float(params.get("l1_ratio", 0.2)),
                    max_iter=20_000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    regressor = model.named_steps["regressor"]
    explainability = pd.DataFrame(
        {
            "feature": list(X_train.columns),
            "importance": regressor.coef_,
        }
    )
    return np.asarray(model.predict(X_eval), dtype=float), explainability


def run_elastic_net_block(
    df: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    model_params: dict[str, object] | None = None,
) -> BlockPrediction:
    X_train = df.loc[train_index, feature_columns]
    y_train = df.loc[train_index, target_column]
    X_test = df.loc[test_index, feature_columns]

    predicted_log_vol, explainability = _fit_elastic_net_predict(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        model_params=model_params,
    )
    actual_log_vol = df.loc[test_index, target_column].to_numpy(dtype=float)
    low_threshold, high_threshold = _train_target_quantiles(y_train)

    return BlockPrediction(
        dates=df.loc[test_index, "date"],
        predicted_log_vol=predicted_log_vol,
        actual_log_vol=actual_log_vol,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        explainability=explainability,
    )


def tune_elastic_net_model(
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    parameter_grid: list[dict[str, object]],
    metric: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    X_train = df.loc[train_index, feature_columns]
    y_train = df.loc[train_index, target_column]
    X_validation = df.loc[validation_index, feature_columns]
    actual_log_vol = df.loc[validation_index, target_column].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    best_params = parameter_grid[0].copy()
    best_score = float("inf")

    for params in parameter_grid:
        predicted_log_vol, _ = _fit_elastic_net_predict(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_validation,
            model_params=params,
        )
        validation_score = _validation_score(actual_log_vol, predicted_log_vol, metric)
        rows.append(
            {
                **params,
                "validation_score": validation_score,
            }
        )
        if validation_score < best_score:
            best_score = validation_score
            best_params = params.copy()

    summary = pd.DataFrame(rows).sort_values("validation_score").reset_index(drop=True)
    return best_params, summary


def _fit_xgboost_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    model_params: dict[str, object] | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    XGBRegressor = _require_xgboost()
    params = model_params or {}

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 3)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        min_child_weight=float(params.get("min_child_weight", 1.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    explainability = pd.DataFrame(
        {
            "feature": list(X_train.columns),
            "importance": model.feature_importances_,
        }
    )
    return np.asarray(model.predict(X_eval), dtype=float), explainability


def run_xgboost_block(
    df: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    model_params: dict[str, object] | None = None,
) -> BlockPrediction:
    X_train = df.loc[train_index, feature_columns]
    y_train = df.loc[train_index, target_column]
    X_test = df.loc[test_index, feature_columns]

    predicted_log_vol, explainability = _fit_xgboost_predict(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        model_params=model_params,
    )
    actual_log_vol = df.loc[test_index, target_column].to_numpy(dtype=float)
    low_threshold, high_threshold = _train_target_quantiles(y_train)

    return BlockPrediction(
        dates=df.loc[test_index, "date"],
        predicted_log_vol=predicted_log_vol,
        actual_log_vol=actual_log_vol,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        explainability=explainability,
    )


def tune_xgboost_model(
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    parameter_grid: list[dict[str, object]],
    metric: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    X_train = df.loc[train_index, feature_columns]
    y_train = df.loc[train_index, target_column]
    X_validation = df.loc[validation_index, feature_columns]
    actual_log_vol = df.loc[validation_index, target_column].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    best_params = parameter_grid[0].copy()
    best_score = float("inf")

    for params in parameter_grid:
        predicted_log_vol, _ = _fit_xgboost_predict(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_validation,
            model_params=params,
        )
        validation_score = _validation_score(actual_log_vol, predicted_log_vol, metric)
        rows.append(
            {
                **params,
                "validation_score": validation_score,
            }
        )
        if validation_score < best_score:
            best_score = validation_score
            best_params = params.copy()

    summary = pd.DataFrame(rows).sort_values("validation_score").reset_index(drop=True)
    return best_params, summary


def _standardize_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_index: np.ndarray,
) -> np.ndarray:
    matrix = df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    train_matrix = matrix[train_index]
    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std


def _standardize_target(
    df: pd.DataFrame,
    target_column: str,
    train_index: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    target = df[target_column].to_numpy(dtype=np.float32)
    train_target = target[train_index]
    mean = float(train_target.mean())
    std = float(train_target.std())
    if std == 0.0:
        std = 1.0
    scaled = (target - mean) / std
    return scaled, mean, std


def _build_sequence_dataset(
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
    indices: np.ndarray,
    lookback_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible_indices = indices[indices >= lookback_window - 1]
    sequences = []
    targets = []
    for index in eligible_indices:
        start = index - lookback_window + 1
        sequences.append(feature_matrix[start : index + 1])
        targets.append(target_vector[index])

    if not sequences:
        raise ValueError("No eligible sequence samples were found. Reduce the lookback window.")

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        eligible_indices,
    )


def _set_torch_seed(torch_module: object, seed: int = 42) -> None:
    torch_module.manual_seed(seed)
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _split_inner_validation_indices(
    train_index: np.ndarray,
    lookback_window: int,
    validation_fraction: float = 0.2,
    min_validation_rows: int = 126,
) -> tuple[np.ndarray, np.ndarray]:
    validation_size = max(min_validation_rows, int(len(train_index) * validation_fraction))
    validation_size = min(validation_size, len(train_index) - lookback_window - 1)
    if validation_size <= 0:
        raise ValueError("Training sample is too short to build an inner validation split.")

    inner_train = train_index[:-validation_size]
    inner_validation = train_index[-validation_size:]
    if len(inner_train) < lookback_window:
        raise ValueError("Inner training split is too short for the selected lookback window.")
    return inner_train, inner_validation


def _fit_torch_sequence_predict(
    model_name: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    evaluation_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_params: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    _set_torch_seed(torch)
    params = model_params or {}

    actual_lookback_window = int(params.get("lookback_window", lookback_window))
    actual_batch_size = int(params.get("batch_size", batch_size))
    actual_learning_rate = float(params.get("learning_rate", learning_rate))
    hidden_size = int(params.get("hidden_size", 32))
    dense_size = int(params.get("dense_size", 16))
    dropout = float(params.get("dropout", 0.0))
    channels = int(params.get("channels", 32))
    kernel_size = int(params.get("kernel_size", 3))

    feature_matrix = _standardize_features(df, feature_columns, train_index)
    target_vector, target_mean, target_std = _standardize_target(df, target_column, train_index)

    X_train, y_train, _ = _build_sequence_dataset(
        feature_matrix,
        target_vector,
        train_index,
        actual_lookback_window,
    )
    X_validation, y_validation, _ = _build_sequence_dataset(
        feature_matrix,
        target_vector,
        validation_index,
        actual_lookback_window,
    )
    X_evaluation, _, eligible_evaluation_index = _build_sequence_dataset(
        feature_matrix,
        target_vector,
        evaluation_index,
        actual_lookback_window,
    )
    actual_log_vol = df.loc[eligible_evaluation_index, target_column].to_numpy(dtype=float)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=actual_batch_size,
        shuffle=True,
    )
    validation_features = torch.from_numpy(X_validation)
    validation_targets = torch.from_numpy(y_validation)
    evaluation_features = torch.from_numpy(X_evaluation)

    input_size = len(feature_columns)

    class LSTMRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, dense_size),
                nn.ReLU(),
                nn.Linear(dense_size, 1),
            )

        def forward(self, inputs: object) -> object:
            outputs, _ = self.lstm(inputs)
            return self.head(outputs[:, -1, :]).squeeze(-1)

    class CNNRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(input_size, channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(channels, dense_size),
                nn.ReLU(),
                nn.Linear(dense_size, 1),
            )

        def forward(self, inputs: object) -> object:
            inputs = inputs.transpose(1, 2)
            features = self.network(inputs)
            return self.head(features).squeeze(-1)

    if model_name == "lstm":
        model = LSTMRegressor()
    elif model_name == "cnn":
        model = CNNRegressor()
    else:
        raise ValueError(f"Unsupported sequence model {model_name}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=actual_learning_rate)
    loss_fn = nn.MSELoss()
    best_state = None
    best_validation_loss = float("inf")
    patience = 5
    patience_left = patience

    for _ in range(epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = model(validation_features)
            validation_loss = float(loss_fn(validation_predictions, validation_targets).item())

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predicted_scaled = model(evaluation_features).cpu().numpy()

    predicted_log_vol = predicted_scaled * target_std + target_mean
    return predicted_log_vol.astype(float), actual_log_vol, eligible_evaluation_index


def run_torch_sequence_block(
    model_name: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    test_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_params: dict[str, object] | None = None,
) -> BlockPrediction:
    predicted_log_vol, actual_log_vol, eligible_test_index = _fit_torch_sequence_predict(
        model_name=model_name,
        df=df,
        train_index=train_index,
        validation_index=validation_index,
        evaluation_index=test_index,
        feature_columns=feature_columns,
        target_column=target_column,
        lookback_window=lookback_window,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_params=model_params,
    )

    train_target = df.loc[train_index, target_column]
    low_threshold, high_threshold = _train_target_quantiles(train_target)
    return BlockPrediction(
        dates=df.loc[eligible_test_index, "date"],
        predicted_log_vol=predicted_log_vol,
        actual_log_vol=actual_log_vol,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def tune_torch_sequence_model(
    model_name: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    parameter_grid: list[dict[str, object]],
    metric: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best_params = parameter_grid[0].copy()
    best_score = float("inf")

    for params in parameter_grid:
        candidate_lookback = int(params.get("lookback_window", lookback_window))
        inner_train_index, inner_validation_index = _split_inner_validation_indices(
            train_index=train_index,
            lookback_window=candidate_lookback,
        )
        predicted_log_vol, actual_log_vol, _ = _fit_torch_sequence_predict(
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
            model_params=params,
        )
        validation_score = _validation_score(actual_log_vol, predicted_log_vol, metric)
        rows.append(
            {
                **params,
                "validation_score": validation_score,
            }
        )
        if validation_score < best_score:
            best_score = validation_score
            best_params = params.copy()

    summary = pd.DataFrame(rows).sort_values("validation_score").reset_index(drop=True)
    return best_params, summary


def sequence_model_names() -> set[str]:
    return SEQUENCE_MODELS.copy()
