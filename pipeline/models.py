"""Model adapters and tuning helpers for Milestone 1 volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import (
    balanced_accuracy,
    classification_accuracy,
    macro_f1_score,
    mean_absolute_error,
    qlike_loss,
    root_mean_squared_error,
)


SEQUENCE_MODELS = {"lstm", "cnn", "ctts"}


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


def _require_sklearn_classifier() -> tuple:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: scikit-learn. Install it with `pip install -r requirements.txt`.",
        ) from exc
    return LogisticRegression, Pipeline, StandardScaler


def _require_xgboost() -> type:
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: xgboost. Install it with `pip install -r requirements.txt`.",
        ) from exc
    return XGBRegressor


def _require_xgboost_classifier() -> type:
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: xgboost. Install it with `pip install -r requirements.txt`.",
        ) from exc
    return XGBClassifier


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
    predicted_log_vol: np.ndarray | None = None
    actual_log_vol: np.ndarray | None = None
    low_threshold: float = float("nan")
    high_threshold: float = float("nan")
    predicted_class: np.ndarray | None = None
    actual_class: np.ndarray | None = None
    class_probabilities: np.ndarray | None = None
    training_history: pd.DataFrame | None = None
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


def _classification_thresholds(train_target: pd.Series) -> tuple[float, float]:
    train_values = train_target.to_numpy(dtype=float)
    return (
        float(np.quantile(train_values, 1.0 / 3.0)),
        float(np.quantile(train_values, 2.0 / 3.0)),
    )


def _encode_classes(values: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    encoded = np.zeros(len(values), dtype=int)
    encoded[values > low_threshold] = 1
    encoded[values > high_threshold] = 2
    return encoded


def _classification_targets(
    df: pd.DataFrame,
    train_index: np.ndarray,
    evaluation_index: np.ndarray,
    source_column: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train_target = df.loc[train_index, source_column]
    low_threshold, high_threshold = _classification_thresholds(train_target)
    train_classes = _encode_classes(
        train_target.to_numpy(dtype=float),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    evaluation_classes = _encode_classes(
        df.loc[evaluation_index, source_column].to_numpy(dtype=float),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    return train_classes, evaluation_classes, low_threshold, high_threshold


def _classification_score(
    actual_class: np.ndarray,
    predicted_class: np.ndarray,
    metric: str,
) -> float:
    if metric == "accuracy":
        return classification_accuracy(actual_class, predicted_class)
    if metric == "balanced_accuracy":
        return balanced_accuracy(actual_class, predicted_class)
    if metric == "macro_f1":
        return macro_f1_score(actual_class, predicted_class)
    raise ValueError(f"Unsupported classification metric {metric}.")


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


def _resolve_torch_device(torch_module: object, requested_device: str) -> object:
    if requested_device == "auto":
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if (
            hasattr(torch_module, "backends")
            and hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return torch_module.device("mps")
        return torch_module.device("cpu")
    if requested_device == "cuda":
        if not hasattr(torch_module, "cuda") or not torch_module.cuda.is_available():
            raise ValueError("CUDA was requested, but it is not available in this environment.")
        return torch_module.device("cuda")
    if requested_device == "mps":
        if (
            not hasattr(torch_module, "backends")
            or not hasattr(torch_module.backends, "mps")
            or not torch_module.backends.mps.is_available()
        ):
            raise ValueError("MPS was requested, but it is not available in this environment.")
        return torch_module.device("mps")
    if requested_device == "cpu":
        return torch_module.device("cpu")
    raise ValueError(f"Unsupported torch device {requested_device}.")


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


def _build_sequence_loss(
    loss_name: str,
    nn: object,
    torch: object,
    target_mean: float,
    target_std: float,
    device: object,
) -> object:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.HuberLoss()
    if loss_name != "qlike":
        raise ValueError(f"Unsupported torch loss {loss_name}.")

    mean_tensor = torch.tensor(target_mean, dtype=torch.float32, device=device)
    std_tensor = torch.tensor(target_std, dtype=torch.float32, device=device)

    def qlike_loss(predicted_scaled: object, target_scaled: object) -> object:
        predicted_log_vol = predicted_scaled * std_tensor + mean_tensor
        actual_log_vol = target_scaled * std_tensor + mean_tensor
        predicted_vol = torch.exp(predicted_log_vol).clamp_min(1e-8)
        actual_vol = torch.exp(actual_log_vol).clamp_min(1e-8)
        predicted_variance = predicted_vol.square().clamp_min(1e-8)
        actual_variance = actual_vol.square().clamp_min(1e-8)
        return torch.mean(torch.log(predicted_variance) + actual_variance / predicted_variance)

    return qlike_loss


def _fit_torch_sequence_predict(
    model_name: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray | None,
    evaluation_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    training_loss: str,
    device_name: str,
    log_epoch_losses: bool,
    model_params: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    _set_torch_seed(torch)
    device = _resolve_torch_device(torch, device_name)
    params = model_params or {}

    actual_lookback_window = int(params.get("lookback_window", lookback_window))
    actual_batch_size = int(params.get("batch_size", batch_size))
    actual_learning_rate = float(params.get("learning_rate", learning_rate))
    hidden_size = int(params.get("hidden_size", 32))
    dense_size = int(params.get("dense_size", 16))
    dropout = float(params.get("dropout", 0.0))
    channels = int(params.get("channels", 32))
    kernel_size = int(params.get("kernel_size", 3))
    embedding_dim = int(params.get("embedding_dim", 64))
    num_heads = int(params.get("num_heads", 4))
    num_layers = int(params.get("num_layers", 2))
    ff_multiplier = int(params.get("ff_multiplier", 4))
    conv_stride = int(params.get("conv_stride", 1))
    weight_decay = float(params.get("weight_decay", 0.0))

    effective_train_index = train_index
    effective_validation_index = validation_index
    if effective_validation_index is None:
        effective_train_index, effective_validation_index = _split_inner_validation_indices(
            train_index=train_index,
            lookback_window=actual_lookback_window,
        )

    feature_matrix = _standardize_features(df, feature_columns, train_index)
    target_vector, target_mean, target_std = _standardize_target(df, target_column, train_index)

    X_train, y_train, _ = _build_sequence_dataset(
        feature_matrix,
        target_vector,
        effective_train_index,
        actual_lookback_window,
    )
    X_validation, y_validation, _ = _build_sequence_dataset(
        feature_matrix,
        target_vector,
        effective_validation_index,
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
    validation_features = torch.from_numpy(X_validation).to(device)
    validation_targets = torch.from_numpy(y_validation).to(device)
    evaluation_features = torch.from_numpy(X_evaluation).to(device)

    input_size = len(feature_columns)

    def conv_output_length(length: int, kernel: int, stride: int, padding: int) -> int:
        return ((length + 2 * padding - kernel) // stride) + 1

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
            return self.head(features).squeeze(-1) #Get outputs

    class CTTSRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if embedding_dim % num_heads != 0:
                raise ValueError("embedding_dim must be divisible by num_heads for CTTS.")

            padding = kernel_size // 2
            token_length = conv_output_length(
                actual_lookback_window,
                kernel_size,
                conv_stride,
                padding,
            )
            if token_length <= 0:
                raise ValueError("CTTS tokenizer produced no tokens. Adjust lookback or kernel settings.")

            self.tokenizer = nn.Conv1d(
                input_size,
                embedding_dim,
                kernel_size=kernel_size,
                stride=conv_stride,
                padding=padding,
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            self.position_embedding = nn.Parameter(
                torch.zeros(1, token_length + 1, embedding_dim),
            )
            self.input_dropout = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * ff_multiplier,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.norm = nn.LayerNorm(embedding_dim)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, dense_size),
                nn.GELU(),
                nn.Linear(dense_size, 1),
            )

        def forward(self, inputs: object) -> object:
            tokens = self.tokenizer(inputs.transpose(1, 2)).transpose(1, 2)
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
            tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
            tokens = self.input_dropout(tokens)
            encoded = self.transformer(tokens)
            pooled = self.norm(encoded[:, 0, :])
            return self.head(pooled).squeeze(-1)

    if model_name == "lstm":
        model = LSTMRegressor()
    elif model_name == "cnn":
        model = CNNRegressor()
    elif model_name == "ctts":
        model = CTTSRegressor()
    else:
        raise ValueError(f"Unsupported sequence model {model_name}.")
    model = model.to(device)

    if model_name == "ctts":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=actual_learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=actual_learning_rate)
    loss_fn = _build_sequence_loss(
        loss_name=training_loss,
        nn=nn,
        torch=torch,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )
    best_state = None
    best_validation_loss = float("inf")
    patience = 10
    patience_left = patience
    history_rows: list[dict[str, float]] = []

    for epoch_index in range(epochs):
        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            if model_name == "ctts":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            validation_predictions = model(validation_features)
            validation_loss = float(loss_fn(validation_predictions, validation_targets).item())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history_rows.append(
            {
                "epoch": float(epoch_index + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if log_epoch_losses:
            print(
                f"{model_name} epoch {epoch_index + 1}/{epochs} "
                f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"
            )

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
    training_history = pd.DataFrame(history_rows)
    return predicted_log_vol.astype(float), actual_log_vol, eligible_evaluation_index, training_history


def run_torch_sequence_block(
    model_name: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    training_loss: str,
    device_name: str,
    log_epoch_losses: bool,
    model_params: dict[str, object] | None = None,
) -> BlockPrediction:
    predicted_log_vol, actual_log_vol, eligible_test_index, training_history = _fit_torch_sequence_predict(
        model_name=model_name,
        df=df,
        train_index=train_index,
        validation_index=None,
        evaluation_index=test_index,
        feature_columns=feature_columns,
        target_column=target_column,
        lookback_window=lookback_window,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_loss=training_loss,
        device_name=device_name,
        log_epoch_losses=log_epoch_losses,
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
        training_history=training_history,
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
    training_loss: str,
    device_name: str,
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


def run_naive_classification_block(
    df: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    source_column: str,
    current_vol_feature: str,
) -> BlockPrediction:
    _, actual_class, low_threshold, high_threshold = _classification_targets(
        df=df,
        train_index=train_index,
        evaluation_index=test_index,
        source_column=source_column,
    )
    predicted_class = _encode_classes(
        df.loc[test_index, current_vol_feature].to_numpy(dtype=float),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    class_probabilities = np.eye(3, dtype=float)[predicted_class]
    return BlockPrediction(
        dates=df.loc[test_index, "date"],
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        predicted_class=predicted_class,
        actual_class=actual_class,
        class_probabilities=class_probabilities,
    )


def tune_naive_classification_model(
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    source_column: str,
    candidate_features: list[str],
    metric: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    unique_features = [feature for feature in dict.fromkeys(candidate_features) if feature in df.columns]
    if not unique_features:
        raise ValueError("No valid naive candidate features were provided for tuning.")

    _, actual_class, low_threshold, high_threshold = _classification_targets(
        df=df,
        train_index=train_index,
        evaluation_index=validation_index,
        source_column=source_column,
    )

    rows: list[dict[str, object]] = []
    best_feature = unique_features[0]
    best_score = float("-inf")

    for feature_name in unique_features:
        predicted_class = _encode_classes(
            df.loc[validation_index, feature_name].to_numpy(dtype=float),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        validation_score = _classification_score(actual_class, predicted_class, metric)
        rows.append(
            {
                "current_vol_feature": feature_name,
                "validation_score": validation_score,
            }
        )
        if validation_score > best_score:
            best_score = validation_score
            best_feature = feature_name

    summary = pd.DataFrame(rows).sort_values("validation_score", ascending=False).reset_index(drop=True)
    return {"current_vol_feature": best_feature}, summary


def _fit_logistic_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    model_params: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    LogisticRegression, Pipeline, StandardScaler = _require_sklearn_classifier()
    params = model_params or {}
    alpha = float(params.get("alpha", 0.001))
    c_value = 1.0 / max(alpha, 1e-6)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=c_value,
                    l1_ratio=float(params.get("l1_ratio", 0.2)),
                    max_iter=5_000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    classifier = model.named_steps["classifier"]
    explainability = pd.DataFrame(
        {
            "feature": list(X_train.columns),
            "importance": np.mean(np.abs(classifier.coef_), axis=0),
        }
    )
    class_probabilities = np.asarray(model.predict_proba(X_eval), dtype=float)
    predicted_class = np.asarray(np.argmax(class_probabilities, axis=1), dtype=int)
    return predicted_class, class_probabilities, explainability


def run_elastic_net_classification_block(
    df: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    feature_columns: list[str],
    source_column: str,
    model_params: dict[str, object] | None = None,
) -> BlockPrediction:
    X_train = df.loc[train_index, feature_columns]
    X_test = df.loc[test_index, feature_columns]
    y_train, actual_class, low_threshold, high_threshold = _classification_targets(
        df=df,
        train_index=train_index,
        evaluation_index=test_index,
        source_column=source_column,
    )
    predicted_class, class_probabilities, explainability = _fit_logistic_predict(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        model_params=model_params,
    )
    return BlockPrediction(
        dates=df.loc[test_index, "date"],
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        predicted_class=predicted_class,
        actual_class=actual_class,
        class_probabilities=class_probabilities,
        explainability=explainability,
    )


def tune_elastic_net_classification_model(
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    feature_columns: list[str],
    source_column: str,
    parameter_grid: list[dict[str, object]],
    metric: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    X_train = df.loc[train_index, feature_columns]
    X_validation = df.loc[validation_index, feature_columns]
    y_train, actual_class, _, _ = _classification_targets(
        df=df,
        train_index=train_index,
        evaluation_index=validation_index,
        source_column=source_column,
    )

    rows: list[dict[str, object]] = []
    best_params = parameter_grid[0].copy()
    best_score = float("-inf")

    for params in parameter_grid:
        predicted_class, _, _ = _fit_logistic_predict(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_validation,
            model_params=params,
        )
        validation_score = _classification_score(actual_class, predicted_class, metric)
        rows.append({**params, "validation_score": validation_score})
        if validation_score > best_score:
            best_score = validation_score
            best_params = params.copy()

    summary = pd.DataFrame(rows).sort_values("validation_score", ascending=False).reset_index(drop=True)
    return best_params, summary


def _fit_xgboost_classifier_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    model_params: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    XGBClassifier = _require_xgboost_classifier()
    params = model_params or {}

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
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

    class_probabilities = np.asarray(model.predict_proba(X_eval), dtype=float)
    predicted_class = np.asarray(np.argmax(class_probabilities, axis=1), dtype=int)
    explainability = pd.DataFrame(
        {
            "feature": list(X_train.columns),
            "importance": model.feature_importances_,
        }
    )
    return predicted_class, class_probabilities, explainability


def run_xgboost_classification_block(
    df: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    feature_columns: list[str],
    source_column: str,
    model_params: dict[str, object] | None = None,
) -> BlockPrediction:
    X_train = df.loc[train_index, feature_columns]
    X_test = df.loc[test_index, feature_columns]
    y_train, actual_class, low_threshold, high_threshold = _classification_targets(
        df=df,
        train_index=train_index,
        evaluation_index=test_index,
        source_column=source_column,
    )
    predicted_class, class_probabilities, explainability = _fit_xgboost_classifier_predict(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        model_params=model_params,
    )
    return BlockPrediction(
        dates=df.loc[test_index, "date"],
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        predicted_class=predicted_class,
        actual_class=actual_class,
        class_probabilities=class_probabilities,
        explainability=explainability,
    )


def tune_xgboost_classification_model(
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    feature_columns: list[str],
    source_column: str,
    parameter_grid: list[dict[str, object]],
    metric: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    X_train = df.loc[train_index, feature_columns]
    X_validation = df.loc[validation_index, feature_columns]
    y_train, actual_class, _, _ = _classification_targets(
        df=df,
        train_index=train_index,
        evaluation_index=validation_index,
        source_column=source_column,
    )

    rows: list[dict[str, object]] = []
    best_params = parameter_grid[0].copy()
    best_score = float("-inf")

    for params in parameter_grid:
        predicted_class, _, _ = _fit_xgboost_classifier_predict(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_validation,
            model_params=params,
        )
        validation_score = _classification_score(actual_class, predicted_class, metric)
        rows.append({**params, "validation_score": validation_score})
        if validation_score > best_score:
            best_score = validation_score
            best_params = params.copy()

    summary = pd.DataFrame(rows).sort_values("validation_score", ascending=False).reset_index(drop=True)
    return best_params, summary


def _build_sequence_classification_dataset(
    feature_matrix: np.ndarray,
    class_vector: np.ndarray,
    indices: np.ndarray,
    lookback_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible_indices = indices[indices >= lookback_window - 1]
    sequences = []
    labels = []
    for index in eligible_indices:
        start = index - lookback_window + 1
        sequences.append(feature_matrix[start : index + 1])
        labels.append(class_vector[index])

    if not sequences:
        raise ValueError("No eligible sequence samples were found. Reduce the lookback window.")

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        eligible_indices,
    )


def _fit_torch_sequence_classifier_predict(
    model_name: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray | None,
    evaluation_index: np.ndarray,
    feature_columns: list[str],
    source_column: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    metric: str,
    device_name: str,
    log_epoch_losses: bool,
    model_params: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, pd.DataFrame]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    _set_torch_seed(torch)
    device = _resolve_torch_device(torch, device_name)
    params = model_params or {}

    actual_lookback_window = int(params.get("lookback_window", lookback_window))
    actual_batch_size = int(params.get("batch_size", batch_size))
    actual_learning_rate = float(params.get("learning_rate", learning_rate))
    hidden_size = int(params.get("hidden_size", 32))
    dense_size = int(params.get("dense_size", 16))
    dropout = float(params.get("dropout", 0.0))
    channels = int(params.get("channels", 32))
    kernel_size = int(params.get("kernel_size", 3))
    embedding_dim = int(params.get("embedding_dim", 64))
    num_heads = int(params.get("num_heads", 4))
    num_layers = int(params.get("num_layers", 2))
    ff_multiplier = int(params.get("ff_multiplier", 4))
    conv_stride = int(params.get("conv_stride", 1))
    weight_decay = float(params.get("weight_decay", 0.0))

    effective_train_index = train_index
    effective_validation_index = validation_index
    if effective_validation_index is None:
        effective_train_index, effective_validation_index = _split_inner_validation_indices(
            train_index=train_index,
            lookback_window=actual_lookback_window,
        )

    _, full_classes, low_threshold, high_threshold = _classification_targets(
        df=df,
        train_index=train_index,
        evaluation_index=np.arange(len(df), dtype=int),
        source_column=source_column,
    )
    feature_matrix = _standardize_features(df, feature_columns, train_index)

    X_train, y_train, _ = _build_sequence_classification_dataset(
        feature_matrix,
        full_classes,
        effective_train_index,
        actual_lookback_window,
    )
    X_validation, y_validation, _ = _build_sequence_classification_dataset(
        feature_matrix,
        full_classes,
        effective_validation_index,
        actual_lookback_window,
    )
    X_evaluation, actual_class, eligible_evaluation_index = _build_sequence_classification_dataset(
        feature_matrix,
        full_classes,
        evaluation_index,
        actual_lookback_window,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=actual_batch_size,
        shuffle=True,
    )
    validation_features = torch.from_numpy(X_validation).to(device)
    validation_targets = torch.from_numpy(y_validation).to(device)
    evaluation_features = torch.from_numpy(X_evaluation).to(device)

    input_size = len(feature_columns)

    def conv_output_length(length: int, kernel: int, stride: int, padding: int) -> int:
        return ((length + 2 * padding - kernel) // stride) + 1

    class LSTMClassifier(nn.Module):
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
                nn.Linear(dense_size, 3),
            )

        def forward(self, inputs: object) -> object:
            outputs, _ = self.lstm(inputs)
            return self.head(outputs[:, -1, :])

    class CNNClassifier(nn.Module):
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
                nn.Linear(dense_size, 3),
            )

        def forward(self, inputs: object) -> object:
            features = self.network(inputs.transpose(1, 2))
            return self.head(features)

    class CTTSClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if embedding_dim % num_heads != 0:
                raise ValueError("embedding_dim must be divisible by num_heads for CTTS.")

            padding = kernel_size // 2
            token_length = conv_output_length(
                actual_lookback_window,
                kernel_size,
                conv_stride,
                padding,
            )
            self.tokenizer = nn.Conv1d(
                input_size,
                embedding_dim,
                kernel_size=kernel_size,
                stride=conv_stride,
                padding=padding,
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            self.position_embedding = nn.Parameter(torch.zeros(1, token_length + 1, embedding_dim))
            self.input_dropout = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * ff_multiplier,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(embedding_dim)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, dense_size),
                nn.GELU(),
                nn.Linear(dense_size, 3),
            )

        def forward(self, inputs: object) -> object:
            tokens = self.tokenizer(inputs.transpose(1, 2)).transpose(1, 2)
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
            tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
            tokens = self.input_dropout(tokens)
            encoded = self.transformer(tokens)
            pooled = self.norm(encoded[:, 0, :])
            return self.head(pooled)

    if model_name == "lstm":
        model = LSTMClassifier()
    elif model_name == "cnn":
        model = CNNClassifier()
    elif model_name == "ctts":
        model = CTTSClassifier()
    else:
        raise ValueError(f"Unsupported sequence model {model_name}.")
    model = model.to(device)

    if model_name == "ctts":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=actual_learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=actual_learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_validation_score = float("-inf")
    patience = 10
    patience_left = patience
    history_rows: list[dict[str, float]] = []

    for epoch_index in range(epochs):
        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_fn(logits, batch_targets)
            loss.backward()
            if model_name == "ctts":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_features)
            validation_ce_loss = float(loss_fn(validation_logits, validation_targets).item())
            validation_predictions = torch.argmax(validation_logits, dim=1).cpu().numpy()
        validation_score = _classification_score(
            validation_targets.cpu().numpy(),
            validation_predictions,
            metric,
        )
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history_rows.append(
            {
                "epoch": float(epoch_index + 1),
                "train_loss": train_loss,
                "validation_loss": validation_ce_loss,
                "validation_score": validation_score,
            }
        )
        if log_epoch_losses:
            print(
                f"{model_name} epoch {epoch_index + 1}/{epochs} "
                f"train_loss={train_loss:.6f} validation_loss={validation_ce_loss:.6f} "
                f"validation_score={validation_score:.6f}"
            )
        if validation_score > best_validation_score:
            best_validation_score = validation_score
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
        evaluation_logits = model(evaluation_features)
        class_probabilities = torch.softmax(evaluation_logits, dim=1).cpu().numpy()
    predicted_class = np.asarray(np.argmax(class_probabilities, axis=1), dtype=int)
    training_history = pd.DataFrame(history_rows)
    return (
        predicted_class,
        actual_class,
        eligible_evaluation_index,
        class_probabilities,
        low_threshold,
        high_threshold,
        training_history,
    )


def run_torch_sequence_classification_block(
    model_name: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    feature_columns: list[str],
    source_column: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    metric: str,
    device_name: str,
    log_epoch_losses: bool,
    model_params: dict[str, object] | None = None,
) -> BlockPrediction:
    (
        predicted_class,
        actual_class,
        eligible_test_index,
        class_probabilities,
        low_threshold,
        high_threshold,
        training_history,
    ) = _fit_torch_sequence_classifier_predict(
        model_name=model_name,
        df=df,
        train_index=train_index,
        validation_index=None,
        evaluation_index=test_index,
        feature_columns=feature_columns,
        source_column=source_column,
        lookback_window=lookback_window,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        metric=metric,
        device_name=device_name,
        log_epoch_losses=log_epoch_losses,
        model_params=model_params,
    )
    return BlockPrediction(
        dates=df.loc[eligible_test_index, "date"],
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        predicted_class=predicted_class,
        actual_class=actual_class,
        class_probabilities=class_probabilities,
        training_history=training_history,
    )


def tune_torch_sequence_classification_model(
    model_name: str,
    df: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    feature_columns: list[str],
    source_column: str,
    parameter_grid: list[dict[str, object]],
    metric: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best_params = parameter_grid[0].copy()
    best_score = float("-inf")

    for params in parameter_grid:
        candidate_lookback = int(params.get("lookback_window", lookback_window))
        inner_train_index, inner_validation_index = _split_inner_validation_indices(
            train_index=train_index,
            lookback_window=candidate_lookback,
        )
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
        )
        validation_score = _classification_score(actual_class, predicted_class, metric)
        rows.append({**params, "validation_score": validation_score})
        if validation_score > best_score:
            best_score = validation_score
            best_params = params.copy()

    summary = pd.DataFrame(rows).sort_values("validation_score", ascending=False).reset_index(drop=True)
    return best_params, summary
