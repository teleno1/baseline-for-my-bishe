from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetScaler:
    """Global target z-score scaler fitted on the training target only."""

    mean: float
    std: float
    n_train: int
    eps: float = 1e-8

    def __post_init__(self) -> None:
        mean = float(self.mean)
        std = float(self.std)
        n_train = int(self.n_train)
        eps = float(self.eps)

        if n_train <= 0:
            raise ValueError("n_train must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not np.isfinite(mean):
            raise ValueError("target scaler mean must be finite")
        if not np.isfinite(std) or std <= eps:
            raise ValueError("training target standard deviation is zero or too close to zero")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "n_train", n_train)
        object.__setattr__(self, "eps", eps)

    @classmethod
    def fit(cls, values: Any, eps: float = 1e-8) -> "TargetScaler":
        train_values = np.asarray(values, dtype=float)
        if train_values.size == 0:
            raise ValueError("training target subset must not be empty")
        if np.isnan(train_values).any():
            raise ValueError("training target subset contains missing values")
        if not np.isfinite(train_values).all():
            raise ValueError("training target subset contains non-finite values")

        return cls(
            mean=float(np.mean(train_values)),
            std=float(np.std(train_values)),
            n_train=int(train_values.size),
            eps=eps,
        )

    def transform_values(self, values: Any) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_values(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean

    def transform_error(self, error: Any) -> np.ndarray:
        return np.asarray(error, dtype=float) / self.std

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "std": self.std,
            "n_train": self.n_train,
            "eps": self.eps,
        }


@dataclass(frozen=True)
class ExogScaler:
    """Global exogenous-feature scaler fitted on training rows only."""

    columns: tuple[str, ...]
    features: dict[str, dict[str, float | str]]
    n_train: int
    eps: float = 1e-8

    def __post_init__(self) -> None:
        columns = tuple(self.columns)
        n_train = int(self.n_train)
        eps = float(self.eps)
        if n_train <= 0:
            raise ValueError("n_train must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        normalized_features: dict[str, dict[str, float | str]] = {}
        for column in columns:
            if column not in self.features:
                raise ValueError(f"missing exog scaler parameters for column {column!r}")
            feature = dict(self.features[column])
            mode = str(feature.get("mode", ""))
            if mode not in {
                "zscore",
                "identity_binary",
                "identity_cyclic",
                "identity_constant",
            }:
                raise ValueError(f"unsupported exog scaler mode for column {column!r}: {mode!r}")
            mean = float(feature.get("mean", 0.0))
            std = float(feature.get("std", 0.0))
            if not np.isfinite(mean):
                raise ValueError(f"exog scaler mean must be finite for column {column!r}")
            if not np.isfinite(std):
                raise ValueError(f"exog scaler std must be finite for column {column!r}")
            if mode == "zscore" and std <= eps:
                raise ValueError(f"exog scaler std is zero or too close to zero for column {column!r}")
            normalized_features[column] = {
                "mode": mode,
                "mean": mean,
                "std": std,
            }

        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "features", normalized_features)
        object.__setattr__(self, "n_train", n_train)
        object.__setattr__(self, "eps", eps)

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        columns: Any,
        eps: float = 1e-8,
    ) -> "ExogScaler":
        ordered_columns = tuple(dict.fromkeys(str(column) for column in columns))
        n_train = int(len(train_df))
        if n_train <= 0:
            raise ValueError("training exog subset must not be empty")

        missing_columns = [column for column in ordered_columns if column not in train_df.columns]
        if missing_columns:
            raise ValueError(f"training exog subset is missing columns: {missing_columns}")

        features: dict[str, dict[str, float | str]] = {}
        for column in ordered_columns:
            series = train_df[column]
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"exog column {column!r} must be numeric before scaling")
            values = series.to_numpy(dtype=float)
            if np.isnan(values).any():
                raise ValueError(f"exog column {column!r} contains missing values")
            if not np.isfinite(values).all():
                raise ValueError(f"exog column {column!r} contains non-finite values")

            mean = float(np.mean(values))
            std = float(np.std(values))
            if cls._is_cyclic_column(column):
                mode = "identity_cyclic"
            elif np.isin(values, [0.0, 1.0]).all():
                mode = "identity_binary"
            elif std <= float(eps):
                mode = "identity_constant"
            else:
                mode = "zscore"
            features[column] = {
                "mode": mode,
                "mean": mean,
                "std": std,
            }

        return cls(
            columns=ordered_columns,
            features=features,
            n_train=n_train,
            eps=eps,
        )

    @staticmethod
    def _is_cyclic_column(column: str) -> bool:
        lower_column = str(column).lower()
        return lower_column.endswith("_sin") or lower_column.endswith("_cos")

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled_df = df.copy()
        for column in self.columns:
            if column not in scaled_df.columns:
                continue
            feature = self.features[column]
            values = scaled_df[column].to_numpy(dtype=float)
            if np.isnan(values).any():
                raise ValueError(f"exog column {column!r} contains missing values")
            if not np.isfinite(values).all():
                raise ValueError(f"exog column {column!r} contains non-finite values")
            if feature["mode"] == "zscore":
                scaled_df[column] = (values - float(feature["mean"])) / float(feature["std"])
            else:
                scaled_df[column] = values
        return scaled_df

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "features": self.features,
            "n_train": self.n_train,
            "eps": self.eps,
        }


@dataclass(frozen=True)
class ForecastMetricCalculator:
    """Shared forecast metrics with train-standardized non-ratio losses."""

    target_scaler: TargetScaler
    eps: float = 1e-8

    def __post_init__(self) -> None:
        eps = float(self.eps)
        if eps <= 0:
            raise ValueError("eps must be positive")
        object.__setattr__(self, "eps", eps)

    @classmethod
    def from_train_actuals(cls, train_actuals: Any, eps: float = 1e-8) -> "ForecastMetricCalculator":
        return cls(target_scaler=TargetScaler.fit(train_actuals, eps=eps), eps=eps)

    def error(self, y_true: Any, y_pred: Any) -> np.ndarray:
        true_values = np.asarray(y_true, dtype=float)
        pred_values = np.asarray(y_pred, dtype=float)
        return pred_values - true_values

    def normalized_error(self, y_true: Any, y_pred: Any) -> np.ndarray:
        return self.target_scaler.transform_error(self.error(y_true, y_pred))

    def absolute_percentage_error(self, y_true: Any, y_pred: Any) -> np.ndarray:
        true_values = np.asarray(y_true, dtype=float)
        return np.abs(self.error(y_true, y_pred)) / np.maximum(np.abs(true_values), self.eps)

    def overall_metrics(
        self,
        y_true: Any,
        y_pred: Any,
        columns: tuple[str, ...] = ("MAE", "RMSE", "WAPE(%)", "Bias", "MAPE(%)"),
    ) -> pd.Series:
        true_values = np.asarray(y_true, dtype=float)
        raw_error = self.error(true_values, y_pred)
        normalized_error = self.target_scaler.transform_error(raw_error)
        metrics = pd.Series(
            {
                "MAE": float(np.mean(np.abs(normalized_error))),
                "RMSE": float(np.sqrt(np.mean(np.square(normalized_error)))),
                "WAPE(%)": float(
                    np.sum(np.abs(raw_error)) / max(float(np.sum(np.abs(true_values))), self.eps) * 100
                ),
                "Bias": float(np.mean(normalized_error)),
                "MAPE(%)": float(
                    np.mean(np.abs(raw_error) / np.maximum(np.abs(true_values), self.eps)) * 100
                ),
            },
            dtype=float,
            name="overall",
        )
        return metrics.loc[list(columns)]

    def horizon_loss_matrix(
        self,
        y_true: Any,
        y_pred: Any,
        horizons: pd.Index,
        columns: tuple[str, ...] = ("MAE", "RMSE", "Bias"),
    ) -> pd.DataFrame:
        normalized_error = self.normalized_error(y_true, y_pred)
        losses = pd.DataFrame(
            {
                "MAE": np.mean(np.abs(normalized_error), axis=0),
                "RMSE": np.sqrt(np.mean(np.square(normalized_error), axis=0)),
                "Bias": np.mean(normalized_error, axis=0),
            },
            index=horizons,
            dtype=float,
        )
        return losses.loc[:, list(columns)]

    def window_loss_matrix(
        self,
        y_true: Any,
        y_pred: Any,
        origin_dates: pd.Index,
        columns: tuple[str, ...] = ("MAE", "RMSE", "MAPE(%)"),
    ) -> pd.DataFrame:
        normalized_error = self.normalized_error(y_true, y_pred)
        losses = pd.DataFrame(
            {
                "MAE": np.mean(np.abs(normalized_error), axis=1),
                "RMSE": np.sqrt(np.mean(np.square(normalized_error), axis=1)),
                "MAPE(%)": np.mean(self.absolute_percentage_error(y_true, y_pred), axis=1) * 100,
            },
            index=origin_dates,
            dtype=float,
        )
        return losses.loc[:, list(columns)]


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """对接近 0 的真实值使用保护分母，避免 MAPE 数值异常。"""

    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(true_values) < 1e-8, 1e-8, np.abs(true_values))
    return float(np.mean(np.abs((true_values - pred_values) / denom)) * 100)


def compute_plot_loss(
    y_true: Any,
    y_pred: Any,
    loss_name: str,
    target_scaler: TargetScaler | None = None,
    eps: float = 1e-8,
) -> float:
    """Compute the window metric used by test-result visualizations."""

    normalized_loss_name = str(loss_name).upper()
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    raw_error = pred_values - true_values

    if normalized_loss_name == "MAPE":
        denom = np.maximum(np.abs(true_values), float(eps))
        return float(np.mean(np.abs(raw_error) / denom) * 100)

    if target_scaler is None:
        raise ValueError(f"target_scaler is required for plot_loss_name={normalized_loss_name!r}")

    normalized_error = target_scaler.transform_error(raw_error)
    if normalized_loss_name == "MAE":
        return float(np.mean(np.abs(normalized_error)))
    if normalized_loss_name == "MSE":
        return float(np.mean(np.square(normalized_error)))

    raise ValueError("plot_loss_name must be one of ('MAPE', 'MAE', 'MSE')")


def format_plot_loss(loss_name: str, value: float) -> str:
    """Format a visualization window metric for legends and titles."""

    normalized_loss_name = str(loss_name).upper()
    if normalized_loss_name == "MAPE":
        return f"{normalized_loss_name}={value:.2f}%"
    if normalized_loss_name in {"MAE", "MSE"}:
        return f"{normalized_loss_name}={value:.4f}"
    raise ValueError("plot_loss_name must be one of ('MAPE', 'MAE', 'MSE')")


def summarize_cv_predictions(
    cv_df: pd.DataFrame,
    prediction_column: str,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    """把 StatsForecast 的 cross_validation 输出整理成统一的 origin 级结果。"""

    origin_records: list[dict[str, Any]] = []
    diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}

    for cutoff, group in cv_df.groupby("cutoff"):
        group = group.sort_values("ds")
        y_true = group["y"].to_numpy()
        y_pred = group[prediction_column].to_numpy()
        origin = pd.to_datetime(group["ds"].iloc[0])
        origin_records.append({"ds_origin": origin, "mape": safe_mape(y_true, y_pred)})
        diagnostics[origin] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "ds": pd.to_datetime(group["ds"]).to_numpy(),
            "cutoff": pd.to_datetime(cutoff),
        }

    return pd.DataFrame(origin_records), diagnostics
