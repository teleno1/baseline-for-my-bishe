from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np
import pandas as pd


class RollingTestAnalyzer:
    REQUIRED_COLUMNS = (
        "origin_id",
        "origin_date",
        "horizon",
        "target_date",
        "y_true",
        "y_pred",
        "error",
    )
    METRIC_COLUMNS = ("MASE", "RMSE", "WAPE", "Bias", "MAPE")

    def __init__(
        self,
        rolling_raw: str | PathLike[str] | pd.DataFrame,
        history_actuals: Sequence[float] | np.ndarray | pd.Series,
        seasonality: int,
        eps: float = 1e-8,
    ) -> None:
        self.eps = self._validate_eps(eps)
        self.seasonality = self._validate_seasonality(seasonality)
        self.rolling_raw = self._load_rolling_raw(rolling_raw)
        self.history_actuals = self._load_history_actuals(history_actuals)
        self.mase_scale = self._build_mase_scale()

        self.origin_ids = pd.Index(self.rolling_raw["origin_id"].unique(), name="origin_id")
        self.horizons = pd.Index(
            np.sort(self.rolling_raw["horizon"].unique()),
            name="horizon",
        )
        self.origin_dates = pd.Index(
            self.rolling_raw.groupby("origin_id", sort=True)["origin_date"]
            .first()
            .reindex(self.origin_ids)
            .tolist(),
            name="origin_date",
        )

        n_origins = len(self.origin_ids)
        n_horizons = len(self.horizons)
        self.error = self.rolling_raw["error"].to_numpy(dtype=float).reshape(n_origins, n_horizons)
        self.y_true = self.rolling_raw["y_true"].to_numpy(dtype=float).reshape(n_origins, n_horizons)
        self.y_pred = self.rolling_raw["y_pred"].to_numpy(dtype=float).reshape(n_origins, n_horizons)

    def overall_metrics(self) -> pd.Series:
        metrics = pd.Series(
            {
                "MASE": float(np.mean(np.abs(self.error)) / self.mase_scale),
                "RMSE": float(np.sqrt(np.mean(np.square(self.error)))),
                "WAPE": float(
                    np.sum(np.abs(self.error))
                    / max(float(np.sum(np.abs(self.y_true))), self.eps)
                    * 100
                ),
                "Bias": float(np.mean(self.error)),
                "MAPE": float(np.mean(self._absolute_percentage_error()) * 100),
            },
            dtype=float,
            name="overall",
        )
        return metrics.loc[list(self.METRIC_COLUMNS)]

    def loss_matrix(self, scope: str) -> pd.DataFrame:
        axis, index = self._resolve_scope(scope)
        losses = pd.DataFrame(
            {
                "MASE": np.mean(np.abs(self.error), axis=axis) / self.mase_scale,
                "RMSE": np.sqrt(np.mean(np.square(self.error), axis=axis)),
                "WAPE": (
                    np.sum(np.abs(self.error), axis=axis)
                    / np.maximum(np.sum(np.abs(self.y_true), axis=axis), self.eps)
                    * 100
                ),
                "Bias": np.mean(self.error, axis=axis),
                "MAPE": np.mean(self._absolute_percentage_error(), axis=axis) * 100,
            },
            index=index,
            dtype=float,
        )
        return losses.loc[:, list(self.METRIC_COLUMNS)]

    def loss_summary(self, scope: str) -> pd.DataFrame:
        losses = self.loss_matrix(scope)
        summary = losses.agg(["min", "max", "mean", "median"]).T
        summary["std"] = losses.std(axis=0, ddof=0)
        return summary.loc[:, ["min", "max", "mean", "median", "std"]]

    def compare_winrate(self, other: "RollingTestAnalyzer", scope: str) -> pd.Series:
        if not isinstance(other, RollingTestAnalyzer):
            raise TypeError("other must be a RollingTestAnalyzer instance")

        left = self.loss_matrix(scope)
        right = other.loss_matrix(scope)
        if not left.index.equals(right.index):
            raise ValueError(
                f"Cannot compare {scope}-wise losses with different indexes: "
                f"{left.index.tolist()} vs {right.index.tolist()}"
            )

        winrate: dict[str, float] = {}
        for metric in self.METRIC_COLUMNS:
            left_values = left[metric]
            right_values = right[metric]
            if metric == "Bias":
                left_values = left_values.abs()
                right_values = right_values.abs()
            winrate[metric] = float((left_values < right_values).mean())

        return pd.Series(winrate, dtype=float, name=f"{scope}_winrate").loc[list(self.METRIC_COLUMNS)]

    def _absolute_percentage_error(self) -> np.ndarray:
        return np.abs(self.error) / np.maximum(np.abs(self.y_true), self.eps)

    def _resolve_scope(self, scope: str) -> tuple[int, pd.Index]:
        if scope == "horizon":
            return 0, self.horizons
        if scope == "window":
            return 1, self.origin_dates
        raise ValueError("scope must be either 'horizon' or 'window'")

    def _build_mase_scale(self) -> float:
        if len(self.history_actuals) <= self.seasonality:
            raise ValueError("history_actuals must be longer than seasonality to compute MASE")

        diffs = np.abs(self.history_actuals[self.seasonality :] - self.history_actuals[: -self.seasonality])
        scale = float(np.mean(diffs))
        if scale <= self.eps:
            raise ValueError("MASE scale is zero or too close to zero")
        return scale

    def _load_history_actuals(
        self,
        history_actuals: Sequence[float] | np.ndarray | pd.Series,
    ) -> np.ndarray:
        values = np.asarray(history_actuals, dtype=float)
        if values.ndim != 1:
            raise ValueError("history_actuals must be a one-dimensional sequence")
        if values.size == 0:
            raise ValueError("history_actuals must not be empty")
        if np.isnan(values).any():
            raise ValueError("history_actuals contains missing values")
        return values

    def _load_rolling_raw(self, rolling_raw: str | PathLike[str] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(rolling_raw, pd.DataFrame):
            df = rolling_raw.copy()
        else:
            df = pd.read_csv(rolling_raw)

        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"rolling_raw is missing required columns: {missing_columns}")

        normalized = df.loc[:, self.REQUIRED_COLUMNS].copy()
        for column in ("origin_id", "horizon"):
            normalized[column] = self._coerce_integer_column(normalized[column], column)
        for column in ("y_true", "y_pred", "error"):
            normalized[column] = self._coerce_float_column(normalized[column], column)
        for column in ("origin_date", "target_date"):
            normalized[column] = self._coerce_date_column(normalized[column], column)

        normalized = normalized.sort_values(["origin_id", "horizon"]).reset_index(drop=True)
        self._validate_rolling_shape(normalized)
        self._validate_error_column(normalized)
        return normalized

    def _validate_rolling_shape(self, rolling_raw: pd.DataFrame) -> None:
        if rolling_raw.empty:
            raise ValueError("rolling_raw must not be empty")
        if (rolling_raw["origin_id"] <= 0).any():
            raise ValueError("origin_id must be positive integers")
        if (rolling_raw["horizon"] <= 0).any():
            raise ValueError("horizon must be positive integers")

        origin_dates_per_origin = rolling_raw.groupby("origin_id")["origin_date"].nunique()
        if not (origin_dates_per_origin == 1).all():
            raise ValueError("Each origin_id must map to exactly one origin_date")

        unique_origin_dates = rolling_raw.groupby("origin_id", sort=True)["origin_date"].first()
        if unique_origin_dates.duplicated().any():
            raise ValueError("origin_date must be unique across origins")

        reference_horizons = np.sort(rolling_raw["horizon"].unique())
        for origin_id, group in rolling_raw.groupby("origin_id", sort=True):
            if group["horizon"].duplicated().any():
                raise ValueError(f"Duplicate horizon values found for origin_id={origin_id}")
            if not np.array_equal(np.sort(group["horizon"].to_numpy()), reference_horizons):
                raise ValueError(
                    f"origin_id={origin_id} does not contain the full shared horizon set"
                )

        expected_rows = rolling_raw["origin_id"].nunique() * len(reference_horizons)
        if len(rolling_raw) != expected_rows:
            raise ValueError("rolling_raw cannot be reshaped into a complete origin x horizon matrix")

    def _validate_error_column(self, rolling_raw: pd.DataFrame) -> None:
        expected_error = rolling_raw["y_pred"] - rolling_raw["y_true"]
        matches = np.allclose(
            rolling_raw["error"].to_numpy(dtype=float),
            expected_error.to_numpy(dtype=float),
            atol=max(self.eps, 1e-6),
            rtol=1e-9,
        )
        if not matches:
            raise ValueError("error column must match y_pred - y_true")

    def _coerce_integer_column(self, values: pd.Series, column_name: str) -> pd.Series:
        numeric = pd.to_numeric(values, errors="raise")
        if numeric.isna().any():
            raise ValueError(f"{column_name} contains missing values")
        rounded = numeric.round()
        if not np.allclose(numeric.to_numpy(dtype=float), rounded.to_numpy(dtype=float)):
            raise ValueError(f"{column_name} must contain integer values")
        return rounded.astype(int)

    def _coerce_float_column(self, values: pd.Series, column_name: str) -> pd.Series:
        numeric = pd.to_numeric(values, errors="raise")
        if numeric.isna().any():
            raise ValueError(f"{column_name} contains missing values")
        return numeric.astype(float)

    def _coerce_date_column(self, values: pd.Series, column_name: str) -> pd.Series:
        parsed = pd.to_datetime(values, errors="raise")
        if parsed.isna().any():
            raise ValueError(f"{column_name} contains missing values")
        return parsed.dt.strftime("%Y-%m-%d")

    def _validate_eps(self, eps: float) -> float:
        eps_value = float(eps)
        if eps_value <= 0:
            raise ValueError("eps must be positive")
        return eps_value

    def _validate_seasonality(self, seasonality: int) -> int:
        seasonality_value = float(seasonality)
        if not seasonality_value.is_integer() or seasonality_value <= 0:
            raise ValueError("seasonality must be a positive integer")
        return int(seasonality_value)

