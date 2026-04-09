from __future__ import annotations

from collections.abc import Sequence
from math import ceil
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
    OVERALL_METRIC_COLUMNS = ("MASE", "RMSE", "WAPE(%)", "Bias", "MAPE(%)")
    HORIZON_METRIC_COLUMNS = ("MAE", "RMSE", "Bias")
    WINDOW_METRIC_COLUMNS = ("MAE", "RMSE", "MAPE(%)")
    WINDOW_SUMMARY_COLUMNS = ("mean", "median", "std", "worst10%", "max")

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
                "MASE": float(np.mean(self._absolute_error()) / self.mase_scale),
                "RMSE": float(np.sqrt(np.mean(np.square(self.error)))),
                "WAPE(%)": float(
                    np.sum(self._absolute_error())
                    / max(float(np.sum(np.abs(self.y_true))), self.eps)
                    * 100
                ),
                "Bias": float(np.mean(self.error)),
                "MAPE(%)": float(np.mean(self._absolute_percentage_error()) * 100),
            },
            dtype=float,
            name="overall",
        )
        return metrics.loc[list(self.OVERALL_METRIC_COLUMNS)]

    def loss_matrix(self, scope: str) -> pd.DataFrame:
        if scope == "horizon":
            return self._horizon_loss_matrix()
        if scope == "window":
            return self._window_loss_matrix()
        raise ValueError("scope must be either 'horizon' or 'window'")

    def loss_summary(
        self,
        scope: str,
        baseline: "RollingTestAnalyzer | None" = None,
    ) -> pd.DataFrame:
        if scope != "window":
            raise ValueError("Only window-wise loss summary is supported")

        losses = self._window_loss_matrix()
        summary = pd.DataFrame(
            {
                "mean": losses.mean(axis=0),
                "median": losses.median(axis=0),
                "std": losses.std(axis=0, ddof=0),
                "worst10%": losses.apply(self._worst_ten_percent_mean, axis=0),
                "max": losses.max(axis=0),
            },
            dtype=float,
        )
        summary = summary.loc[
            list(self.WINDOW_METRIC_COLUMNS),
            list(self.WINDOW_SUMMARY_COLUMNS),
        ]

        if baseline is not None:
            summary["win_rate"] = self._window_win_rate(baseline)

        return summary

    def _horizon_loss_matrix(self) -> pd.DataFrame:
        losses = pd.DataFrame(
            {
                "MAE": np.mean(self._absolute_error(), axis=0),
                "RMSE": np.sqrt(np.mean(np.square(self.error), axis=0)),
                "Bias": np.mean(self.error, axis=0),
            },
            index=self.horizons,
            dtype=float,
        )
        return losses.loc[:, list(self.HORIZON_METRIC_COLUMNS)]

    def _window_loss_matrix(self) -> pd.DataFrame:
        losses = pd.DataFrame(
            {
                "MAE": np.mean(self._absolute_error(), axis=1),
                "RMSE": np.sqrt(np.mean(np.square(self.error), axis=1)),
                "MAPE(%)": np.mean(self._absolute_percentage_error(), axis=1) * 100,
            },
            index=self.origin_dates,
            dtype=float,
        )
        return losses.loc[:, list(self.WINDOW_METRIC_COLUMNS)]

    def _window_win_rate(self, baseline: "RollingTestAnalyzer") -> pd.Series:
        if not isinstance(baseline, RollingTestAnalyzer):
            raise TypeError("baseline must be a RollingTestAnalyzer instance")

        left = self._window_loss_matrix()
        right = baseline._window_loss_matrix()
        if not left.index.equals(right.index):
            raise ValueError(
                "Cannot compare window-wise losses with different indexes: "
                f"{left.index.tolist()} vs {right.index.tolist()}"
            )

        wins = {
            metric: float((left[metric] < right[metric]).mean())
            for metric in self.WINDOW_METRIC_COLUMNS
        }
        return pd.Series(wins, dtype=float).loc[list(self.WINDOW_METRIC_COLUMNS)]

    def _absolute_error(self) -> np.ndarray:
        return np.abs(self.error)

    def _absolute_percentage_error(self) -> np.ndarray:
        return self._absolute_error() / np.maximum(np.abs(self.y_true), self.eps)

    def _worst_ten_percent_mean(self, values: pd.Series) -> float:
        worst_count = max(1, ceil(len(values) * 0.1))
        return float(values.nlargest(worst_count).mean())

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
