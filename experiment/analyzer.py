from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from os import PathLike

import numpy as np
import pandas as pd


class RollingTestAnalyzer:
    """分析 rolling_test_raw.csv 的滚动测试结果。

    这个类假设输入数据已经是单序列 rolling forecast 的原始逐点结果，
    每条记录对应一个 origin 窗口下某个 horizon 的预测。

    核心思路：
    1. 先把表按 origin_id / horizon 排序。
    2. 再把 error、y_true、y_pred reshape 成 O x H 矩阵。
    3. 基于矩阵分别计算 overall、horizon-wise、window-wise 指标。
    """

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
        """初始化分析器并完成输入校验与矩阵重建。

        参数
        ----------
        rolling_raw
            rolling_test_raw.csv 的路径，或同结构的 DataFrame。
        history_actuals
            首个测试目标点之前的真实历史序列，按时间升序排列。
            这里只用于计算 overall MASE 的全局缩放分母。
        seasonality
            MASE 使用的季节周期，例如日频周季节性通常取 7。
        eps
            百分比类指标的分母保护常数，避免 y_true 接近 0 时数值爆炸。
        """

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

        # 三个矩阵共享同一套 O x H 形状：
        # 行表示 rolling origin，列表示 horizon。
        n_origins = len(self.origin_ids)
        n_horizons = len(self.horizons)
        self.error = self.rolling_raw["error"].to_numpy(dtype=float).reshape(n_origins, n_horizons)
        self.y_true = self.rolling_raw["y_true"].to_numpy(dtype=float).reshape(n_origins, n_horizons)
        self.y_pred = self.rolling_raw["y_pred"].to_numpy(dtype=float).reshape(n_origins, n_horizons)

    def overall_metrics(self) -> pd.Series:
        """返回整体误差指标。

        overall 保留 5 个指标：
        - MASE
        - RMSE
        - WAPE(%)
        - Bias
        - MAPE(%)
        """

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
        """返回指定粒度下的误差矩阵。

        - scope='horizon'：索引是 horizon，列为 MAE / RMSE / Bias
        - scope='window'：索引是 origin_date，列为 MAE / RMSE / MAPE(%)
        """

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
        """返回 window-wise 指标的统计摘要。

        当前只支持 scope='window'，因为 horizon-wise 的目标是直接观察
        不同 horizon 的变化趋势，不再额外做统计汇总。

        若传入 baseline，则在统计表中追加一列 win_rate：
        对每个窗口逐项比较，当前模型指标更小记为 1，否则记为 0，
        最后对所有窗口求平均。
        """

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
        """计算 horizon-wise 指标表。

        每一列固定一个 horizon，聚合所有 origin 上该 horizon 的误差。
        """

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
        """计算 window-wise 指标表。

        每一行对应一个 origin 窗口，先在该窗口内部对 H 个 horizon 聚合。
        """

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
        """计算当前模型相对 baseline 的 window-wise 胜率。"""

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
            # 统一采用“误差越小越好”的规则；平局记为 0。
            metric: float((left[metric] < right[metric]).mean())
            for metric in self.WINDOW_METRIC_COLUMNS
        }
        return pd.Series(wins, dtype=float).loc[list(self.WINDOW_METRIC_COLUMNS)]

    def _absolute_error(self) -> np.ndarray:
        """返回逐点绝对误差矩阵 |E|。"""
        return np.abs(self.error)

    def _absolute_percentage_error(self) -> np.ndarray:
        """返回逐点绝对百分比误差矩阵 |E| / max(|Y|, eps)。"""
        return self._absolute_error() / np.maximum(np.abs(self.y_true), self.eps)

    def _worst_ten_percent_mean(self, values: pd.Series) -> float:
        """计算最差 10% 样本的平均值。

        这里“最差”定义为数值最大的那部分窗口，因为 window-wise 统计里
        只保留了误差型指标，值越大代表表现越差。
        """

        worst_count = max(1, ceil(len(values) * 0.1))
        return float(values.nlargest(worst_count).mean())

    def _build_mase_scale(self) -> float:
        """基于历史真实值构造 overall MASE 的全局缩放分母。"""

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
        """读取并校验 history_actuals。"""

        values = np.asarray(history_actuals, dtype=float)
        if values.ndim != 1:
            raise ValueError("history_actuals must be a one-dimensional sequence")
        if values.size == 0:
            raise ValueError("history_actuals must not be empty")
        if np.isnan(values).any():
            raise ValueError("history_actuals contains missing values")
        return values

    def _load_rolling_raw(self, rolling_raw: str | PathLike[str] | pd.DataFrame) -> pd.DataFrame:
        """读取 rolling 原始表，并完成类型归一化与结构校验。"""

        if isinstance(rolling_raw, pd.DataFrame):
            df = rolling_raw.copy()
        else:
            df = pd.read_csv(rolling_raw)

        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"rolling_raw is missing required columns: {missing_columns}")

        # 这里只保留分析器真正需要的列，避免额外列影响后续 reshape。
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
        """校验 rolling 原始表能否稳定重建成完整的 O x H 矩阵。"""

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
        """校验 error 列与 y_pred - y_true 是否一致。"""

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
        """将整数字段安全转换为 int，并拦截缺失值与非整数值。"""

        numeric = pd.to_numeric(values, errors="raise")
        if numeric.isna().any():
            raise ValueError(f"{column_name} contains missing values")
        rounded = numeric.round()
        if not np.allclose(numeric.to_numpy(dtype=float), rounded.to_numpy(dtype=float)):
            raise ValueError(f"{column_name} must contain integer values")
        return rounded.astype(int)

    def _coerce_float_column(self, values: pd.Series, column_name: str) -> pd.Series:
        """将数值字段安全转换为 float。"""

        numeric = pd.to_numeric(values, errors="raise")
        if numeric.isna().any():
            raise ValueError(f"{column_name} contains missing values")
        return numeric.astype(float)

    def _coerce_date_column(self, values: pd.Series, column_name: str) -> pd.Series:
        """将日期字段标准化成 YYYY-MM-DD 字符串。"""

        parsed = pd.to_datetime(values, errors="raise")
        if parsed.isna().any():
            raise ValueError(f"{column_name} contains missing values")
        return parsed.dt.strftime("%Y-%m-%d")

    def _validate_eps(self, eps: float) -> float:
        """校验 eps 为正数。"""

        eps_value = float(eps)
        if eps_value <= 0:
            raise ValueError("eps must be positive")
        return eps_value

    def _validate_seasonality(self, seasonality: int) -> int:
        """校验 seasonality 为正整数。"""

        seasonality_value = float(seasonality)
        if not seasonality_value.is_integer() or seasonality_value <= 0:
            raise ValueError("seasonality must be a positive integer")
        return int(seasonality_value)

