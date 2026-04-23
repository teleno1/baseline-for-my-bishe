from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..metrics import safe_mape
from ..types import ExecutorOutput, PhaseOutput, ExecutionContext
from .base import BaseExecutor


class ResidualMLForecastExecutor(BaseExecutor):
    """
    残差预测执行器，使用 MLForecast 预测残差。

    工作流程:
    1. 对每个 rolling origin，取历史窗口最后一天的值 y_last
    2. 将预测目标转换为 residual[h] = y_true[h] - y_last
    3. 用 MLForecast 模型预测残差
    4. 最终预测 = y_last + residual_pred
    """

    def run(self) -> ExecutorOutput:
        """执行残差预测流程。"""
        self._validate_supported_exog()

        # Validation: use cross-validation on trainval
        val_phase = self._cross_validate_residual_phase(
            df=self.context.split_data.trainval,
            n_points=self.context.split_data.n_val,
            skip_early_origins=True,
        )

        # Test: train on trainval and predict on test set
        test_phase = self._predict_test_phase()

        return ExecutorOutput(val_phase=val_phase, test_phase=test_phase)

    def _validate_supported_exog(self) -> None:
        if self.context.hist_exog:
            raise ValueError(
                "ResidualMLForecastExecutor only supports no-exog or future-known exogenous "
                "features. Historical exogenous columns should be disabled."
            )

    def _cross_validate_residual_phase(
        self,
        df: pd.DataFrame,
        n_points: int,
        skip_early_origins: bool = False,
    ) -> PhaseOutput:
        """对残差进行 cross-validation。"""
        # Build forecaster once and reuse
        forecaster = self._build_forecaster()

        # Origins are relative to df
        origins = list(range(n_points))

        origin_records = []
        diagnostics = {}

        for origin_idx in origins:
            # Skip early origins if they don't have enough history
            if skip_early_origins and origin_idx < 2:
                continue

            cutoff_idx = origin_idx - 1
            if cutoff_idx < 0:
                continue

            # Get y_last from history
            y_last = float(df.iloc[cutoff_idx]["y"])
            cutoff_date = pd.to_datetime(df.iloc[cutoff_idx]["ds"])

            # Get future horizon values
            future_rows = df.iloc[origin_idx : origin_idx + self.context.run_config.horizon]
            if len(future_rows) < self.context.run_config.horizon:
                continue
            y_true = future_rows["y"].to_numpy()
            ds_forecast = pd.to_datetime(future_rows["ds"]).to_numpy()

            # Get history for training
            history_df = df.iloc[:origin_idx].copy()

            # Create difference target
            history_df["y_diff"] = history_df["y"].diff()

            # Drop first row (NaN from diff)
            fit_df = history_df[["unique_id", "ds"]].iloc[1:].copy()
            fit_df["y"] = history_df["y_diff"].iloc[1:]

            if len(fit_df) < self.context.run_config.input_size:
                continue

            # Train forecaster on y_diff
            forecaster.fit(fit_df, static_features=[])

            # Predict next horizon steps
            y_diff_pred = forecaster.predict(h=self.context.run_config.horizon)

            # Get prediction column
            pred_col = self._prediction_column(y_diff_pred, excluded={"unique_id", "ds"})
            y_diff_pred_values = y_diff_pred[pred_col].to_numpy()

            # Accumulate y_diff predictions to get residual prediction
            residual_pred = np.cumsum(y_diff_pred_values)

            # Final prediction = y_last + residual_pred
            final_pred = y_last + residual_pred

            # Compute MAPE using original y_true
            origin = pd.to_datetime(future_rows["ds"].iloc[0])
            mape = safe_mape(y_true, final_pred)
            origin_records.append({"ds_origin": origin, "mape": mape})

            diagnostics[origin] = {
                "y_true": y_true,
                "y_pred": final_pred,
                "ds": ds_forecast,
                "cutoff": cutoff_date,
            }

        return PhaseOutput(
            origin_mape_df=pd.DataFrame(origin_records),
            diagnostics=diagnostics,
        )

    def _predict_test_phase(self) -> PhaseOutput:
        """Train on trainval and predict on test set."""
        forecaster = self._build_forecaster()

        trainval = self.context.split_data.trainval
        test = self.context.split_data.test
        full_df = self.context.dataset.df

        n_train = self.context.split_data.n_train
        n_val = self.context.split_data.n_val

        origin_records = []
        diagnostics = {}

        # For test: origins are relative to test set, so we need to map to full df
        test_start_idx = n_train + n_val  # = 375
        n_test = self.context.split_data.n_test

        origins = list(range(n_test))

        for origin_idx in origins:
            cutoff_idx = test_start_idx + origin_idx - 1
            if cutoff_idx < 0:
                continue

            # Get y_last from full df (history before test set)
            y_last = float(full_df.iloc[cutoff_idx]["y"])
            cutoff_date = pd.to_datetime(full_df.iloc[cutoff_idx]["ds"])

            # Get future horizon values from test set
            test_position = origin_idx + test_start_idx
            future_rows = full_df.iloc[test_position : test_position + self.context.run_config.horizon]
            if len(future_rows) < self.context.run_config.horizon:
                continue
            y_true = future_rows["y"].to_numpy()
            ds_forecast = pd.to_datetime(future_rows["ds"]).to_numpy()

            # Get history for training (all trainval data up to cutoff)
            history_df = full_df.iloc[:cutoff_idx].copy()

            # Create difference target
            history_df["y_diff"] = history_df["y"].diff()

            # Drop first row (NaN from diff)
            fit_df = history_df[["unique_id", "ds"]].iloc[1:].copy()
            fit_df["y"] = history_df["y_diff"].iloc[1:]

            if len(fit_df) < self.context.run_config.input_size:
                continue

            # Train forecaster on y_diff
            forecaster.fit(fit_df, static_features=[])

            # Predict next horizon steps
            y_diff_pred = forecaster.predict(h=self.context.run_config.horizon)

            # Get prediction column
            pred_col = self._prediction_column(y_diff_pred, excluded={"unique_id", "ds"})
            y_diff_pred_values = y_diff_pred[pred_col].to_numpy()

            # Accumulate y_diff predictions to get residual prediction
            residual_pred = np.cumsum(y_diff_pred_values)

            # Final prediction = y_last + residual_pred
            final_pred = y_last + residual_pred

            # Compute MAPE using original y_true
            origin = pd.to_datetime(future_rows["ds"].iloc[0])
            mape = safe_mape(y_true, final_pred)
            origin_records.append({"ds_origin": origin, "mape": mape})

            diagnostics[origin] = {
                "y_true": y_true,
                "y_pred": final_pred,
                "ds": ds_forecast,
                "cutoff": cutoff_date,
            }

        return PhaseOutput(
            origin_mape_df=pd.DataFrame(origin_records),
            diagnostics=diagnostics,
        )

    def _build_forecaster(self) -> Any:
        """构建 MLForecast forecaster for difference prediction."""
        from mlforecast import MLForecast

        model_params = dict(self.context.model_spec.model_params)
        models = model_params.get("models")

        # Use shorter lags since we're predicting differences on limited history
        lags = [1, 2, 3]
        lag_transforms = {}
        date_features = ["dayofweek"]
        num_threads = model_params.get("num_threads", 1)

        return MLForecast(
            models=models,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            num_threads=num_threads,
            freq=self.context.run_config.freq,
        )