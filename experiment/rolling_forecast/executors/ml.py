from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..metrics import safe_mape
from ..types import ExecutorOutput, PhaseOutput
from .base import BaseExecutor


class MLExecutor(BaseExecutor):
    def run(self) -> ExecutorOutput:
        self._validate_single_series()
        model = self._fit_model()
        val_phase = self._evaluate_phase(
            model=model,
            base_history_df=self.context.split_data.train,
            target_df=self.context.split_data.val,
            origins=self.context.split_data.val_origins,
        )
        test_phase = self._evaluate_phase(
            model=model,
            base_history_df=self.context.split_data.trainval,
            target_df=self.context.split_data.test,
            origins=self.context.split_data.test_origins,
        )
        return ExecutorOutput(val_phase=val_phase, test_phase=test_phase)

    def _validate_single_series(self) -> None:
        if self.context.dataset.df["unique_id"].nunique() != 1:
            raise NotImplementedError("MLExecutor currently supports a single unique_id series.")

    def _fit_model(self):
        model = self.context.model_spec.model_cls(**self.context.model_spec.model_params)
        x_train, y_train = self._build_supervised_arrays(
            df=self.context.split_data.train,
            target_start_idx=0,
        )
        if len(x_train) == 0:
            raise ValueError("Training split does not produce any supervised ML samples.")

        trainval_df = self.context.split_data.trainval.reset_index(drop=True)
        x_val_fit, y_val_fit = self._build_supervised_arrays(
            df=trainval_df,
            target_start_idx=self.context.split_data.n_train,
        )

        if self._is_lightgbm_model(model):
            return self._fit_lightgbm_model(model, x_train, y_train, x_val_fit, y_val_fit)

        model.fit(x_train, y_train)
        return model

    def _build_supervised_arrays(
        self,
        df: pd.DataFrame,
        target_start_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        input_size = self.context.run_config.input_size
        feature_count = (
            input_size
            + input_size * len(self.context.hist_exog)
            + len(self.context.futr_exog)
        )
        y_values = df["y"].to_numpy(dtype=float)
        hist_exog_values = (
            df[self.context.hist_exog].to_numpy(dtype=float)
            if self.context.hist_exog
            else None
        )
        futr_exog_values = (
            df[self.context.futr_exog].to_numpy(dtype=float)
            if self.context.futr_exog
            else None
        )
        start_idx = max(input_size, target_start_idx)
        if start_idx >= len(df):
            return np.empty((0, feature_count), dtype=float), np.empty((0,), dtype=float)

        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        for idx in range(start_idx, len(df)):
            feature_row = y_values[idx - input_size : idx][::-1].tolist()
            if hist_exog_values is not None:
                feature_row.extend(
                    hist_exog_values[idx - input_size : idx][::-1].reshape(-1).tolist()
                )
            if futr_exog_values is not None:
                feature_row.extend(futr_exog_values[idx].tolist())
            x_rows.append(feature_row)
            y_rows.append(float(y_values[idx]))

        return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)

    def _is_lightgbm_model(self, model: Any) -> bool:
        module_name = getattr(model.__class__, "__module__", "")
        class_name = getattr(model.__class__, "__name__", "")
        return module_name.startswith("lightgbm") or class_name.startswith("LGBM")

    def _fit_lightgbm_model(
        self,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        fit_kwargs: dict[str, Any] = {}
        if len(x_val) > 0:
            fit_kwargs["eval_set"] = [(x_val, y_val)]

        rounds = self.context.run_config.ml_early_stopping_rounds
        if rounds and rounds > 0 and len(x_val) > 0:
            try:
                from lightgbm import early_stopping, log_evaluation

                fit_kwargs["callbacks"] = [
                    early_stopping(rounds, verbose=False),
                    log_evaluation(period=0),
                ]
                model.fit(x_train, y_train, **fit_kwargs)
                return model
            except TypeError:
                fit_kwargs.pop("callbacks", None)
                fit_kwargs["early_stopping_rounds"] = rounds
            except ImportError:
                fit_kwargs["early_stopping_rounds"] = rounds

        model.fit(x_train, y_train, **fit_kwargs)
        return model

    def _evaluate_phase(
        self,
        model: Any,
        base_history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        origins: list[int],
    ) -> PhaseOutput:
        origin_records: list[dict[str, Any]] = []
        diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}

        for origin_offset in origins:
            observed_df = pd.concat(
                [base_history_df, target_df.iloc[:origin_offset]],
                ignore_index=True,
            )
            future_df = target_df.iloc[
                origin_offset : origin_offset + self.context.run_config.horizon
            ].copy()
            y_true = future_df["y"].to_numpy(dtype=float)
            y_pred = self._recursive_predict(
                model=model,
                observed_df=observed_df,
                future_df=future_df,
            )
            ds_forecast = pd.to_datetime(future_df["ds"]).to_numpy()
            origin = pd.to_datetime(target_df.iloc[origin_offset]["ds"])
            origin_records.append({"ds_origin": origin, "mape": safe_mape(y_true, y_pred)})
            diagnostics[origin] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "ds": ds_forecast,
                "cutoff": pd.to_datetime(observed_df.iloc[-1]["ds"]),
            }

        return PhaseOutput(
            origin_mape_df=pd.DataFrame(origin_records),
            diagnostics=diagnostics,
        )

    def _recursive_predict(
        self,
        model: Any,
        observed_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> np.ndarray:
        input_size = self.context.run_config.input_size
        history_values = observed_df["y"].astype(float).tolist()
        if len(history_values) < input_size:
            raise ValueError("Observed history is shorter than input_size.")

        hist_exog_window = self._build_hist_exog_window(observed_df)
        future_exog = (
            future_df[self.context.futr_exog].to_numpy(dtype=float)
            if self.context.futr_exog
            else None
        )
        predictions: list[float] = []
        for step in range(len(future_df)):
            lag_features = np.asarray(history_values[-input_size:][::-1], dtype=float)
            feature_parts = [lag_features]
            if hist_exog_window is not None:
                feature_parts.append(hist_exog_window)
            if future_exog is not None:
                feature_parts.append(future_exog[step])
            features = np.concatenate(feature_parts)
            prediction = model.predict(features.reshape(1, -1))
            prediction_value = float(np.asarray(prediction).reshape(-1)[0])
            predictions.append(prediction_value)
            history_values.append(prediction_value)

        return np.asarray(predictions, dtype=float)

    def _build_hist_exog_window(self, observed_df: pd.DataFrame) -> np.ndarray | None:
        if not self.context.hist_exog:
            return None

        input_size = self.context.run_config.input_size
        hist_exog_values = observed_df[self.context.hist_exog].to_numpy(dtype=float)
        return np.asarray(hist_exog_values[-input_size:][::-1].reshape(-1), dtype=float)
