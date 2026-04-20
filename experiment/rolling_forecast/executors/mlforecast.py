from __future__ import annotations

from collections.abc import Sized
from typing import Any

import pandas as pd

from ..metrics import summarize_cv_predictions
from ..types import ExecutorOutput, PhaseOutput
from .base import BaseExecutor


class MLForecastExecutor(BaseExecutor):
    """Run a Nixtla MLForecast pipeline through the shared rolling evaluator."""

    def run(self) -> ExecutorOutput:
        self._validate_supported_exog()
        val_phase = self._cross_validate_phase(
            df=self.context.split_data.trainval,
            origins=self.context.split_data.val_origins,
        )
        test_phase = self._cross_validate_phase(
            df=self.context.dataset.df,
            origins=self.context.split_data.test_origins,
        )
        return ExecutorOutput(val_phase=val_phase, test_phase=test_phase)

    def _validate_supported_exog(self) -> None:
        if self.context.hist_exog:
            raise ValueError(
                "MLForecastExecutor only supports no-exog or future-known exogenous "
                "features. Historical exogenous columns should be disabled or converted "
                "into lagged features before using model_type='mlforecast'."
            )

    def _cross_validate_phase(self, df: pd.DataFrame, origins: list[int]) -> PhaseOutput:
        cv_input = self._select_cv_columns(df)
        forecaster = self._build_forecaster()
        cv_df = forecaster.cross_validation(
            df=cv_input,
            h=self.context.run_config.horizon,
            n_windows=len(origins),
            step_size=self.context.run_config.sliding_step_size,
            refit=False,
            static_features=[],
        )
        prediction_column = self._prediction_column(
            cv_df,
            excluded={"unique_id", "ds", "cutoff", "y"},
        )
        origin_mape_df, diagnostics = summarize_cv_predictions(
            cv_df=cv_df,
            prediction_column=prediction_column,
        )
        return PhaseOutput(origin_mape_df=origin_mape_df, diagnostics=diagnostics)

    def _select_cv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = ["unique_id", "ds", "y"] + list(self.context.futr_exog)
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"MLForecast input is missing required columns: {missing}")
        return df[columns].copy()

    def _build_forecaster(self) -> Any:
        model_params = dict(self.context.model_spec.model_params)
        self._validate_single_mlforecast_model(model_params.get("models"))
        model_params.setdefault("freq", self.context.run_config.freq)
        return self.context.model_spec.model_cls(**model_params)

    def _validate_single_mlforecast_model(self, models: Any) -> None:
        if models is None:
            raise ValueError("MLForecast ModelSpec.model_params must include a 'models' entry.")
        if isinstance(models, dict):
            model_count = len(models)
        elif isinstance(models, (str, bytes)):
            model_count = 1
        elif isinstance(models, Sized) and not hasattr(models, "fit"):
            model_count = len(models)
        else:
            model_count = 1

        if model_count != 1:
            raise ValueError(
                "MLForecastExecutor expects exactly one estimator per ModelSpec "
                "so prediction columns remain unambiguous."
            )
