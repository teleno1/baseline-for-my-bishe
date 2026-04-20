from __future__ import annotations

from typing import Any

import pandas as pd

from ..metrics import summarize_cv_predictions
from ..types import ExecutorOutput, PhaseOutput
from .base import BaseExecutor


class MLForecastExecutor(BaseExecutor):
    def run(self) -> ExecutorOutput:
        val_df = pd.concat(
            [self.context.split_data.train, self.context.split_data.val],
            ignore_index=True,
        )
        test_df = pd.concat(
            [self.context.split_data.trainval, self.context.split_data.test],
            ignore_index=True,
        )
        return ExecutorOutput(
            val_phase=self._cross_validate_phase(
                df=val_df,
                origins=self.context.split_data.val_origins,
            ),
            test_phase=self._cross_validate_phase(
                df=test_df,
                origins=self.context.split_data.test_origins,
            ),
        )

    def _build_forecaster(self) -> Any:
        params = dict(self.context.model_spec.model_params)
        models = params.get("models")
        if isinstance(models, (list, tuple)):
            if len(models) != 1:
                raise ValueError("MLForecastExecutor requires exactly one estimator in model_params['models']")
            params["models"] = models[0]
        params.setdefault("freq", self.context.run_config.freq)
        return self.context.model_spec.model_cls(**params)

    def _cross_validate_phase(self, df: pd.DataFrame, origins: list[int]) -> PhaseOutput:
        if not origins:
            return PhaseOutput(
                origin_mape_df=pd.DataFrame(columns=["ds_origin", "mape"]),
                diagnostics={},
            )

        forecaster = self._build_forecaster()
        prediction_df = forecaster.cross_validation(
            df=self._training_frame(df),
            h=self.context.run_config.horizon,
            n_windows=len(origins),
            step_size=self.context.run_config.sliding_step_size,
            refit=False,
            static_features=[],
        )
        origin_mape_df, diagnostics = summarize_cv_predictions(
            cv_df=prediction_df,
            prediction_column=self.context.model_name,
        )
        return PhaseOutput(origin_mape_df=origin_mape_df, diagnostics=diagnostics)

    def _training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = ["unique_id", "ds", "y", *self.context.futr_exog]
        return df.loc[:, columns].copy()

__all__ = ["MLForecastExecutor"]
