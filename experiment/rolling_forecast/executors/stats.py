from __future__ import annotations

import pandas as pd

from ..metrics import summarize_cv_predictions
from ..types import ExecutorOutput, PhaseOutput
from .base import BaseExecutor


class StatsExecutor(BaseExecutor):
    """统计模型在 val 和 test 阶段分别运行 cross_validation。"""

    def run(self) -> ExecutorOutput:
        val_phase = self._cross_validate_phase(
            df=self.context.split_data.trainval[["unique_id", "ds", "y"]],
            test_size=self.context.split_data.n_val,
        )
        test_phase = self._cross_validate_phase(
            df=self.context.dataset.df[["unique_id", "ds", "y"]],
            test_size=self.context.split_data.n_test,
        )
        return ExecutorOutput(val_phase=val_phase, test_phase=test_phase)

    def _cross_validate_phase(self, df: pd.DataFrame, test_size: int) -> PhaseOutput:
        from statsforecast import StatsForecast

        model = self.context.model_spec.model_cls(**self.context.model_spec.model_params)
        sf = StatsForecast(
            models=[model],
            freq=self.context.run_config.freq,
            n_jobs=1,
        )
        cv_df = sf.cross_validation(
            df=df,
            h=self.context.run_config.horizon,
            test_size=test_size,
            step_size=self.context.run_config.sliding_step_size,
            refit=self._cross_validation_refit(model),
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

    @staticmethod
    def _cross_validation_refit(model: object) -> bool:
        return False if hasattr(model, "forward") else True
