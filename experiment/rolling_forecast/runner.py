from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import ExperimentResult, ModelSpec, RunConfig
from ..data import PreparedDataset
from .artifacts import finalize_phase
from .executors import BaseExecutor, MLExecutor, NeuralExecutor, StatsExecutor
from .runtime import set_random_seed, suppress_lightning_logs
from .types import ExecutionContext, SplitData


class RollingForecastRunner:
    """rolling forecast 的统一实验入口，负责切分数据、选择执行器、运行评估并汇总结果。"""

    def __init__(self, prepared_dataset: PreparedDataset, run_config: RunConfig):
        self.prepared_dataset = PreparedDataset(
            df=prepared_dataset.df.sort_values(["unique_id", "ds"]).reset_index(drop=True),
            futr_exog=list(prepared_dataset.futr_exog),
            csv_path=prepared_dataset.csv_path,
            unique_id=prepared_dataset.unique_id,
        )
        self.run_config = run_config

    def run(self, model_spec: ModelSpec, run_name: Optional[str] = None) -> ExperimentResult:
        if self._should_skip(model_spec):
            return ExperimentResult.skipped_result(
                model_name=model_spec.name,
                reason=(
                    f"{model_spec.name} does not support future exogenous variables "
                    "while RunConfig.use_exog=True."
                ),
                requested_use_exog=self.run_config.use_exog,
            )

        futr_exog = self._resolve_future_exog(model_spec)
        split_data = self._split_dataset(self.prepared_dataset.df)
        artifact_dir = self._create_artifact_dir(
            model_name=run_name or model_spec.name,
            used_exog=bool(futr_exog),
        )
        context = ExecutionContext(
            dataset=self.prepared_dataset,
            run_config=self.run_config,
            model_spec=model_spec,
            futr_exog=futr_exog,
            future_cols=["unique_id", "ds"] + futr_exog,
            artifact_dir=artifact_dir,
            split_data=split_data,
        )

        set_random_seed(self.run_config.random_seed)
        executor = self._build_executor(context)

        with suppress_lightning_logs():
            output = executor.run()

        val_phase = finalize_phase(
            context=context,
            phase_name="val",
            target_df=split_data.val,
            phase_output=output.val_phase,
        )
        test_phase = finalize_phase(
            context=context,
            phase_name="test",
            target_df=split_data.test,
            phase_output=output.test_phase,
        )

        return ExperimentResult(
            model_name=model_spec.name,
            val_origin_mape_df=val_phase.origin_mape_df,
            val_overall_mape=val_phase.overall_mape,
            val_best_origin=val_phase.best_origin,
            val_worst_origin=val_phase.worst_origin,
            origin_mape_df=test_phase.origin_mape_df,
            overall_mape=test_phase.overall_mape,
            best_origin=test_phase.best_origin,
            worst_origin=test_phase.worst_origin,
            artifact_dir=str(artifact_dir),
            best_model_path=output.best_model_path,
            metrics_path=output.metrics_path,
            loss_plot_path=output.loss_plot_path,
            forecast_plot_path=test_phase.forecast_plot_path,
            rolling_raw_path=test_phase.rolling_raw_path,
            skipped=False,
            skip_reason=None,
            used_exog=bool(futr_exog),
            requested_use_exog=self.run_config.use_exog,
            val_diagnostics=val_phase.diagnostics,
            diagnostics=test_phase.diagnostics,
            train_size=split_data.n_train,
            val_size=split_data.n_val,
            test_size=split_data.n_test,
            val_n_origins=len(split_data.val_origins),
            n_origins=len(split_data.test_origins),
        )

    def _build_executor(self, context: ExecutionContext) -> BaseExecutor:
        """根据模型类型构造对应的执行器实例。"""

        executor_map: dict[str, type[BaseExecutor]] = {
            "stats": StatsExecutor,
            "ml": MLExecutor,
            "neural": NeuralExecutor,
        }
        return executor_map[context.model_spec.model_type](context=context)

    def _should_skip(self, model_spec: ModelSpec) -> bool:
        return (
            self.run_config.use_exog
            and bool(self.prepared_dataset.futr_exog)
            and model_spec.model_type in {"ml", "neural"}
            and not model_spec.supports_future_exog
        )

    def _resolve_future_exog(self, model_spec: ModelSpec) -> list[str]:
        if not self.run_config.use_exog:
            return []
        if model_spec.model_type not in {"ml", "neural"}:
            return []
        if not model_spec.supports_future_exog:
            return []
        return list(self.prepared_dataset.futr_exog)

    def _split_dataset(self, df: pd.DataFrame) -> SplitData:
        """按 train / val / test 顺序切分数据，并为 val 和 test 构造 rolling 起点。"""

        n_total = len(df)
        train_ratio, _, test_ratio = self.run_config.normalized_split_ratio()
        n_train = int(n_total * train_ratio)
        n_test = int(n_total * test_ratio)
        n_val = n_total - n_train - n_test

        if min(n_train, n_val, n_test) <= 0:
            raise ValueError("train/val/test split produced a non-positive subset length")
        if n_train < self.run_config.input_size:
            raise ValueError("train_size must be >= input_size")
        if n_val < self.run_config.horizon:
            raise ValueError("val_size must be >= horizon")
        if n_test < self.run_config.horizon:
            raise ValueError("test_size must be >= horizon")

        train = df.iloc[:n_train].copy()
        test = df.iloc[n_total - n_test :].copy()
        val = df.iloc[n_train : n_total - n_test].copy()
        trainval = pd.concat([train, val], ignore_index=True)

        val_origins = self._build_origins(n_points=n_val, phase_name="val")
        test_origins = self._build_origins(n_points=n_test, phase_name="test")

        return SplitData(
            train=train,
            val=val,
            test=test,
            trainval=trainval,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            val_origins=val_origins,
            test_origins=test_origins,
        )

    def _build_origins(self, n_points: int, phase_name: str) -> list[int]:
        origins = list(
            range(0, n_points - self.run_config.horizon + 1, self.run_config.sliding_step_size)
        )
        if not origins:
            raise ValueError(
                f"No valid rolling {phase_name} origins. "
                "Check horizon, split ratio and sliding_step_size."
            )
        return origins

    def _create_artifact_dir(self, model_name: str, used_exog: bool) -> Path:
        """为当前模型运行创建独立的 artifact 目录。"""

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"{model_name}_{'feat' if used_exog else 'no_feat'}"
        artifact_dir = Path(self.run_config.save_dir) / run_tag / run_stamp
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir
