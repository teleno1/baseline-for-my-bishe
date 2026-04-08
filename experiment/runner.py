from __future__ import annotations

import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentResult, ModelSpec, RunConfig
from .data import PreparedDataset


@contextmanager
# 临时压低 Lightning 训练日志，避免 notebook/终端输出过于冗长。
def suppress_lightning_logs():
    lightning_logger = logging.getLogger("lightning")
    pl_logger = logging.getLogger("pytorch_lightning")

    old_lightning_level = lightning_logger.level
    old_pl_level = pl_logger.level

    lightning_logger.setLevel(logging.ERROR)
    pl_logger.setLevel(logging.ERROR)

    try:
        yield
    finally:
        lightning_logger.setLevel(old_lightning_level)
        pl_logger.setLevel(old_pl_level)


@dataclass(frozen=True)
# 保存一次固定切分后的 train / val / test 及其派生信息。
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    trainval: pd.DataFrame
    n_train: int
    n_val: int
    n_test: int
    val_origins: list[int]
    test_origins: list[int]

    @property
    def origins(self) -> list[int]:
        return self.test_origins


@dataclass(frozen=True)
# 将一次实验运行所需的公共上下文集中打包，便于不同执行器复用。
class ExecutionContext:
    dataset: PreparedDataset
    run_config: RunConfig
    model_spec: ModelSpec
    futr_exog: list[str]
    history_cols: list[str]
    future_cols: list[str]
    artifact_dir: Path
    split_data: SplitData

    @property
    def model_name(self) -> str:
        return self.model_spec.name


@dataclass
# 单个阶段（val 或 test）的原始 rolling 结果。
class PhaseOutput:
    origin_mape_df: pd.DataFrame
    diagnostics: dict[pd.Timestamp, dict[str, Any]]


@dataclass
# 执行器返回的中间结果，最终会再被封装成 ExperimentResult。
class ExecutorOutput:
    val_phase: PhaseOutput
    test_phase: PhaseOutput
    best_model_path: Optional[str] = None
    metrics_path: Optional[str] = None
    loss_plot_path: Optional[str] = None


@dataclass(frozen=True)
# runner 汇总后的阶段结果，包含表格、指标和图路径。
class FinalizedPhase:
    origin_mape_df: pd.DataFrame
    overall_mape: float
    best_origin: Optional[pd.Timestamp]
    worst_origin: Optional[pd.Timestamp]
    diagnostics: dict[pd.Timestamp, dict[str, Any]]
    forecast_plot_path: Optional[str]
    rolling_raw_path: Optional[str]


# 三类执行器的公共基类，主要提供上下文和通用辅助方法。
class BaseExecutor:
    def __init__(self, runner: "RollingForecastRunner", context: ExecutionContext):
        self.runner = runner
        self.context = context

    @property
    def model_name(self) -> str:
        return self.context.model_name

    # 不同库输出的预测列名可能不同，这里统一负责定位真正的预测列。
    def _prediction_column(self, df: pd.DataFrame, excluded: set[str]) -> str:
        if self.model_name in df.columns:
            return self.model_name

        candidate_columns = [column for column in df.columns if column not in excluded]
        if len(candidate_columns) == 1:
            return candidate_columns[0]
        raise ValueError(
            f"Unable to determine prediction column for {self.model_name}. "
            f"Available columns: {df.columns.tolist()}"
        )

    def run(self) -> ExecutorOutput:
        raise NotImplementedError


class StatsExecutor(BaseExecutor):
    # 统计模型分别对 val 和 test 跑独立的 cross_validation。
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

        sf = StatsForecast(
            models=[self.context.model_spec.model_cls(**self.context.model_spec.model_params)],
            freq=self.context.run_config.freq,
            n_jobs=1,
        )
        cv_df = sf.cross_validation(
            df=df,
            h=self.context.run_config.horizon,
            test_size=test_size,
            step_size=self.context.run_config.sliding_step_size,
            refit=False,
        )
        prediction_column = self._prediction_column(
            cv_df,
            excluded={"unique_id", "ds", "cutoff", "y"},
        )
        origin_mape_df, diagnostics = self.runner._summarize_cv_predictions(
            cv_df=cv_df,
            prediction_column=prediction_column,
        )
        return PhaseOutput(origin_mape_df=origin_mape_df, diagnostics=diagnostics)


class MLExecutor(BaseExecutor):
    # 机器学习模型显式分成 train -> val -> test 两阶段评估。
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
        feature_count = input_size + len(self.context.futr_exog)
        y_values = df["y"].to_numpy(dtype=float)
        exog_values = (
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
            if exog_values is not None:
                feature_row.extend(exog_values[idx].tolist())
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
            y_pred = self._recursive_predict(model=model, observed_df=observed_df, future_df=future_df)
            ds_forecast = pd.to_datetime(future_df["ds"]).to_numpy()
            origin = pd.to_datetime(target_df.iloc[origin_offset]["ds"])
            origin_records.append(
                {"ds_origin": origin, "mape": self.runner._safe_mape(y_true, y_pred)}
            )
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

        future_exog = (
            future_df[self.context.futr_exog].to_numpy(dtype=float)
            if self.context.futr_exog
            else None
        )
        predictions: list[float] = []
        for step in range(len(future_df)):
            lag_features = np.asarray(history_values[-input_size:][::-1], dtype=float)
            if future_exog is not None:
                features = np.concatenate([lag_features, future_exog[step]])
            else:
                features = lag_features
            prediction = model.predict(features.reshape(1, -1))
            prediction_value = float(np.asarray(prediction).reshape(-1)[0])
            predictions.append(prediction_value)
            history_values.append(prediction_value)

        return np.asarray(predictions, dtype=float)


class NeuralExecutor(BaseExecutor):
    # 神经网络执行器除了训练和 checkpoint，还会分别产出 val/test 的 rolling 结果。
    def run(self) -> ExecutorOutput:
        from neuralforecast import NeuralForecast
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger

        checkpoint_dir = self.context.artifact_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        csv_logger = CSVLogger(
            save_dir=str(self.context.artifact_dir),
            name="logs",
            version="main",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="ptl/val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )

        neural_params = dict(self.context.model_spec.model_params)
        neural_params.setdefault("alias", self.model_name)
        neural_params.setdefault("enable_progress_bar", False)
        neural_params.setdefault("enable_model_summary", False)
        trainer_callbacks = list(neural_params.pop("callbacks", []))
        trainer_logger = neural_params.pop("logger", None)
        trainer_callbacks.append(checkpoint_callback)

        model = self.context.model_spec.model_cls(
            h=self.context.run_config.horizon,
            input_size=self.context.run_config.input_size,
            futr_exog_list=self.context.futr_exog,
            loss=self.runner._build_neural_loss(),
            valid_loss=self.runner._build_neural_loss(),
            random_seed=self.context.run_config.random_seed,
            early_stop_patience_steps=self.context.run_config.early_stop_patience_steps,
            val_check_steps=self.context.run_config.val_check_steps,
            enable_checkpointing=True,
            logger=trainer_logger or csv_logger,
            callbacks=trainer_callbacks,
            default_root_dir=str(self.context.artifact_dir),
            **neural_params,
        )

        nf = NeuralForecast(models=[model], freq=self.context.run_config.freq)
        nf.fit(df=self.context.split_data.trainval, val_size=self.context.split_data.n_val)

        checkpoint_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or None
        if checkpoint_path:
            best_model_path = str(checkpoint_path)
        else:
            fallback_checkpoint = checkpoint_dir / "final.ckpt"
            nf.models[0].save(str(fallback_checkpoint))
            best_model_path = str(fallback_checkpoint)

        metrics_path = None
        loss_plot_path = None
        metrics_file = Path(csv_logger.log_dir) / "metrics.csv"
        if metrics_file.exists():
            _, loss_plot_path = self.runner._build_loss_artifact(
                metrics_csv_path=metrics_file,
                artifact_root=self.context.artifact_dir,
                model_name=self.model_name,
            )
            metrics_path = str(metrics_file)

        if best_model_path:
            best_model = self.context.model_spec.model_cls.load(best_model_path)
            best_model.alias = self.model_name
            nf.models[0] = best_model

        nf.models[0] = self._prepare_model_for_prediction(nf.models[0])

        val_phase = self._evaluate_phase(
            nf=nf,
            base_history_df=self.context.split_data.train,
            target_df=self.context.split_data.val,
            origins=self.context.split_data.val_origins,
        )
        test_phase = self._evaluate_phase(
            nf=nf,
            base_history_df=self.context.split_data.trainval,
            target_df=self.context.split_data.test,
            origins=self.context.split_data.test_origins,
        )

        return ExecutorOutput(
            val_phase=val_phase,
            test_phase=test_phase,
            best_model_path=best_model_path,
            metrics_path=metrics_path,
            loss_plot_path=loss_plot_path,
        )

    def _prepare_model_for_prediction(self, model: Any) -> Any:
        if hasattr(model, "callbacks"):
            model.callbacks = []
        for attr in ("trainer_kwargs", "_trainer_kwargs"):
            trainer_kwargs = getattr(model, attr, None)
            if isinstance(trainer_kwargs, dict):
                trainer_kwargs["callbacks"] = []
                trainer_kwargs["logger"] = False
                trainer_kwargs["enable_progress_bar"] = False
                trainer_kwargs["enable_model_summary"] = False
        return model

    def _evaluate_phase(
        self,
        nf: Any,
        base_history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        origins: list[int],
    ) -> PhaseOutput:
        origin_records: list[dict[str, Any]] = []
        diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}

        for origin_offset in origins:
            history = pd.concat(
                [base_history_df, target_df.iloc[:origin_offset]],
                ignore_index=True,
            )
            history_df = history.iloc[-self.context.run_config.input_size :].copy()
            future_df = target_df.iloc[
                origin_offset : origin_offset + self.context.run_config.horizon
            ][self.context.future_cols].copy()
            pred = nf.predict(df=history_df, futr_df=future_df)

            prediction_column = self._prediction_column(pred, excluded={"unique_id", "ds"})
            y_true = target_df.iloc[
                origin_offset : origin_offset + self.context.run_config.horizon
            ]["y"].to_numpy()
            y_pred = pred[prediction_column].to_numpy()
            ds_forecast = pd.to_datetime(
                target_df.iloc[
                    origin_offset : origin_offset + self.context.run_config.horizon
                ]["ds"]
            ).to_numpy()
            origin = pd.to_datetime(target_df.iloc[origin_offset]["ds"])
            origin_records.append(
                {"ds_origin": origin, "mape": self.runner._safe_mape(y_true, y_pred)}
            )
            diagnostics[origin] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "ds": ds_forecast,
                "cutoff": pd.to_datetime(history.iloc[-1]["ds"]),
            }

        return PhaseOutput(
            origin_mape_df=pd.DataFrame(origin_records),
            diagnostics=diagnostics,
        )


# 对 notebook 暴露的统一实验入口：负责调度、切分、执行、汇总和可视化。
class RollingForecastRunner:
    # 初始化时先保证数据按时间排序，避免后续滚动切分出现顺序问题。
    def __init__(self, prepared_dataset: PreparedDataset, run_config: RunConfig):
        self.prepared_dataset = PreparedDataset(
            df=prepared_dataset.df.sort_values(["unique_id", "ds"]).reset_index(drop=True),
            futr_exog=list(prepared_dataset.futr_exog),
            csv_path=prepared_dataset.csv_path,
            unique_id=prepared_dataset.unique_id,
        )
        self.run_config = run_config

    # 这是 notebook 调用的主入口：单次调用对应一个模型实验。
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
            history_cols=["unique_id", "ds", "y"] + futr_exog,
            future_cols=["unique_id", "ds"] + futr_exog,
            artifact_dir=artifact_dir,
            split_data=split_data,
        )

        self._set_random_seed()
        executor = self._build_executor(context)

        with suppress_lightning_logs():
            output = executor.run()

        val_phase = self._finalize_phase(
            context=context,
            phase_name="val",
            target_df=split_data.val,
            phase_output=output.val_phase,
        )
        test_phase = self._finalize_phase(
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

    # 根据模型类型选择对应的执行策略类。
    def _build_executor(self, context: ExecutionContext) -> BaseExecutor:
        executor_map: dict[str, type[BaseExecutor]] = {
            "stats": StatsExecutor,
            "ml": MLExecutor,
            "neural": NeuralExecutor,
        }
        return executor_map[context.model_spec.model_type](runner=self, context=context)

    # 某些模型不支持 future exog，例如 PatchTST，此时直接跳过实验。
    def _should_skip(self, model_spec: ModelSpec) -> bool:
        return (
            self.run_config.use_exog
            and bool(self.prepared_dataset.futr_exog)
            and model_spec.model_type in {"ml", "neural"}
            and not model_spec.supports_future_exog
        )

    # 综合全局开关、模型类型和模型能力，决定本次实验实际使用的外生变量。
    def _resolve_future_exog(self, model_spec: ModelSpec) -> list[str]:
        if not self.run_config.use_exog:
            return []
        if model_spec.model_type not in {"ml", "neural"}:
            return []
        if not model_spec.supports_future_exog:
            return []
        return list(self.prepared_dataset.futr_exog)

    # 按“先 train、再 test、剩余给 val”的顺序切分，并分别生成 val/test 上的 rolling origins。
    def _split_dataset(self, df: pd.DataFrame) -> SplitData:
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

    # 根据 horizon 和滑动步长，为某个阶段生成全部 rolling origins。
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

    # 将单个阶段的原始结果汇总成表格、指标和图路径。
    def _finalize_phase(
        self,
        context: ExecutionContext,
        phase_name: str,
        target_df: pd.DataFrame,
        phase_output: PhaseOutput,
    ) -> FinalizedPhase:
        origin_mape_df = phase_output.origin_mape_df.sort_values("ds_origin").reset_index(drop=True)
        rolling_raw_path = None
        if phase_name == "test":
            rolling_raw_df = self._build_rolling_raw_df(
                origin_mape_df=origin_mape_df,
                diagnostics=phase_output.diagnostics,
            )
            rolling_raw_path = self._save_rolling_raw_artifact(
                rolling_raw_df=rolling_raw_df,
                artifact_dir=context.artifact_dir,
            )

        if origin_mape_df.empty:
            overall_mape = float("nan")
            best_origin = None
            worst_origin = None
        else:
            overall_mape = float(origin_mape_df["mape"].mean())
            best_origin = pd.to_datetime(
                origin_mape_df.loc[origin_mape_df["mape"].idxmin(), "ds_origin"]
            )
            worst_origin = pd.to_datetime(
                origin_mape_df.loc[origin_mape_df["mape"].idxmax(), "ds_origin"]
            )

        forecast_plot_path = None
        if phase_name == "test":
            forecast_plot_path = self._plot_forecast(
                context=context,
                phase_name=phase_name,
                target_df=target_df,
                diagnostics=phase_output.diagnostics,
                overall_mape=overall_mape,
                best_origin=best_origin,
                worst_origin=worst_origin,
            )

        return FinalizedPhase(
            origin_mape_df=origin_mape_df,
            overall_mape=overall_mape,
            best_origin=best_origin,
            worst_origin=worst_origin,
            diagnostics=phase_output.diagnostics,
            forecast_plot_path=forecast_plot_path,
            rolling_raw_path=rolling_raw_path,
        )

    def _build_rolling_raw_df(
        self,
        origin_mape_df: pd.DataFrame,
        diagnostics: dict[pd.Timestamp, dict[str, Any]],
    ) -> pd.DataFrame:
        columns = [
            "origin_id",
            "origin_date",
            "horizon",
            "target_date",
            "y_true",
            "y_pred",
            "error",
        ]
        raw_records: list[dict[str, Any]] = []

        for origin_id, ds_origin in enumerate(pd.to_datetime(origin_mape_df["ds_origin"]), start=1):
            origin = pd.Timestamp(ds_origin)
            if origin not in diagnostics:
                raise KeyError(f"Missing diagnostics for rolling origin {origin}")

            diagnostic = diagnostics[origin]
            cutoff = diagnostic.get("cutoff")
            if cutoff is None:
                raise ValueError(f"Missing cutoff for rolling origin {origin}")

            y_true = np.asarray(diagnostic["y_true"], dtype=float)
            y_pred = np.asarray(diagnostic["y_pred"], dtype=float)
            ds_forecast = pd.to_datetime(diagnostic["ds"])
            if len(y_true) != len(y_pred) or len(y_true) != len(ds_forecast):
                raise ValueError(f"Inconsistent rolling diagnostics for origin {origin}")

            origin_date = pd.to_datetime(cutoff).strftime("%Y-%m-%d")
            for horizon_idx, (target_date, true_value, pred_value) in enumerate(
                zip(ds_forecast, y_true, y_pred),
                start=1,
            ):
                raw_records.append(
                    {
                        "origin_id": origin_id,
                        "origin_date": origin_date,
                        "horizon": horizon_idx,
                        "target_date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
                        "y_true": float(true_value),
                        "y_pred": float(pred_value),
                        "error": float(pred_value - true_value),
                    }
                )

        return pd.DataFrame(raw_records, columns=columns)

    def _save_rolling_raw_artifact(
        self,
        rolling_raw_df: pd.DataFrame,
        artifact_dir: Path,
    ) -> str:
        rolling_raw_path = artifact_dir / "rolling_test_raw.csv"
        rolling_raw_df.to_csv(rolling_raw_path, index=False, encoding="utf-8-sig")
        return str(rolling_raw_path)

    # 为每次实验创建独立产物目录，便于保存图、日志和模型文件。
    def _create_artifact_dir(self, model_name: str, used_exog: bool) -> Path:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"{model_name}_{'feat' if used_exog else 'no_feat'}"
        artifact_dir = Path(self.run_config.save_dir) / run_tag / run_stamp
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    # 统一设置 random / numpy / torch / lightning 的随机种子。
    # Build the neural training/validation loss from notebook hyperparameters.
    def _build_neural_loss(self):
        from neuralforecast.losses import pytorch as loss_module

        loss_registry = {
            "MAE": loss_module.MAE,
            "MSE": loss_module.MSE,
            "RMSE": loss_module.RMSE,
            "MAPE": loss_module.MAPE,
            "SMAPE": loss_module.SMAPE,
            "HuberLoss": loss_module.HuberLoss,
            "TukeyLoss": loss_module.TukeyLoss,
            "MASE": loss_module.MASE,
        }

        loss_name = self.run_config.neural_loss_name
        loss_params = dict(self.run_config.neural_loss_params)
        if loss_name == "MASE" and "seasonality" not in loss_params:
            raise ValueError(
                "MASE requires NEURAL_LOSS_PARAMS to include 'seasonality', "
                "for example {'seasonality': 7}."
            )

        loss_cls = loss_registry[loss_name]
        try:
            return loss_cls(**loss_params)
        except TypeError as exc:
            raise ValueError(
                f"Failed to initialize neural loss {loss_name!r} with params {loss_params}. "
                "Update NEURAL_LOSS_PARAMS in the notebook to match the selected loss constructor."
            ) from exc

    def _set_random_seed(self) -> None:
        import torch
        from pytorch_lightning import seed_everything

        seed_everything(self.run_config.random_seed, workers=True)
        random.seed(self.run_config.random_seed)
        np.random.seed(self.run_config.random_seed)
        torch.manual_seed(self.run_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.run_config.random_seed)

    # 绘制某个阶段的真实值、最佳窗口预测和最差窗口预测。
    def _plot_forecast(
        self,
        context: ExecutionContext,
        phase_name: str,
        target_df: pd.DataFrame,
        diagnostics: dict[pd.Timestamp, dict[str, Any]],
        overall_mape: float,
        best_origin: Optional[pd.Timestamp],
        worst_origin: Optional[pd.Timestamp],
    ) -> Optional[str]:
        if not self.run_config.plot_forecast or best_origin is None or worst_origin is None:
            return None

        phase_label = "Validation" if phase_name == "val" else "Test"
        file_name = "val_forecast_plot.png" if phase_name == "val" else "forecast_plot.png"

        best = diagnostics[best_origin]
        worst = diagnostics[worst_origin]
        best_mape = self._safe_mape(best["y_true"], best["y_pred"])
        worst_mape = self._safe_mape(worst["y_true"], worst["y_pred"])

        plt.figure(figsize=(14, 5))
        plt.plot(
            target_df["ds"],
            target_df["y"],
            color="black",
            linewidth=2,
            label=f"True Load ({phase_label} Set)",
        )
        plt.plot(
            best["ds"],
            best["y_pred"],
            marker="o",
            linestyle="--",
            label=(
                f"Best Forecast (origin={best_origin.date()}, "
                f"MAPE={best_mape:.2f}%)"
            ),
        )
        plt.plot(
            worst["ds"],
            worst["y_pred"],
            marker="o",
            linestyle="--",
            label=(
                f"Worst Forecast (origin={worst_origin.date()}, "
                f"MAPE={worst_mape:.2f}%)"
            ),
        )
        plt.title(
            f"Rolling {phase_label} Forecast ({context.model_name})\n"
            f"Horizon={self.run_config.horizon}, Mean Origin MAPE={overall_mape:.2f}%"
        )
        plt.xlabel("Date")
        plt.ylabel("Daily Electricity Load")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        forecast_plot_path = None
        if self.run_config.save_plots:
            forecast_plot_path = context.artifact_dir / file_name
            plt.savefig(forecast_plot_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        return str(forecast_plot_path) if forecast_plot_path else None

    # 不同版本日志列名可能不同，这里从候选列里挑出实际存在的一列。
    def _choose_metric_column(
        self,
        metrics_df: pd.DataFrame,
        candidates: list[str],
    ) -> Optional[str]:
        for column in candidates:
            if column in metrics_df.columns and metrics_df[column].notna().any():
                return column
        return None

    # 从 Lightning 生成的 metrics.csv 中提取训练/验证损失并绘制 loss 图。
    def _build_loss_artifact(
        self,
        metrics_csv_path: Path,
        artifact_root: Path,
        model_name: str,
    ) -> tuple[pd.DataFrame, Optional[str]]:
        metrics_df = pd.read_csv(metrics_csv_path)
        step_col = "step" if "step" in metrics_df.columns else None
        train_col = self._choose_metric_column(
            metrics_df,
            ["train_loss_epoch", "train_loss", "ptl/train_loss", "train_loss_step"],
        )
        valid_col = self._choose_metric_column(
            metrics_df,
            ["valid_loss", "ptl/val_loss", "val_loss"],
        )

        loss_plot_path = None
        if self.run_config.plot_loss and (train_col or valid_col):
            plt.figure(figsize=(10, 4))
            if train_col is not None:
                train_curve = metrics_df[[train_col] + ([step_col] if step_col else [])].dropna()
                train_x = train_curve[step_col] if step_col else np.arange(len(train_curve))
                plt.plot(train_x, train_curve[train_col], label="Train Loss")
            if valid_col is not None:
                valid_curve = metrics_df[[valid_col] + ([step_col] if step_col else [])].dropna()
                valid_x = valid_curve[step_col] if step_col else np.arange(len(valid_curve))
                plt.plot(valid_x, valid_curve[valid_col], label="Valid Loss")
            plt.title(f"Training Curve ({model_name})")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if self.run_config.save_plots:
                loss_plot_path = artifact_root / "loss_curve.png"
                plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
            plt.show()
            plt.close()

        return metrics_df, str(loss_plot_path) if loss_plot_path else None

    # 将 StatsForecast 的 cross_validation 输出整理成统一的 origin 级结果。
    def _summarize_cv_predictions(
        self,
        cv_df: pd.DataFrame,
        prediction_column: str,
    ) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
        origin_records: list[dict[str, Any]] = []
        diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}

        for cutoff, group in cv_df.groupby("cutoff"):
            group = group.sort_values("ds")
            y_true = group["y"].to_numpy()
            y_pred = group[prediction_column].to_numpy()
            origin = pd.to_datetime(group["ds"].iloc[0])
            origin_records.append({"ds_origin": origin, "mape": self._safe_mape(y_true, y_pred)})
            diagnostics[origin] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "ds": pd.to_datetime(group["ds"]).to_numpy(),
                "cutoff": pd.to_datetime(cutoff),
            }

        return pd.DataFrame(origin_records), diagnostics

    # 对分母接近 0 的情况做保护，避免 MAPE 出现数值问题。
    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_values = np.asarray(y_true, dtype=float)
        pred_values = np.asarray(y_pred, dtype=float)
        denom = np.where(np.abs(true_values) < 1e-8, 1e-8, np.abs(true_values))
        return float(np.mean(np.abs((true_values - pred_values) / denom)) * 100)
