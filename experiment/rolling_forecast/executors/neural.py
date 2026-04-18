from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from ...config import RunConfig
from ..artifacts import build_loss_artifact
from ..metrics import safe_mape
from ..types import ExecutionContext, ExecutorOutput, PhaseOutput
from .base import BaseExecutor

FORECASTER_REGISTRY: dict[str, Any] = {}


def build_neural_loss(run_config: RunConfig):
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

    loss_name = run_config.neural_loss_name
    loss_params = dict(run_config.neural_loss_params)
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


def prepare_model_for_prediction(model: Any) -> Any:
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


def prediction_column_for_model(
    prediction_df: pd.DataFrame,
    model_name: str,
    excluded: set[str],
) -> str:
    if model_name in prediction_df.columns:
        return model_name

    candidate_columns = [column for column in prediction_df.columns if column not in excluded]
    if len(candidate_columns) == 1:
        return candidate_columns[0]
    raise ValueError(
        f"Unable to determine prediction column for {model_name}. "
        f"Available columns: {prediction_df.columns.tolist()}"
    )


def clone_forecaster_for_prediction(source_forecaster: Any, model: Any) -> Any:
    from neuralforecast import NeuralForecast

    prediction_forecaster = NeuralForecast(models=[model], freq=source_forecaster.freq)
    copied_state = dict(source_forecaster.__dict__)
    copied_state["models"] = [model]
    copied_state["_fitted"] = True
    prediction_forecaster.__dict__.update(copied_state)
    return prediction_forecaster


def evaluate_phase_with_forecaster(
    context: ExecutionContext,
    model_name: str,
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
        history_df = history.iloc[-context.run_config.input_size :].copy()
        future_df = target_df.iloc[
            origin_offset : origin_offset + context.run_config.horizon
        ][context.future_cols].copy()
        pred = nf.predict(df=history_df, futr_df=future_df)

        prediction_column = prediction_column_for_model(
            pred,
            model_name=model_name,
            excluded={"unique_id", "ds"},
        )
        y_true = target_df.iloc[
            origin_offset : origin_offset + context.run_config.horizon
        ]["y"].to_numpy(dtype=float)
        y_pred = pred[prediction_column].to_numpy(dtype=float)
        ds_forecast = pd.to_datetime(
            target_df.iloc[
                origin_offset : origin_offset + context.run_config.horizon
            ]["ds"]
        ).to_numpy()
        origin = pd.to_datetime(target_df.iloc[origin_offset]["ds"])
        origin_records.append({"ds_origin": origin, "mape": safe_mape(y_true, y_pred)})
        diagnostics[origin] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_insample": history_df["y"].to_numpy(dtype=float),
            "ds": ds_forecast,
            "cutoff": pd.to_datetime(history.iloc[-1]["ds"]),
        }

    return PhaseOutput(
        origin_mape_df=pd.DataFrame(origin_records),
        diagnostics=diagnostics,
    )


def compute_phase_loss(run_config: RunConfig, phase_output: PhaseOutput) -> float:
    import torch

    if not phase_output.diagnostics:
        return float("nan")

    loss_fn = build_neural_loss(run_config)
    ordered_origins = sorted(phase_output.diagnostics)
    y_true = np.stack(
        [np.asarray(phase_output.diagnostics[origin]["y_true"], dtype=float) for origin in ordered_origins]
    )
    y_pred = np.stack(
        [np.asarray(phase_output.diagnostics[origin]["y_pred"], dtype=float) for origin in ordered_origins]
    )
    y_tensor = torch.tensor(y_true, dtype=torch.float32).unsqueeze(-1)
    y_hat_tensor = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(-1)

    if run_config.neural_loss_name == "MASE":
        y_insample = np.stack(
            [
                np.asarray(phase_output.diagnostics[origin]["y_insample"], dtype=float)
                for origin in ordered_origins
            ]
        )
        loss_value = loss_fn(
            y=y_tensor,
            y_hat=y_hat_tensor,
            y_insample=torch.tensor(y_insample, dtype=torch.float32),
        )
    else:
        loss_value = loss_fn(y=y_tensor, y_hat=y_hat_tensor)

    return float(loss_value.detach().cpu().item())


class RollingTestCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        *,
        context: ExecutionContext,
        model_cls: type,
        model_name: str,
        checkpoint_dir: Path,
    ) -> None:
        super().__init__(
            dirpath=str(checkpoint_dir),
            filename="test_best",
            monitor="ptl/test_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
        )
        self.context = context
        self.model_cls = model_cls
        self.model_name = model_name
        self.registry_key = str(checkpoint_dir.resolve())
        self.metric_name = "ptl/test_loss"
        self.temp_checkpoint_path = checkpoint_dir / "_current_test_monitor.ckpt"

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        return None

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        import torch

        source_forecaster = FORECASTER_REGISTRY.get(self.registry_key)
        if trainer.sanity_checking or source_forecaster is None:
            return None

        pl_module.save(str(self.temp_checkpoint_path))
        try:
            monitor_model = self.model_cls.load(str(self.temp_checkpoint_path))
            monitor_model.alias = self.model_name
            monitor_model = prepare_model_for_prediction(monitor_model)
            monitor_forecaster = clone_forecaster_for_prediction(
                source_forecaster,
                monitor_model,
            )
            phase_output = evaluate_phase_with_forecaster(
                context=self.context,
                model_name=self.model_name,
                nf=monitor_forecaster,
                base_history_df=self.context.split_data.trainval,
                target_df=self.context.split_data.test,
                origins=self.context.split_data.test_origins,
            )
            test_loss = compute_phase_loss(self.context.run_config, phase_output)
            metric = torch.tensor(test_loss, dtype=torch.float32, device=pl_module.device)
            trainer.callback_metrics[self.metric_name] = metric.detach()
            trainer.logged_metrics[self.metric_name] = metric.detach()
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {self.metric_name: float(metric.detach().cpu().item())},
                    step=trainer.global_step,
                )
                if hasattr(trainer.logger, "save"):
                    trainer.logger.save()
        finally:
            if self.temp_checkpoint_path.exists():
                self.temp_checkpoint_path.unlink()

        super().on_validation_end(trainer, pl_module)


def build_rolling_test_loss_callback(
    *,
    context: ExecutionContext,
    model_cls: type,
    model_name: str,
    checkpoint_dir: Path,
) -> Any:
    return RollingTestCheckpoint(
        context=context,
        model_cls=model_cls,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
    )


class NeuralExecutor(BaseExecutor):
    def _normalize_neural_params(self) -> dict[str, Any]:
        neural_params = dict(self.context.model_spec.model_params)

        if self.context.model_spec.model_cls.__name__ != "TimeLLM":
            return neural_params

        if importlib.util.find_spec("transformers") is None:
            raise ImportError(
                "TimeLLM requires the optional dependency `transformers`. "
                "Install it before running TimeLLM experiments."
            )

        neural_params.setdefault("llm", "openai-community/gpt2")
        neural_params.setdefault("enc_in", 1)
        neural_params.setdefault("dec_in", 1)
        return neural_params

    def run(self) -> ExecutorOutput:
        from neuralforecast import NeuralForecast
        from pytorch_lightning.loggers import CSVLogger

        checkpoint_dir = self.context.artifact_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        csv_logger = CSVLogger(
            save_dir=str(self.context.artifact_dir),
            name="logs",
            version="main",
        )
        val_checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="val_best",
            monitor="ptl/val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
        test_checkpoint_callback = build_rolling_test_loss_callback(
            context=self.context,
            model_cls=self.context.model_spec.model_cls,
            model_name=self.model_name,
            checkpoint_dir=checkpoint_dir,
        )

        neural_params = self._normalize_neural_params()
        neural_params.setdefault("alias", self.model_name)
        neural_params.setdefault("enable_progress_bar", False)
        neural_params.setdefault("enable_model_summary", False)
        trainer_callbacks = list(neural_params.pop("callbacks", []))
        trainer_logger = neural_params.pop("logger", None)
        trainer_callbacks.extend([test_checkpoint_callback, val_checkpoint_callback])

        model = self.context.model_spec.model_cls(
            h=self.context.run_config.horizon,
            input_size=self.context.run_config.input_size,
            hist_exog_list=self.context.hist_exog,
            futr_exog_list=self.context.futr_exog,
            loss=build_neural_loss(self.context.run_config),
            valid_loss=build_neural_loss(self.context.run_config),
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
        registry_key = str(checkpoint_dir.resolve())
        FORECASTER_REGISTRY[registry_key] = nf
        try:
            nf.fit(df=self.context.split_data.trainval, val_size=self.context.split_data.n_val)
        finally:
            FORECASTER_REGISTRY.pop(registry_key, None)

        val_best_model_path = self._resolve_checkpoint_path(
            val_checkpoint_callback=val_checkpoint_callback,
            test_checkpoint_callback=test_checkpoint_callback,
            checkpoint_dir=checkpoint_dir,
            preferred_mode="val_best",
        )
        if val_best_model_path is None:
            val_best_model_path = self._ensure_final_checkpoint(
                nf=nf,
                checkpoint_dir=checkpoint_dir,
                save_if_missing=True,
            )

        test_best_model_path = self._resolve_checkpoint_path(
            val_checkpoint_callback=val_checkpoint_callback,
            test_checkpoint_callback=test_checkpoint_callback,
            checkpoint_dir=checkpoint_dir,
            preferred_mode="test_best",
        )
        if test_best_model_path is None:
            test_best_model_path = self._ensure_final_checkpoint(
                nf=nf,
                checkpoint_dir=checkpoint_dir,
                save_if_missing=True,
            )

        last_model_path = self._resolve_checkpoint_path(
            val_checkpoint_callback=val_checkpoint_callback,
            test_checkpoint_callback=test_checkpoint_callback,
            checkpoint_dir=checkpoint_dir,
            preferred_mode="last",
        )
        if last_model_path is None:
            last_model_path = self._ensure_final_checkpoint(
                nf=nf,
                checkpoint_dir=checkpoint_dir,
                save_if_missing=True,
            )

        metrics_path = None
        loss_plot_path = None
        metrics_file = Path(csv_logger.log_dir) / "metrics.csv"
        if metrics_file.exists():
            loss_plot_path = build_loss_artifact(
                metrics_csv_path=metrics_file,
                artifact_root=self.context.artifact_dir,
                model_name=self.model_name,
                run_config=self.context.run_config,
            )
            metrics_path = str(metrics_file)

        self._load_model_for_prediction(nf=nf, checkpoint_path=val_best_model_path)
        val_phase = self._evaluate_phase(
            nf=nf,
            base_history_df=self.context.split_data.train,
            target_df=self.context.split_data.val,
            origins=self.context.split_data.val_origins,
        )

        compare_test_phase = None
        compare_test_checkpoint_label = None
        if self.context.run_config.neural_test_checkpoint_mode == "best":
            main_test_checkpoint_path = test_best_model_path
            if val_best_model_path != test_best_model_path:
                self._load_model_for_prediction(nf=nf, checkpoint_path=val_best_model_path)
                compare_test_phase = self._evaluate_phase(
                    nf=nf,
                    base_history_df=self.context.split_data.trainval,
                    target_df=self.context.split_data.test,
                    origins=self.context.split_data.test_origins,
                )
                compare_test_checkpoint_label = Path(val_best_model_path).name
        else:
            main_test_checkpoint_path = last_model_path

        self._load_model_for_prediction(nf=nf, checkpoint_path=main_test_checkpoint_path)
        test_phase = self._evaluate_phase(
            nf=nf,
            base_history_df=self.context.split_data.trainval,
            target_df=self.context.split_data.test,
            origins=self.context.split_data.test_origins,
        )

        return ExecutorOutput(
            val_phase=val_phase,
            test_phase=test_phase,
            best_model_path=main_test_checkpoint_path,
            val_best_model_path=val_best_model_path,
            test_best_model_path=test_best_model_path,
            metrics_path=metrics_path,
            loss_plot_path=loss_plot_path,
            test_checkpoint_label=Path(main_test_checkpoint_path).name,
            compare_test_phase=compare_test_phase,
            compare_test_checkpoint_label=compare_test_checkpoint_label,
        )

    def _ensure_final_checkpoint(
        self,
        nf: Any,
        checkpoint_dir: Path,
        save_if_missing: bool,
    ) -> str:
        final_checkpoint_path = checkpoint_dir / "final.ckpt"
        if save_if_missing and not final_checkpoint_path.exists():
            nf.models[0].save(str(final_checkpoint_path))
        return str(final_checkpoint_path)

    def _checkpoint_candidates(
        self,
        val_checkpoint_callback: Any,
        test_checkpoint_callback: Any,
        checkpoint_dir: Path,
        preferred_mode: str,
    ) -> list[Optional[str]]:
        val_best_path = getattr(val_checkpoint_callback, "best_model_path", None) or None
        test_best_path = getattr(test_checkpoint_callback, "best_model_path", None) or None
        last_path = getattr(val_checkpoint_callback, "last_model_path", None) or None
        explicit_val_best = str(checkpoint_dir / "val_best.ckpt")
        explicit_test_best = str(checkpoint_dir / "test_best.ckpt")
        explicit_last = str(checkpoint_dir / "last.ckpt")

        if preferred_mode == "val_best":
            return [val_best_path, explicit_val_best, last_path, explicit_last]
        if preferred_mode == "test_best":
            return [
                test_best_path,
                explicit_test_best,
                val_best_path,
                explicit_val_best,
                last_path,
                explicit_last,
            ]
        if preferred_mode == "last":
            return [last_path, explicit_last, val_best_path, explicit_val_best]
        raise ValueError(f"Unsupported checkpoint mode: {preferred_mode!r}")

    def _resolve_checkpoint_path(
        self,
        val_checkpoint_callback: Any,
        test_checkpoint_callback: Any,
        checkpoint_dir: Path,
        preferred_mode: str,
    ) -> Optional[str]:
        for candidate in self._checkpoint_candidates(
            val_checkpoint_callback=val_checkpoint_callback,
            test_checkpoint_callback=test_checkpoint_callback,
            checkpoint_dir=checkpoint_dir,
            preferred_mode=preferred_mode,
        ):
            if candidate and Path(candidate).exists():
                return str(candidate)
        return None

    def _load_model_for_prediction(self, nf: Any, checkpoint_path: Optional[str]) -> None:
        if not checkpoint_path:
            nf.models[0] = prepare_model_for_prediction(nf.models[0])
            return

        model = self.context.model_spec.model_cls.load(checkpoint_path)
        model.alias = self.model_name
        nf.models[0] = prepare_model_for_prediction(model)

    def _prepare_model_for_prediction(self, model: Any) -> Any:
        return prepare_model_for_prediction(model)

    def _evaluate_phase(
        self,
        nf: Any,
        base_history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        origins: list[int],
    ) -> PhaseOutput:
        return evaluate_phase_with_forecaster(
            context=self.context,
            model_name=self.model_name,
            nf=nf,
            base_history_df=base_history_df,
            target_df=target_df,
            origins=origins,
        )
