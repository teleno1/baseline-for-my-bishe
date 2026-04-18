from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

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
    if not origins:
        return PhaseOutput(
            origin_mape_df=pd.DataFrame(columns=["ds_origin", "mape"]),
            diagnostics={},
        )

    history_parts: list[pd.DataFrame] = []
    future_parts: list[pd.DataFrame] = []
    phase_records: list[dict[str, Any]] = []
    base_id = str(base_history_df["unique_id"].iloc[-1])

    for origin_id, origin_offset in enumerate(origins):
        history = pd.concat(
            [base_history_df, target_df.iloc[:origin_offset]],
            ignore_index=True,
        )
        history_df = history.iloc[-context.run_config.input_size :].copy()
        synthetic_id = f"{base_id}__origin_{origin_id}"
        history_df["unique_id"] = synthetic_id

        future_df = target_df.iloc[
            origin_offset : origin_offset + context.run_config.horizon
        ][context.future_cols].copy()
        future_df["unique_id"] = synthetic_id
        history_parts.append(history_df)
        future_parts.append(future_df)

        y_true = target_df.iloc[
            origin_offset : origin_offset + context.run_config.horizon
        ]["y"].to_numpy(dtype=float)
        ds_forecast = pd.to_datetime(
            target_df.iloc[
                origin_offset : origin_offset + context.run_config.horizon
            ]["ds"]
        ).to_numpy()
        origin = pd.to_datetime(target_df.iloc[origin_offset]["ds"])
        phase_records.append(
            {
                "synthetic_id": synthetic_id,
                "origin": origin,
                "y_true": y_true,
                "y_insample": history_df["y"].to_numpy(dtype=float),
                "ds": ds_forecast,
                "cutoff": pd.to_datetime(history.iloc[-1]["ds"]),
            }
        )

    batched_history_df = pd.concat(history_parts, ignore_index=True)
    batched_future_df = pd.concat(future_parts, ignore_index=True)
    pred = nf.predict(df=batched_history_df, futr_df=batched_future_df)

    prediction_column = prediction_column_for_model(
        pred,
        model_name=model_name,
        excluded={"unique_id", "ds"},
    )
    pred_by_id = {
        str(unique_id): group.sort_values("ds")
        for unique_id, group in pred.groupby("unique_id", sort=False)
    }

    origin_records: list[dict[str, Any]] = []
    diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
    for record in phase_records:
        pred_group = pred_by_id.get(record["synthetic_id"])
        if pred_group is None:
            raise ValueError(f"Missing predictions for {record['synthetic_id']}")
        y_pred = pred_group[prediction_column].to_numpy(dtype=float)
        if len(y_pred) != len(record["y_true"]):
            raise ValueError(
                f"Prediction length mismatch for {record['synthetic_id']}: "
                f"expected {len(record['y_true'])}, got {len(y_pred)}"
            )
        origin = record["origin"]
        origin_records.append({"ds_origin": origin, "mape": safe_mape(record["y_true"], y_pred)})
        diagnostics[origin] = {
            "y_true": record["y_true"],
            "y_pred": y_pred,
            "y_insample": record["y_insample"],
            "ds": record["ds"],
            "cutoff": record["cutoff"],
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


def compute_normalized_phase_loss(
    run_config: RunConfig,
    phase_output: PhaseOutput,
    scaler_type: str,
) -> float:
    import torch
    from neuralforecast.common._scalers import TemporalNorm

    if not phase_output.diagnostics:
        return float("nan")

    loss_fn = build_neural_loss(run_config)
    scaler = TemporalNorm(scaler_type=scaler_type, dim=1, num_features=1)
    ordered_origins = sorted(phase_output.diagnostics)
    y_true = np.stack(
        [
            np.asarray(phase_output.diagnostics[origin]["y_true"], dtype=float)
            for origin in ordered_origins
        ]
    )
    y_pred = np.stack(
        [
            np.asarray(phase_output.diagnostics[origin]["y_pred"], dtype=float)
            for origin in ordered_origins
        ]
    )
    y_insample = np.stack(
        [
            np.asarray(phase_output.diagnostics[origin]["y_insample"], dtype=float)
            for origin in ordered_origins
        ]
    )

    temporal = np.concatenate([y_insample, y_true], axis=1)
    temporal_tensor = torch.tensor(temporal, dtype=torch.float32).unsqueeze(-1)
    mask = torch.ones_like(temporal_tensor)
    mask[:, -run_config.horizon :, :] = 0.0
    normalized = scaler.transform(x=temporal_tensor, mask=mask)
    normalized_y_true = normalized[:, -run_config.horizon :, :]

    shift = scaler.x_shift
    scale = scaler.x_scale
    y_hat_tensor = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(-1)
    normalized_y_pred = scaler.scaler(y_hat_tensor, shift, scale)

    if run_config.neural_loss_name == "MASE":
        normalized_y_insample = normalized[:, : y_insample.shape[1], 0]
        loss_value = loss_fn(
            y=normalized_y_true,
            y_hat=normalized_y_pred,
            y_insample=normalized_y_insample,
        )
    else:
        loss_value = loss_fn(y=normalized_y_true, y_hat=normalized_y_pred)

    return float(loss_value.detach().cpu().item())


class RollingPhaseLossLogger(Callback):
    def __init__(
        self,
        *,
        context: ExecutionContext,
        model_cls: type,
        model_name: str,
        checkpoint_dir: Path,
    ) -> None:
        super().__init__()
        self.context = context
        self.model_cls = model_cls
        self.model_name = model_name
        self.registry_key = str(checkpoint_dir.resolve())
        self.val_metric_name = "ptl/val_loss_norm"
        self.test_metric_name = "ptl/test_loss_norm"
        self.temp_checkpoint_path = checkpoint_dir / "_current_phase_loss.ckpt"

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
            scaler_type = getattr(getattr(monitor_model, "scaler", None), "scaler_type", "identity")
            val_phase_output = evaluate_phase_with_forecaster(
                context=self.context,
                model_name=self.model_name,
                nf=monitor_forecaster,
                base_history_df=self.context.split_data.train,
                target_df=self.context.split_data.val,
                origins=self.context.split_data.val_origins,
            )
            test_phase_output = evaluate_phase_with_forecaster(
                context=self.context,
                model_name=self.model_name,
                nf=monitor_forecaster,
                base_history_df=self.context.split_data.trainval,
                target_df=self.context.split_data.test,
                origins=self.context.split_data.test_origins,
            )
            val_loss = compute_normalized_phase_loss(
                self.context.run_config,
                val_phase_output,
                scaler_type=scaler_type,
            )
            test_loss = compute_normalized_phase_loss(
                self.context.run_config,
                test_phase_output,
                scaler_type=scaler_type,
            )
            val_metric = torch.tensor(val_loss, dtype=torch.float32, device=pl_module.device)
            test_metric = torch.tensor(test_loss, dtype=torch.float32, device=pl_module.device)
            trainer.callback_metrics[self.val_metric_name] = val_metric.detach()
            trainer.callback_metrics[self.test_metric_name] = test_metric.detach()
            trainer.logged_metrics[self.val_metric_name] = val_metric.detach()
            trainer.logged_metrics[self.test_metric_name] = test_metric.detach()
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {
                        "epoch": int(trainer.current_epoch),
                        self.val_metric_name: float(val_metric.detach().cpu().item()),
                        self.test_metric_name: float(test_metric.detach().cpu().item()),
                    },
                    step=trainer.global_step,
                )
                if hasattr(trainer.logger, "save"):
                    trainer.logger.save()
        finally:
            if self.temp_checkpoint_path.exists():
                self.temp_checkpoint_path.unlink()

def build_rolling_phase_loss_logger(
    *,
    context: ExecutionContext,
    model_cls: type,
    model_name: str,
    checkpoint_dir: Path,
) -> Any:
    return RollingPhaseLossLogger(
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

    def _coerce_positive_int(self, value: Any, param_name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{param_name} must be a positive integer")
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{param_name} must be a positive integer") from exc
        if int_value <= 0:
            raise ValueError(f"{param_name} must be a positive integer")
        return int_value

    def _steps_per_epoch(self, neural_params: dict[str, Any]) -> int:
        n_series = int(self.context.split_data.trainval["unique_id"].nunique())
        batch_size = self._coerce_positive_int(neural_params.get("batch_size", 32), "batch_size")
        if getattr(self.context.model_spec.model_cls, "MULTIVARIATE", False):
            batch_size = max(batch_size, n_series)

        drop_last = bool(neural_params.get("drop_last_loader", False))
        if drop_last:
            steps_per_epoch = n_series // batch_size
            if steps_per_epoch < 1:
                raise ValueError(
                    "drop_last_loader=True would leave no training batches for "
                    f"{n_series} series and batch_size={batch_size}."
                )
            return steps_per_epoch

        return max(1, math.ceil(n_series / batch_size))

    def _apply_epoch_training_params(self, neural_params: dict[str, Any]) -> tuple[int, int]:
        max_epochs = neural_params.pop("max_epochs", None)
        legacy_max_steps = neural_params.pop("max_steps", None)
        neural_params.pop("val_check_steps", None)
        neural_params.pop("early_stop_patience_steps", None)

        steps_per_epoch = self._steps_per_epoch(neural_params)
        if max_epochs is not None and legacy_max_steps is not None:
            raise ValueError("Use max_epochs only; max_steps is derived internally.")
        if max_epochs is None:
            if legacy_max_steps is None:
                max_epochs = 1000
            else:
                legacy_steps = self._coerce_positive_int(legacy_max_steps, "max_steps")
                max_epochs = max(1, math.ceil(legacy_steps / steps_per_epoch))

        max_epochs = self._coerce_positive_int(max_epochs, "max_epochs")
        neural_params["max_steps"] = max_epochs * steps_per_epoch
        neural_params["val_check_steps"] = steps_per_epoch
        return max_epochs, steps_per_epoch

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
            monitor="ptl/val_loss_norm",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
        phase_loss_logger = build_rolling_phase_loss_logger(
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
        self._apply_epoch_training_params(neural_params)
        trainer_callbacks.append(phase_loss_logger)
        if self.context.run_config.early_stop_patience_epochs > 0:
            trainer_callbacks.append(
                EarlyStopping(
                    monitor="ptl/val_loss_norm",
                    patience=self.context.run_config.early_stop_patience_epochs,
                    mode="min",
                )
            )
        trainer_callbacks.append(val_checkpoint_callback)

        model = self.context.model_spec.model_cls(
            h=self.context.run_config.horizon,
            input_size=self.context.run_config.input_size,
            hist_exog_list=self.context.hist_exog,
            futr_exog_list=self.context.futr_exog,
            loss=build_neural_loss(self.context.run_config),
            valid_loss=build_neural_loss(self.context.run_config),
            random_seed=self.context.run_config.random_seed,
            early_stop_patience_steps=-1,
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
            test_checkpoint_callback=None,
            checkpoint_dir=checkpoint_dir,
            preferred_mode="val_best",
        )
        if val_best_model_path is None:
            val_best_model_path = self._ensure_final_checkpoint(
                nf=nf,
                checkpoint_dir=checkpoint_dir,
                save_if_missing=True,
            )

        test_best_model_path = None

        last_model_path = self._resolve_checkpoint_path(
            val_checkpoint_callback=val_checkpoint_callback,
            test_checkpoint_callback=None,
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
        if self.context.run_config.neural_checkpoint_mode == "last":
            main_test_checkpoint_path = last_model_path
        else:
            main_test_checkpoint_path = val_best_model_path

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
        last_path = getattr(val_checkpoint_callback, "last_model_path", None) or None
        explicit_val_best = str(checkpoint_dir / "val_best.ckpt")
        explicit_last = str(checkpoint_dir / "last.ckpt")

        if preferred_mode == "val_best":
            return [val_best_path, explicit_val_best, last_path, explicit_last]
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
