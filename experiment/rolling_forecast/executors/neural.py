from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from ...config import RunConfig
from ..artifacts import build_loss_artifact
from ..metrics import safe_mape
from ..types import ExecutorOutput, PhaseOutput
from .base import BaseExecutor


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

        neural_params = self._normalize_neural_params()
        neural_params.setdefault("alias", self.model_name)
        neural_params.setdefault("enable_progress_bar", False)
        neural_params.setdefault("enable_model_summary", False)
        trainer_callbacks = list(neural_params.pop("callbacks", []))
        trainer_logger = neural_params.pop("logger", None)
        trainer_callbacks.append(checkpoint_callback)

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
        nf.fit(df=self.context.split_data.trainval, val_size=self.context.split_data.n_val)

        checkpoint_path = (
            checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or None
        )
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
            loss_plot_path = build_loss_artifact(
                metrics_csv_path=metrics_file,
                artifact_root=self.context.artifact_dir,
                model_name=self.model_name,
                run_config=self.context.run_config,
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
            origin_records.append({"ds_origin": origin, "mape": safe_mape(y_true, y_pred)})
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
