from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

SUPPORTED_NEURAL_LOSS_NAMES = (
    "MAE",
    "MSE",
    "RMSE",
    "MAPE",
    "SMAPE",
    "HuberLoss",
    "TukeyLoss",
    "MASE",
)
SUPPORTED_PLOT_LOSS_NAMES = ("MAPE", "MAE", "MSE")


@dataclass(frozen=True)
class RunConfig:
    input_size: int
    horizon: int
    split_ratio: tuple[float, float, float] = (7, 1, 2)
    sliding_step_size: int = 1
    use_hist_exog: bool = True
    use_futr_exog: bool = True
    save_plots: bool = True
    random_seed: int = 42
    early_stop_patience_epochs: int = 20
    ml_early_stopping_rounds: Optional[int] = 20
    neural_loss_name: str = "MAPE"
    neural_loss_params: dict[str, Any] = field(default_factory=dict)
    neural_checkpoint_mode: str = "last"
    save_dir: str = "./artifacts"
    plot_forecast: bool = True
    plot_loss: bool = True
    plot_loss_name: str = "MAPE"
    freq: str = "D"

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.sliding_step_size <= 0:
            raise ValueError("sliding_step_size must be positive")
        if len(self.split_ratio) != 3:
            raise ValueError("split_ratio must contain exactly three values")
        if any(value <= 0 for value in self.split_ratio):
            raise ValueError("split_ratio values must be positive")
        if self.early_stop_patience_epochs < -1:
            raise ValueError("early_stop_patience_epochs must be >= -1")
        if self.ml_early_stopping_rounds is not None and self.ml_early_stopping_rounds < 0:
            raise ValueError("ml_early_stopping_rounds must be >= 0 or None")
        if self.neural_loss_name not in SUPPORTED_NEURAL_LOSS_NAMES:
            raise ValueError(
                "neural_loss_name must be one of "
                f"{SUPPORTED_NEURAL_LOSS_NAMES}, got {self.neural_loss_name!r}"
            )
        if not isinstance(self.neural_loss_params, dict):
            raise ValueError("neural_loss_params must be a dict")
        if self.neural_checkpoint_mode not in {"last", "val_best"}:
            raise ValueError("neural_checkpoint_mode must be 'last' or 'val_best'")
        normalized_plot_loss_name = str(self.plot_loss_name).upper()
        if normalized_plot_loss_name not in SUPPORTED_PLOT_LOSS_NAMES:
            raise ValueError(
                "plot_loss_name must be one of "
                f"{SUPPORTED_PLOT_LOSS_NAMES}, got {self.plot_loss_name!r}"
            )
        object.__setattr__(self, "plot_loss_name", normalized_plot_loss_name)

    def normalized_split_ratio(self) -> tuple[float, float, float]:
        total = float(sum(self.split_ratio))
        return tuple(value / total for value in self.split_ratio)

    @property
    def use_exog(self) -> bool:
        return self.use_hist_exog or self.use_futr_exog


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
    model_cls: type
    model_params: dict[str, Any] = field(default_factory=dict)
    supports_hist_exog: bool = True
    supports_future_exog: bool = True

    def __post_init__(self) -> None:
        if self.model_type not in {"stats", "ml", "mlforecast", "neural"}:
            raise ValueError("model_type must be 'stats', 'ml', 'mlforecast', or 'neural'")


@dataclass
class ExperimentResult:
    model_name: str
    val_origin_mape_df: pd.DataFrame
    val_overall_mape: float
    val_best_origin: Optional[pd.Timestamp]
    val_worst_origin: Optional[pd.Timestamp]
    origin_mape_df: pd.DataFrame
    overall_mape: float
    best_origin: Optional[pd.Timestamp]
    worst_origin: Optional[pd.Timestamp]
    artifact_dir: Optional[str]
    best_model_path: Optional[str]
    val_best_model_path: Optional[str]
    test_best_model_path: Optional[str]
    metrics_path: Optional[str]
    loss_plot_path: Optional[str]
    forecast_plot_path: Optional[str]
    rolling_raw_path: Optional[str]
    overlay_plot_path: Optional[str]
    skipped: bool = False
    skip_reason: Optional[str] = None
    used_exog: bool = False
    requested_use_exog: bool = False
    val_diagnostics: dict[Any, Any] = field(default_factory=dict)
    diagnostics: dict[Any, Any] = field(default_factory=dict)
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    test_size: Optional[int] = None
    val_n_origins: Optional[int] = None
    n_origins: Optional[int] = None

    @classmethod
    def skipped_result(
        cls,
        model_name: str,
        reason: str,
        requested_use_exog: bool,
    ) -> "ExperimentResult":
        empty_origin_df = pd.DataFrame(columns=["ds_origin", "mape"])
        return cls(
            model_name=model_name,
            val_origin_mape_df=empty_origin_df.copy(),
            val_overall_mape=float("nan"),
            val_best_origin=None,
            val_worst_origin=None,
            origin_mape_df=empty_origin_df.copy(),
            overall_mape=float("nan"),
            best_origin=None,
            worst_origin=None,
            artifact_dir=None,
            best_model_path=None,
            val_best_model_path=None,
            test_best_model_path=None,
            metrics_path=None,
            loss_plot_path=None,
            forecast_plot_path=None,
            rolling_raw_path=None,
            overlay_plot_path=None,
            skipped=True,
            skip_reason=reason,
            used_exog=False,
            requested_use_exog=requested_use_exog,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "val_overall_mape": self.val_overall_mape,
            "overall_mape": self.overall_mape,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "artifact_dir": self.artifact_dir,
            "best_model_path": self.best_model_path,
            "val_best_model_path": self.val_best_model_path,
            "test_best_model_path": self.test_best_model_path,
            "rolling_raw_path": self.rolling_raw_path,
            "overlay_plot_path": self.overlay_plot_path,
        }

    def __repr__(self) -> str:
        return f"ExperimentResult({self.summary()})"

