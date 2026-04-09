from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import RunConfig
from .metrics import safe_mape
from .types import ExecutionContext, FinalizedPhase, PhaseOutput


def finalize_phase(
    context: ExecutionContext,
    phase_name: str,
    target_df: pd.DataFrame,
    phase_output: PhaseOutput,
) -> FinalizedPhase:
    """汇总单个阶段的 rolling 结果，并在 test 阶段生成明细表与图像产物。"""

    if phase_name not in {"val", "test"}:
        raise ValueError(f"Unsupported phase_name: {phase_name}")

    origin_mape_df = phase_output.origin_mape_df.sort_values("ds_origin").reset_index(drop=True)
    rolling_raw_path = None
    if phase_name == "test":
        rolling_raw_df = build_rolling_raw_df(
            origin_mape_df=origin_mape_df,
            diagnostics=phase_output.diagnostics,
        )
        rolling_raw_path = save_rolling_raw_artifact(
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
        forecast_plot_path = plot_test_forecast(
            context=context,
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


def build_rolling_raw_df(
    origin_mape_df: pd.DataFrame,
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
) -> pd.DataFrame:
    """把 test 阶段的 diagnostics 还原成逐 origin、逐 horizon 的原始明细表。"""

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


def save_rolling_raw_artifact(
    rolling_raw_df: pd.DataFrame,
    artifact_dir: Path,
) -> str:
    """把 rolling_test_raw.csv 保存到当前实验的 artifact 目录中。"""

    rolling_raw_path = artifact_dir / "rolling_test_raw.csv"
    rolling_raw_df.to_csv(rolling_raw_path, index=False, encoding="utf-8-sig")
    return str(rolling_raw_path)


def plot_test_forecast(
    context: ExecutionContext,
    target_df: pd.DataFrame,
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
    overall_mape: float,
    best_origin: Optional[pd.Timestamp],
    worst_origin: Optional[pd.Timestamp],
) -> Optional[str]:
    """绘制 test 集的真实值、最佳窗口预测和最差窗口预测。"""

    if not context.run_config.plot_forecast or best_origin is None or worst_origin is None:
        return None

    best = diagnostics[best_origin]
    worst = diagnostics[worst_origin]
    best_mape = safe_mape(best["y_true"], best["y_pred"])
    worst_mape = safe_mape(worst["y_true"], worst["y_pred"])

    plt.figure(figsize=(14, 5))
    plt.plot(
        target_df["ds"],
        target_df["y"],
        color="black",
        linewidth=2,
        label="True Load (Test Set)",
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
        f"Rolling Test Forecast ({context.model_name})\n"
        f"Horizon={context.run_config.horizon}, Mean Origin MAPE={overall_mape:.2f}%"
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Electricity Load")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    forecast_plot_path = None
    if context.run_config.save_plots:
        forecast_plot_path = context.artifact_dir / "forecast_plot.png"
        plt.savefig(forecast_plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return str(forecast_plot_path) if forecast_plot_path else None


def choose_metric_column(
    metrics_df: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    """从候选指标列中选出第一个实际存在且包含有效数值的列。"""

    for column in candidates:
        if column in metrics_df.columns and metrics_df[column].notna().any():
            return column
    return None


def build_loss_artifact(
    metrics_csv_path: Path,
    artifact_root: Path,
    model_name: str,
    run_config: RunConfig,
) -> Optional[str]:
    """从 Lightning 生成的 metrics.csv 中提取训练与验证损失，并绘制 loss 曲线。"""

    metrics_df = pd.read_csv(metrics_csv_path)
    step_col = "step" if "step" in metrics_df.columns else None
    train_col = choose_metric_column(
        metrics_df,
        ["train_loss_epoch", "train_loss", "ptl/train_loss", "train_loss_step"],
    )
    valid_col = choose_metric_column(
        metrics_df,
        ["valid_loss", "ptl/val_loss", "val_loss"],
    )

    loss_plot_path = None
    if run_config.plot_loss and (train_col or valid_col):
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
        if run_config.save_plots:
            loss_plot_path = artifact_root / "loss_curve.png"
            plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    return str(loss_plot_path) if loss_plot_path else None
