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
    checkpoint_label: Optional[str] = None,
    artifact_suffix: str = "",
) -> FinalizedPhase:
    """Summarize a rolling phase and build test artifacts."""

    if phase_name not in {"val", "test"}:
        raise ValueError(f"Unsupported phase_name: {phase_name}")

    origin_mape_df = phase_output.origin_mape_df.sort_values("ds_origin").reset_index(drop=True)
    rolling_raw_path = None
    overlay_plot_path = None
    rolling_raw_df = pd.DataFrame()
    if phase_name == "test":
        rolling_raw_df = build_rolling_raw_df(
            origin_mape_df=origin_mape_df,
            diagnostics=phase_output.diagnostics,
        )
        rolling_raw_path = save_rolling_raw_artifact(
            rolling_raw_df=rolling_raw_df,
            artifact_dir=context.artifact_dir,
            artifact_suffix=artifact_suffix,
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
        overlay_plot_path = plot_test_overlay(
            context=context,
            target_df=target_df,
            rolling_raw_df=rolling_raw_df,
            checkpoint_label=checkpoint_label,
            artifact_suffix=artifact_suffix,
        )
        forecast_plot_path = plot_test_forecast(
            context=context,
            target_df=target_df,
            diagnostics=phase_output.diagnostics,
            overall_mape=overall_mape,
            best_origin=best_origin,
            worst_origin=worst_origin,
            checkpoint_label=checkpoint_label,
            artifact_suffix=artifact_suffix,
        )

    return FinalizedPhase(
        origin_mape_df=origin_mape_df,
        overall_mape=overall_mape,
        best_origin=best_origin,
        worst_origin=worst_origin,
        diagnostics=phase_output.diagnostics,
        forecast_plot_path=forecast_plot_path,
        rolling_raw_path=rolling_raw_path,
        overlay_plot_path=overlay_plot_path,
    )


def build_rolling_raw_df(
    origin_mape_df: pd.DataFrame,
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
) -> pd.DataFrame:
    """Expand per-origin diagnostics into a row-level rolling detail table."""

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
    artifact_suffix: str = "",
) -> str:
    """Save the rolling test detail CSV for the current run."""

    suffix = artifact_suffix or ""
    rolling_raw_path = artifact_dir / f"rolling_test_raw{suffix}.csv"
    rolling_raw_df.to_csv(rolling_raw_path, index=False, encoding="utf-8-sig")
    return str(rolling_raw_path)


def select_non_overlapping_windows(
    rolling_raw_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    window_size: int,
    window_step: int,
) -> pd.DataFrame:
    """Select non-overlapping forecast windows that cover the full test range."""

    required_columns = [
        "origin_id",
        "origin_date",
        "horizon",
        "target_date",
        "y_true",
        "y_pred",
        "error",
    ]
    missing_columns = [column for column in required_columns if column not in rolling_raw_df.columns]
    if missing_columns:
        raise ValueError(f"rolling_raw_df is missing required columns: {missing_columns}")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_step <= 0:
        raise ValueError("window_step must be positive")

    normalized = rolling_raw_df.copy()
    normalized["origin_date"] = pd.to_datetime(normalized["origin_date"])
    normalized["target_date"] = pd.to_datetime(normalized["target_date"])
    normalized = normalized.sort_values(["origin_date", "horizon"]).reset_index(drop=True)

    expected_target_dates = pd.DatetimeIndex(pd.to_datetime(target_dates)).sort_values().unique()
    if expected_target_dates.empty:
        return pd.DataFrame(
            columns=required_columns + ["group_id", "group_label", "is_tail_patch"]
        )

    expected_horizons = list(range(1, window_size + 1))
    horizons_by_origin = normalized.groupby("origin_date", sort=True)["horizon"].apply(list)
    bad_origins = horizons_by_origin[horizons_by_origin.map(lambda values: values != expected_horizons)]
    if not bad_origins.empty:
        raise ValueError(
            "Found origins without the expected horizon sequence "
            f"1..{window_size}: {bad_origins.to_dict()}"
        )

    unique_origins = pd.DatetimeIndex(sorted(normalized["origin_date"].drop_duplicates()))
    full_group_count, remainder_days = divmod(len(expected_target_dates), window_size)
    selected_origins = unique_origins[::window_step][:full_group_count]
    if len(selected_origins) != full_group_count:
        raise ValueError("Not enough rolling origins to build non-overlapping overlay windows.")

    plot_columns = ["origin_date", "target_date", "horizon", "y_true", "y_pred", "error"]
    groups: list[pd.DataFrame] = []

    for group_id, origin_date in enumerate(selected_origins, start=1):
        group_df = (
            normalized.loc[normalized["origin_date"] == origin_date, plot_columns]
            .sort_values("target_date")
            .reset_index(drop=True)
            .copy()
        )
        if len(group_df) != window_size:
            raise ValueError(
                f"Origin {origin_date.date()} does not contain {window_size} forecast rows."
            )
        group_df["group_id"] = group_id
        group_df["group_label"] = f"origin={origin_date:%Y-%m-%d}"
        group_df["is_tail_patch"] = False
        groups.append(group_df)

    if remainder_days:
        tail_origin = unique_origins[-1]
        tail_patch_df = (
            normalized.loc[normalized["origin_date"] == tail_origin, plot_columns]
            .sort_values("target_date")
            .tail(remainder_days)
            .reset_index(drop=True)
            .copy()
        )
        tail_patch_df["group_id"] = full_group_count + 1
        tail_patch_df["group_label"] = f"origin={tail_origin:%Y-%m-%d} tail({remainder_days}d)"
        tail_patch_df["is_tail_patch"] = True
        groups.append(tail_patch_df)

    if not groups:
        raise ValueError("No overlay windows were selected from rolling_raw_df.")

    selected_df = pd.concat(groups, ignore_index=True)
    selected_df = selected_df.sort_values(["target_date", "group_id"]).reset_index(drop=True)

    selected_target_dates = pd.DatetimeIndex(selected_df["target_date"])
    if not selected_target_dates.is_unique:
        raise ValueError("Selected target_date values must be unique.")
    if not selected_target_dates.equals(expected_target_dates):
        raise ValueError("Selected target_date values do not exactly cover the full test range.")

    return selected_df


def plot_test_overlay(
    context: ExecutionContext,
    target_df: pd.DataFrame,
    rolling_raw_df: pd.DataFrame,
    checkpoint_label: Optional[str] = None,
    artifact_suffix: str = "",
) -> Optional[str]:
    """Plot non-overlapping rolling predictions over the full test interval."""

    if not context.run_config.plot_forecast or target_df.empty or rolling_raw_df.empty:
        return None

    target_slice = target_df[["ds", "y"]].copy()
    target_slice["ds"] = pd.to_datetime(target_slice["ds"])
    target_slice = target_slice.sort_values("ds").reset_index(drop=True)

    selected_windows = select_non_overlapping_windows(
        rolling_raw_df=rolling_raw_df,
        target_dates=pd.DatetimeIndex(target_slice["ds"]),
        window_size=context.run_config.horizon,
        window_step=context.run_config.horizon,
    )

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(
        target_slice["ds"],
        target_slice["y"],
        color="black",
        linewidth=3,
        label="True Load (Test Set)",
    )

    group_labels = selected_windows["group_label"].drop_duplicates().tolist()
    colors = plt.cm.tab20(np.linspace(0, 1, len(group_labels)))
    for color, group_label in zip(colors, group_labels):
        group_df = (
            selected_windows.loc[selected_windows["group_label"] == group_label]
            .sort_values("target_date")
            .reset_index(drop=True)
        )
        group_mape = safe_mape(group_df["y_true"], group_df["y_pred"])
        group_plot_label = f"{group_label} (MAPE={group_mape:.2f}%)"
        ax.plot(
            group_df["target_date"],
            group_df["y_pred"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=group_plot_label,
        )

    title = f"Rolling Test Prediction Overlay ({context.model_name})"
    if checkpoint_label:
        title = f"{title}\nfrom {checkpoint_label}"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Electricity Load")
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.28)

    overlay_plot_path = None
    if context.run_config.save_plots:
        suffix = artifact_suffix or ""
        overlay_plot_path = context.artifact_dir / f"rolling_test_overlay{suffix}.png"
        fig.savefig(overlay_plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(overlay_plot_path) if overlay_plot_path else None


def plot_test_forecast(
    context: ExecutionContext,
    target_df: pd.DataFrame,
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
    overall_mape: float,
    best_origin: Optional[pd.Timestamp],
    worst_origin: Optional[pd.Timestamp],
    checkpoint_label: Optional[str] = None,
    artifact_suffix: str = "",
) -> Optional[str]:
    """Plot the best and worst rolling forecasts for the test phase."""

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
    checkpoint_suffix = f", from {checkpoint_label}" if checkpoint_label else ""
    if context.model_spec.model_type == "neural":
        plt.title(
            f"Rolling Test Forecast ({context.model_name}{checkpoint_suffix})\n"
            f"Loss Function:{context.run_config.neural_loss_name},"
            f"Horizon={context.run_config.horizon}, Mean Origin MAPE={overall_mape:.2f}%"
        )
    else:
        plt.title(
            f"Rolling Test Forecast ({context.model_name}{checkpoint_suffix})\n"
            f"Horizon={context.run_config.horizon}, Mean Origin MAPE={overall_mape:.2f}%"
        )
    plt.xlabel("Date")
    plt.ylabel("Daily Electricity Load")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    forecast_plot_path = None
    if context.run_config.save_plots:
        suffix = artifact_suffix or ""
        forecast_plot_path = context.artifact_dir / f"forecast_plot{suffix}.png"
        plt.savefig(forecast_plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return str(forecast_plot_path) if forecast_plot_path else None


def choose_metric_column(
    metrics_df: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    """Return the first metrics column that exists and has non-null values."""

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
    """Build the training-loss artifact from the Lightning metrics CSV."""

    metrics_df = pd.read_csv(metrics_csv_path)
    x_col = "epoch" if "epoch" in metrics_df.columns else None
    if x_col is None and "step" in metrics_df.columns:
        x_col = "step"
    x_label = "Epoch" if x_col == "epoch" else "Step"
    train_col = choose_metric_column(
        metrics_df,
        ["train_loss_epoch", "train_loss", "ptl/train_loss", "train_loss_step"],
    )
    valid_col = choose_metric_column(
        metrics_df,
        ["ptl/val_loss_train_norm"],
    )
    test_col = choose_metric_column(
        metrics_df,
        ["ptl/test_loss_train_norm"],
    )

    loss_plot_path = None
    if run_config.plot_loss and (train_col or valid_col or test_col):
        plt.figure(figsize=(10, 4))

        def plot_metric(column: str, label: str) -> None:
            curve_cols = [column] + ([x_col] if x_col else [])
            curve = metrics_df[curve_cols].dropna(subset=[column])
            if curve.empty:
                return
            if x_col:
                curve = curve.groupby(x_col, as_index=False)[column].last()
                x_values = curve[x_col]
            else:
                x_values = np.arange(len(curve))
            plt.plot(x_values, curve[column], label=label)

        if train_col is not None:
            plot_metric(train_col, "Train Loss")
        if valid_col is not None:
            plot_metric(valid_col, "Valid Loss")
        if test_col is not None:
            plot_metric(test_col, "Test Loss")
        plt.title(f"Training Curve ({model_name})")
        plt.xlabel(x_label)
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

