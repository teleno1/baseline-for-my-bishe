from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """对接近 0 的真实值使用保护分母，避免 MAPE 数值异常。"""

    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(true_values) < 1e-8, 1e-8, np.abs(true_values))
    return float(np.mean(np.abs((true_values - pred_values) / denom)) * 100)


def summarize_cv_predictions(
    cv_df: pd.DataFrame,
    prediction_column: str,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    """把 StatsForecast 的 cross_validation 输出整理成统一的 origin 级结果。"""

    origin_records: list[dict[str, Any]] = []
    diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}

    for cutoff, group in cv_df.groupby("cutoff"):
        group = group.sort_values("ds")
        y_true = group["y"].to_numpy()
        y_pred = group[prediction_column].to_numpy()
        origin = pd.to_datetime(group["ds"].iloc[0])
        origin_records.append({"ds_origin": origin, "mape": safe_mape(y_true, y_pred)})
        diagnostics[origin] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "ds": pd.to_datetime(group["ds"]).to_numpy(),
            "cutoff": pd.to_datetime(cutoff),
        }

    return pd.DataFrame(origin_records), diagnostics
