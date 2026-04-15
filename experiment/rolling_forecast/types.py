from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..config import ModelSpec, RunConfig
from ..data import PreparedDataset


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ExecutionContext:
    dataset: PreparedDataset
    run_config: RunConfig
    model_spec: ModelSpec
    hist_exog: list[str]
    futr_exog: list[str]
    future_cols: list[str]
    artifact_dir: Path
    split_data: SplitData

    @property
    def model_name(self) -> str:
        return self.model_spec.name


@dataclass
class PhaseOutput:
    origin_mape_df: pd.DataFrame
    diagnostics: dict[pd.Timestamp, dict[str, Any]]


@dataclass
class ExecutorOutput:
    val_phase: PhaseOutput
    test_phase: PhaseOutput
    best_model_path: Optional[str] = None
    metrics_path: Optional[str] = None
    loss_plot_path: Optional[str] = None


@dataclass(frozen=True)
class FinalizedPhase:
    origin_mape_df: pd.DataFrame
    overall_mape: float
    best_origin: Optional[pd.Timestamp]
    worst_origin: Optional[pd.Timestamp]
    diagnostics: dict[pd.Timestamp, dict[str, Any]]
    forecast_plot_path: Optional[str]
    rolling_raw_path: Optional[str]
