from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..config import ModelSpec, RunConfig
from ..data import PreparedDataset


@dataclass(frozen=True)
class SplitData:
    """保存一次固定切分后的 train / val / test 数据，以及对应的 rolling 起点索引。"""

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
    """封装单次实验运行所需的公共上下文，避免在执行器之间重复传参。"""

    dataset: PreparedDataset
    run_config: RunConfig
    model_spec: ModelSpec
    futr_exog: list[str]
    future_cols: list[str]
    artifact_dir: Path
    split_data: SplitData

    @property
    def model_name(self) -> str:
        return self.model_spec.name


@dataclass
class PhaseOutput:
    """表示单个阶段（val 或 test）的 rolling 评估结果。"""

    origin_mape_df: pd.DataFrame
    diagnostics: dict[pd.Timestamp, dict[str, Any]]


@dataclass
class ExecutorOutput:
    """执行器的原始输出，最终会进一步组装成 ExperimentResult。"""

    val_phase: PhaseOutput
    test_phase: PhaseOutput
    best_model_path: Optional[str] = None
    metrics_path: Optional[str] = None
    loss_plot_path: Optional[str] = None


@dataclass(frozen=True)
class FinalizedPhase:
    """表示 runner 汇总后的阶段结果，附带整体指标和 artifact 路径。"""

    origin_mape_df: pd.DataFrame
    overall_mape: float
    best_origin: Optional[pd.Timestamp]
    worst_origin: Optional[pd.Timestamp]
    diagnostics: dict[pd.Timestamp, dict[str, Any]]
    forecast_plot_path: Optional[str]
    rolling_raw_path: Optional[str]
