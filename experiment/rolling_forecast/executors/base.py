from __future__ import annotations

import pandas as pd

from ..types import ExecutionContext, ExecutorOutput


class BaseExecutor:
    """三类执行器的公共基类，封装共享上下文和通用辅助逻辑。"""

    def __init__(self, context: ExecutionContext):
        self.context = context

    @property
    def model_name(self) -> str:
        return self.context.model_name

    def _prediction_column(self, df: pd.DataFrame, excluded: set[str]) -> str:
        """从模型输出中定位真正的预测列，屏蔽不同库的列名差异。"""

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
