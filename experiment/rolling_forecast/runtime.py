from __future__ import annotations

import logging
import random
from contextlib import contextmanager

import numpy as np


@contextmanager
def suppress_lightning_logs():
    """临时降低 Lightning 的日志级别，避免训练输出在 notebook 或终端中过于冗长。"""

    lightning_logger = logging.getLogger("lightning")
    pl_logger = logging.getLogger("pytorch_lightning")

    old_lightning_level = lightning_logger.level
    old_pl_level = pl_logger.level

    lightning_logger.setLevel(logging.ERROR)
    pl_logger.setLevel(logging.ERROR)

    try:
        yield
    finally:
        lightning_logger.setLevel(old_lightning_level)
        pl_logger.setLevel(old_pl_level)


def set_random_seed(random_seed: int) -> None:
    """统一设置 random、numpy、torch 与 Lightning 的随机种子。"""

    import torch
    from pytorch_lightning import seed_everything

    seed_everything(random_seed, workers=True)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
