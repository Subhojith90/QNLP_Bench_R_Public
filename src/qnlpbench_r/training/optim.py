from typing import Iterable

import torch


def build_optimizer(parameters: Iterable[torch.nn.Parameter], learning_rate: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))
