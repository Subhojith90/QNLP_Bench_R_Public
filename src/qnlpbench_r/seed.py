from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch where available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed function for PyTorch DataLoader workers."""
    if torch is None:
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> Optional["torch.Generator"]:
    """Create a seeded torch Generator when torch is available."""
    if torch is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g
