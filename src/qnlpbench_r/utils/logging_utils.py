from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(name: str, run_dir: str | Path, level: str = "INFO") -> logging.Logger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def write_warnings(run_dir: str | Path, warnings: list[str]) -> None:
    with (Path(run_dir) / "warnings.txt").open("w", encoding="utf-8") as f:
        for w in warnings:
            f.write(str(w).rstrip() + "\n")
