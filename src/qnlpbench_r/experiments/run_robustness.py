from __future__ import annotations

from pathlib import Path

from qnlpbench_r.config import load_config
from qnlpbench_r.experiments.run_experiment import run_experiment_from_config


def run_robustness(config_path: str | Path) -> list[Path]:
    cfg = load_config(config_path)
    cfg.setdefault("evaluation", {}).setdefault("robustness", {})["enabled"] = True
    cfg["experiment"]["run_group"] = "robustness"
    return run_experiment_from_config(cfg)
