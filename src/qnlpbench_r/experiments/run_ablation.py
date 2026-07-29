from __future__ import annotations

import copy
from pathlib import Path

from qnlpbench_r.config import deep_update, load_config
from qnlpbench_r.experiments.run_experiment import run_experiment_from_config


def run_ablation(config_path: str | Path) -> list[Path]:
    base = load_config(config_path)
    grid = base.get("ablation", {}).get("grid", [])
    if not grid:
        raise ValueError("Ablation config must contain ablation.grid entries.")
    outputs: list[Path] = []
    base_model = copy.deepcopy(base["models"][0])
    for item in grid:
        cfg = copy.deepcopy(base)
        ablation_name = str(item.get("name", "unnamed_ablation"))
        model = copy.deepcopy(base_model)
        if "model_overrides" in item:
            deep_update(model, item["model_overrides"])
        model["name"] = f"{base_model['name']}__{ablation_name}"
        cfg["models"] = [model]
        cfg["experiment"]["name"] = f"{base['experiment']['name']}__{ablation_name}"
        cfg["experiment"]["run_group"] = "ablation"
        outputs.extend(run_experiment_from_config(cfg))
    return outputs
