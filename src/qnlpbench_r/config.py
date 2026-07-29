from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


class ConfigError(ValueError):
    """Raised when a YAML configuration is missing required keys or has invalid values."""


REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "seed_list",
    "device",
    "dataset",
    "models",
    "training",
    "evaluation",
    "output",
    "logging",
    "reproducibility",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration at {config_path} must be a YAML mapping.")
    validate_config(data)
    data["_config_path"] = str(config_path)
    return data


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate required keys and high-level consistency."""
    missing = REQUIRED_TOP_LEVEL_KEYS.difference(config.keys())
    if missing:
        raise ConfigError(f"Missing required top-level config keys: {sorted(missing)}")
    if not isinstance(config.get("seed_list"), list) or not config["seed_list"]:
        raise ConfigError("seed_list must be a non-empty list of integers.")
    if not all(isinstance(s, int) for s in config["seed_list"]):
        raise ConfigError("Every entry in seed_list must be an integer.")
    exp = config["experiment"]
    if not isinstance(exp, Mapping) or not exp.get("name"):
        raise ConfigError("experiment.name is required.")
    dataset = config["dataset"]
    if not isinstance(dataset, Mapping) or not dataset.get("name"):
        raise ConfigError("dataset.name is required.")
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ConfigError("models must be a non-empty list.")
    for m in models:
        if not isinstance(m, Mapping) or not m.get("name") or not m.get("type"):
            raise ConfigError("Each model entry must contain name and type.")
    training = config["training"]
    if int(training.get("batch_size", 0)) <= 0:
        raise ConfigError("training.batch_size must be positive.")
    if int(training.get("max_epochs", 0)) <= 0:
        raise ConfigError("training.max_epochs must be positive.")
    if float(training.get("learning_rate", 0.0)) <= 0.0:
        raise ConfigError("training.learning_rate must be positive.")
    evaluation = config["evaluation"]
    if not isinstance(evaluation.get("splits", []), list) or not evaluation["splits"]:
        raise ConfigError("evaluation.splits must be a non-empty list.")
    output = config["output"]
    if not output.get("base_dir"):
        raise ConfigError("output.base_dir is required.")


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a config dictionary to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = copy.deepcopy(dict(config))
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False)


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable short hash for a config mapping."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively update a dictionary and return it."""
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def with_single_seed_and_model(config: Mapping[str, Any], seed: int, model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Create an isolated config for one seed and one model."""
    out = copy.deepcopy(dict(config))
    out["seed_list"] = [int(seed)]
    out["seed"] = int(seed)
    out["models"] = [copy.deepcopy(dict(model_config))]
    return out
