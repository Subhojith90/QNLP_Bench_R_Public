from __future__ import annotations

import platform
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import numpy as np

from qnlpbench_r import __version__
from qnlpbench_r.config import load_config, save_config, validate_config, with_single_seed_and_model
from qnlpbench_r.data.datasets import dataset_summary, load_dataset
from qnlpbench_r.data.preprocessing import fit_preprocessor, make_arrays
from qnlpbench_r.device import cuda_memory_summary, get_device_info, select_device
from qnlpbench_r.evaluation.evaluate import evaluate_splits
from qnlpbench_r.evaluation.robustness import evaluate_robustness
from qnlpbench_r.models import build_model
from qnlpbench_r.seed import set_global_seed
from qnlpbench_r.training.train import train_model
from qnlpbench_r.utils.io import disk_free_gb, metrics_to_csv, project_root_from_file, safe_copy_manifest, write_json
from qnlpbench_r.utils.logging_utils import setup_logger, write_warnings
from qnlpbench_r.utils.timing import elapsed_timer


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_dir(base_dir: str | Path, experiment_name: str, model_name: str, seed: int) -> Path:
    return Path(base_dir) / f"{_timestamp()}_{experiment_name.replace(' ', '_')}_{model_name.replace(' ', '_')}_seed{seed}"


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for module_name in ["numpy", "pandas", "sklearn", "torch", "yaml", "matplotlib"]:
        try:
            mod = __import__(module_name)
            packages[module_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[module_name] = "not_importable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "host_label": os.environ.get(
            "QNLPBENCH_HOST_LABEL", "redacted-local-host"
        ),
        "packages": packages,
        "command": " ".join(sys.argv),
    }


def write_environment_text(snapshot: dict[str, Any], run_dir: Path) -> None:
    with (run_dir / "environment_snapshot.txt").open("w", encoding="utf-8") as f:
        for k, v in snapshot.items():
            f.write(f"{k}: {v}\n")


def run_single_model(single_config: dict[str, Any], config_path: str | Path | None = None) -> Path:
    validate_config(single_config)
    seed = int(single_config.get("seed", single_config["seed_list"][0]))
    model_config = single_config["models"][0]
    exp_cfg = single_config["experiment"]
    run_dir = _run_dir(single_config["output"]["base_dir"], exp_cfg["name"], model_config["name"], seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger("qnlpbench_r", run_dir, level=str(single_config["logging"].get("level", "INFO")))
    warnings: list[str] = []
    metadata: dict[str, Any] = {"project_version": __version__, "experiment_name": exp_cfg["name"], "run_group": exp_cfg.get("run_group"), "model_name": model_config["name"], "model_type": model_config["type"], "seed": seed, "config_path": str(config_path) if config_path is not None else single_config.get("_config_path"), "status": "started", "start_time": datetime.now().isoformat()}
    with elapsed_timer() as elapsed:
        try:
            if disk_free_gb(run_dir) < 1.0:
                warnings.append("Less than 1 GiB free disk space detected near output directory.")
            set_global_seed(seed, deterministic_torch=bool(single_config["reproducibility"].get("deterministic_torch", True)))
            device = select_device(preference=str(single_config["device"].get("preference", "auto")), require_cuda=bool(single_config["device"].get("require_cuda", False)))
            if device.type == "cpu" and str(single_config["device"].get("preference", "auto")) == "auto":
                warnings.append("CUDA unavailable or not selected; running on CPU.")
            logger.info("Selected device: %s", device)
            save_config(single_config, run_dir / "config.yaml")
            safe_copy_manifest(project_root_from_file(), run_dir)
            env = environment_snapshot()
            write_environment_text(env, run_dir)
            metadata["environment"] = env
            metadata["device_info"] = get_device_info(device).to_dict()

            bundle = load_dataset(single_config, seed=seed, run_dir=run_dir)
            preprocessor = fit_preprocessor(bundle)
            arrays = make_arrays(bundle, preprocessor)
            if bool(model_config.get("random_label_training", False)):
                rng = np.random.default_rng(seed + 90417)
                X_train, y_train = arrays["train"]
                shuffled = rng.permutation(y_train).astype(np.int64)
                arrays["train"] = (X_train, shuffled)
                metadata["random_label_training"] = True
                write_json({"enabled": True, "seed": seed + 90417, "original_positive_rate": float(np.mean(y_train)), "shuffled_positive_rate": float(np.mean(shuffled))}, run_dir / "data" / "random_label_training_manifest.json")
            write_json(preprocessor.to_dict(), run_dir / "data" / "preprocessor.json")
            metadata["dataset_summary"] = dataset_summary(bundle)
            n_classes = int(bundle.all_data[bundle.label_column].nunique())
            if n_classes != 2:
                raise ValueError(f"This release supports binary classification; found {n_classes} classes.")
            model_config_for_build = dict(model_config)
            model_config_for_build.setdefault("seed", seed)
            model = build_model(model_config_for_build, input_dim=len(bundle.feature_columns), n_classes=n_classes, feature_columns=bundle.feature_columns)
            metadata["model_summary"] = model.model_summary() if hasattr(model, "model_summary") else {"parameter_count": sum(p.numel() for p in model.parameters())}
            logger.info("Model summary: %s", metadata["model_summary"])
            model, _, train_info = train_model(model, arrays, single_config["training"], device, seed, run_dir, logger)
            metadata["train_info"] = train_info
            metrics = evaluate_splits(model, arrays, device, list(single_config["evaluation"].get("splits", ["train", "val", "test", "ood"])), int(single_config["evaluation"].get("batch_size", 256)), run_dir, bool(single_config["evaluation"].get("quantum_metrics", True)), int(single_config["evaluation"].get("kernel_sample_cap", 256)), seed=seed, kernel_sampling_strategy=str(single_config["evaluation"].get("kernel_sampling_strategy", "stratified")))
            original_frames = {split: bundle.split(split) for split in ["train", "val", "test", "ood"]}
            metrics.update(evaluate_robustness(model, arrays, original_frames, preprocessor, bundle.feature_columns, device, single_config["evaluation"], seed))
            metrics.update({f"model_{k}": v for k, v in metadata["model_summary"].items() if isinstance(v, (int, float, bool))})
            metrics["elapsed_seconds"] = elapsed()
            metrics_to_csv(metrics, run_dir / "metrics.csv")
            write_json(metrics, run_dir / "metrics.json")
            metadata["status"] = "completed"
            metadata["elapsed_seconds"] = elapsed()
            metadata["cuda_memory_summary_end"] = cuda_memory_summary()
            write_warnings(run_dir, warnings)
            write_json(metadata, run_dir / "run_metadata.json")
            logger.info("Completed run in %.2f seconds", metadata["elapsed_seconds"])
            return run_dir
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["elapsed_seconds"] = elapsed()
            metadata["error"] = str(exc)
            metadata["traceback"] = traceback.format_exc()
            write_warnings(run_dir, warnings + [f"RUN FAILED: {exc}"])
            write_json(metadata, run_dir / "run_metadata.json")
            logger.exception("Run failed.")
            if not bool(single_config["output"].get("continue_on_error", False)):
                raise
            return run_dir


def run_experiment(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    outputs = []
    for seed in config["seed_list"]:
        for model_config in config["models"]:
            outputs.append(run_single_model(with_single_seed_and_model(config, int(seed), model_config), config_path=config_path))
    return outputs


def run_experiment_from_config(config: dict[str, Any]) -> list[Path]:
    validate_config(config)
    outputs = []
    for seed in config["seed_list"]:
        for model_config in config["models"]:
            outputs.append(run_single_model(with_single_seed_and_model(config, int(seed), model_config), config_path=config.get("_config_path")))
    return outputs
