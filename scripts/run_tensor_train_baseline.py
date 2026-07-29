from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from qnlpbench_r.models.tensor_train import TensorTrainClassifier  # noqa: E402
from qnlpbench_r.seed import set_global_seed  # noqa: E402
from qnlpbench_r.training.train import train_model  # noqa: E402
from stage8a_common import index_runs, kernel_matrix, load_arrays, median_gamma  # noqa: E402


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direct_predict(model: TensorTrainClassifier, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (
            model(torch.from_numpy(x.astype(np.float32)))
            .argmax(dim=1)
            .cpu()
            .numpy()
        )


def representation(model: TensorTrainClassifier, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (
            model.representation(torch.from_numpy(x.astype(np.float32)))
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage8a.yaml")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument(
        "--indices", type=Path, default=ROOT / "results/index_matched/indices"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/tensor_train")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    seeds = list(cfg["seeds"])
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    index_root = args.indices if args.indices.is_absolute() else ROOT / args.indices
    output.mkdir(parents=True, exist_ok=True)
    model_output = output / "models"
    model_output.mkdir(exist_ok=True)

    frozen_runs = ROOT / "inputs/stage6d_frozen/results/stage6d/runs"
    classical_runs = index_runs(frozen_runs / "classical")
    mlp_name = cfg["learned_kernels"]["mlp_model"]
    training_cfg = {
        **cfg["tensor_train"],
        "mixed_precision": False,
        "num_workers": 0,
        "gradient_accumulation_steps": 1,
    }
    logger = logging.getLogger("stage8a_tensor_train")
    logger.addHandler(logging.NullHandler())
    selection_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for seed in seeds:
        reference_dir = classical_runs[(mlp_name, seed)]
        arrays = load_arrays(reference_dir)
        feature_columns = read_json(reference_dir / "data/preprocessor.json")[
            "feature_columns"
        ]
        indices = {
            split: np.load(index_root / f"seed_{seed}/{split}.npy", allow_pickle=False)
            for split in ["train", "val", "test", "ood_composition", "ood_depth"]
        }
        capped_arrays = {
            split: (arrays[split][0][index], arrays[split][1][index])
            for split, index in indices.items()
        }
        candidates: list[tuple[float, int, TensorTrainClassifier, dict[str, Any], float]] = []
        for bond_dim in cfg["tensor_train"]["bond_dimensions"]:
            candidate_seed = seed + int(bond_dim) * 1000
            set_global_seed(candidate_seed, deterministic_torch=True)
            torch.use_deterministic_algorithms(True)
            model = TensorTrainClassifier(
                feature_columns=feature_columns,
                bond_dim=int(bond_dim),
                n_classes=2,
            )
            candidate_dir = model_output / f"seed_{seed}/bond_{bond_dim}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            model, history, train_info = train_model(
                model,
                capped_arrays,
                training_cfg,
                torch.device("cpu"),
                candidate_seed,
                candidate_dir,
                logger,
            )
            runtime = time.perf_counter() - start
            validation_prediction = direct_predict(model, capped_arrays["val"][0])
            validation_ba = float(
                balanced_accuracy_score(capped_arrays["val"][1], validation_prediction)
            )
            converged = len(history) < int(training_cfg["max_epochs"])
            record = {
                "seed": seed,
                "bond_dimension": int(bond_dim),
                "validation_balanced_accuracy": validation_ba,
                "parameter_count": model.parameter_count(),
                "runtime_seconds": runtime,
                "best_epoch": int(train_info["best_epoch"]),
                "epochs_run": len(history),
                "converged_by_early_stopping": bool(converged),
                "selected": False,
                "checkpoint_sha256": sha256_file(candidate_dir / "model_best.pt"),
            }
            selection_rows.append(record)
            candidates.append((validation_ba, int(bond_dim), model, record, runtime))
        selected = max(candidates, key=lambda item: (item[0], -item[1]))
        _, selected_bond, selected_model, selected_record, selected_runtime = selected
        selected_record["selected"] = True

        train_rep = representation(selected_model, capped_arrays["train"][0])
        val_rep = representation(selected_model, capped_arrays["val"][0])
        eval_rep = {
            split: representation(selected_model, capped_arrays[split][0])
            for split in ["test", "ood_composition", "ood_depth"]
        }
        y_train = capped_arrays["train"][1]
        y_val = capped_arrays["val"][1]
        gamma_base = median_gamma(train_rep)
        best_kernel: dict[str, Any] | None = None
        for multiplier in cfg["tensor_train"]["kernel_gamma_multipliers"]:
            gamma = gamma_base * float(multiplier)
            params = {"gamma": gamma}
            k_train = kernel_matrix(train_rep, train_rep, "rbf", params)
            k_val = kernel_matrix(val_rep, train_rep, "rbf", params)
            for c_value in cfg["svm_c"]:
                classifier = SVC(kernel="precomputed", C=float(c_value))
                classifier.fit(k_train, y_train)
                score = float(
                    balanced_accuracy_score(y_val, classifier.predict(k_val))
                )
                candidate = {
                    "gamma_multiplier": float(multiplier),
                    "gamma": gamma,
                    "C": float(c_value),
                    "validation_balanced_accuracy": score,
                    "K_train": k_train,
                }
                if (
                    best_kernel is None
                    or score > best_kernel["validation_balanced_accuracy"]
                ):
                    best_kernel = candidate
        if best_kernel is None:
            raise RuntimeError("No tensor-train kernel candidate selected")
        classifier = SVC(kernel="precomputed", C=best_kernel["C"])
        classifier.fit(best_kernel["K_train"], y_train)
        kernel_payload: dict[str, np.ndarray] = {
            "tensor_train_train": best_kernel["K_train"],
            "y_train": y_train,
        }
        for split in ["test", "ood_composition", "ood_depth"]:
            x_eval, y_eval = capped_arrays[split]
            direct_ba = float(
                balanced_accuracy_score(y_eval, direct_predict(selected_model, x_eval))
            )
            k_eval = kernel_matrix(
                eval_rep[split],
                train_rep,
                "rbf",
                {"gamma": best_kernel["gamma"]},
            )
            kernel_ba = float(
                balanced_accuracy_score(y_eval, classifier.predict(k_eval))
            )
            kernel_payload[f"tensor_train_{split}"] = k_eval
            result_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "selected_bond_dimension": selected_bond,
                    "parameter_count": selected_model.parameter_count(),
                    "training_runtime_seconds": selected_runtime,
                    "direct_balanced_accuracy": direct_ba,
                    "kernel_balanced_accuracy": kernel_ba,
                    "kernel_gamma_multiplier": best_kernel["gamma_multiplier"],
                    "kernel_gamma": best_kernel["gamma"],
                    "kernel_C": best_kernel["C"],
                    "kernel_validation_balanced_accuracy": best_kernel[
                        "validation_balanced_accuracy"
                    ],
                }
            )
        np.savez_compressed(output / f"seed_{seed}_tensor_train_kernels.npz", **kernel_payload)

    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output / "stage8a_tensor_train_selection.csv", index=False)
    results = pd.DataFrame(result_rows)
    results.to_csv(output / "stage8a_tensor_train_seed_level.csv", index=False)
    summary = (
        results.groupby("split", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            direct_balanced_accuracy_mean=("direct_balanced_accuracy", "mean"),
            kernel_balanced_accuracy_mean=("kernel_balanced_accuracy", "mean"),
            parameter_count_mean=("parameter_count", "mean"),
            training_runtime_seconds_mean=("training_runtime_seconds", "mean"),
        )
    )
    summary.to_csv(output / "stage8a_tensor_train_summary.csv", index=False)
    manifest = {
        "stage": "8A",
        "analysis": "tensor_train_baseline",
        "seeds": seeds,
        "bond_dimension_grid": cfg["tensor_train"]["bond_dimensions"],
        "selection_metric": "validation_balanced_accuracy",
        "sample_cap": cfg["sample_cap"],
        "seed_level_sha256": sha256_file(output / "stage8a_tensor_train_seed_level.csv"),
        "selection_sha256": sha256_file(output / "stage8a_tensor_train_selection.csv"),
    }
    with (output / "stage8a_tensor_train_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Completed tensor-train baseline for {len(seeds)} seeds.")


if __name__ == "__main__":
    main()
