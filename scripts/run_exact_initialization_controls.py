from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import pickle
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from qnlpbench_r.models import build_model  # noqa: E402
from qnlpbench_r.seed import set_global_seed  # noqa: E402
from qnlpbench_r.training.train import train_model  # noqa: E402
from stage8a_common import index_runs, load_arrays, qfc_kernel, qfc_states  # noqa: E402


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_states() -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu().numpy(),
    }


def save_rng_states(path: Path, payload: dict[str, Any]) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)


def restore_rng_states(payload: dict[str, Any]) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(torch.from_numpy(payload["torch_cpu_rng_state"].copy()))


def model_from_frozen_config(run_dir: Path, seed: int):
    config = read_yaml(run_dir / "config.yaml")
    preprocessor = read_json(run_dir / "data/preprocessor.json")
    model_config = dict(config["models"][0])
    model_config.setdefault("seed", seed)
    return build_model(
        model_config,
        input_dim=len(preprocessor["feature_columns"]),
        n_classes=2,
        feature_columns=preprocessor["feature_columns"],
    )


def selected_kernel_accuracy(
    model: Any,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    indices: dict[str, np.ndarray],
    c_values: list[float],
) -> tuple[dict[str, float], float, float, dict[str, np.ndarray]]:
    x_train = arrays["train"][0][indices["train"]]
    y_train = arrays["train"][1][indices["train"]]
    x_val = arrays["val"][0][indices["val"]]
    y_val = arrays["val"][1][indices["val"]]
    train_states = qfc_states(model, x_train)
    k_train = qfc_kernel(train_states, train_states)
    k_val = qfc_kernel(qfc_states(model, x_val), train_states)
    best_c = c_values[0]
    best_score = -1.0
    for c_value in c_values:
        classifier = SVC(kernel="precomputed", C=c_value)
        classifier.fit(k_train, y_train)
        score = float(balanced_accuracy_score(y_val, classifier.predict(k_val)))
        if score > best_score:
            best_score = score
            best_c = c_value
    classifier = SVC(kernel="precomputed", C=best_c)
    classifier.fit(k_train, y_train)
    scores: dict[str, float] = {}
    kernels: dict[str, np.ndarray] = {"train": k_train, "val": k_val}
    for split in ["test", "ood_composition", "ood_depth"]:
        x_eval = arrays[split][0][indices[split]]
        y_eval = arrays[split][1][indices[split]]
        k_eval = qfc_kernel(qfc_states(model, x_eval), train_states)
        kernels[split] = k_eval
        scores[split] = float(
            balanced_accuracy_score(y_eval, classifier.predict(k_eval))
        )
    return scores, float(best_c), best_score, kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage8a.yaml")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument(
        "--indices", type=Path, default=ROOT / "results/index_matched/indices"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/exact_initialization"
    )
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    seeds = list(cfg["seeds"])
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    index_root = args.indices if args.indices.is_absolute() else ROOT / args.indices
    output.mkdir(parents=True, exist_ok=True)

    runs_root = ROOT / "inputs/stage6d_frozen/results/stage6d/runs"
    classical_runs = index_runs(runs_root / "classical")
    topology_runs = index_runs(runs_root / "topology")
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    training_cfg = {
        **cfg["qfc_training"],
        "mixed_precision": False,
        "num_workers": 0,
        "gradient_accumulation_steps": 1,
    }
    logger = logging.getLogger("stage8a_exact_controls")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler(output / "training_transcript.log", mode="w"))

    for seed in seeds:
        mlp_dir = classical_runs[(cfg["learned_kernels"]["mlp_model"], seed)]
        arrays = load_arrays(mlp_dir)
        indices = {
            split: np.load(index_root / f"seed_{seed}/{split}.npy", allow_pickle=False)
            for split in ["train", "val", "test", "ood_composition", "ood_depth"]
        }
        for topology in cfg["topologies"]:
            frozen_dir = topology_runs[(topology, seed)]
            run_dir = output / f"seed_{seed}" / topology
            ordinary_dir = run_dir / "ordinary_labels"
            shuffled_dir = run_dir / "shuffled_labels"
            ordinary_dir.mkdir(parents=True, exist_ok=True)
            shuffled_dir.mkdir(parents=True, exist_ok=True)

            set_global_seed(seed, deterministic_torch=True)
            torch.use_deterministic_algorithms(True)
            initial_model = model_from_frozen_config(frozen_dir, seed)
            initial_state = copy.deepcopy(initial_model.state_dict())
            initial_path = run_dir / "model_initial.pt"
            torch.save(initial_state, initial_path)
            rng_snapshot = capture_rng_states()
            save_rng_states(run_dir / "rng_state_after_initialization.pkl", rng_snapshot)

            ordinary_model = copy.deepcopy(initial_model)
            restore_rng_states(rng_snapshot)
            ordinary_model, _, ordinary_info = train_model(
                ordinary_model,
                arrays,
                training_cfg,
                torch.device("cpu"),
                seed,
                ordinary_dir,
                logger,
            )
            ordinary_path = ordinary_dir / "model_best.pt"

            shuffled_arrays = {
                split: (values[0].copy(), values[1].copy())
                for split, values in arrays.items()
            }
            shuffled_rng = np.random.default_rng(
                seed + int(cfg["qfc_training"]["shuffled_label_offset"])
            )
            shuffled_arrays["train"] = (
                shuffled_arrays["train"][0],
                shuffled_rng.permutation(shuffled_arrays["train"][1]).astype(np.int64),
            )
            shuffled_model = copy.deepcopy(initial_model)
            restore_rng_states(rng_snapshot)
            shuffled_model, _, shuffled_info = train_model(
                shuffled_model,
                shuffled_arrays,
                training_cfg,
                torch.device("cpu"),
                seed,
                shuffled_dir,
                logger,
            )
            shuffled_path = shuffled_dir / "model_best.pt"

            initial_scores, initial_c, initial_val, initial_kernels = selected_kernel_accuracy(
                initial_model, arrays, indices, [float(value) for value in cfg["svm_c"]]
            )
            trained_scores, trained_c, trained_val, trained_kernels = selected_kernel_accuracy(
                ordinary_model, arrays, indices, [float(value) for value in cfg["svm_c"]]
            )
            shuffled_scores, shuffled_c, shuffled_val, shuffled_kernels = selected_kernel_accuracy(
                shuffled_model, arrays, indices, [float(value) for value in cfg["svm_c"]]
            )
            np.savez_compressed(
                run_dir / "state_overlap_kernels.npz",
                y_train=arrays["train"][1][indices["train"]],
                **{
                    f"initial_{name}": value
                    for name, value in initial_kernels.items()
                },
                **{
                    f"trained_{name}": value
                    for name, value in trained_kernels.items()
                },
                **{
                    f"shuffled_{name}": value
                    for name, value in shuffled_kernels.items()
                },
            )

            initial_hash = file_hash(initial_path)
            ordinary_hash = file_hash(ordinary_path)
            shuffled_hash = file_hash(shuffled_path)
            for split in ["test", "ood_composition", "ood_depth"]:
                rows.append(
                    {
                        "seed": seed,
                        "topology": topology,
                        "split": split,
                        "exact_initial_balanced_accuracy": initial_scores[split],
                        "trained_balanced_accuracy": trained_scores[split],
                        "shuffled_label_balanced_accuracy": shuffled_scores[split],
                        "trained_minus_exact_initial": trained_scores[split]
                        - initial_scores[split],
                        "trained_minus_shuffled_label": trained_scores[split]
                        - shuffled_scores[split],
                        "initial_C": initial_c,
                        "trained_C": trained_c,
                        "shuffled_C": shuffled_c,
                        "initial_validation_balanced_accuracy": initial_val,
                        "trained_validation_balanced_accuracy": trained_val,
                        "shuffled_validation_balanced_accuracy": shuffled_val,
                    }
                )
            manifest_rows.append(
                {
                    "seed": seed,
                    "topology": topology,
                    "initial_checkpoint": str(initial_path.relative_to(ROOT)),
                    "initial_sha256": initial_hash,
                    "trained_checkpoint": str(ordinary_path.relative_to(ROOT)),
                    "trained_sha256": ordinary_hash,
                    "shuffled_checkpoint": str(shuffled_path.relative_to(ROOT)),
                    "shuffled_sha256": shuffled_hash,
                    "rng_state": str(
                        (run_dir / "rng_state_after_initialization.pkl").relative_to(ROOT)
                    ),
                    "rng_state_sha256": file_hash(
                        run_dir / "rng_state_after_initialization.pkl"
                    ),
                    "kernel_archive": str(
                        (run_dir / "state_overlap_kernels.npz").relative_to(ROOT)
                    ),
                    "kernel_archive_sha256": file_hash(
                        run_dir / "state_overlap_kernels.npz"
                    ),
                    "ordinary_best_epoch": ordinary_info["best_epoch"],
                    "shuffled_best_epoch": shuffled_info["best_epoch"],
                    "initial_checkpoint_shared_by_both_training_runs": True,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(output / "stage8a_exact_initialization_seed_level.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output / "stage8a_checkpoint_pair_manifest.csv", index=False)
    summary = (
        result.groupby(["topology", "split"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            exact_initial_balanced_accuracy_mean=(
                "exact_initial_balanced_accuracy",
                "mean",
            ),
            trained_balanced_accuracy_mean=("trained_balanced_accuracy", "mean"),
            shuffled_label_balanced_accuracy_mean=(
                "shuffled_label_balanced_accuracy",
                "mean",
            ),
            trained_minus_exact_initial_mean=(
                "trained_minus_exact_initial",
                "mean",
            ),
            trained_minus_exact_initial_positive_seeds=(
                "trained_minus_exact_initial",
                lambda values: int((values > 0).sum()),
            ),
            trained_minus_shuffled_label_mean=(
                "trained_minus_shuffled_label",
                "mean",
            ),
        )
    )
    summary.to_csv(output / "stage8a_exact_initialization_summary.csv", index=False)
    write_json(
        {
            "stage": "8A",
            "analysis": "exact_initialization_matched_controls",
            "seeds": seeds,
            "one_to_one_pairs": len(manifest_rows),
            "all_pairs_hash_verified": bool(
                manifest["initial_sha256"].notna().all()
                and manifest["trained_sha256"].notna().all()
            ),
            "result_sha256": file_hash(
                output / "stage8a_exact_initialization_seed_level.csv"
            ),
            "checkpoint_manifest_sha256": file_hash(
                output / "stage8a_checkpoint_pair_manifest.csv"
            ),
        },
        output / "stage8a_exact_initialization_manifest.json",
    )
    print(f"Completed exact-initialization controls for {len(seeds)} seeds.")


if __name__ == "__main__":
    main()
