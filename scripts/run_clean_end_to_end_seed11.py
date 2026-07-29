from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from qnlpbench_r.data.datasets import load_dataset  # noqa: E402
from qnlpbench_r.data.preprocessing import fit_preprocessor, make_arrays  # noqa: E402
from qnlpbench_r.models import build_model  # noqa: E402
from qnlpbench_r.models.tensor_train import TensorTrainClassifier  # noqa: E402
from qnlpbench_r.seed import set_global_seed  # noqa: E402
from qnlpbench_r.training.train import train_model  # noqa: E402
from qnlpbench_r.utils.io import write_json  # noqa: E402
from run_index_matched_comparison import (  # noqa: E402
    choose_svc,
    generic_candidates,
    learned_config,
    qfc_candidate,
    save_index,
)
from stage8a_common import (  # noqa: E402
    candidate_embeddings,
    index_runs,
    kernel_matrix,
    load_arrays,
    median_gamma,
    select_indices,
)
from stage8a_dataset_identity import compare_regenerated_dataset  # noqa: E402

REFERENCE_MLP_INITIAL_STATE_SHA256 = (
    "e6b1af8834687f965dea156ccb1cf03be67b6f85086c1f642f0ae7bfbcf9b158"
)
REFERENCE_MLP_56_EPOCH_SAMPLE_ORDER_SHA256 = (
    "a9b1f0c83f1348e8ad100e648236d73b4bdc67ebbe982fbb23d94d54b7ad3bcb"
)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(model: Any) -> str:
    return state_mapping_sha256(model.state_dict())


def state_mapping_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sample_order_sha256(
    n_samples: int,
    batch_size: int,
    seed: int,
    epochs: int,
) -> str:
    """Hash the exact DataLoader sample order independently of model arithmetic."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.arange(n_samples, dtype=torch.int64)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    digest = hashlib.sha256()
    for _ in range(epochs):
        for (indices,) in loader:
            digest.update(indices.numpy().tobytes(order="C"))
    return digest.hexdigest()


def installed_package_inventory() -> list[str]:
    return sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )


def direct_predict(model: Any, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x.astype(np.float32))).argmax(1).cpu().numpy()


def tensor_representation(model: TensorTrainClassifier, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (
            model.representation(torch.from_numpy(x.astype(np.float32)))
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/end_to_end_seed11",
        help="Directory for replay evidence; relative paths are resolved from the repository root.",
    )
    args = parser.parse_args()
    start_total = time.perf_counter()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage8a_end_to_end")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.FileHandler(output / "command_transcript.log", mode="w"))
    failure_path = output / "failure.log"
    failure_path.write_text("", encoding="utf-8")
    try:
        cfg = read_yaml(ROOT / "configs/stage8a.yaml")
        base = read_yaml(ROOT / "configs/stage6d_replication_base.yaml")
        seed = int(cfg["end_to_end_seed"])
        set_global_seed(seed, deterministic_torch=True)
        generated_dir = output / "generated_dataset"
        generated_dir.mkdir(exist_ok=True)
        bundle = load_dataset(base, seed=seed, run_dir=generated_dir)
        preprocessor = fit_preprocessor(bundle)
        arrays = make_arrays(bundle, preprocessor)
        write_json(preprocessor.to_dict(), generated_dir / "data/preprocessor.json")

        frozen_runs_root = ROOT / "inputs/stage6d_frozen/results/stage6d/runs"
        frozen_classical = index_runs(frozen_runs_root / "classical")
        frozen_reference = frozen_classical[(cfg["learned_kernels"]["mlp_model"], seed)]
        generated_csv = generated_dir / "data/dataset.csv"
        frozen_csv = frozen_reference / "data/dataset.csv"
        dataset_identity = compare_regenerated_dataset(
            generated_csv,
            frozen_csv,
            frozen_reference / "data/dataset_manifest.json",
            bundle.feature_columns,
        )
        write_json(dataset_identity, output / "generated_data_identity_gate.json")
        if dataset_identity["status"] != "pass":
            raise RuntimeError(
                "Freshly generated dataset failed the registered identity gate"
            )
        frozen_arrays = load_arrays(frozen_reference)
        for split in ["train", "val", "test", "ood_composition", "ood_depth"]:
            if not (
                np.array_equal(arrays[split][0], frozen_arrays[split][0])
                and np.array_equal(arrays[split][1], frozen_arrays[split][1])
            ):
                raise RuntimeError(f"Fresh array mismatch for split {split}")

        cap = int(cfg["sample_cap"])
        offsets = cfg["sampling"]
        indices = {
            "train": select_indices(
                arrays["train"][1], cap, seed + int(offsets["train_offset"]), "stratified"
            ),
            "val": select_indices(
                arrays["val"][1],
                cap,
                seed + int(offsets["validation_offset"]),
                "stratified",
            ),
        }
        for split in cfg["splits"]:
            indices[split] = select_indices(
                arrays[split][1],
                cap,
                seed + int(offsets["split_base_offset"]) + len(split),
                "stratified",
            )
        index_hashes = {}
        for split, index in indices.items():
            path = output / "indices" / f"{split}.npy"
            index_hashes[split] = save_index(path, index)
            primary_path = ROOT / f"results/index_matched/indices/seed_{seed}/{split}.npy"
            if sha256_file(path) != sha256_file(primary_path):
                raise RuntimeError(f"Fresh index hash mismatch for {split}")

        training_cfg = {
            **cfg["qfc_training"],
            "mixed_precision": False,
            "num_workers": 0,
            "gradient_accumulation_steps": 1,
        }
        mlp_config = {
            "name": "sequence_mlp_256x3",
            "type": "sequence_mlp",
            "hidden_dim": 256,
            "depth": 3,
            "dropout": 0.05,
            "seed": seed,
        }
        set_global_seed(seed, deterministic_torch=True)
        mlp = build_model(
            mlp_config,
            input_dim=len(bundle.feature_columns),
            n_classes=2,
            feature_columns=bundle.feature_columns,
        )
        mlp_initial_state_sha256 = state_dict_sha256(mlp)
        mlp_dir = output / "models/sequence_mlp_256x3"
        mlp_dir.mkdir(parents=True, exist_ok=True)
        mlp, mlp_history, mlp_train_info = train_model(
            mlp, arrays, training_cfg, torch.device("cpu"), seed, mlp_dir, log
        )
        mlp_trained_state_sha256 = state_dict_sha256(mlp)
        frozen_mlp_state = torch.load(
            frozen_reference / "model_best.pt",
            map_location="cpu",
            weights_only=True,
        )
        frozen_mlp_trained_state_sha256 = state_mapping_sha256(frozen_mlp_state)

        x_train = arrays["train"][0][indices["train"]]
        y_train = arrays["train"][1][indices["train"]]
        x_val = arrays["val"][0][indices["val"]]
        y_val = arrays["val"][1][indices["val"]]
        x_eval = {
            split: arrays[split][0][indices[split]] for split in cfg["splits"]
        }
        y_eval = {
            split: arrays[split][1][indices[split]] for split in cfg["splits"]
        }
        c_values = [float(value) for value in cfg["svm_c"]]
        generic, _ = choose_svc(
            generic_candidates(x_train, x_val, x_eval, cfg),
            y_train,
            y_val,
            c_values,
            seed,
            "generic",
        )
        learned, learned_selection = choose_svc(
            candidate_embeddings(
                x_train,
                x_val,
                x_eval,
                mlp,
                y_train,
                learned_config(cfg),
                seed,
            ),
            y_train,
            y_val,
            c_values,
            seed,
            "learned",
        )
        generic_clf = SVC(kernel="precomputed", C=float(generic["C"])).fit(
            generic["K_train"], y_train
        )
        learned_clf = SVC(kernel="precomputed", C=float(learned["C"])).fit(
            learned["K_train"], y_train
        )

        topology_specs = {
            "qfc_all_to_all": "all_to_all",
            "qfc_random_01": "random_fixed_01",
        }
        rows: list[dict[str, Any]] = []
        for topology, entanglement in topology_specs.items():
            set_global_seed(seed, deterministic_torch=True)
            qfc_model = build_model(
                {
                    "name": topology,
                    "type": "grammar_structured_quantum_feature",
                    "n_qubits": 6,
                    "n_layers": 2,
                    "structured_entanglement": entanglement,
                    "data_reuploading": True,
                    "seed": seed,
                },
                input_dim=len(bundle.feature_columns),
                n_classes=2,
                feature_columns=bundle.feature_columns,
            )
            qfc_dir = output / f"models/{topology}"
            qfc_dir.mkdir(parents=True, exist_ok=True)
            qfc_model, _, _ = train_model(
                qfc_model,
                arrays,
                training_cfg,
                torch.device("cpu"),
                seed,
                qfc_dir,
                log,
            )
            qfc, _ = choose_svc(
                [qfc_candidate(qfc_model, x_train, x_val, x_eval)],
                y_train,
                y_val,
                c_values,
                seed,
                "qfc",
            )
            qfc_clf = SVC(kernel="precomputed", C=float(qfc["C"])).fit(
                qfc["K_train"], y_train
            )
            for split in cfg["splits"]:
                rows.append(
                    {
                        "seed": seed,
                        "topology": topology,
                        "split": split,
                        "qfc_balanced_accuracy": float(
                            balanced_accuracy_score(
                                y_eval[split], qfc_clf.predict(qfc["K_eval"][split])
                            )
                        ),
                        "generic_balanced_accuracy": float(
                            balanced_accuracy_score(
                                y_eval[split],
                                generic_clf.predict(generic["K_eval"][split]),
                            )
                        ),
                        "learned_balanced_accuracy": float(
                            balanced_accuracy_score(
                                y_eval[split],
                                learned_clf.predict(learned["K_eval"][split]),
                            )
                        ),
                    }
                )

        tensor_candidates: list[tuple[float, int, TensorTrainClassifier]] = []
        tensor_cfg = {
            **cfg["tensor_train"],
            "mixed_precision": False,
            "num_workers": 0,
            "gradient_accumulation_steps": 1,
        }
        capped_arrays = {
            split: (arrays[split][0][index], arrays[split][1][index])
            for split, index in indices.items()
        }
        for bond_dim in cfg["tensor_train"]["bond_dimensions"]:
            candidate_seed = seed + int(bond_dim) * 1000
            set_global_seed(candidate_seed, deterministic_torch=True)
            model = TensorTrainClassifier(bundle.feature_columns, int(bond_dim))
            run_dir = output / f"models/tensor_train_bond_{bond_dim}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model, _, _ = train_model(
                model,
                capped_arrays,
                tensor_cfg,
                torch.device("cpu"),
                candidate_seed,
                run_dir,
                log,
            )
            val_ba = float(
                balanced_accuracy_score(
                    capped_arrays["val"][1],
                    direct_predict(model, capped_arrays["val"][0]),
                )
            )
            tensor_candidates.append((val_ba, int(bond_dim), model))
        _, selected_bond, tensor_model = max(
            tensor_candidates, key=lambda value: (value[0], -value[1])
        )
        train_rep = tensor_representation(tensor_model, x_train)
        val_rep = tensor_representation(tensor_model, x_val)
        gamma0 = median_gamma(train_rep)
        tensor_best: dict[str, Any] | None = None
        for multiplier in cfg["tensor_train"]["kernel_gamma_multipliers"]:
            gamma = gamma0 * float(multiplier)
            k_train = kernel_matrix(
                train_rep, train_rep, "rbf", {"gamma": gamma}
            )
            k_val = kernel_matrix(
                val_rep, train_rep, "rbf", {"gamma": gamma}
            )
            for c_value in c_values:
                clf = SVC(kernel="precomputed", C=c_value).fit(k_train, y_train)
                val_ba = float(
                    balanced_accuracy_score(y_val, clf.predict(k_val))
                )
                candidate = {
                    "validation_ba": val_ba,
                    "gamma": gamma,
                    "C": c_value,
                    "K_train": k_train,
                }
                if tensor_best is None or val_ba > tensor_best["validation_ba"]:
                    tensor_best = candidate
        assert tensor_best is not None
        tensor_clf = SVC(
            kernel="precomputed", C=float(tensor_best["C"])
        ).fit(tensor_best["K_train"], y_train)
        tensor_scores = {}
        for split in cfg["splits"]:
            k_eval = kernel_matrix(
                tensor_representation(tensor_model, x_eval[split]),
                train_rep,
                "rbf",
                {"gamma": tensor_best["gamma"]},
            )
            tensor_scores[split] = float(
                balanced_accuracy_score(y_eval[split], tensor_clf.predict(k_eval))
            )
        for row in rows:
            row["tensor_train_balanced_accuracy"] = tensor_scores[row["split"]]
            row["selected_tensor_train_bond_dimension"] = selected_bond
        observed = pd.DataFrame(rows)
        observed.to_csv(output / "end_to_end_seed11_results.csv", index=False)

        expected_primary = pd.read_csv(
            ROOT / "results/index_matched/stage8a_index_matched_seed_level.csv"
        )
        expected_primary = expected_primary[expected_primary.seed == seed]
        expected_tensor = pd.read_csv(
            ROOT / "results/tensor_train/stage8a_tensor_train_seed_level.csv"
        )
        expected_tensor = expected_tensor[expected_tensor.seed == seed][
            ["seed", "split", "kernel_balanced_accuracy"]
        ]
        comparison = observed.merge(
            expected_primary[
                [
                    "seed",
                    "topology",
                    "split",
                    "qfc_balanced_accuracy",
                    "generic_balanced_accuracy",
                    "learned_balanced_accuracy",
                ]
            ],
            on=["seed", "topology", "split"],
            suffixes=("_rerun", "_expected"),
        ).merge(expected_tensor, on=["seed", "split"])
        for metric in [
            "qfc_balanced_accuracy",
            "generic_balanced_accuracy",
            "learned_balanced_accuracy",
        ]:
            comparison[f"{metric}_absolute_error"] = np.abs(
                comparison[f"{metric}_rerun"] - comparison[f"{metric}_expected"]
            )
        comparison["tensor_train_absolute_error"] = np.abs(
            comparison["tensor_train_balanced_accuracy"]
            - comparison["kernel_balanced_accuracy"]
        )
        comparison.to_csv(output / "end_to_end_numerical_comparison.csv", index=False)
        error_columns = [column for column in comparison if column.endswith("_absolute_error")]
        max_error = float(comparison[error_columns].to_numpy().max())
        expected_history = pd.read_csv(frozen_reference / "history.csv")
        expected_run_metadata = json.loads(
            (frozen_reference / "run_metadata.json").read_text(encoding="utf-8")
        )
        common_history_columns = [
            column
            for column in expected_history.columns
            if column in mlp_history.columns
            and pd.api.types.is_numeric_dtype(expected_history[column])
            and pd.api.types.is_numeric_dtype(mlp_history[column])
        ]
        comparable_epochs = min(len(expected_history), len(mlp_history))
        history_differences: dict[str, float] = {}
        first_history_divergence: dict[str, Any] | None = None
        for column in common_history_columns:
            expected_values = expected_history[column].to_numpy(dtype=np.float64)[
                :comparable_epochs
            ]
            observed_values = mlp_history[column].to_numpy(dtype=np.float64)[
                :comparable_epochs
            ]
            differences = np.abs(observed_values - expected_values)
            maximum = float(np.nanmax(differences))
            history_differences[column] = maximum
            divergent = np.flatnonzero(
                ~np.isclose(
                    observed_values,
                    expected_values,
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=True,
                )
            )
            if divergent.size and first_history_divergence is None:
                index = int(divergent[0])
                first_history_divergence = {
                    "epoch": index + 1,
                    "column": column,
                    "expected": float(expected_values[index]),
                    "observed": float(observed_values[index]),
                    "absolute_difference": float(differences[index]),
                }

        selected_learned = {
            key: learned[key]
            for key in [
                "source",
                "kernel_family",
                "kernel_name",
                "parameters_json",
                "C",
                "validation_balanced_accuracy",
                "candidate_order",
                "c_order",
            ]
        }
        expected_seed_rows = expected_primary[
            expected_primary["seed"] == seed
        ]
        expected_selection = {
            "source": str(expected_seed_rows.iloc[0]["learned_source"]),
            "kernel_name": str(expected_seed_rows.iloc[0]["learned_kernel"]),
            "C": float(expected_seed_rows.iloc[0]["learned_C"]),
        }
        selection_match = (
            selected_learned["source"] == expected_selection["source"]
            and selected_learned["kernel_name"] == expected_selection["kernel_name"]
            and float(selected_learned["C"]) == expected_selection["C"]
        )
        reference_history_epochs = len(expected_history)
        sample_order_digest = sample_order_sha256(
            arrays["train"][0].shape[0],
            int(training_cfg["batch_size"]),
            seed,
            reference_history_epochs,
        )
        expected_train_info = expected_run_metadata["train_info"]
        causal_trace = {
            "reference_environment": {
                "python_version": expected_run_metadata["device_info"][
                    "python_version"
                ],
                "platform": expected_run_metadata["device_info"]["platform"],
                "torch_version": expected_run_metadata["device_info"][
                    "torch_version"
                ],
            },
            "initialization": {
                "registered_sha256": REFERENCE_MLP_INITIAL_STATE_SHA256,
                "observed_sha256": mlp_initial_state_sha256,
                "exact_match": (
                    mlp_initial_state_sha256
                    == REFERENCE_MLP_INITIAL_STATE_SHA256
                ),
            },
            "minibatch_order": {
                "epochs_hashed": reference_history_epochs,
                "registered_sha256": (
                    REFERENCE_MLP_56_EPOCH_SAMPLE_ORDER_SHA256
                ),
                "observed_sha256": sample_order_digest,
                "exact_match": (
                    sample_order_digest
                    == REFERENCE_MLP_56_EPOCH_SAMPLE_ORDER_SHA256
                ),
            },
            "training_history": {
                "first_divergence_at_1e-12": first_history_divergence,
                "reference_rows": reference_history_epochs,
                "observed_rows": len(mlp_history),
                "maximum_absolute_difference_by_column": history_differences,
            },
            "early_stopping_and_checkpoint": {
                "reference_train_info": expected_train_info,
                "observed_train_info": mlp_train_info,
                "reference_trained_state_sha256": (
                    frozen_mlp_trained_state_sha256
                ),
                "observed_trained_state_sha256": mlp_trained_state_sha256,
                "trained_state_exact_match": (
                    frozen_mlp_trained_state_sha256
                    == mlp_trained_state_sha256
                ),
            },
            "learned_kernel_selection": {
                "reference": expected_selection,
                "observed": selected_learned,
                "exact_match": selection_match,
            },
            "interpretation": (
                "When initialization and minibatch order match but the training "
                "history diverges, the first unresolved boundary is floating-point "
                "model arithmetic or its backward/optimizer path. Later early-"
                "stopping, checkpoint, and learned-kernel differences are downstream."
            ),
        }
        write_json(causal_trace, output / "mlp_portability_diagnosis.json")
        mlp_trace = {
            "initial_state_sha256": mlp_initial_state_sha256,
            "trained_state_sha256": mlp_trained_state_sha256,
            "train_info": mlp_train_info,
            "observed_history_rows": len(mlp_history),
            "expected_history_rows": len(expected_history),
            "history_maximum_absolute_difference_by_column": history_differences,
            "first_history_divergence_at_1e-12": first_history_divergence,
            "selected_learned_candidate": selected_learned,
            "expected_learned_candidate": expected_selection,
            "learned_selection_match": selection_match,
            "selection_candidate_count": len(learned_selection),
            "causal_trace_file": "mlp_portability_diagnosis.json",
        }
        write_json(mlp_trace, output / "mlp_replay_trace.json")
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "host_label": os.environ.get(
                "QNLPBENCH_HOST_LABEL", "redacted-local-host"
            ),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "cpu_count": __import__("os").cpu_count(),
            "device": "cpu",
        }
        write_json(environment, output / "environment.json")
        (output / "installed_packages.txt").write_text(
            "\n".join(installed_package_inventory()) + "\n",
            encoding="utf-8",
        )
        deterministic_settings = {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "python_hash_seed_note": (
                "PYTHONHASHSEED is read from the process environment and only "
                "governs hash randomization when set before interpreter startup."
            ),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get(
                "VECLIB_MAXIMUM_THREADS"
            ),
            "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        }
        write_json(
            deterministic_settings, output / "deterministic_settings.json"
        )
        replay_passed = (
            dataset_identity["status"] == "pass"
            and selection_match
            and max_error <= 1.0e-12
        )
        report = {
            "stage": "8A",
            "seed": seed,
            "status": "pass" if replay_passed else "fail",
            "dataset_identity_gate_status": dataset_identity["status"],
            "dataset_bytewise_hash_match": dataset_identity[
                "bytewise_hash_match"
            ],
            "frozen_dataset_sha256": dataset_identity[
                "observed_frozen_sha256"
            ],
            "index_hashes": index_hashes,
            "maximum_absolute_metric_error": max_error,
            "tolerance": 1.0e-12,
            "runtime_seconds": time.perf_counter() - start_total,
            "selected_tensor_train_bond_dimension": selected_bond,
            "learned_selection_match": selection_match,
            "first_mlp_history_divergence_at_1e-12": first_history_divergence,
        }
        write_json(report, output / "end_to_end_report.json")
        if report["status"] != "pass":
            raise RuntimeError(
                "End-to-end replay failed: "
                f"dataset_gate={dataset_identity['status']}, "
                f"learned_selection_match={selection_match}, "
                f"maximum_absolute_metric_error={max_error}"
            )
        manifest = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                manifest[str(path.relative_to(output))] = {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
        write_json(manifest, output / "artifact_manifest.json")
        print(
            f"End-to-end seed {seed} PASS; maximum absolute error={max_error:.3e}."
        )
    except Exception as error:
        failure_path.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
