from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def center_kernel(kernel: np.ndarray) -> np.ndarray:
    n = kernel.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    return centering @ kernel @ centering


def frobenius_alignment(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left, "fro") * np.linalg.norm(right, "fro")
    if denominator <= 1.0e-15:
        return float("nan")
    return float(np.sum(left * right) / denominator)


def kernel_statistics(
    kernel: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    tolerance: float,
    condition_floor: float,
) -> tuple[dict[str, float], np.ndarray]:
    symmetric = (kernel + kernel.T) / 2.0
    centered = center_kernel(symmetric)
    eigenvalues = np.linalg.eigvalsh(centered)[::-1]
    positive = eigenvalues[eigenvalues > tolerance]
    total = float(positive.sum())
    if total > 0:
        probabilities = positive / total
        effective_rank = float(
            np.exp(-np.sum(probabilities * np.log(probabilities.clip(min=1.0e-300))))
        )
        dominant_fraction = float(positive[0] / total)
        stable_rank = float(np.sum(positive**2) / positive[0] ** 2)
        condition_number = float(positive[0] / max(positive[-1], condition_floor))
    else:
        effective_rank = dominant_fraction = stable_rank = condition_number = float("nan")
    signed_labels = np.where(labels == 1, 1.0, -1.0)
    target = np.outer(signed_labels, signed_labels)
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    same_class = (labels[:, None] == labels[None, :]) & off_diagonal
    different_class = (labels[:, None] != labels[None, :]) & off_diagonal
    classifier = SVC(kernel="precomputed", C=float(c_value))
    classifier.fit(symmetric, labels)
    decision = classifier.decision_function(symmetric)
    signed_margin = signed_labels * decision
    values = {
        "centered_kernel_target_alignment": frobenius_alignment(
            centered, center_kernel(target)
        ),
        "effective_rank_entropy": effective_rank,
        "stable_rank": stable_rank,
        "dominant_eigenvalue_fraction": dominant_fraction,
        "condition_number_positive_spectrum": condition_number,
        "positive_eigenvalue_count": int(len(positive)),
        "within_class_similarity_mean": float(symmetric[same_class].mean()),
        "between_class_similarity_mean": float(symmetric[different_class].mean()),
        "within_minus_between_similarity": float(
            symmetric[same_class].mean() - symmetric[different_class].mean()
        ),
        "support_vector_fraction": float(len(classifier.support_) / len(labels)),
        "signed_margin_mean": float(signed_margin.mean()),
        "signed_margin_q05": float(np.quantile(signed_margin, 0.05)),
        "signed_margin_q25": float(np.quantile(signed_margin, 0.25)),
        "signed_margin_median": float(np.quantile(signed_margin, 0.50)),
        "signed_margin_q75": float(np.quantile(signed_margin, 0.75)),
        "signed_margin_q95": float(np.quantile(signed_margin, 0.95)),
    }
    return values, eigenvalues


def normalized_displacement(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first, "fro")
    return float(np.linalg.norm(second - first, "fro") / max(denominator, 1.0e-15))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage8a.yaml")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument(
        "--index-results",
        type=Path,
        default=ROOT / "results/index_matched/stage8a_index_matched_seed_level.csv",
    )
    parser.add_argument(
        "--index-kernels", type=Path, default=ROOT / "results/index_matched/kernels"
    )
    parser.add_argument(
        "--exact-root", type=Path, default=ROOT / "results/exact_initialization"
    )
    parser.add_argument(
        "--tensor-root", type=Path, default=ROOT / "results/tensor_train"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    index_results = pd.read_csv(args.index_results)
    tensor_results = pd.read_csv(args.tensor_root / "stage8a_tensor_train_seed_level.csv")
    exact_results = pd.read_csv(
        args.exact_root / "stage8a_exact_initialization_seed_level.csv"
    )
    tolerance = float(cfg["diagnostics"]["eigenvalue_tolerance"])
    condition_floor = float(cfg["diagnostics"]["condition_floor"])

    diagnostic_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    seeds = list(cfg["seeds"])
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    for seed in seeds:
        tensor_archive = np.load(
            args.tensor_root / f"seed_{seed}_tensor_train_kernels.npz"
        )
        for topology in cfg["topologies"]:
            primary_archive = np.load(
                args.index_kernels / f"seed_{seed}_{topology}.npz"
            )
            exact_archive = np.load(
                args.exact_root
                / f"seed_{seed}/{topology}/state_overlap_kernels.npz"
            )
            labels = primary_archive["y_train"]
            if not np.array_equal(labels, tensor_archive["y_train"]):
                raise RuntimeError(f"Tensor-train label mismatch for seed {seed}")
            if not np.array_equal(labels, exact_archive["y_train"]):
                raise RuntimeError(f"Exact-control label mismatch for seed {seed}, {topology}")
            primary_row = index_results[
                (index_results.seed == seed)
                & (index_results.topology == topology)
            ].iloc[0]
            tensor_row = tensor_results[tensor_results.seed == seed].iloc[0]
            exact_row = exact_results[
                (exact_results.seed == seed)
                & (exact_results.topology == topology)
            ].iloc[0]
            kernels = {
                "frozen_trained_qfc": (
                    primary_archive["qfc_train"],
                    float(primary_row.qfc_C),
                ),
                "selected_learned_classical": (
                    primary_archive["learned_train"],
                    float(primary_row.learned_C),
                ),
                "selected_tensor_train": (
                    tensor_archive["tensor_train_train"],
                    float(tensor_row.kernel_C),
                ),
                "exact_initial_qfc": (
                    exact_archive["initial_train"],
                    float(exact_row.initial_C),
                ),
                "new_trained_qfc": (
                    exact_archive["trained_train"],
                    float(exact_row.trained_C),
                ),
                "label_shuffled_qfc": (
                    exact_archive["shuffled_train"],
                    float(exact_row.shuffled_C),
                ),
            }
            for family, (kernel, c_value) in kernels.items():
                stats, eigenvalues = kernel_statistics(
                    kernel, labels, c_value, tolerance, condition_floor
                )
                diagnostic_rows.append(
                    {
                        "seed": seed,
                        "topology": topology,
                        "kernel": family,
                        "C": c_value,
                        **stats,
                    }
                )
                for rank, eigenvalue in enumerate(eigenvalues, start=1):
                    spectrum_rows.append(
                        {
                            "seed": seed,
                            "topology": topology,
                            "kernel": family,
                            "eigenvalue_rank": rank,
                            "eigenvalue": float(eigenvalue),
                        }
                    )
            primary_centered = center_kernel(primary_archive["qfc_train"])
            learned_centered = center_kernel(primary_archive["learned_train"])
            tensor_centered = center_kernel(tensor_archive["tensor_train_train"])
            exact_initial = exact_archive["initial_train"]
            exact_trained = exact_archive["trained_train"]
            comparison_rows.append(
                {
                    "seed": seed,
                    "topology": topology,
                    "qfc_learned_centered_alignment": frobenius_alignment(
                        primary_centered, learned_centered
                    ),
                    "qfc_tensor_train_centered_alignment": frobenius_alignment(
                        primary_centered, tensor_centered
                    ),
                    "exact_training_kernel_displacement": normalized_displacement(
                        exact_initial, exact_trained
                    ),
                    "exact_initial_trained_centered_alignment": frobenius_alignment(
                        center_kernel(exact_initial), center_kernel(exact_trained)
                    ),
                }
            )

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(output / "stage8a_kernel_diagnostics_seed_level.csv", index=False)
    pd.DataFrame(spectrum_rows).to_csv(
        output / "stage8a_kernel_eigenspectra.csv", index=False
    )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(
        output / "stage8a_cross_kernel_comparisons_seed_level.csv", index=False
    )
    metric_columns = [
        column
        for column in diagnostics.columns
        if column not in {"seed", "topology", "kernel"}
    ]
    diagnostics.groupby(["topology", "kernel"], as_index=False)[metric_columns].mean().to_csv(
        output / "stage8a_kernel_diagnostics_summary.csv", index=False
    )
    comparison_metrics = [
        column for column in comparisons.columns if column not in {"seed", "topology"}
    ]
    comparisons.groupby("topology", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{
            column: (column, "mean")
            for column in comparison_metrics
        },
    ).to_csv(output / "stage8a_cross_kernel_comparisons_summary.csv", index=False)
    manifest = {
        "stage": "8A",
        "analysis": "focused_kernel_diagnostics",
        "eigenvalue_tolerance": tolerance,
        "condition_floor": condition_floor,
        "seed_level_sha256": sha256_file(
            output / "stage8a_kernel_diagnostics_seed_level.csv"
        ),
        "eigenspectra_sha256": sha256_file(
            output / "stage8a_kernel_eigenspectra.csv"
        ),
        "cross_kernel_sha256": sha256_file(
            output / "stage8a_cross_kernel_comparisons_seed_level.csv"
        ),
    }
    with (output / "stage8a_kernel_diagnostics_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Completed Stage 8A kernel diagnostics.")


if __name__ == "__main__":
    main()
