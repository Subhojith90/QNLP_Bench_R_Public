from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from stage8a_common import (  # noqa: E402
    bootstrap_ci,
    candidate_embeddings,
    index_runs,
    kernel_matrix,
    load_arrays,
    median_gamma,
    model_from_run,
    qfc_kernel,
    qfc_states,
    select_indices,
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


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_index(path: Path, index: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(index, dtype=np.int64), allow_pickle=False)
    loaded = np.load(path, allow_pickle=False)
    if loaded.dtype != np.int64 or not np.array_equal(loaded, index):
        raise RuntimeError(f"Index round-trip failed: {path}")
    return sha256_file(path)


def choose_svc(
    candidates: Iterable[tuple[str, str, str, dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    c_values: list[float],
    seed: int,
    family: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for order, (source, kernel_family, name, params, k_train, k_val, k_eval) in enumerate(candidates):
        for c_order, c_value in enumerate(c_values):
            clf = SVC(kernel="precomputed", C=c_value)
            clf.fit(k_train, y_train)
            score = float(balanced_accuracy_score(y_val, clf.predict(k_val)))
            record = {
                "candidate_family": family,
                "source": source,
                "kernel_family": kernel_family,
                "kernel_name": name,
                "parameters_json": json.dumps(params, sort_keys=True),
                "C": c_value,
                "validation_balanced_accuracy": score,
                "candidate_order": order,
                "c_order": c_order,
                "selected": False,
            }
            rows.append(record)
            if best is None or score > best["validation_balanced_accuracy"]:
                best = {
                    **record,
                    "K_train": k_train,
                    "K_val": k_val,
                    "K_eval": k_eval,
                }
    if best is None:
        raise RuntimeError(f"No {family} candidate was generated for seed {seed}")
    for row in rows:
        if (
            row["candidate_order"] == best["candidate_order"]
            and row["c_order"] == best["c_order"]
        ):
            row["selected"] = True
            break
    return best, rows


def generic_candidates(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_eval: dict[str, np.ndarray],
    cfg: dict[str, Any],
):
    base_gamma = median_gamma(x_train)
    for family in cfg["generic_kernels"]["families"]:
        if family in {"linear", "cosine"}:
            params: dict[str, Any] = {}
            yield (
                "primitive_input",
                family,
                family,
                params,
                kernel_matrix(x_train, x_train, family, params),
                kernel_matrix(x_val, x_train, family, params),
                {name: kernel_matrix(x, x_train, family, params) for name, x in x_eval.items()},
            )
        elif family in {"rbf", "laplacian"}:
            for multiplier in cfg["generic_kernels"]["gamma_multipliers"]:
                params = {
                    "gamma": float(base_gamma * multiplier),
                    "gamma_multiplier": float(multiplier),
                }
                yield (
                    "primitive_input",
                    family,
                    f"{family}_gamma_x{multiplier:g}",
                    params,
                    kernel_matrix(x_train, x_train, family, params),
                    kernel_matrix(x_val, x_train, family, params),
                    {name: kernel_matrix(x, x_train, family, params) for name, x in x_eval.items()},
                )
        elif family == "poly":
            for degree in cfg["generic_kernels"]["polynomial_degrees"]:
                params = {"degree": int(degree), "gamma": 1.0 / x_train.shape[1]}
                yield (
                    "primitive_input",
                    family,
                    f"poly_degree_{degree}",
                    params,
                    kernel_matrix(x_train, x_train, family, params),
                    kernel_matrix(x_val, x_train, family, params),
                    {name: kernel_matrix(x, x_train, family, params) for name, x in x_eval.items()},
                )


def learned_config(cfg: dict[str, Any]) -> dict[str, Any]:
    learned = cfg["learned_kernels"]
    return {
        "gamma_multipliers": learned["gamma_multipliers"],
        "poly_degrees": learned["polynomial_degrees"],
        "rff_components": learned["rff_components"],
        "tree_estimators": learned["tree_estimators"],
        "classical_kernel_families": learned["families"],
    }


def fit_selected(candidate: dict[str, Any], y_train: np.ndarray) -> SVC:
    classifier = SVC(kernel="precomputed", C=float(candidate["C"]))
    classifier.fit(candidate["K_train"], y_train)
    return classifier


def qfc_candidate(
    model: Any,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_eval: dict[str, np.ndarray],
) -> tuple[str, str, str, dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train_states = qfc_states(model, x_train)
    val_states = qfc_states(model, x_val)
    eval_states = {name: qfc_states(model, values) for name, values in x_eval.items()}
    return (
        "trained_qfc",
        "state_overlap",
        model.__class__.__name__,
        {},
        qfc_kernel(train_states, train_states),
        qfc_kernel(val_states, train_states),
        {name: qfc_kernel(states, train_states) for name, states in eval_states.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage8a.yaml")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results/index_matched")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    seeds = list(cfg["seeds"])
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    split_names = list(cfg["splits"])
    cap = int(cfg["sample_cap"])
    c_values = [float(value) for value in cfg["svm_c"]]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    kernel_dir = output / "kernels"
    kernel_dir.mkdir(exist_ok=True)

    frozen_runs = ROOT / "inputs/stage6d_frozen/results/stage6d/runs"
    classical_runs = index_runs(frozen_runs / "classical")
    topology_runs = index_runs(frozen_runs / "topology")
    mlp_name = cfg["learned_kernels"]["mlp_model"]

    result_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []

    for seed in seeds:
        if (mlp_name, seed) not in classical_runs:
            raise FileNotFoundError(f"Missing frozen MLP run for seed {seed}")
        mlp_dir = classical_runs[(mlp_name, seed)]
        arrays = load_arrays(mlp_dir)
        mlp = model_from_run(mlp_dir, trained=True)
        dataset = pd.read_csv(mlp_dir / "data/dataset.csv")

        offsets = cfg["sampling"]
        indices: dict[str, np.ndarray] = {
            "train": select_indices(
                arrays["train"][1], cap, seed + int(offsets["train_offset"]), "stratified"
            ),
            "val": select_indices(
                arrays["val"][1], cap, seed + int(offsets["validation_offset"]), "stratified"
            ),
        }
        for split in split_names:
            indices[split] = select_indices(
                arrays[split][1],
                cap,
                seed + int(offsets["split_base_offset"]) + len(split),
                "stratified",
            )
        index_hashes: dict[str, str] = {}
        for split, values in indices.items():
            path = output / "indices" / f"seed_{seed}" / f"{split}.npy"
            index_hashes[split] = save_index(path, values)
            labels = arrays[split][1][values]
            index_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "count": len(values),
                    "class_0": int((labels == 0).sum()),
                    "class_1": int((labels == 1).sum()),
                    "sha256": index_hashes[split],
                    "path": str(path.relative_to(ROOT)),
                }
            )

        x_train = arrays["train"][0][indices["train"]]
        y_train = arrays["train"][1][indices["train"]]
        x_val = arrays["val"][0][indices["val"]]
        y_val = arrays["val"][1][indices["val"]]
        x_eval = {split: arrays[split][0][indices[split]] for split in split_names}
        y_eval = {split: arrays[split][1][indices[split]] for split in split_names}

        generic, generic_selection = choose_svc(
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
        for record in generic_selection + learned_selection:
            selection_rows.append({"seed": seed, **record})
        generic_clf = fit_selected(generic, y_train)
        learned_clf = fit_selected(learned, y_train)

        for topology in cfg["topologies"]:
            if (topology, seed) not in topology_runs:
                raise FileNotFoundError(f"Missing frozen {topology} run for seed {seed}")
            qfc_model = model_from_run(topology_runs[(topology, seed)], trained=True)
            qfc, qfc_selection = choose_svc(
                [qfc_candidate(qfc_model, x_train, x_val, x_eval)],
                y_train,
                y_val,
                c_values,
                seed,
                "qfc",
            )
            for record in qfc_selection:
                selection_rows.append({"seed": seed, "topology": topology, **record})
            qfc_clf = fit_selected(qfc, y_train)

            np.savez_compressed(
                kernel_dir / f"seed_{seed}_{topology}.npz",
                qfc_train=qfc["K_train"],
                generic_train=generic["K_train"],
                learned_train=learned["K_train"],
                y_train=y_train,
                **{f"qfc_{split}": qfc["K_eval"][split] for split in split_names},
                **{f"generic_{split}": generic["K_eval"][split] for split in split_names},
                **{f"learned_{split}": learned["K_eval"][split] for split in split_names},
            )

            for split in split_names:
                qfc_pred = qfc_clf.predict(qfc["K_eval"][split])
                generic_pred = generic_clf.predict(generic["K_eval"][split])
                learned_pred = learned_clf.predict(learned["K_eval"][split])
                qfc_ba = float(balanced_accuracy_score(y_eval[split], qfc_pred))
                generic_ba = float(balanced_accuracy_score(y_eval[split], generic_pred))
                learned_ba = float(balanced_accuracy_score(y_eval[split], learned_pred))
                result_rows.append(
                    {
                        "seed": seed,
                        "topology": topology,
                        "split": split,
                        "sample_count": len(y_eval[split]),
                        "train_index_sha256": index_hashes["train"],
                        "validation_index_sha256": index_hashes["val"],
                        "evaluation_index_sha256": index_hashes[split],
                        "qfc_balanced_accuracy": qfc_ba,
                        "generic_balanced_accuracy": generic_ba,
                        "learned_balanced_accuracy": learned_ba,
                        "qfc_minus_generic": qfc_ba - generic_ba,
                        "qfc_minus_learned": qfc_ba - learned_ba,
                        "learned_minus_generic": learned_ba - generic_ba,
                        "qfc_C": qfc["C"],
                        "generic_kernel": generic["kernel_name"],
                        "generic_C": generic["C"],
                        "learned_source": learned["source"],
                        "learned_kernel": learned["kernel_name"],
                        "learned_C": learned["C"],
                    }
                )

                if split == "ood_composition":
                    split_frame = dataset[dataset["split"] == split].reset_index(drop=True)
                    selected_frame = split_frame.iloc[indices[split]].reset_index(drop=True)
                    for stratum, mask in {
                        "modifier_count_lt_3": selected_frame["modifier_count"].to_numpy() < 3,
                        "modifier_count_ge_3": selected_frame["modifier_count"].to_numpy() >= 3,
                    }.items():
                        stratum_y = y_eval[split][mask]
                        qfc_stratum = qfc_pred[mask]
                        learned_stratum = learned_pred[mask]
                        both_classes = np.unique(stratum_y).size == 2
                        q_ba = (
                            float(balanced_accuracy_score(stratum_y, qfc_stratum))
                            if both_classes
                            else np.nan
                        )
                        l_ba = (
                            float(balanced_accuracy_score(stratum_y, learned_stratum))
                            if both_classes
                            else np.nan
                        )
                        depth_rows.append(
                            {
                                "seed": seed,
                                "topology": topology,
                                "stratum": stratum,
                                "sample_count": int(mask.sum()),
                                "class_0": int((stratum_y == 0).sum()),
                                "class_1": int((stratum_y == 1).sum()),
                                "qfc_balanced_accuracy": q_ba,
                                "learned_balanced_accuracy": l_ba,
                                "qfc_minus_learned": q_ba - l_ba,
                                "evaluation_index_sha256": index_hashes[split],
                            }
                        )

    results = pd.DataFrame(result_rows)
    results.to_csv(output / "stage8a_index_matched_seed_level.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        output / "stage8a_validation_selection_candidates.csv", index=False
    )
    pd.DataFrame(index_rows).to_csv(output / "stage8a_index_manifest.csv", index=False)
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(output / "stage8a_depth_stratified_seed_level.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    difference_columns = ["qfc_minus_generic", "qfc_minus_learned", "learned_minus_generic"]
    for (topology, split), group in results.groupby(["topology", "split"], sort=True):
        base = {
            "topology": topology,
            "split": split,
            "n_seeds": int(group["seed"].nunique()),
            "qfc_balanced_accuracy_mean": float(group["qfc_balanced_accuracy"].mean()),
            "generic_balanced_accuracy_mean": float(group["generic_balanced_accuracy"].mean()),
            "learned_balanced_accuracy_mean": float(group["learned_balanced_accuracy"].mean()),
        }
        for position, column in enumerate(difference_columns):
            mean, low, high = bootstrap_ci(
                group[column].to_numpy(),
                int(cfg["bootstrap"]["resamples"]),
                int(cfg["bootstrap"]["seed"]) + len(topology) + len(split) + position * 100,
            )
            base[f"{column}_mean"] = mean
            base[f"{column}_ci_low"] = low
            base[f"{column}_ci_high"] = high
            base[f"{column}_positive_seeds"] = int((group[column] > 0).sum())
        summary_rows.append(base)
    pd.DataFrame(summary_rows).to_csv(output / "stage8a_index_matched_summary.csv", index=False)

    depth_summary: list[dict[str, Any]] = []
    for (topology, stratum), group in depth.groupby(["topology", "stratum"], sort=True):
        mean, low, high = bootstrap_ci(
            group["qfc_minus_learned"].to_numpy(),
            int(cfg["bootstrap"]["resamples"]),
            int(cfg["bootstrap"]["seed"]) + len(topology) + len(stratum),
        )
        depth_summary.append(
            {
                "topology": topology,
                "stratum": stratum,
                "n_seeds": int(group["seed"].nunique()),
                "sample_count_mean": float(group["sample_count"].mean()),
                "sample_count_min": int(group["sample_count"].min()),
                "sample_count_max": int(group["sample_count"].max()),
                "qfc_balanced_accuracy_mean": float(group["qfc_balanced_accuracy"].mean()),
                "learned_balanced_accuracy_mean": float(
                    group["learned_balanced_accuracy"].mean()
                ),
                "qfc_minus_learned_mean": mean,
                "qfc_minus_learned_ci_low": low,
                "qfc_minus_learned_ci_high": high,
                "positive_seeds": int((group["qfc_minus_learned"] > 0).sum()),
            }
        )
    pd.DataFrame(depth_summary).to_csv(
        output / "stage8a_depth_stratified_summary.csv", index=False
    )

    manifest = {
        "stage": "8A",
        "analysis": "index_matched_comparator_escalation",
        "config": str(args.config.relative_to(ROOT)),
        "seeds": seeds,
        "topologies": cfg["topologies"],
        "sample_cap": cap,
        "all_models_share_saved_indices": True,
        "output_hashes": {},
    }
    for path in sorted(output.glob("*.csv")):
        manifest["output_hashes"][path.name] = sha256_file(path)
    write_json(manifest, output / "stage8a_index_matched_manifest.json")
    print(f"Completed index-matched analysis for {len(seeds)} seeds.")


if __name__ == "__main__":
    main()
