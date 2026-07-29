from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qnlpbench_r.data.latent import generate_latent_compositional_dataset
from qnlpbench_r.models.tensor_train import (
    TensorTrainClassifier,
    primitive_site_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_protocol_hashes_match_seal():
    seal = json.loads((ROOT / "docs/STAGE8A_PROTOCOL_SEAL.json").read_text())
    for relative_path, expected in seal["files"].items():
        assert sha256(ROOT / relative_path) == expected


def test_tensor_train_uses_eight_nonoverlapping_primitive_sites():
    _, report = generate_latent_compositional_dataset(
        n_samples=100,
        seed=11,
        semantic_seed=314159,
        latent_dim=4,
        min_abs_margin=0.10,
        max_modifiers=4,
    )
    groups = primitive_site_indices(report.feature_columns)
    assert list(groups) == [
        "subject",
        "verb",
        "object",
        "negation",
        "modifier_0",
        "modifier_1",
        "modifier_2",
        "modifier_3",
    ]
    flattened = [index for values in groups.values() for index in values]
    assert len(flattened) == len(set(flattened)) == 49


def test_tensor_train_forward_and_gradient_are_finite():
    _, report = generate_latent_compositional_dataset(
        n_samples=100,
        seed=23,
        semantic_seed=314159,
        latent_dim=4,
        min_abs_margin=0.10,
        max_modifiers=4,
    )
    model = TensorTrainClassifier(report.feature_columns, bond_dim=4)
    inputs = torch.randn(7, len(report.feature_columns))
    logits = model(inputs)
    assert logits.shape == (7, 2)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_kernel_diagnostic_invariants():
    module = load_script("run_kernel_diagnostics.py")
    kernel = np.array(
        [[1.0, 0.8, 0.2, 0.1], [0.8, 1.0, 0.1, 0.2], [0.2, 0.1, 1.0, 0.7], [0.1, 0.2, 0.7, 1.0]]
    )
    labels = np.array([0, 0, 1, 1])
    stats, eigenvalues = module.kernel_statistics(
        kernel, labels, c_value=1.0, tolerance=1.0e-10, condition_floor=1.0e-12
    )
    assert eigenvalues.shape == (4,)
    assert stats["centered_kernel_target_alignment"] > 0
    assert stats["within_minus_between_similarity"] > 0
    assert 0 < stats["support_vector_fraction"] <= 1


def test_index_round_trip_is_byte_hashed(tmp_path):
    module = load_script("run_index_matched_comparison.py")
    index = np.array([8, 1, 5, 3], dtype=np.int64)
    path = tmp_path / "index.npy"
    digest = module.save_index(path, index)
    assert digest == sha256(path)
    assert np.array_equal(np.load(path, allow_pickle=False), index)


def test_regenerated_data_gate_tolerates_only_audit_float_serialization(tmp_path):
    module = load_script("stage8a_dataset_identity.py")
    feature_columns = [f"feature_{index}" for index in range(49)]
    rows = []
    for index in range(3):
        row = {
            "id": f"example-{index}",
            "text": f"surface {index}",
            "split": "train",
            "label": index % 2,
            "clean_label": index % 2,
            "audit_latent_score": 0.1 * index,
            "audit_latent_margin": 0.2 * index,
        }
        row.update(
            {column: int(position == index) for position, column in enumerate(feature_columns)}
        )
        rows.append(row)
    frozen = pd.DataFrame(rows)
    regenerated = frozen.copy()
    regenerated.loc[1, "audit_latent_score"] += 8.0e-16
    regenerated.loc[2, "audit_latent_margin"] -= 8.0e-16

    frozen_path = tmp_path / "frozen.csv"
    regenerated_path = tmp_path / "regenerated.csv"
    manifest_path = tmp_path / "dataset_manifest.json"
    frozen.to_csv(frozen_path, index=False)
    regenerated.to_csv(regenerated_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "generation_report": {
                    "content_hash": sha256(frozen_path),
                    "feature_columns": feature_columns,
                }
            }
        )
    )

    report = module.compare_regenerated_dataset(
        regenerated_path,
        frozen_path,
        manifest_path,
        feature_columns,
    )
    assert report["status"] == "pass"
    assert not report["bytewise_hash_match"]
    assert report["model_visible_features_exact"]
    assert report["labels_exact"]
    assert report["audit_floats_within_tolerance"]


def test_regenerated_data_gate_rejects_model_visible_drift(tmp_path):
    module = load_script("stage8a_dataset_identity.py")
    feature_columns = [f"feature_{index}" for index in range(49)]
    row = {
        "id": "example-0",
        "text": "surface 0",
        "split": "train",
        "label": 0,
        "clean_label": 0,
        "audit_latent_score": 0.0,
        "audit_latent_margin": 0.1,
        **{column: 0 for column in feature_columns},
    }
    frozen = pd.DataFrame([row])
    regenerated = frozen.copy()
    regenerated.loc[0, feature_columns[0]] = 1

    frozen_path = tmp_path / "frozen.csv"
    regenerated_path = tmp_path / "regenerated.csv"
    manifest_path = tmp_path / "dataset_manifest.json"
    frozen.to_csv(frozen_path, index=False)
    regenerated.to_csv(regenerated_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "generation_report": {
                    "content_hash": sha256(frozen_path),
                    "feature_columns": feature_columns,
                }
            }
        )
    )

    report = module.compare_regenerated_dataset(
        regenerated_path,
        frozen_path,
        manifest_path,
        feature_columns,
    )
    assert report["status"] == "fail"
    assert not report["model_visible_features_exact"]
