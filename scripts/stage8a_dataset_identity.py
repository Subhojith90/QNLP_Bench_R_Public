from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


AUDIT_FLOAT_COLUMNS = ("audit_latent_score", "audit_latent_margin")
AUDIT_FLOAT_ABSOLUTE_TOLERANCE = 2.0e-15
ROW_IDENTITY_COLUMNS = ("id", "text", "split")
LABEL_COLUMNS = ("label", "clean_label")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_columns_equal(
    regenerated: pd.DataFrame,
    frozen: pd.DataFrame,
    columns: Iterable[str],
) -> bool:
    selected = list(columns)
    return regenerated[selected].equals(frozen[selected])


def compare_regenerated_dataset(
    regenerated_csv: Path,
    frozen_csv: Path,
    frozen_manifest: Path,
    model_visible_feature_columns: Iterable[str],
    *,
    audit_float_columns: Iterable[str] = AUDIT_FLOAT_COLUMNS,
    audit_float_absolute_tolerance: float = AUDIT_FLOAT_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Apply the Stage 8A regenerated-data identity policy.

    The immutable frozen CSV is always verified against its registered content
    hash. A regenerated CSV is not required to be byte-identical because
    platform-specific floating serialization may affect the two explicitly
    audit-only latent floating columns. Everything used by a model is compared
    exactly.
    """

    regenerated = pd.read_csv(regenerated_csv)
    frozen = pd.read_csv(frozen_csv)
    manifest = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    registered_frozen_sha256 = manifest["generation_report"]["content_hash"]
    observed_frozen_sha256 = sha256_file(frozen_csv)
    regenerated_sha256 = sha256_file(regenerated_csv)

    feature_columns = list(model_visible_feature_columns)
    float_columns = list(audit_float_columns)
    expected_columns = list(frozen.columns)
    schema_exact = list(regenerated.columns) == expected_columns
    row_count_exact = regenerated.shape[0] == frozen.shape[0]

    required_columns = set(
        feature_columns
        + float_columns
        + list(ROW_IDENTITY_COLUMNS)
        + list(LABEL_COLUMNS)
    )
    required_columns_present = required_columns.issubset(regenerated.columns) and (
        required_columns.issubset(frozen.columns)
    )

    row_order_exact = False
    labels_exact = False
    model_visible_features_exact = False
    nonfloating_metadata_exact = False
    float_differences: dict[str, dict[str, Any]] = {}
    audit_floats_within_tolerance = False

    if schema_exact and row_count_exact and required_columns_present:
        row_order_exact = _exact_columns_equal(
            regenerated, frozen, ROW_IDENTITY_COLUMNS
        )
        labels_exact = _exact_columns_equal(regenerated, frozen, LABEL_COLUMNS)
        model_visible_features_exact = np.array_equal(
            regenerated[feature_columns].to_numpy(),
            frozen[feature_columns].to_numpy(),
        )

        exact_metadata_columns = [
            column for column in expected_columns if column not in float_columns
        ]
        nonfloating_metadata_exact = _exact_columns_equal(
            regenerated, frozen, exact_metadata_columns
        )

        audit_floats_within_tolerance = True
        for column in float_columns:
            regenerated_values = regenerated[column].to_numpy(dtype=np.float64)
            frozen_values = frozen[column].to_numpy(dtype=np.float64)
            absolute_difference = np.abs(regenerated_values - frozen_values)
            maximum = float(np.nanmax(absolute_difference))
            within = bool(
                np.allclose(
                    regenerated_values,
                    frozen_values,
                    rtol=0.0,
                    atol=audit_float_absolute_tolerance,
                    equal_nan=True,
                )
            )
            audit_floats_within_tolerance &= within
            float_differences[column] = {
                "maximum_absolute_difference": maximum,
                "within_tolerance": within,
            }

    frozen_hash_preserved = (
        observed_frozen_sha256 == registered_frozen_sha256
    )
    passed = all(
        [
            frozen_hash_preserved,
            schema_exact,
            row_count_exact,
            required_columns_present,
            row_order_exact,
            labels_exact,
            model_visible_features_exact,
            nonfloating_metadata_exact,
            audit_floats_within_tolerance,
        ]
    )

    return {
        "policy": "stage8a_regenerated_data_identity_v2",
        "status": "pass" if passed else "fail",
        "frozen_csv": str(frozen_csv),
        "regenerated_csv": str(regenerated_csv),
        "registered_frozen_sha256": registered_frozen_sha256,
        "observed_frozen_sha256": observed_frozen_sha256,
        "frozen_hash_preserved": frozen_hash_preserved,
        "regenerated_sha256": regenerated_sha256,
        "bytewise_hash_match": regenerated_sha256 == observed_frozen_sha256,
        "schema_exact": schema_exact,
        "row_count_exact": row_count_exact,
        "required_columns_present": required_columns_present,
        "row_order_columns": list(ROW_IDENTITY_COLUMNS),
        "row_order_exact": row_order_exact,
        "label_columns": list(LABEL_COLUMNS),
        "labels_exact": labels_exact,
        "model_visible_feature_count": len(feature_columns),
        "model_visible_features_exact": model_visible_features_exact,
        "nonfloating_metadata_exact": nonfloating_metadata_exact,
        "audit_only_float_columns": float_columns,
        "audit_float_absolute_tolerance": audit_float_absolute_tolerance,
        "audit_float_relative_tolerance": 0.0,
        "audit_float_differences": float_differences,
        "audit_floats_within_tolerance": audit_floats_within_tolerance,
    }
