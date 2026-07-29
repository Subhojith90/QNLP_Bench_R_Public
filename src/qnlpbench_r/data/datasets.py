from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qnlpbench_r.data.synthetic import generate_synthetic_compositional_dataset
from qnlpbench_r.data.latent import generate_latent_compositional_dataset
from qnlpbench_r.utils.io import sha256_file, write_json


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    ood: pd.DataFrame
    all_data: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    manifest: dict[str, Any]
    ood_composition: pd.DataFrame | None = None
    ood_depth: pd.DataFrame | None = None

    def split(self, name: str) -> pd.DataFrame:
        if name == "train":
            return self.train
        if name == "val":
            return self.val
        if name == "test":
            return self.test
        if name == "ood":
            return self.ood
        if name == "ood_composition":
            return self.ood_composition if self.ood_composition is not None else pd.DataFrame(columns=self.all_data.columns)
        if name == "ood_depth":
            return self.ood_depth if self.ood_depth is not None else pd.DataFrame(columns=self.all_data.columns)
        raise KeyError(f"Unknown split: {name}")


def _primitive_columns(columns: list[str]) -> list[str]:
    prefixes = ("ps_", "pv_", "po_", "pn_", "pd_")
    return [c for c in columns if c.startswith(prefixes)]


def select_feature_columns(all_feature_columns: list[str], ds_cfg: dict[str, Any]) -> list[str]:
    """Select model-visible columns while preserving hidden audit metadata.

    Stage 4D-R / Stage 5 policy:
    - primitive_slot_sequence is the primary information-preserving non-engineered input.
      It contains observable slot/token indicators and ordered distractor-token occupancy.
    - primitive_no_context is a deliberate negative control that removes negation and
      distractor sequence information required for the compositional rule.
    - engineered_rule_positive_control is deliberately shortcut-exposing and is only
      valid as a saturation/positive control.
    - legacy feature sets remain available only for provenance and should not be used
      for scientific claims.
    """
    feature_set = str(ds_cfg.get("feature_set", "full")).strip().lower()
    columns = list(all_feature_columns)
    legacy_f = [c for c in columns if c.startswith("f") and c[1:].isdigit()]
    primitive = _primitive_columns(columns)
    if feature_set in {"full", "all", "default"}:
        # Preserve prior experiment semantics: legacy full means f-columns only.
        selected = legacy_f
    elif feature_set in {"primitive_slot_sequence", "primitive_tokens_plus_sequence", "primitive_slot_onehot"}:
        selected = primitive
    elif feature_set in {"primitive_no_context", "primitive_lexical_no_context", "bag_of_tokens_negative_control"}:
        selected = [c for c in primitive if c.startswith(("ps_", "pv_", "po_"))]
    elif feature_set in {"primitive_without_negation"}:
        selected = [c for c in primitive if not c.startswith("pn_")]
    elif feature_set in {"engineered_rule_positive_control", "rule_features_only", "f7_f8_f9_only"}:
        selected = [c for c in ["f7", "f8", "f9"] if c in columns]
    elif feature_set in {"no_rule_features", "no_f7_f8_f9", "remove_f7_f8_f9_legacy"}:
        selected = [c for c in legacy_f if c not in {"f7", "f8", "f9"}]
    elif feature_set == "transformed_rule_control":
        selected = [c for c in ["f4", "f10", "f11"] if c in columns]
    elif feature_set == "strict_partial_semantics":
        selected = [c for c in ["f0", "f1", "f2", "f3", "f12", "f13"] if c in columns]
    elif feature_set == "strict_projection_only":
        selected = [c for c in ["f12", "f13"] if c in columns]
    elif feature_set == "raw_ids_only":
        selected = [c for c in ["f0", "f1", "f2", "f3", "f4"] if c in columns]
    elif feature_set == "custom":
        selected = [str(c) for c in ds_cfg.get("feature_columns", [])]
    else:
        raise ValueError(f"Unsupported dataset.feature_set: {feature_set}")
    if not selected:
        raise ValueError(f"Feature-set {feature_set!r} selected no feature columns.")
    missing = set(selected).difference(columns)
    if missing:
        raise ValueError(f"Requested feature columns missing from generated dataset: {sorted(missing)}")
    return selected


def _split_audit(df: pd.DataFrame, label_column: str) -> dict[str, Any]:
    """Return split-level counts and leakage indicators for reproducibility manifests."""
    audit: dict[str, Any] = {
        "split_counts": df["split"].value_counts(dropna=False).to_dict(),
        "label_counts_by_split": {},
    }
    for split_name, frame in df.groupby("split"):
        audit["label_counts_by_split"][str(split_name)] = frame[label_column].value_counts(dropna=False).to_dict()
    if {"subject_id", "object_id"}.issubset(df.columns):
        train_pairs = set(map(tuple, df.loc[df["split"] == "train", ["subject_id", "object_id"]].to_numpy()))
        ood_mask = df["split"].astype(str).str.startswith("ood")
        ood_pairs = set(map(tuple, df.loc[ood_mask, ["subject_id", "object_id"]].to_numpy()))
        audit["train_ood_subject_object_overlap_count"] = len(train_pairs.intersection(ood_pairs))
    if "text" in df.columns:
        split_names = sorted(df["split"].astype(str).unique())
        text_by_split = {s: set(df.loc[df["split"].astype(str) == s, "text"].astype(str)) for s in split_names}
        audit["exact_text_duplicate_overlap"] = {}
        for i, left in enumerate(split_names):
            for right in split_names[i + 1:]:
                audit["exact_text_duplicate_overlap"][f"{left}-{right}"] = len(text_by_split[left].intersection(text_by_split[right]))
        audit["duplicate_text_count_total"] = int(df["text"].duplicated().sum())
    if "is_heldout_composition" in df.columns:
        for split_name in ["train", "val", "test", "ood_composition", "ood_depth"]:
            audit[f"heldout_compositions_in_{split_name}"] = int(((df["split"] == split_name) & (df["is_heldout_composition"].astype(bool))).sum())
        if {"subject_id", "object_id"}.issubset(df.columns):
            pair_sets = {name: set(map(tuple, df.loc[df["split"] == name, ["subject_id", "object_id"]].to_numpy())) for name in ["train", "val", "test", "ood_composition", "ood_depth"]}
            audit["ood_composition_pair_overlap_with_train"] = len(pair_sets["ood_composition"].intersection(pair_sets["train"]))
            audit["ood_composition_pair_overlap_with_val"] = len(pair_sets["ood_composition"].intersection(pair_sets["val"]))
            audit["ood_composition_pair_overlap_with_test"] = len(pair_sets["ood_composition"].intersection(pair_sets["test"]))
    return audit


def _bundle_from_dataframe(df: pd.DataFrame, feature_columns: list[str], label_column: str, manifest: dict[str, Any]) -> DatasetBundle:
    required = set(feature_columns + [label_column, "split"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    allowed_splits = {"train", "val", "test", "ood", "ood_composition", "ood_depth"}
    observed_splits = set(df["split"].astype(str).unique())
    unknown_splits = observed_splits.difference(allowed_splits)
    if unknown_splits:
        raise ValueError(f"Unsupported split labels found: {sorted(unknown_splits)}")
    splits = {name: df[df["split"] == name].reset_index(drop=True) for name in ["train", "val", "test", "ood", "ood_composition", "ood_depth"]}
    if splits["ood"].empty and (not splits["ood_composition"].empty or not splits["ood_depth"].empty):
        splits["ood"] = pd.concat([splits["ood_composition"], splits["ood_depth"]], ignore_index=True)
    for name, frame in splits.items():
        if frame.empty and name in {"train", "val", "test"}:
            raise ValueError(f"Required split '{name}' is empty. Check split fractions or CSV split column.")
    manifest = dict(manifest)
    manifest["split_audit"] = _split_audit(df, label_column)
    if manifest["split_audit"].get("heldout_compositions_in_train", 0) != 0:
        raise ValueError("Held-out synthetic compositions leaked into the training split.")
    if bool(manifest.get("require_heldout_exclusion_from_id_splits", False)):
        for split_name in ["val", "test"]:
            if manifest["split_audit"].get(f"heldout_compositions_in_{split_name}", 0) != 0:
                raise ValueError(f"Held-out synthetic compositions leaked into strict ID split: {split_name}.")
    return DatasetBundle(splits["train"], splits["val"], splits["test"], splits["ood"], df.reset_index(drop=True), feature_columns, label_column, manifest, splits["ood_composition"], splits["ood_depth"])


def load_dataset(config: dict[str, Any], seed: int, run_dir: str | Path | None = None) -> DatasetBundle:
    """Load or generate a dataset according to config."""
    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    if name == "synthetic_compositional":
        df, report = generate_synthetic_compositional_dataset(
            n_samples=int(ds_cfg.get("n_samples", 1000)),
            seed=int(seed),
            feature_dim=int(ds_cfg.get("feature_dim", 12)),
            validation_fraction=float(ds_cfg.get("validation_fraction", 0.15)),
            test_fraction=float(ds_cfg.get("test_fraction", 0.20)),
            ood_fraction=float(ds_cfg.get("ood_fraction", 0.20)),
            max_distractors=int(ds_cfg.get("max_distractors", 5)),
            holdout_compositions=bool(ds_cfg.get("holdout_compositions", True)),
            label_noise=float(ds_cfg.get("label_noise", 0.0)),
            unique_examples=bool(ds_cfg.get("unique_examples", False)),
            ood_mode=str(ds_cfg.get("ood_mode", "mixed")),
            ood_composition_fraction=ds_cfg.get("ood_composition_fraction"),
            ood_depth_fraction=ds_cfg.get("ood_depth_fraction"),
            require_ood_purity=bool(ds_cfg.get("require_ood_purity", False)),
            isolate_heldout_compositions_from_id_splits=bool(ds_cfg.get("isolate_heldout_compositions_from_id_splits", False)),
        )
        feature_columns = select_feature_columns(report.feature_columns, ds_cfg)
        manifest = {
            "dataset_name": name,
            "generation_report": report.to_dict(),
            "seed": int(seed),
            "leakage_policy": "Held-out subject-object compositions are excluded from train; Stage 5 strict inputs prohibit derived rule-component columns and allow primitive observable token/slot inputs.",
            "selected_feature_columns": feature_columns,
            "feature_set": str(ds_cfg.get("feature_set", "full")),
            "model_visible_input_policy": "primitive_observable_inputs" if str(ds_cfg.get("feature_set", "full")).startswith("primitive") else "legacy_or_control_inputs",
            "audit_only_metadata_columns": ["subject_id", "verb_id", "object_id", "negation", "distractor_count", "grammar_depth", "clean_label", "is_heldout_composition", "is_high_depth"],
            "require_heldout_exclusion_from_id_splits": bool(ds_cfg.get("isolate_heldout_compositions_from_id_splits", False)),
            "prohibited_engineered_columns_for_stage5_claims": ["f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
        }
        bundle = _bundle_from_dataframe(df, feature_columns, report.label_column, manifest)
    elif name == "latent_compositional_semantics":
        df, report = generate_latent_compositional_dataset(
            n_samples=int(ds_cfg.get("n_samples", 1600)),
            seed=int(seed),
            semantic_seed=int(ds_cfg.get("semantic_seed", 314159)),
            latent_dim=int(ds_cfg.get("latent_dim", 6)),
            validation_fraction=float(ds_cfg.get("validation_fraction", 0.15)),
            test_fraction=float(ds_cfg.get("test_fraction", 0.20)),
            ood_composition_fraction=float(ds_cfg.get("ood_composition_fraction", 0.10)),
            ood_depth_fraction=float(ds_cfg.get("ood_depth_fraction", 0.10)),
            max_modifiers=int(ds_cfg.get("max_modifiers", 4)),
            ood_depth_min=int(ds_cfg.get("ood_depth_min", 3)),
            min_abs_margin=float(ds_cfg.get("min_abs_margin", 0.05)),
            label_noise=float(ds_cfg.get("label_noise", 0.0)),
            unique_examples=bool(ds_cfg.get("unique_examples", True)),
        )
        feature_columns = select_feature_columns(report.feature_columns, {**ds_cfg, "feature_set": ds_cfg.get("feature_set", "primitive_slot_sequence")})
        manifest = {
            "dataset_name": name,
            "generation_report": report.to_dict(),
            "seed": int(seed),
            "feature_set": str(ds_cfg.get("feature_set", "primitive_slot_sequence")),
            "selected_feature_columns": feature_columns,
            "model_visible_input_policy": "primitive_observable_inputs_only",
            "hidden_semantic_program": {"semantic_seed": int(ds_cfg.get("semantic_seed", 314159)), "latent_dim": int(ds_cfg.get("latent_dim", 6)), "stored_as_audit_only": True},
            "audit_only_metadata_columns": ["audit_latent_score", "audit_latent_margin", "audit_semantic_seed", "clean_label", "subject_id", "verb_id", "object_id", "negation", "modifier_count", "grammar_depth", "is_heldout_composition", "is_high_depth"],
            "prohibited_model_visible_columns": ["audit_latent_score", "audit_latent_margin", "audit_semantic_seed", "clean_label"],
            "require_heldout_exclusion_from_id_splits": True,
            "leakage_policy": "Latent semantic operators/scores are hidden audit metadata; models receive primitive observable token-slot sequence indicators only. Held-out composition families occur only in OOD-composition.",
        }
        bundle = _bundle_from_dataframe(df, feature_columns, report.label_column, manifest)
        prohibited = set(manifest["prohibited_model_visible_columns"]).intersection(feature_columns)
        if prohibited:
            raise ValueError(f"Latent audit-only columns exposed as model inputs: {sorted(prohibited)}")
    elif name == "csv":
        path = Path(ds_cfg.get("csv_path", ""))
        if not path.exists():
            raise FileNotFoundError(f"CSV dataset path not found: {path}")
        df = pd.read_csv(path)
        all_feature_columns = list(ds_cfg.get("feature_columns", [])) or [c for c in df.columns if c.startswith("f")]
        feature_columns = select_feature_columns(all_feature_columns, ds_cfg)
        label_column = str(ds_cfg.get("label_column", "label"))
        split_column = str(ds_cfg.get("split_column", "split"))
        if split_column not in df.columns:
            raise ValueError("CSV datasets must provide a split column for this release to prevent leakage.")
        if split_column != "split":
            df = df.rename(columns={split_column: "split"})
        manifest = {"dataset_name": name, "csv_path": str(path), "csv_sha256": sha256_file(path), "feature_columns": feature_columns, "label_column": label_column, "split_column": split_column}
        bundle = _bundle_from_dataframe(df, feature_columns, label_column, manifest)
    else:
        raise ValueError(f"Unsupported dataset.name: {name}")
    if run_dir is not None:
        data_dir = Path(run_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        bundle.all_data.to_csv(data_dir / "dataset.csv", index=False)
        write_json(bundle.manifest, data_dir / "dataset_manifest.json")
    return bundle


def dataset_summary(bundle: DatasetBundle) -> dict[str, Any]:
    return {"n_train": len(bundle.train), "n_val": len(bundle.val), "n_test": len(bundle.test), "n_ood": len(bundle.ood), "n_ood_composition": len(bundle.split("ood_composition")), "n_ood_depth": len(bundle.split("ood_depth")), "feature_dim": len(bundle.feature_columns), "label_column": bundle.label_column}
