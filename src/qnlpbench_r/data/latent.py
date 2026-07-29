from __future__ import annotations

"""Latent compositional semantic benchmark generator (Stage 6).

The generator stores a hidden semantic program for auditability, but exposes only
primitive observable token/slot indicators to trainable models. Labels are derived
from latent vector/operator composition, not a hand-specified Boolean rule over
visible feature columns.
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from qnlpbench_r.data.synthetic import SUBJECTS, VERBS, OBJECTS, DISTRACTORS


@dataclass(frozen=True)
class LatentGenerationReport:
    n_samples: int
    n_train: int
    n_val: int
    n_test: int
    n_ood: int
    feature_columns: list[str]
    label_column: str
    content_hash: str
    heldout_compositions: list[tuple[int, int]]
    semantic_seed: int
    latent_dim: int
    semantic_threshold: float
    min_abs_margin: float
    split_policy: str = "strict_latent_compositional"
    duplicate_text_count: int = 0
    exact_duplicate_overlap: dict[str, int] | None = None
    ood_condition_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples, "n_train": self.n_train, "n_val": self.n_val,
            "n_test": self.n_test, "n_ood": self.n_ood, "feature_columns": self.feature_columns,
            "label_column": self.label_column, "content_hash": self.content_hash,
            "heldout_compositions": [list(x) for x in self.heldout_compositions],
            "semantic_seed": self.semantic_seed, "latent_dim": self.latent_dim,
            "semantic_threshold": self.semantic_threshold, "min_abs_margin": self.min_abs_margin,
            "split_policy": self.split_policy, "duplicate_text_count": self.duplicate_text_count,
            "exact_duplicate_overlap": self.exact_duplicate_overlap or {},
            "ood_condition_counts": self.ood_condition_counts or {},
            "model_visible_policy": "primitive tokens/slots only; latent vectors, operators, score and margin are audit-only",
        }


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def _near_identity_operator(rng: np.random.Generator, dim: int, scale: float) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=(dim, dim))
    q, _ = np.linalg.qr(raw)
    return (1.0 - scale) * np.eye(dim) + scale * q


@lru_cache(maxsize=32)
def latent_semantic_program(semantic_seed: int, latent_dim: int) -> dict[str, np.ndarray | float | list[tuple[int, int]]]:
    rng = np.random.default_rng(int(semantic_seed))
    subjects = _unit_rows(rng.normal(size=(len(SUBJECTS), latent_dim)))
    objects = _unit_rows(rng.normal(size=(len(OBJECTS), latent_dim)))
    verbs = np.stack([_near_identity_operator(rng, latent_dim, 0.70) for _ in VERBS])
    modifiers = np.stack([_near_identity_operator(rng, latent_dim, 0.38) for _ in DISTRACTORS])
    # Reflection-like semantic negation operator.
    v = rng.normal(size=(latent_dim,)); v /= np.linalg.norm(v).clip(min=1e-8)
    negation = np.eye(latent_dim) - 2.0 * np.outer(v, v)
    all_pairs = [(s, o) for s in range(len(SUBJECTS)) for o in range(len(OBJECTS))]
    pair_idx = rng.choice(len(all_pairs), size=8, replace=False)
    heldout = [all_pairs[int(i)] for i in pair_idx]
    # Threshold estimated once from hidden semantic samples, independent of dataset seed.
    scores = []
    for _ in range(10000):
        sid = int(rng.integers(len(SUBJECTS))); vid = int(rng.integers(len(VERBS))); oid = int(rng.integers(len(OBJECTS)))
        neg = int(rng.integers(2)); depth = int(rng.integers(5))
        mods = [int(rng.integers(len(DISTRACTORS))) for _ in range(depth)]
        scores.append(_latent_score(subjects, verbs, objects, modifiers, negation, sid, vid, oid, neg, mods))
    threshold = float(np.median(scores))
    return {"subjects": subjects, "objects": objects, "verbs": verbs, "modifiers": modifiers,
            "negation": negation, "threshold": threshold, "heldout": heldout}


def _latent_score(subjects: np.ndarray, verbs: np.ndarray, objects: np.ndarray, modifiers: np.ndarray, negation_op: np.ndarray,
                  sid: int, vid: int, oid: int, neg: int, modifier_ids: list[int]) -> float:
    state = objects[oid]
    for mid in modifier_ids:
        state = modifiers[mid] @ state
    if neg:
        state = negation_op @ state
    related = verbs[vid] @ state
    score = float(subjects[sid] @ related)
    return score


def _text(sid: int, vid: int, oid: int, neg: int, modifier_ids: list[int]) -> str:
    mods = " ".join(DISTRACTORS[i] for i in modifier_ids)
    prefix = f"{SUBJECTS[sid]}" + (f" {mods}" if mods else "")
    return f"{prefix}{' not' if neg else ''} {VERBS[vid]} {OBJECTS[oid]}"


def _primitive_features(sid: int, vid: int, oid: int, neg: int, modifier_ids: list[int], max_modifiers: int) -> dict[str, float]:
    vals: dict[str, float] = {}
    for i, token in enumerate(SUBJECTS): vals[f"ps_{token}"] = float(i == sid)
    for i, token in enumerate(VERBS): vals[f"pv_{token}"] = float(i == vid)
    for i, token in enumerate(OBJECTS): vals[f"po_{token}"] = float(i == oid)
    vals["pn_not"] = float(neg)
    for pos in range(max_modifiers):
        for i, token in enumerate(DISTRACTORS):
            vals[f"pd_pos{pos}_{token}"] = float(pos < len(modifier_ids) and modifier_ids[pos] == i)
    return vals


def _primitive_columns(max_modifiers: int) -> list[str]:
    cols = [f"ps_{t}" for t in SUBJECTS] + [f"pv_{t}" for t in VERBS] + [f"po_{t}" for t in OBJECTS] + ["pn_not"]
    cols += [f"pd_pos{p}_{t}" for p in range(max_modifiers) for t in DISTRACTORS]
    return cols


def _overlap(df: pd.DataFrame) -> dict[str, int]:
    names = sorted(df["split"].unique())
    sets = {s: set(df.loc[df["split"] == s, "text"]) for s in names}
    return {f"{a}-{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}


def generate_latent_compositional_dataset(
    n_samples: int,
    seed: int,
    semantic_seed: int = 314159,
    latent_dim: int = 6,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.20,
    ood_composition_fraction: float = 0.10,
    ood_depth_fraction: float = 0.10,
    max_modifiers: int = 4,
    ood_depth_min: int = 3,
    min_abs_margin: float = 0.05,
    label_noise: float = 0.0,
    unique_examples: bool = True,
) -> tuple[pd.DataFrame, LatentGenerationReport]:
    if n_samples < 80: raise ValueError("n_samples must be >= 80 for strict Stage 6 splits")
    if not 0 <= label_noise < 0.5: raise ValueError("label_noise must be in [0,0.5)")
    if max_modifiers < ood_depth_min: raise ValueError("max_modifiers must be >= ood_depth_min")
    program = latent_semantic_program(int(semantic_seed), int(latent_dim))
    threshold = float(program["threshold"]); heldout = list(program["heldout"])
    rng = np.random.default_rng(int(seed))
    n_val = max(1, int(round(n_samples * validation_fraction)))
    n_test = max(1, int(round(n_samples * test_fraction)))
    n_comp = max(1, int(round(n_samples * ood_composition_fraction)))
    n_depth = max(1, int(round(n_samples * ood_depth_fraction)))
    n_train = n_samples - n_val - n_test - n_comp - n_depth
    if n_train < 1: raise ValueError("Split fractions leave no training data")
    targets = {"train": n_train, "val": n_val, "test": n_test, "ood_composition": n_comp, "ood_depth": n_depth}
    pools: dict[str, list[dict[str, Any]]] = {k: [] for k in targets}
    id_rows: list[dict[str, Any]] = []
    seen: set[str] = set(); attempts = 0
    needed_id = n_train + n_val + n_test
    while len(id_rows) < needed_id or len(pools["ood_composition"]) < n_comp or len(pools["ood_depth"]) < n_depth:
        attempts += 1
        if attempts > n_samples * 500:
            raise RuntimeError("Unable to populate latent split pools; reduce min_abs_margin or n_samples")
        sid = int(rng.integers(len(SUBJECTS))); vid = int(rng.integers(len(VERBS))); oid = int(rng.integers(len(OBJECTS)))
        neg = int(rng.integers(2)); depth = int(rng.integers(max_modifiers + 1))
        modifiers = [int(rng.integers(len(DISTRACTORS))) for _ in range(depth)]
        text = _text(sid, vid, oid, neg, modifiers)
        if unique_examples and text in seen: continue
        score = _latent_score(program["subjects"], program["verbs"], program["objects"], program["modifiers"], program["negation"], sid, vid, oid, neg, modifiers)
        margin = score - threshold
        if abs(margin) < float(min_abs_margin): continue
        seen.add(text)
        clean = int(score >= threshold); label = clean if rng.random() >= label_noise else 1 - clean
        row: dict[str, Any] = {
            "text": text, "label": int(label), "clean_label": int(clean), "subject": SUBJECTS[sid], "verb": VERBS[vid], "object": OBJECTS[oid],
            "subject_id": sid, "verb_id": vid, "object_id": oid, "negation": neg, "modifier_count": depth, "distractor_count": depth,
            "grammar_depth": 2 + depth + neg, "is_heldout_composition": (sid, oid) in heldout, "is_high_depth": depth >= ood_depth_min,
            "audit_latent_score": float(score), "audit_latent_margin": float(margin), "audit_semantic_seed": int(semantic_seed),
        }
        row.update(_primitive_features(sid, vid, oid, neg, modifiers, max_modifiers))
        if row["is_heldout_composition"]:
            if len(pools["ood_composition"]) < n_comp: pools["ood_composition"].append(row)
        elif row["is_high_depth"]:
            if len(pools["ood_depth"]) < n_depth: pools["ood_depth"].append(row)
        elif len(id_rows) < needed_id:
            id_rows.append(row)
    rng.shuffle(id_rows)
    pools["train"] = id_rows[:n_train]; pools["val"] = id_rows[n_train:n_train+n_val]; pools["test"] = id_rows[n_train+n_val:needed_id]
    rows: list[dict[str, Any]] = []
    for split, selected in pools.items():
        for row in selected:
            row = dict(row); row["split"] = split; row["ood_condition"] = split if split.startswith("ood") else "id"; rows.append(row)
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    df.insert(0, "id", [f"latent_{i:06d}" for i in range(len(df))])
    cols = _primitive_columns(max_modifiers)
    content_hash = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
    counts = df["split"].value_counts().to_dict()
    report = LatentGenerationReport(
        n_samples=len(df), n_train=int(counts.get("train", 0)), n_val=int(counts.get("val", 0)), n_test=int(counts.get("test", 0)),
        n_ood=int(counts.get("ood_composition", 0) + counts.get("ood_depth", 0)), feature_columns=cols, label_column="label",
        content_hash=content_hash, heldout_compositions=heldout, semantic_seed=int(semantic_seed), latent_dim=int(latent_dim),
        semantic_threshold=threshold, min_abs_margin=float(min_abs_margin), duplicate_text_count=int(df["text"].duplicated().sum()),
        exact_duplicate_overlap=_overlap(df), ood_condition_counts={"ood_composition": int(counts.get("ood_composition",0)), "ood_depth": int(counts.get("ood_depth",0))},
    )
    return df, report
