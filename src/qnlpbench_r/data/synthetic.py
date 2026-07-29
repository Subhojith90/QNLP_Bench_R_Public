from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SUBJECTS = ["alice", "bob", "chemist", "robot", "doctor", "student", "judge", "engineer"]
VERBS = ["trusts", "doubts", "examines", "rejects", "supports", "questions", "verifies", "challenges"]
OBJECTS = ["proof", "paper", "model", "dataset", "claim", "patient", "contract", "theorem"]
DISTRACTORS = ["carefully", "quickly", "silently", "during_review", "after_revision", "with_context"]


@dataclass(frozen=True)
class SyntheticGenerationReport:
    n_samples: int
    n_train: int
    n_val: int
    n_test: int
    n_ood: int
    feature_columns: list[str]
    label_column: str
    content_hash: str
    heldout_compositions: list[tuple[int, int]]
    split_policy: str = "mixed"
    duplicate_text_count: int = 0
    exact_duplicate_overlap: dict[str, int] | None = None
    ood_condition_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_ood": self.n_ood,
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
            "content_hash": self.content_hash,
            "heldout_compositions": [list(x) for x in self.heldout_compositions],
            "split_policy": self.split_policy,
            "duplicate_text_count": self.duplicate_text_count,
            "exact_duplicate_overlap": self.exact_duplicate_overlap or {},
            "ood_condition_counts": self.ood_condition_counts or {},
        }


def _composition_label(subject_id: int, verb_id: int, object_id: int, negation: int, distractors: int) -> int:
    """Known non-local synthetic label rule."""
    subject_group = subject_id % 2
    object_group = (object_id // 2) % 2
    verb_polarity = 1 if verb_id in {0, 2, 4, 6} else 0
    long_range = (subject_group ^ object_group) ^ negation
    depth_effect = 1 if distractors >= 3 else 0
    return int((long_range ^ depth_effect) == verb_polarity)


def _make_text(subject_id: int, verb_id: int, object_id: int, negation: int, distractors: int) -> str:
    ds = [DISTRACTORS[(subject_id + verb_id + object_id + i) % len(DISTRACTORS)] for i in range(distractors)]
    neg = " not" if negation else ""
    middle = " ".join(ds)
    if middle:
        return f"{SUBJECTS[subject_id]} {middle}{neg} {VERBS[verb_id]} {OBJECTS[object_id]}"
    return f"{SUBJECTS[subject_id]}{neg} {VERBS[verb_id]} {OBJECTS[object_id]}"


def _features(subject_id: int, verb_id: int, object_id: int, negation: int, distractors: int, feature_dim: int) -> list[float]:
    subject_group = subject_id % 2
    object_group = (object_id // 2) % 2
    verb_polarity = 1 if verb_id in {0, 2, 4, 6} else 0
    long_range = (subject_group ^ object_group) ^ negation
    depth_effect = 1 if distractors >= 3 else 0
    # The first features remain simple numeric grammar metadata. The later features expose
    # rule components for the Stage-3 learnability diagnostics while avoiding direct labels.
    base = [
        subject_id / (len(SUBJECTS) - 1),
        verb_id / (len(VERBS) - 1),
        object_id / (len(OBJECTS) - 1),
        float(negation),
        distractors / 8.0,
        float(subject_group),
        float(object_group),
        float(verb_polarity),
        float(long_range),
        float(depth_effect),
        float(subject_group ^ object_group),
        float((verb_polarity ^ negation)),
        np.sin((subject_id + 1) * (verb_id + 1) / 3.0),
        np.cos((object_id + 1) * (negation + 1)),
    ]
    if feature_dim <= len(base):
        return [float(x) for x in base[:feature_dim]]
    extra = [float(np.sin((k + 1) * (subject_id + 1) + (object_id + 1))) for k in range(feature_dim - len(base))]
    return [float(x) for x in base + extra]


def _primitive_features(subject_id: int, verb_id: int, object_id: int, negation: int, distractors: int, max_distractors: int) -> dict[str, float]:
    """Return model-visible primitive token/slot features.

    These columns encode observable lexical/structural inputs only: subject, verb,
    object, the explicit negation token, and the ordered distractor-token sequence.
    They deliberately do not expose the generator's derived rule components such as
    long_range, depth_effect, verb_polarity or XOR combinations.
    """
    values: dict[str, float] = {}
    for i, token in enumerate(SUBJECTS):
        values[f"ps_{token}"] = float(i == subject_id)
    for i, token in enumerate(VERBS):
        values[f"pv_{token}"] = float(i == verb_id)
    for i, token in enumerate(OBJECTS):
        values[f"po_{token}"] = float(i == object_id)
    values["pn_not"] = float(negation)
    # Ordered token occupancy encodes the visible distractor sequence rather than
    # a precomputed depth threshold. A learner must infer any useful interaction.
    sequence = [DISTRACTORS[(subject_id + verb_id + object_id + pos) % len(DISTRACTORS)] for pos in range(distractors)]
    for pos in range(max_distractors):
        for token in DISTRACTORS:
            values[f"pd_pos{pos}_{token}"] = float(pos < len(sequence) and sequence[pos] == token)
    return values


def _primitive_feature_columns(max_distractors: int) -> list[str]:
    cols = [f"ps_{t}" for t in SUBJECTS] + [f"pv_{t}" for t in VERBS] + [f"po_{t}" for t in OBJECTS] + ["pn_not"]
    cols.extend(f"pd_pos{pos}_{token}" for pos in range(max_distractors) for token in DISTRACTORS)
    return cols


def _duplicate_overlap(df: pd.DataFrame) -> dict[str, int]:
    splits = sorted(df["split"].astype(str).unique())
    out: dict[str, int] = {}
    text_by_split = {s: set(df.loc[df["split"].astype(str) == s, "text"].astype(str)) for s in splits}
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            out[f"{left}-{right}"] = len(text_by_split[left].intersection(text_by_split[right]))
    return out


def _assign_split(df: pd.DataFrame, rng: np.random.Generator, *, n_val: int, n_test: int, n_ood: int, ood_mode: str, holdout_compositions: bool, max_distractors: int, ood_composition_fraction: float | None, ood_depth_fraction: float | None, require_ood_purity: bool, isolate_heldout_compositions_from_id_splits: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["split"] = "train"
    df["ood_condition"] = "id"

    if ood_mode == "separate":
        n_comp = max(1, int(round(len(df) * (ood_composition_fraction if ood_composition_fraction is not None else 0.05))))
        n_depth = max(1, int(round(len(df) * (ood_depth_fraction if ood_depth_fraction is not None else 0.15))))
        comp_candidates = df.index[df["is_heldout_composition"]].to_numpy(copy=True)
        depth_candidates = df.index[df["is_high_depth"] & ~df["is_heldout_composition"]].to_numpy(copy=True)
        rng.shuffle(comp_candidates)
        rng.shuffle(depth_candidates)
        if isolate_heldout_compositions_from_id_splits:
            # Stage 5B policy: every example from a held-out subject/object family belongs
            # only to composition-OOD. Validation and ordinary test are drawn exclusively
            # from seen-composition families, so model selection cannot observe OOD families.
            comp_selected = comp_candidates
        else:
            if require_ood_purity and len(comp_candidates) < n_comp:
                raise ValueError(
                    f"Insufficient candidates for pure OOD-composition split: {len(comp_candidates)}/{n_comp}. "
                    "Lower ood_composition_fraction or increase n_samples."
                )
            comp_selected = comp_candidates[: min(n_comp, len(comp_candidates))]
        if require_ood_purity and len(depth_candidates) < n_depth:
            raise ValueError(
                f"Insufficient candidates for pure OOD-depth split: {len(depth_candidates)}/{n_depth}. "
                "Lower ood_depth_fraction or increase n_samples."
            )
        depth_selected = depth_candidates[: min(n_depth, len(depth_candidates))]
        df.loc[comp_selected, "split"] = "ood_composition"
        df.loc[comp_selected, "ood_condition"] = "heldout_composition"
        df.loc[depth_selected, "split"] = "ood_depth"
        df.loc[depth_selected, "ood_condition"] = "high_depth"
    else:
        ood_candidates = df.index[df["is_heldout_composition"]].to_numpy(copy=True)
        if len(ood_candidates) < n_ood:
            high_depth = df.index[df["is_high_depth"] & ~df["is_heldout_composition"]].to_numpy(copy=True)
            rng.shuffle(high_depth)
            ood_candidates = np.concatenate([ood_candidates, high_depth[: max(0, n_ood - len(ood_candidates))]])
        rng.shuffle(ood_candidates)
        selected = ood_candidates[:n_ood]
        df.loc[selected, "split"] = "ood"
        df.loc[df["split"] == "ood", "ood_condition"] = np.where(
            df.loc[df["split"] == "ood", "is_heldout_composition"], "heldout_composition", "high_depth_fallback"
        )

    remaining = df.index[df["split"] == "train"].to_numpy(copy=True)
    rng.shuffle(remaining)
    df.loc[remaining[:n_test], "split"] = "test"
    df.loc[remaining[n_test:n_test + n_val], "split"] = "val"

    if holdout_compositions:
        leaked = df[(df["split"] == "train") & (df["is_heldout_composition"])]
        if not leaked.empty:
            target = "ood_composition" if ood_mode == "separate" else "ood"
            df.loc[leaked.index, "split"] = target
            df.loc[leaked.index, "ood_condition"] = "heldout_composition"
    return df


def generate_synthetic_compositional_dataset(
    n_samples: int,
    seed: int,
    feature_dim: int = 12,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.20,
    ood_fraction: float = 0.20,
    max_distractors: int = 5,
    holdout_compositions: bool = True,
    label_noise: float = 0.0,
    unique_examples: bool = False,
    ood_mode: str = "mixed",
    ood_composition_fraction: float | None = None,
    ood_depth_fraction: float | None = None,
    require_ood_purity: bool = False,
    isolate_heldout_compositions_from_id_splits: bool = False,
) -> tuple[pd.DataFrame, SyntheticGenerationReport]:
    """Generate a controlled compositional classification dataset with OOD splits.

    ood_mode="mixed" preserves the original single OOD split. ood_mode="separate"
    creates pure ood_composition and ood_depth splits and never hides high-depth fallback
    examples inside the heldout-composition condition.
    """
    if n_samples < 40:
        raise ValueError("n_samples must be at least 40 to create train/val/test/OOD splits.")
    if not 0 <= label_noise < 0.5:
        raise ValueError("label_noise must be in [0, 0.5).")
    if ood_mode not in {"mixed", "separate"}:
        raise ValueError("ood_mode must be either 'mixed' or 'separate'.")
    max_possible = len(SUBJECTS) * len(VERBS) * len(OBJECTS) * 2 * (max_distractors + 1)
    if unique_examples and n_samples > max_possible:
        raise ValueError(f"unique_examples=True allows at most {max_possible} examples for this grammar.")

    rng = np.random.default_rng(seed)
    heldout = [(0, 0), (1, 3), (4, 6), (7, 2)] if holdout_compositions else []
    rows: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    attempts = 0
    while len(rows) < n_samples:
        attempts += 1
        if attempts > max(10000, n_samples * 50):
            raise RuntimeError("Could not generate enough unique synthetic examples. Lower n_samples or disable unique_examples.")
        sid = int(rng.integers(len(SUBJECTS)))
        vid = int(rng.integers(len(VERBS)))
        oid = int(rng.integers(len(OBJECTS)))
        neg = int(rng.integers(2))
        dist = int(rng.integers(max_distractors + 1))
        text = _make_text(sid, vid, oid, neg, dist)
        if unique_examples and text in seen_texts:
            continue
        seen_texts.add(text)
        clean_label = _composition_label(sid, vid, oid, neg, dist)
        label = clean_label
        if label_noise > 0 and rng.random() < label_noise:
            label = 1 - label
        row = {
            "id": f"syn_{len(rows):06d}",
            "text": text,
            "label": label,
            "clean_label": clean_label,
            "subject": SUBJECTS[sid],
            "verb": VERBS[vid],
            "object": OBJECTS[oid],
            "subject_id": sid,
            "verb_id": vid,
            "object_id": oid,
            "negation": neg,
            "distractor_count": dist,
            "grammar_depth": 2 + dist + neg,
            "is_heldout_composition": (sid, oid) in heldout,
            "is_high_depth": dist >= max(3, max_distractors - 1),
        }
        for j, value in enumerate(_features(sid, vid, oid, neg, dist, feature_dim)):
            row[f"f{j}"] = value
        row.update(_primitive_features(sid, vid, oid, neg, dist, max_distractors))
        rows.append(row)

    df = pd.DataFrame(rows)
    n_ood = max(1, int(round(n_samples * ood_fraction)))
    n_test = max(1, int(round(n_samples * test_fraction)))
    n_val = max(1, int(round(n_samples * validation_fraction)))
    df = _assign_split(
        df,
        rng,
        n_val=n_val,
        n_test=n_test,
        n_ood=n_ood,
        ood_mode=ood_mode,
        holdout_compositions=holdout_compositions,
        max_distractors=max_distractors,
        ood_composition_fraction=ood_composition_fraction,
        ood_depth_fraction=ood_depth_fraction,
        require_ood_purity=require_ood_purity,
        isolate_heldout_compositions_from_id_splits=isolate_heldout_compositions_from_id_splits,
    )
    feature_columns = [f"f{j}" for j in range(feature_dim)] + _primitive_feature_columns(max_distractors)
    content_hash = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
    split_counts = df["split"].value_counts().to_dict()
    report = SyntheticGenerationReport(
        n_samples=len(df),
        n_train=int(split_counts.get("train", 0)),
        n_val=int(split_counts.get("val", 0)),
        n_test=int(split_counts.get("test", 0)),
        n_ood=int(sum(v for k, v in split_counts.items() if str(k).startswith("ood"))),
        feature_columns=feature_columns,
        label_column="label",
        content_hash=content_hash,
        heldout_compositions=heldout,
        split_policy=ood_mode,
        duplicate_text_count=int(df["text"].duplicated().sum()),
        exact_duplicate_overlap=_duplicate_overlap(df),
        ood_condition_counts=df.loc[df["split"].astype(str).str.startswith("ood"), "ood_condition"].value_counts().to_dict(),
    )
    return df.reset_index(drop=True), report
