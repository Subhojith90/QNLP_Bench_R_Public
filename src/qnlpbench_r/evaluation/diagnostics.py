from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from qnlpbench_r.data.synthetic import _composition_label


@dataclass(frozen=True)
class BaselineProbeSpec:
    name: str
    estimator: Any


def exact_duplicate_audit(df: pd.DataFrame, text_column: str = "text", split_column: str = "split") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_names = sorted(df[split_column].astype(str).unique())
    text_by_split = {s: set(df.loc[df[split_column].astype(str) == s, text_column].astype(str)) for s in split_names}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            rows.append({"left_split": left, "right_split": right, "exact_text_overlap": len(text_by_split[left] & text_by_split[right])})
    return pd.DataFrame(rows)


def split_summary(df: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame in df.groupby("split"):
        row: dict[str, Any] = {
            "split": split,
            "n": len(frame),
            "label_1_rate": float(frame[label_column].mean()) if len(frame) else np.nan,
            "unique_texts": int(frame["text"].nunique()) if "text" in frame else np.nan,
            "duplicate_texts_within_split": int(frame["text"].duplicated().sum()) if "text" in frame else np.nan,
        }
        for col in ["is_heldout_composition", "is_high_depth"]:
            if col in frame.columns:
                row[f"{col}_count"] = int(frame[col].astype(bool).sum())
        if "ood_condition" in frame.columns:
            for cond, n_cond in frame["ood_condition"].value_counts(dropna=False).items():
                row[f"ood_condition_{cond}"] = int(n_cond)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def feature_label_summary(df: pd.DataFrame, feature_columns: list[str], label_column: str = "label", seed: int = 0) -> pd.DataFrame:
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[label_column].to_numpy(dtype=int)
    try:
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=seed)
    except Exception:
        mi = np.full(len(feature_columns), np.nan)
    rows = []
    for i, col in enumerate(feature_columns):
        series = df[col].astype(float)
        corr = float(series.corr(df[label_column].astype(float))) if series.std() > 0 else np.nan
        rows.append({"feature": col, "pearson_corr_with_label": corr, "abs_corr": abs(corr) if np.isfinite(corr) else np.nan, "mutual_information": float(mi[i]) if i < len(mi) else np.nan})
    return pd.DataFrame(rows).sort_values(["mutual_information", "abs_corr"], ascending=False).reset_index(drop=True)


def oracle_rule_predictions(frame: pd.DataFrame) -> np.ndarray:
    preds = []
    for row in frame.itertuples(index=False):
        preds.append(_composition_label(int(row.subject_id), int(row.verb_id), int(row.object_id), int(row.negation), int(row.distractor_count)))
    return np.asarray(preds, dtype=int)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def baseline_probe_specs(seed: int = 0) -> list[BaselineProbeSpec]:
    return [
        BaselineProbeSpec("majority_train", None),
        BaselineProbeSpec("sk_logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))),
        BaselineProbeSpec("decision_tree", DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=seed)),
        BaselineProbeSpec("rf_200", RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)),
        BaselineProbeSpec("extra_trees_200", ExtraTreesClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1)),
        BaselineProbeSpec("grad_boost", GradientBoostingClassifier(random_state=seed)),
        BaselineProbeSpec("rbf_svm", make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", class_weight="balanced", random_state=seed))),
    ]


def baseline_probe_summary(bundle: Any, seed: int = 0, splits: list[str] | None = None) -> pd.DataFrame:
    if splits is None:
        splits = ["test", "ood", "ood_composition", "ood_depth"]
    train = bundle.train
    X_train = train[bundle.feature_columns].to_numpy(dtype=float)
    y_train = train[bundle.label_column].to_numpy(dtype=int)
    majority_label = int(pd.Series(y_train).mode().iloc[0])
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for spec in baseline_probe_specs(seed):
        if spec.name == "majority_train":
            fitted[spec.name] = None
        else:
            model = spec.estimator
            model.fit(X_train, y_train)
            fitted[spec.name] = model
    for split in splits:
        frame = bundle.split(split)
        if frame.empty:
            continue
        X = frame[bundle.feature_columns].to_numpy(dtype=float)
        y = frame[bundle.label_column].to_numpy(dtype=int)
        clean_y = frame["clean_label"].to_numpy(dtype=int) if "clean_label" in frame.columns else y
        for name, model in fitted.items():
            pred = np.full_like(y, majority_label) if model is None else model.predict(X).astype(int)
            row = {"probe": name, "split": split, "seed": seed, **_score(y, pred)}
            rows.append(row)
        oracle_pred = oracle_rule_predictions(frame)
        rows.append({"probe": "oracle_rule_without_noise", "split": split, "seed": seed, **_score(y, oracle_pred)})
        if "clean_label" in frame.columns:
            rows.append({"probe": "oracle_vs_clean_label", "split": split, "seed": seed, **_score(clean_y, oracle_pred)})
    return pd.DataFrame(rows)
