from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from qnlpbench_r.data.preprocessing import FeaturePreprocessor
from qnlpbench_r.evaluation.evaluate import predict_proba
from qnlpbench_r.evaluation.metrics import classification_metrics


def perturb_frame(df: pd.DataFrame, feature_columns: list[str], variant: str, severity: float, seed: int) -> pd.DataFrame:
    """Create a perturbed copy of a dataframe while preserving labels."""
    rng = np.random.default_rng(seed)
    out = df.copy(deep=True)
    if variant == "feature_noise":
        noise = rng.normal(loc=0.0, scale=float(severity), size=(len(out), len(feature_columns)))
        out.loc[:, feature_columns] = out[feature_columns].to_numpy(dtype=float) + noise
    elif variant == "parser_corruption":
        n_cols = max(1, int(round(float(severity) * len(feature_columns))))
        cols = list(rng.choice(feature_columns, size=n_cols, replace=False))
        shuffled = out[cols].to_numpy(dtype=float).copy()
        for j in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, j])
        out.loc[:, cols] = shuffled
    elif variant == "word_order_proxy":
        if "distractor_count" in out.columns:
            jitter = rng.integers(-1, 2, size=len(out))
            out["distractor_count"] = np.clip(out["distractor_count"].to_numpy() + jitter, 0, None)
        if feature_columns:
            out.loc[:, feature_columns[-1]] = rng.permutation(out[feature_columns[-1]].to_numpy())
    else:
        raise ValueError(f"Unsupported feature/text perturbation variant: {variant}")
    out["perturbation_variant"] = variant
    out["perturbation_severity"] = severity
    return out


def apply_prediction_noise(probs: np.ndarray, variant: str, severity: float, seed: int, shots: int | None = None) -> np.ndarray:
    """Apply prediction-level approximations for shot and depolarizing noise."""
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    if variant == "depolarizing_prediction_noise":
        n_classes = probs.shape[1]
        mixed = (1.0 - severity) * probs + severity * np.ones_like(probs) / n_classes
        return mixed / mixed.sum(axis=1, keepdims=True)
    if variant == "shot_noise":
        if shots is None or shots <= 0:
            raise ValueError("shot_noise requires a positive shot count.")
        if probs.shape[1] != 2:
            raise ValueError("shot_noise approximation currently supports binary classification only.")
        counts = rng.binomial(int(shots), probs[:, 1])
        p1 = counts / float(shots)
        return np.stack([1.0 - p1, p1], axis=1)
    raise ValueError(f"Unsupported prediction noise variant: {variant}")


def evaluate_robustness(model: nn.Module, clean_arrays: dict[str, tuple[np.ndarray, np.ndarray]], original_frames: dict[str, pd.DataFrame], preprocessor: FeaturePreprocessor, feature_columns: list[str], device: torch.device, eval_config: dict[str, Any], seed: int) -> dict[str, float]:
    """Evaluate robustness variants and return metrics with descriptive prefixes."""
    rob_cfg = eval_config.get("robustness", {})
    if not rob_cfg.get("enabled", False):
        return {}
    variants = list(rob_cfg.get("variants", []))
    severities = [float(s) for s in rob_cfg.get("severities", [0.1])]
    shots_list = [int(s) for s in rob_cfg.get("shots", [512])]
    eval_splits = [s for s in eval_config.get("splits", ["test"]) if s in original_frames]
    out: dict[str, float] = {}
    for split in eval_splits:
        clean_X, clean_y = clean_arrays[split]
        if clean_X.shape[0] == 0:
            continue
        clean_probs = predict_proba(model, clean_X, device=device, batch_size=int(eval_config.get("batch_size", 256)))
        clean_acc = classification_metrics(clean_y, clean_probs, prefix="").get("accuracy", np.nan)
        for variant in variants:
            if variant in {"feature_noise", "parser_corruption", "word_order_proxy"}:
                for severity in severities:
                    pert = perturb_frame(original_frames[split], feature_columns, variant, severity, seed=seed + int(severity * 1000))
                    Xp, yp = preprocessor.transform_frame(pert)
                    probs = predict_proba(model, Xp, device=device, batch_size=int(eval_config.get("batch_size", 256)))
                    prefix = f"robust_{split}_{variant}_sev{severity:.3f}_"
                    metrics = classification_metrics(yp, probs, prefix=prefix)
                    out.update(metrics)
                    out[f"{prefix}accuracy_gap_from_clean"] = float(clean_acc - metrics.get(f"{prefix}accuracy", np.nan))
            elif variant == "depolarizing_prediction_noise":
                for severity in severities:
                    probs = apply_prediction_noise(clean_probs, variant, severity, seed=seed)
                    prefix = f"robust_{split}_{variant}_sev{severity:.3f}_"
                    metrics = classification_metrics(clean_y, probs, prefix=prefix)
                    out.update(metrics)
                    out[f"{prefix}accuracy_gap_from_clean"] = float(clean_acc - metrics.get(f"{prefix}accuracy", np.nan))
            elif variant == "shot_noise":
                for shots in shots_list:
                    probs = apply_prediction_noise(clean_probs, variant, severity=0.0, seed=seed + shots, shots=shots)
                    prefix = f"robust_{split}_{variant}_shots{shots}_"
                    metrics = classification_metrics(clean_y, probs, prefix=prefix)
                    out.update(metrics)
                    out[f"{prefix}accuracy_gap_from_clean"] = float(clean_acc - metrics.get(f"{prefix}accuracy", np.nan))
            else:
                raise ValueError(f"Unknown robustness variant: {variant}")
    return out
