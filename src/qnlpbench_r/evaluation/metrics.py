from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score


def _safe_probs(y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-8, 1.0)
    return y_prob / y_prob.sum(axis=1, keepdims=True)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = _safe_probs(y_prob)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _safe_probs(y_prob)
    onehot = np.zeros_like(y_prob)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, prefix: str = "", y_pred: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _safe_probs(np.asarray(y_prob, dtype=float))
    if len(y_true) == 0:
        return {}
    y_pred = y_prob.argmax(axis=1) if y_pred is None else np.asarray(y_pred, dtype=int)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_pred shape {y_pred.shape} does not match y_true shape {y_true.shape}.")
    labels = np.unique(y_true)
    out = {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}brier": brier_score_multiclass(y_true, y_prob),
        f"{prefix}ece_10": expected_calibration_error(y_true, y_prob, n_bins=10),
    }
    try:
        out[f"{prefix}nll"] = float(log_loss(y_true, y_prob, labels=list(range(y_prob.shape[1]))))
    except Exception:
        out[f"{prefix}nll"] = float("nan")
    if y_prob.shape[1] == 2 and len(labels) == 2:
        try:
            out[f"{prefix}roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            out[f"{prefix}roc_auc"] = float("nan")
    return out


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=int)
    signed = 2 * y - 1
    target = np.outer(signed, signed)
    denom = np.linalg.norm(K, ord="fro") * np.linalg.norm(target, ord="fro")
    return float(np.sum(K * target) / denom) if denom > 0 else 0.0


def effective_rank_from_eigenvalues(evals: np.ndarray) -> float:
    vals = np.maximum(np.asarray(evals, dtype=float), 0.0)
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    entropy = -np.sum([pi * math.log(pi) for pi in p if pi > 0])
    return float(math.exp(entropy))


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int = 1000, seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=values.size, replace=True).mean() for _ in range(int(n_bootstrap))]
    alpha = (1.0 - ci) / 2.0
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if values.size > 1 else 0.0, "ci_low": float(np.quantile(means, alpha)), "ci_high": float(np.quantile(means, 1.0 - alpha)), "n": int(values.size)}
