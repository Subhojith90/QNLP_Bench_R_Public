import numpy as np
import pandas as pd


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-8, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    rows = []
    for idx, (lo, hi) in enumerate(zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:])):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        rows.append({"bin": idx, "confidence_low": lo, "confidence_high": hi, "n": int(mask.sum()), "mean_confidence": float(conf[mask].mean()) if mask.any() else float("nan"), "empirical_accuracy": float(correct[mask].mean()) if mask.any() else float("nan")})
    return pd.DataFrame(rows)
