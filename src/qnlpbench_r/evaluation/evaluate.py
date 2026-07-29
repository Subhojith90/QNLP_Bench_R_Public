from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from qnlpbench_r.evaluation.metrics import classification_metrics, effective_rank_from_eigenvalues, kernel_target_alignment


def _stable_split_seed(seed: int, split: str) -> int:
    """Derive a deterministic per-split seed without relying on Python's randomized hash."""
    digest = hashlib.sha256(f"{int(seed)}::{split}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def select_kernel_sample_indices(
    y: np.ndarray,
    sample_cap: int = 256,
    seed: int = 0,
    strategy: str = "stratified",
) -> np.ndarray:
    """Select deterministic indices for capped quantum-kernel diagnostics.

    The earlier implementation used the first ``sample_cap`` examples, which made
    kernel diagnostics sensitive to dataset ordering. This selector supports a
    deterministic stratified sample by default, preserving class coverage while
    remaining fully reproducible across runs.
    """
    y = np.asarray(y, dtype=int)
    n_total = int(y.shape[0])
    n_sample = min(max(int(sample_cap), 0), n_total)
    if n_sample == 0:
        return np.empty(0, dtype=int)
    if n_sample == n_total:
        return np.arange(n_total, dtype=int)

    rng = np.random.default_rng(int(seed))
    strategy = str(strategy).lower().strip()

    if strategy in {"first", "head"}:
        return np.arange(n_sample, dtype=int)

    if strategy in {"random", "uniform"}:
        return np.sort(rng.choice(n_total, size=n_sample, replace=False).astype(int))

    if strategy != "stratified":
        raise ValueError(f"Unknown kernel sampling strategy: {strategy!r}. Use 'stratified', 'random', or 'first'.")

    classes, counts = np.unique(y, return_counts=True)
    if classes.size <= 1:
        return np.sort(rng.choice(n_total, size=n_sample, replace=False).astype(int))

    # Start from proportional allocation, with at least one example per class
    # when the cap permits it.
    raw = counts / counts.sum() * n_sample
    allocation = np.floor(raw).astype(int)
    if n_sample >= classes.size:
        allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts)

    # Adjust allocation to exactly match n_sample.
    while allocation.sum() > n_sample:
        candidates = np.where(allocation > 0)[0]
        # Remove from the class with the smallest fractional need first, while
        # preserving one per class when possible.
        removable = candidates[allocation[candidates] > (1 if n_sample >= classes.size else 0)]
        if removable.size == 0:
            removable = candidates
        idx = removable[np.argmin(raw[removable] - np.floor(raw[removable]))]
        allocation[idx] -= 1

    while allocation.sum() < n_sample:
        remaining = counts - allocation
        candidates = np.where(remaining > 0)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmax(raw[candidates] - allocation[candidates])]
        allocation[idx] += 1

    selected: list[np.ndarray] = []
    for cls, k in zip(classes, allocation):
        if k <= 0:
            continue
        cls_indices = np.flatnonzero(y == cls)
        selected.append(rng.choice(cls_indices, size=int(k), replace=False))

    if not selected:
        return np.empty(0, dtype=int)
    indices = np.concatenate(selected).astype(int)
    if indices.size > n_sample:
        indices = rng.choice(indices, size=n_sample, replace=False).astype(int)
    return np.sort(indices)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Predict class probabilities for a NumPy feature matrix."""
    if X.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    if hasattr(model, "predict_proba_np"):
        return model.predict_proba_np(X.astype(np.float32))
    model.eval()
    probs = []
    for start in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
        logits = model(xb)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits encountered during evaluation.")
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.vstack(probs)




def predict_labels(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Predict class labels using a model's authoritative label path when available.

    For sklearn estimators such as SVC, direct ``predict`` is the class-label
    source of truth. Probability argmax is retained only for probabilistic metrics
    because Platt-calibrated probabilities can disagree with SVC decision labels.
    """
    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if hasattr(model, "predict_labels_np"):
        return np.asarray(model.predict_labels_np(X.astype(np.float32)), dtype=np.int64)
    return predict_proba(model, X, device=device, batch_size=batch_size).argmax(axis=1).astype(np.int64)


@torch.no_grad()
def quantum_kernel_metrics(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    sample_cap: int = 256,
    seed: int = 0,
    sampling_strategy: str = "stratified",
) -> tuple[dict[str, float], np.ndarray]:
    """Compute quantum-state kernel diagnostics for models exposing quantum_states."""
    if not hasattr(model, "quantum_states") or X.shape[0] == 0:
        return {}, np.empty(0, dtype=int)

    indices = select_kernel_sample_indices(y, sample_cap=sample_cap, seed=seed, strategy=sampling_strategy)
    if indices.size == 0:
        return {}, indices

    xb = torch.from_numpy(X[indices].astype(np.float32)).to(device)
    states = model.quantum_states(xb).detach().cpu().numpy()
    K = np.abs(states @ states.conj().T) ** 2
    K_sym = (K + K.T) / 2.0
    evals = np.linalg.eigvalsh(K_sym)
    y_sample = y[indices]
    return {
        "quantum_kernel_alignment": kernel_target_alignment(K, y_sample),
        "quantum_kernel_effective_rank": effective_rank_from_eigenvalues(evals),
        "quantum_kernel_trace": float(np.trace(K)),
        "quantum_kernel_mean_offdiag": float((K.sum() - np.trace(K)) / max(1, indices.size * (indices.size - 1))),
        "quantum_kernel_sample_n": float(indices.size),
        "quantum_kernel_sample_class0_n": float(np.sum(y_sample == 0)),
        "quantum_kernel_sample_class1_n": float(np.sum(y_sample == 1)),
        "quantum_kernel_sampling_strategy": float({"first": 0, "head": 0, "random": 1, "uniform": 1, "stratified": 2}.get(str(sampling_strategy).lower().strip(), -1)),
    }, indices


def evaluate_splits(
    model: nn.Module,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    splits: list[str],
    batch_size: int,
    output_dir: str | Path,
    quantum_metrics_enabled: bool = True,
    kernel_sample_cap: int = 256,
    seed: int = 0,
    kernel_sampling_strategy: str = "stratified",
) -> dict[str, float]:
    """Evaluate model on named splits and save prediction CSV files."""
    output_dir = Path(output_dir)
    all_metrics = {}
    for split in splits:
        X, y = arrays[split]
        if X.shape[0] == 0:
            continue
        probs = predict_proba(model, X, device=device, batch_size=batch_size)
        labels = predict_labels(model, X, device=device, batch_size=batch_size)
        all_metrics.update(classification_metrics(y, probs, prefix=f"{split}_", y_pred=labels))
        pd.DataFrame({"y_true": y, "y_pred": labels, "p_class0": probs[:, 0], "p_class1": probs[:, 1] if probs.shape[1] > 1 else np.nan}).to_csv(output_dir / f"predictions_{split}.csv", index=False)
        if quantum_metrics_enabled and split in {"test", "ood", "ood_composition", "ood_depth"}:
            split_seed = _stable_split_seed(seed, split)
            qm, indices = quantum_kernel_metrics(
                model,
                X,
                y,
                device=device,
                sample_cap=kernel_sample_cap,
                seed=split_seed,
                sampling_strategy=kernel_sampling_strategy,
            )
            if indices.size:
                pd.DataFrame({"split": split, "local_index": indices, "y_true": y[indices]}).to_csv(output_dir / f"quantum_kernel_sample_{split}.csv", index=False)
            all_metrics.update({f"{split}_{k}": v for k, v in qm.items()})
    return all_metrics
