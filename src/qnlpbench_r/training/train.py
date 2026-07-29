from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from qnlpbench_r.data.preprocessing import make_loader
from qnlpbench_r.evaluation.evaluate import predict_labels, predict_proba
from qnlpbench_r.evaluation.metrics import classification_metrics
from qnlpbench_r.training.losses import build_loss
from qnlpbench_r.training.optim import build_optimizer


def _epoch_gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item() ** 2)
    return float(total ** 0.5)


def train_model(model: nn.Module, arrays: dict[str, tuple[np.ndarray, np.ndarray]], training_config: dict[str, Any], device: torch.device, seed: int, run_dir: str | Path, logger: logging.Logger) -> tuple[nn.Module, pd.DataFrame, dict[str, float]]:
    """Train a torch model with deterministic data loading and early stopping."""
    run_dir = Path(run_dir)
    model = model.to(device)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    if X_train.shape[0] == 0:
        raise ValueError("Training split is empty.")
    if hasattr(model, "fit_labels"):
        model.fit_labels(torch.from_numpy(y_train).to(device))
    if hasattr(model, "fit_features"):
        model.fit_features(X_train, y_train)
    if not getattr(model, "is_trainable", True):
        probs = predict_proba(model, X_val, device=device, batch_size=int(training_config.get("batch_size", 128)))
        labels = predict_labels(model, X_val, device=device, batch_size=int(training_config.get("batch_size", 128)))
        val_metrics = classification_metrics(y_val, probs, prefix="val_", y_pred=labels)
        history = pd.DataFrame([{ "epoch": 0, "train_loss": np.nan, "val_loss": np.nan, **val_metrics }])
        history.to_csv(run_dir / "history.csv", index=False)
        return model, history, {"best_epoch": 0, "best_val_accuracy": val_metrics.get("val_accuracy", np.nan)}

    loader = make_loader(X_train, y_train, batch_size=int(training_config.get("batch_size", 128)), shuffle=True, seed=seed, num_workers=int(training_config.get("num_workers", 0)))
    loss_fn = build_loss(str(training_config.get("loss", "cross_entropy")))
    optimizer = build_optimizer(model.parameters(), learning_rate=float(training_config.get("learning_rate", 0.003)), weight_decay=float(training_config.get("weight_decay", 0.0)))
    max_epochs = int(training_config.get("max_epochs", 50))
    patience = int(training_config.get("early_stopping_patience", 10))
    grad_clip = float(training_config.get("gradient_clip_norm", 0.0))
    grad_accum_steps = max(1, int(training_config.get("gradient_accumulation_steps", 1)))
    use_amp = bool(training_config.get("mixed_precision", False)) and device.type == "cuda" and bool(getattr(model, "is_amp_safe", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -np.inf
    best_epoch = -1
    stale = 0
    rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        last_grad_norm = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            amp_context = torch.amp.autocast("cuda", enabled=True) if use_amp and hasattr(torch, "amp") else nullcontext()
            with amp_context:
                logits = model(xb)
                loss = loss_fn(logits, yb) / grad_accum_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                last_grad_norm = _epoch_gradient_norm(model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.item()) * grad_accum_steps * xb.shape[0]
            n_seen += xb.shape[0]
        train_loss = running_loss / max(1, n_seen)
        val_probs = predict_proba(model, X_val, device=device, batch_size=int(training_config.get("batch_size", 128)))
        val_metrics = classification_metrics(y_val, val_probs, prefix="val_")
        row = {"epoch": epoch, "train_loss": train_loss, "gradient_norm": last_grad_norm, **val_metrics}
        rows.append(row)
        logger.info("epoch=%s train_loss=%.6f val_accuracy=%.4f grad_norm=%.4f", epoch, train_loss, val_metrics.get("val_accuracy", float("nan")), last_grad_norm)
        monitor = val_metrics.get("val_balanced_accuracy", val_metrics.get("val_accuracy", -np.inf))
        if monitor > best_metric + 1e-6:
            best_metric = float(monitor)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            torch.save(best_state, run_dir / "model_best.pt")
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %s after %s stale epochs.", epoch, patience)
                break
    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    history.to_csv(run_dir / "history.csv", index=False)
    return model, history, {"best_epoch": int(best_epoch), "best_val_balanced_accuracy": float(best_metric)}
