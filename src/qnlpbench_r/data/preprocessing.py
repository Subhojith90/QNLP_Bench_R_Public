from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from qnlpbench_r.data.datasets import DatasetBundle
from qnlpbench_r.seed import make_torch_generator, seed_worker


@dataclass
class FeaturePreprocessor:
    mean: np.ndarray
    std: np.ndarray
    feature_columns: list[str]
    label_column: str

    def transform_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = set(self.feature_columns + [self.label_column]).difference(df.columns)
        if missing:
            raise ValueError(f"Frame is missing required columns: {sorted(missing)}")
        X = df[self.feature_columns].to_numpy(dtype=np.float32)
        y = df[self.label_column].to_numpy(dtype=np.int64)
        X = (X - self.mean) / self.std
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values found after feature preprocessing.")
        return X.astype(np.float32), y

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "feature_columns": self.feature_columns, "label_column": self.label_column}


def fit_preprocessor(bundle: DatasetBundle) -> FeaturePreprocessor:
    X = bundle.train[bundle.feature_columns].to_numpy(dtype=np.float32)
    mean = X.mean(axis=0)
    std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0))
    return FeaturePreprocessor(mean, std, bundle.feature_columns, bundle.label_column)


def make_arrays(bundle: DatasetBundle, preprocessor: FeaturePreprocessor) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays = {}
    for split_name in ["train", "val", "test", "ood", "ood_composition", "ood_depth"]:
        frame = bundle.split(split_name)
        if frame.empty:
            arrays[split_name] = (np.empty((0, len(bundle.feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.int64))
        else:
            arrays[split_name] = preprocessor.transform_frame(frame)
    return arrays


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0) -> DataLoader:
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=int(num_workers), worker_init_fn=seed_worker if num_workers else None, generator=make_torch_generator(seed))
