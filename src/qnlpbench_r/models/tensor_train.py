from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn


def primitive_site_indices(feature_columns: list[str]) -> OrderedDict[str, list[int]]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    groups["subject"] = [i for i, name in enumerate(feature_columns) if name.startswith("ps_")]
    groups["verb"] = [i for i, name in enumerate(feature_columns) if name.startswith("pv_")]
    groups["object"] = [i for i, name in enumerate(feature_columns) if name.startswith("po_")]
    groups["negation"] = [i for i, name in enumerate(feature_columns) if name == "pn_not"]
    for position in range(4):
        groups[f"modifier_{position}"] = [
            i for i, name in enumerate(feature_columns) if name.startswith(f"pd_pos{position}_")
        ]
    missing = [name for name, indices in groups.items() if not indices]
    if missing:
        raise ValueError(f"Missing primitive feature groups for tensor train: {missing}")
    flattened = [index for indices in groups.values() for index in indices]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Tensor-train feature groups overlap")
    return groups


class TensorTrainClassifier(nn.Module):
    """Tensor-train classifier over ordered primitive semantic sites.

    Each site vector is augmented by a constant bias coordinate. Contracting the
    input with rank-3 cores yields a fixed-bond representation, followed by a
    linear class readout. This is a classical multilinear comparator rather than
    a quantum simulation.
    """

    is_trainable = True
    is_amp_safe = True

    def __init__(
        self,
        feature_columns: list[str],
        bond_dim: int,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.site_indices = primitive_site_indices(self.feature_columns)
        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive")
        cores: list[nn.Parameter] = []
        for site_number, indices in enumerate(self.site_indices.values()):
            left_rank = 1 if site_number == 0 else self.bond_dim
            right_rank = self.bond_dim
            physical_dim = len(indices) + 1
            scale = (left_rank * physical_dim) ** -0.5
            core = nn.Parameter(
                scale * torch.randn(left_rank, physical_dim, right_rank)
            )
            cores.append(core)
        self.cores = nn.ParameterList(cores)
        self.readout = nn.Linear(self.bond_dim, n_classes)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        state: torch.Tensor | None = None
        for core, indices in zip(self.cores, self.site_indices.values()):
            site = x[:, indices].float()
            site = torch.cat(
                [torch.ones(site.shape[0], 1, dtype=site.dtype, device=site.device), site],
                dim=1,
            )
            if state is None:
                state = torch.einsum("lpr,bp->br", core, site)
            else:
                state = torch.einsum("bl,lpr,bp->br", state, core, site)
            state = state / torch.linalg.vector_norm(state, dim=1, keepdim=True).clamp_min(
                1.0e-8
            )
        if state is None:
            raise RuntimeError("Tensor train contains no sites")
        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.representation(x))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def model_summary(self) -> dict[str, Any]:
        return {
            "model_type": "tensor_train_classifier",
            "bond_dimension": self.bond_dim,
            "site_count": len(self.site_indices),
            "site_names": list(self.site_indices),
            "parameter_count": self.parameter_count(),
        }
