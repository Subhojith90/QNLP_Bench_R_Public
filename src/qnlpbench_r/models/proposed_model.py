from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qnlpbench_r.device import estimate_statevector_memory_gb


class QuantumFeatureCircuitClassifier(nn.Module):
    """Differentiable small statevector circuit classifier for grammar-derived features."""

    is_trainable = True
    is_amp_safe = False

    def __init__(self, input_dim: int, n_classes: int = 2, n_qubits: int = 6, n_layers: int = 2, entanglement: str = "linear", data_reuploading: bool = True) -> None:
        super().__init__()
        if n_qubits < 1 or n_qubits > 12:
            raise ValueError("n_qubits must be between 1 and 12 for this laptop-scale differentiable simulator.")
        if n_layers < 1:
            raise ValueError("n_layers must be positive.")
        if entanglement not in {"linear", "none"}:
            raise ValueError("entanglement must be 'linear' or 'none'.")
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.entanglement = entanglement
        self.data_reuploading = bool(data_reuploading)
        self.state_dim = 2**self.n_qubits
        angle_dim = self.n_layers * self.n_qubits if self.data_reuploading else self.n_qubits
        self.input_projection = nn.Linear(self.input_dim, angle_dim)
        self.trainable_rx = nn.Parameter(0.05 * torch.randn(self.n_layers, self.n_qubits))
        self.trainable_rz = nn.Parameter(0.05 * torch.randn(self.n_layers, self.n_qubits))
        self.readout = nn.Linear(self.n_qubits, self.n_classes)
        self.register_buffer("cz_phase", self._make_cz_phase(), persistent=False)
        self.register_buffer("z_signs", self._make_z_signs(), persistent=False)

    def _make_cz_phase(self) -> torch.Tensor:
        phase = torch.ones(self.state_dim, dtype=torch.complex64)
        if self.entanglement == "none":
            return phase
        for basis in range(self.state_dim):
            sign = 1.0
            for q in range(self.n_qubits - 1):
                if ((basis >> q) & 1) and ((basis >> (q + 1)) & 1):
                    sign *= -1.0
            phase[basis] = complex(sign, 0.0)
        return phase

    def _make_z_signs(self) -> torch.Tensor:
        signs = torch.empty(self.n_qubits, self.state_dim, dtype=torch.float32)
        for q in range(self.n_qubits):
            for basis in range(self.state_dim):
                signs[q, basis] = 1.0 if ((basis >> q) & 1) == 0 else -1.0
        return signs

    def _initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        state = torch.zeros(batch_size, self.state_dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j
        return state

    def _state_view_to_last_qubit(self, state: torch.Tensor, qubit: int) -> tuple[torch.Tensor, list[int]]:
        batch = state.shape[0]
        tensor = state.reshape(batch, *([2] * self.n_qubits))
        axes = [0] + [i + 1 for i in range(self.n_qubits) if i != qubit] + [qubit + 1]
        inv_axes = [0] * len(axes)
        for i, a in enumerate(axes):
            inv_axes[a] = i
        return tensor.permute(*axes), inv_axes

    def _restore_from_last_qubit(self, tensor: torch.Tensor, inv_axes: list[int]) -> torch.Tensor:
        return tensor.permute(*inv_axes).contiguous().reshape(tensor.shape[0], self.state_dim)

    def _apply_rx(self, state: torch.Tensor, qubit: int, theta: torch.Tensor) -> torch.Tensor:
        view, inv_axes = self._state_view_to_last_qubit(state, qubit)
        c = torch.cos(theta / 2).reshape(-1, *([1] * (view.ndim - 2))).to(state.dtype)
        s = torch.sin(theta / 2).reshape(-1, *([1] * (view.ndim - 2))).to(state.dtype)
        old0, old1 = view[..., 0], view[..., 1]
        minus_i = torch.tensor(-1j, dtype=torch.complex64, device=state.device)
        out = torch.stack([c * old0 + minus_i * s * old1, minus_i * s * old0 + c * old1], dim=-1)
        return self._restore_from_last_qubit(out, inv_axes)

    def _apply_rz(self, state: torch.Tensor, qubit: int, theta: torch.Tensor) -> torch.Tensor:
        view, inv_axes = self._state_view_to_last_qubit(state, qubit)
        phase0 = torch.exp((-0.5j * theta).to(torch.complex64)).reshape(-1, *([1] * (view.ndim - 2)))
        phase1 = torch.exp((0.5j * theta).to(torch.complex64)).reshape(-1, *([1] * (view.ndim - 2)))
        out = torch.stack([phase0 * view[..., 0], phase1 * view[..., 1]], dim=-1)
        return self._restore_from_last_qubit(out, inv_axes)

    def quantum_states(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        batch = x.shape[0]
        angles = self.input_projection(x)
        if self.data_reuploading:
            angles = angles.reshape(batch, self.n_layers, self.n_qubits)
        else:
            angles = angles.reshape(batch, 1, self.n_qubits).expand(batch, self.n_layers, self.n_qubits)
        state = self._initial_state(batch, x.device)
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                theta_z = angles[:, layer, q] + self.trainable_rz[layer, q]
                theta_rx = torch.sin(angles[:, layer, q]) + self.trainable_rx[layer, q]
                state = self._apply_rz(state, q, theta_z)
                state = self._apply_rx(state, q, theta_rx)
            if self.entanglement == "linear":
                state = state * self.cz_phase.to(state.device).unsqueeze(0)
            state = state / torch.linalg.vector_norm(state, dim=1, keepdim=True).clamp_min(1e-8)
        return state

    def z_expectations(self, x: torch.Tensor) -> torch.Tensor:
        state = self.quantum_states(x)
        probs = state.abs().pow(2).real
        return probs @ self.z_signs.to(probs.device).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.z_expectations(x))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def circuit_summary(self) -> dict[str, Any]:
        two_qubit_per_layer = max(0, self.n_qubits - 1) if self.entanglement == "linear" else 0
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "state_dim": self.state_dim,
            "entanglement": self.entanglement,
            "data_reuploading": self.data_reuploading,
            "parameter_count": self.parameter_count(),
            "single_qubit_rotation_count_proxy": 2 * self.n_qubits * self.n_layers,
            "two_qubit_gate_count_proxy": two_qubit_per_layer * self.n_layers,
            "depth_proxy": self.n_layers * (3 if self.entanglement == "linear" else 2),
            "bare_statevector_memory_gb_per_sample_complex64": estimate_statevector_memory_gb(self.n_qubits, complex_bytes=8, batch_size=1),
        }

    def model_summary(self) -> dict[str, Any]:
        out = {"model_type": "quantum_feature"}
        out.update(self.circuit_summary())
        return out


class GrammarStructuredQuantumCircuitClassifier(QuantumFeatureCircuitClassifier):
    """QFC with role-aware encoding and grammar-aligned entanglement.

    Model-visible inputs are expected to be primitive slot/token indicators. Separate
    role encoders map subject, predicate, object, negation and distractor-sequence
    inputs to circuit angles; the entanglement graph then represents specified
    grammatical interactions rather than a generic nearest-neighbour chain.
    """

    is_trainable = True
    is_amp_safe = False

    def __init__(
        self,
        input_dim: int,
        feature_columns: list[str],
        n_classes: int = 2,
        n_qubits: int = 6,
        n_layers: int = 2,
        structured_entanglement: str = "grammar",
        data_reuploading: bool = True,
    ) -> None:
        # Base operations and readout are reused; the generic projection is not used.
        super().__init__(input_dim=input_dim, n_classes=n_classes, n_qubits=n_qubits, n_layers=n_layers, entanglement="none", data_reuploading=data_reuploading)
        self.feature_columns = list(feature_columns)
        self.structured_entanglement = str(structured_entanglement)
        self.input_projection = nn.Identity()
        self.role_indices = self._make_role_indices(self.feature_columns)
        missing = [name for name in ["subject", "verb", "object"] if not self.role_indices[name]]
        self.primitive_role_encoding_active = not bool(missing)
        if missing:
            # Positive/negative control feature sets may not carry role-prefixed primitive columns.
            # Permit them as generic circuit controls while recording that grammar-role encoding is inactive.
            self.role_indices = {"generic_input": list(range(self.input_dim))}
        angle_dim = self.n_layers * self.n_qubits if self.data_reuploading else self.n_qubits
        self.role_projections = nn.ModuleDict({
            name: nn.Linear(len(indices), angle_dim, bias=True)
            for name, indices in self.role_indices.items() if indices
        })
        self.grammar_edges = self._edges_for_pattern(self.structured_entanglement)
        self.register_buffer("structured_cz_phase", self._make_edge_phase(self.grammar_edges), persistent=False)

    @staticmethod
    def _make_role_indices(columns: list[str]) -> dict[str, list[int]]:
        groups = {"subject": [], "verb": [], "object": [], "negation": [], "distractor_sequence": []}
        for idx, col in enumerate(columns):
            if col.startswith("ps_"):
                groups["subject"].append(idx)
            elif col.startswith("pv_"):
                groups["verb"].append(idx)
            elif col.startswith("po_"):
                groups["object"].append(idx)
            elif col.startswith("pn_"):
                groups["negation"].append(idx)
            elif col.startswith("pd_"):
                groups["distractor_sequence"].append(idx)
        return groups

    def _edges_for_pattern(self, pattern: str) -> list[tuple[int, int]]:
        if pattern in {"none", "no_entanglement"}:
            return []
        if pattern in {"linear", "generic_linear"}:
            return [(q, q + 1) for q in range(self.n_qubits - 1)]
        if pattern == "ring":
            return [(q, q + 1) for q in range(self.n_qubits - 1)] + ([(self.n_qubits - 1, 0)] if self.n_qubits > 2 else [])
        if pattern in {"star_verb", "verb_centered_star"}:
            base = [(1, q) for q in range(self.n_qubits) if q != 1]
            return base
        if pattern in {"star_context", "context_centered_star"}:
            centre = min(4, self.n_qubits - 1)
            return [(centre, q) for q in range(self.n_qubits) if q != centre]
        if pattern == "all_to_all":
            return [(a, b) for a in range(self.n_qubits) for b in range(a + 1, self.n_qubits)]
        random_graphs = {
            "random_fixed": [(0, 2), (1, 4), (3, 5), (0, 5)],
            "random_fixed_01": [(0, 2), (1, 4), (3, 5), (0, 5)],
            "random_fixed_02": [(0, 3), (1, 5), (2, 4), (1, 3)],
            "random_fixed_03": [(0, 4), (1, 2), (2, 5), (0, 3)],
            "random_fixed_04": [(0, 5), (1, 3), (2, 4), (2, 5)],
            "random_fixed_05": [(0, 2), (1, 3), (1, 5), (3, 4)],
            "random_fixed_06": [(0, 4), (0, 5), (1, 2), (3, 5)],
            "random_fixed_07": [(0, 1), (2, 4), (3, 5), (1, 5)],
            "random_fixed_08": [(0, 3), (1, 4), (2, 5), (0, 2)],
        }
        if pattern in random_graphs:
            return [(a, b) for a, b in random_graphs[pattern] if a < self.n_qubits and b < self.n_qubits]
        if pattern != "grammar":
            raise ValueError(f"Unsupported structured_entanglement: {pattern}")
        # Logical mapping: q0 subject, q1 verb, q2 object, q3 negation,
        # q4 distractor/depth context, q5 compositional readout channel.
        base = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 4), (4, 5)]
        return [(a, b) for a, b in base if a < self.n_qubits and b < self.n_qubits]

    def _make_edge_phase(self, edges: list[tuple[int, int]]) -> torch.Tensor:
        phase = torch.ones(self.state_dim, dtype=torch.complex64)
        for basis in range(self.state_dim):
            sign = 1.0
            for a, b in edges:
                if ((basis >> a) & 1) and ((basis >> b) & 1):
                    sign *= -1.0
            phase[basis] = complex(sign, 0.0)
        return phase

    def _role_angles(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        angle_dim = self.n_layers * self.n_qubits if self.data_reuploading else self.n_qubits
        angles = torch.zeros(batch, angle_dim, device=x.device, dtype=x.dtype)
        for name, indices in self.role_indices.items():
            if indices:
                angles = angles + self.role_projections[name](x[:, indices])
        return angles

    def quantum_states(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        batch = x.shape[0]
        angles = self._role_angles(x)
        if self.data_reuploading:
            angles = angles.reshape(batch, self.n_layers, self.n_qubits)
        else:
            angles = angles.reshape(batch, 1, self.n_qubits).expand(batch, self.n_layers, self.n_qubits)
        state = self._initial_state(batch, x.device)
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                theta_z = angles[:, layer, q] + self.trainable_rz[layer, q]
                theta_rx = torch.sin(angles[:, layer, q]) + self.trainable_rx[layer, q]
                state = self._apply_rz(state, q, theta_z)
                state = self._apply_rx(state, q, theta_rx)
            if self.grammar_edges:
                state = state * self.structured_cz_phase.to(state.device).unsqueeze(0)
            state = state / torch.linalg.vector_norm(state, dim=1, keepdim=True).clamp_min(1e-8)
        return state

    def circuit_summary(self) -> dict[str, Any]:
        base = super().circuit_summary()
        base.update({
            "architecture": "grammar_structured_qfc",
            "structured_entanglement": self.structured_entanglement,
            "grammar_edges": [list(edge) for edge in self.grammar_edges],
            "two_qubit_gate_count_proxy": len(self.grammar_edges) * self.n_layers,
            "depth_proxy": self.n_layers * (3 if self.grammar_edges else 2),
            "role_feature_counts": {name: len(indices) for name, indices in self.role_indices.items()},
            "primitive_role_encoding_active": self.primitive_role_encoding_active,
        })
        return base

    def model_summary(self) -> dict[str, Any]:
        out = {"model_type": "grammar_structured_quantum_feature"}
        out.update(self.circuit_summary())
        return out
