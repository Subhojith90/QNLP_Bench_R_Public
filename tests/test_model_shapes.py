import torch
from qnlpbench_r.models.baseline_models import build_model


def test_baseline_forward_shapes():
    model = build_model({"name": "logistic", "type": "logistic_regression"}, input_dim=10, n_classes=2)
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 2)


def test_quantum_feature_forward_shapes():
    model = build_model({"name": "qfc", "type": "quantum_feature", "n_qubits": 3, "n_layers": 1}, input_dim=10, n_classes=2)
    x = torch.randn(5, 10)
    logits = model(x)
    states = model.quantum_states(x)
    assert logits.shape == (5, 2)
    assert states.shape == (5, 8)
    assert torch.allclose(torch.linalg.vector_norm(states, dim=1), torch.ones(5), atol=1e-5)
