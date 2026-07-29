from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from qnlpbench_r.models.proposed_model import QuantumFeatureCircuitClassifier, GrammarStructuredQuantumCircuitClassifier


class MajorityClassifier(nn.Module):
    is_trainable = False
    is_amp_safe = True

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = int(n_classes)
        self.register_buffer("class_probs", torch.ones(self.n_classes) / self.n_classes)

    def fit_labels(self, y: torch.Tensor) -> None:
        counts = torch.bincount(y.cpu(), minlength=self.n_classes).float()
        self.class_probs = (counts / counts.sum().clamp_min(1.0)).to(self.class_probs.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.class_probs.to(x.device).clamp_min(1e-8)).unsqueeze(0).expand(x.shape[0], -1)

    def parameter_count(self) -> int:
        return 0

    def model_summary(self) -> dict[str, Any]:
        return {"model_type": "majority", "parameter_count": 0}


class TorchLogisticRegression(nn.Module):
    is_trainable = True
    is_amp_safe = True

    def __init__(self, input_dim: int, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def model_summary(self) -> dict[str, Any]:
        return {"model_type": "logistic_regression", "parameter_count": self.parameter_count()}


class MLPClassifier(nn.Module):
    is_trainable = True
    is_amp_safe = True

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_classes: int = 2, dropout: float = 0.0, depth: int = 1):
        super().__init__()
        depth = max(1, int(depth))
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def model_summary(self) -> dict[str, Any]:
        return {"model_type": "mlp", "parameter_count": self.parameter_count(), "depth": self.depth}


class PrimitiveDecoderMixin:
    """Decode primitive observable one-hot columns after train-fitted standardisation.

    One-hot members remain ordered under affine standardisation, so role token identity is
    recovered by argmax within each role group. The oracle is reported as an upper bound,
    not as a learned model or evidence of leakage.
    """
    def _init_decoder(self, feature_columns: list[str]) -> None:
        self.feature_columns = list(feature_columns)
        self.role_indices = {
            "subject": [i for i, c in enumerate(feature_columns) if c.startswith("ps_")],
            "verb": [i for i, c in enumerate(feature_columns) if c.startswith("pv_")],
            "object": [i for i, c in enumerate(feature_columns) if c.startswith("po_")],
            "negation": [i for i, c in enumerate(feature_columns) if c == "pn_not"],
        }
        positions: dict[int, list[int]] = {}
        for i, c in enumerate(feature_columns):
            if c.startswith("pd_pos"):
                pos = int(c.split("_")[1].replace("pos", ""))
                positions.setdefault(pos, []).append(i)
        self.position_indices = positions
        required = ["subject", "verb", "object", "negation"]
        if any(not self.role_indices[x] for x in required):
            raise ValueError("Primitive-rule controls require primitive_slot_sequence columns.")

    def _decode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sid = np.argmax(X[:, self.role_indices["subject"]], axis=1)
        vid = np.argmax(X[:, self.role_indices["verb"]], axis=1)
        oid = np.argmax(X[:, self.role_indices["object"]], axis=1)
        neg = (X[:, self.role_indices["negation"][0]] > 0).astype(int)
        distractors = np.zeros(X.shape[0], dtype=int)
        for pos, idx in sorted(self.position_indices.items()):
            distractors += (np.max(X[:, idx], axis=1) > 0).astype(int)
        return sid, vid, oid, neg, distractors

    def _oracle_labels(self, X: np.ndarray) -> np.ndarray:
        sid, vid, oid, neg, distractors = self._decode(X)
        subject_group = sid % 2
        object_group = (oid // 2) % 2
        verb_polarity = np.isin(vid, [0, 2, 4, 6]).astype(int)
        long_range = (subject_group ^ object_group) ^ neg
        depth_effect = (distractors >= 3).astype(int)
        return ((long_range ^ depth_effect) == verb_polarity).astype(int)

    def _interaction_matrix(self, X: np.ndarray) -> np.ndarray:
        sid, vid, oid, neg, distractors = self._decode(X)
        sg = (sid % 2).astype(float)
        og = ((oid // 2) % 2).astype(float)
        vp = np.isin(vid, [0, 2, 4, 6]).astype(float)
        de = (distractors >= 3).astype(float)
        # Explicitly capable but trainable classical representation family.
        terms = [sg, og, vp, neg.astype(float), de]
        primitives = [sg, og, vp, neg.astype(float), de]
        for i in range(len(primitives)):
            for j in range(i + 1, len(primitives)):
                terms.append(primitives[i] * primitives[j])
        terms.extend([
            (sg != og).astype(float),
            (sg != neg).astype(float),
            (og != neg).astype(float),
            ((sg.astype(int) ^ og.astype(int) ^ neg) != de.astype(int)).astype(float),
            self._oracle_labels(X).astype(float),
        ])
        return np.column_stack(terms).astype(np.float32)


class PrimitiveRuleOracleClassifier(nn.Module, PrimitiveDecoderMixin):
    is_trainable = False
    is_amp_safe = True

    def __init__(self, feature_columns: list[str], n_classes: int = 2, name: str = "exact_rule_oracle"):
        super().__init__()
        self.n_classes = n_classes
        self.name = name
        self._init_decoder(feature_columns)

    def predict_labels_np(self, X: np.ndarray) -> np.ndarray:
        return self._oracle_labels(X)

    def predict_proba_np(self, X: np.ndarray) -> np.ndarray:
        pred = self.predict_labels_np(X)
        probs = np.full((len(pred), self.n_classes), 1e-6, dtype=np.float32)
        probs[np.arange(len(pred)), pred] = 1.0 - 1e-6
        return probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.from_numpy(self.predict_proba_np(x.detach().cpu().numpy())).to(x.device))

    def model_summary(self) -> dict[str, Any]:
        return {"model_type": self.name, "parameter_count": 0, "oracle_upper_bound": True}


class PrimitiveInteractionLogisticClassifier(nn.Module, PrimitiveDecoderMixin):
    is_trainable = False
    is_amp_safe = True

    def __init__(self, feature_columns: list[str], n_classes: int = 2, seed: int = 0, C: float = 100.0):
        super().__init__()
        from sklearn.linear_model import LogisticRegression
        self.n_classes = n_classes
        self._init_decoder(feature_columns)
        self.estimator = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit_features(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(self._interaction_matrix(X), y)

    def predict_labels_np(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self._interaction_matrix(X)).astype(int)

    def predict_proba_np(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self._interaction_matrix(X)).astype(np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.from_numpy(self.predict_proba_np(x.detach().cpu().numpy())).to(x.device).clamp_min(1e-8))

    def model_summary(self) -> dict[str, Any]:
        return {"model_type": "interaction_logistic", "parameter_count": 0, "rule_capable_baseline": True}


class SklearnBaselineClassifier(nn.Module):
    """First-class sklearn baseline with direct-label and probability evaluation paths."""
    is_trainable = False
    is_amp_safe = True

    def __init__(self, kind: str, input_dim: int, n_classes: int = 2, seed: int = 0, **kwargs: Any) -> None:
        super().__init__()
        self.kind = str(kind)
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.kwargs = dict(kwargs)
        self.estimator = self._make_estimator()

    def _make_estimator(self):
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        if self.kind == "random_forest":
            return RandomForestClassifier(n_estimators=int(self.kwargs.get("n_estimators", 300)), max_depth=self.kwargs.get("max_depth", None), min_samples_leaf=int(self.kwargs.get("min_samples_leaf", 1)), random_state=self.seed, n_jobs=int(self.kwargs.get("n_jobs", 1)))
        if self.kind == "extra_trees":
            return ExtraTreesClassifier(n_estimators=int(self.kwargs.get("n_estimators", 300)), max_depth=self.kwargs.get("max_depth", None), min_samples_leaf=int(self.kwargs.get("min_samples_leaf", 1)), random_state=self.seed, n_jobs=int(self.kwargs.get("n_jobs", 1)))
        if self.kind == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=int(self.kwargs.get("n_estimators", 150)), learning_rate=float(self.kwargs.get("learning_rate", 0.05)), max_depth=int(self.kwargs.get("max_depth", 3)), random_state=self.seed)
        if self.kind == "decision_tree":
            return DecisionTreeClassifier(max_depth=self.kwargs.get("max_depth", None), min_samples_leaf=int(self.kwargs.get("min_samples_leaf", 1)), random_state=self.seed)
        if self.kind == "rbf_svm":
            return make_pipeline(StandardScaler(with_mean=False), SVC(C=float(self.kwargs.get("C", 10.0)), gamma=self.kwargs.get("gamma", "scale"), kernel="rbf", probability=True, random_state=self.seed))
        if self.kind == "poly_svm":
            return make_pipeline(StandardScaler(with_mean=False), SVC(C=float(self.kwargs.get("C", 10.0)), gamma=self.kwargs.get("gamma", "scale"), degree=int(self.kwargs.get("degree", 3)), kernel="poly", probability=True, random_state=self.seed))
        if self.kind == "linear_svm":
            return make_pipeline(StandardScaler(with_mean=False), SVC(C=float(self.kwargs.get("C", 10.0)), kernel="linear", probability=True, random_state=self.seed))
        raise ValueError(f"Unsupported sklearn baseline kind: {self.kind}")

    def fit_features(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(X, y)

    def predict_labels_np(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X).astype(int)

    def predict_proba_np(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.empty((0, self.n_classes), dtype=np.float32)
        probs = self.estimator.predict_proba(X)
        out = np.zeros((X.shape[0], self.n_classes), dtype=np.float32)
        classes = getattr(self.estimator, "classes_", None)
        if classes is None and hasattr(self.estimator, "steps"):
            classes = self.estimator.steps[-1][1].classes_
        for j, cls in enumerate(classes):
            out[:, int(cls)] = probs[:, j]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba_np(x.detach().cpu().numpy().astype(np.float32))
        return torch.log(torch.from_numpy(probs).to(x.device).clamp_min(1e-8))

    def parameter_count(self) -> int:
        return 0

    def model_summary(self) -> dict[str, Any]:
        return {"model_type": self.kind, "parameter_count": 0, "sklearn_baseline": True, "direct_label_evaluation": True, "input_dim": self.input_dim}


def build_model(model_config: dict[str, Any], input_dim: int, n_classes: int = 2, feature_columns: list[str] | None = None) -> nn.Module:
    model_type = model_config["type"]
    if model_type == "majority":
        return MajorityClassifier(n_classes=n_classes)
    if model_type == "logistic_regression":
        return TorchLogisticRegression(input_dim=input_dim, n_classes=n_classes)
    if model_type in {"mlp", "sequence_mlp"}:
        return MLPClassifier(input_dim=input_dim, hidden_dim=int(model_config.get("hidden_dim", 64)), n_classes=n_classes, dropout=float(model_config.get("dropout", 0.0)), depth=int(model_config.get("depth", 1 if model_type == "mlp" else 3)))
    if model_type == "quantum_feature":
        return QuantumFeatureCircuitClassifier(input_dim=input_dim, n_classes=n_classes, n_qubits=int(model_config.get("n_qubits", 6)), n_layers=int(model_config.get("n_layers", 2)), entanglement=str(model_config.get("entanglement", "linear")), data_reuploading=bool(model_config.get("data_reuploading", True)))
    if model_type == "grammar_structured_quantum_feature":
        if not feature_columns:
            raise ValueError("grammar_structured_quantum_feature requires feature_columns for role-aware encoding.")
        return GrammarStructuredQuantumCircuitClassifier(input_dim=input_dim, feature_columns=feature_columns, n_classes=n_classes, n_qubits=int(model_config.get("n_qubits", 6)), n_layers=int(model_config.get("n_layers", 2)), structured_entanglement=str(model_config.get("structured_entanglement", "grammar")), data_reuploading=bool(model_config.get("data_reuploading", True)))
    if model_type in {"exact_rule_oracle", "parity_oracle"}:
        if not feature_columns:
            raise ValueError(f"{model_type} requires primitive feature columns.")
        return PrimitiveRuleOracleClassifier(feature_columns=feature_columns, n_classes=n_classes, name=model_type)
    if model_type == "interaction_logistic":
        if not feature_columns:
            raise ValueError("interaction_logistic requires primitive feature columns.")
        return PrimitiveInteractionLogisticClassifier(feature_columns=feature_columns, n_classes=n_classes, seed=int(model_config.get("seed", 0)), C=float(model_config.get("C", 100.0)))
    if model_type in {"random_forest", "extra_trees", "gradient_boosting", "rbf_svm", "poly_svm", "linear_svm", "decision_tree"}:
        return SklearnBaselineClassifier(model_type, input_dim=input_dim, n_classes=n_classes, seed=int(model_config.get("seed", 0)), **{k: v for k, v in model_config.items() if k not in {"name", "type", "seed"}})
    raise ValueError(f"Unsupported model type: {model_type}")
