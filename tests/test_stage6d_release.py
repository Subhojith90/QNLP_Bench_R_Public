from pathlib import Path
import numpy as np
import torch
import yaml
from qnlpbench_r.config import load_config
from qnlpbench_r.data.datasets import load_dataset
from qnlpbench_r.data.preprocessing import fit_preprocessor, make_arrays
from qnlpbench_r.evaluation.evaluate import predict_labels, predict_proba
from qnlpbench_r.models import build_model


def test_stage6d_selected_regime_and_output_layout():
    cfg = load_config('configs/stage6d_replication_base.yaml')
    assert cfg['dataset']['latent_dim'] == 4
    assert cfg['dataset']['min_abs_margin'] == 0.10
    assert cfg['dataset']['feature_set'] == 'primitive_slot_sequence'
    for p in ['configs/stage6d_classical.yaml', 'configs/stage6d_topologies.yaml', 'configs/stage6d_kernel.yaml']:
        text = Path(p).read_text(encoding='utf-8')
        assert 'paper_assets' not in text
        assert 'results/stage6d' in text or 'figures' in text


def test_stage6d_split_integrity_is_preserved():
    cfg = load_config('configs/stage6d_replication_base.yaml')
    bundle = load_dataset(cfg, seed=11)
    audit = bundle.manifest['split_audit']
    assert audit['heldout_compositions_in_train'] == 0
    assert audit['heldout_compositions_in_val'] == 0
    assert audit['heldout_compositions_in_test'] == 0
    assert audit['heldout_compositions_in_ood_composition'] > 0
    assert max(audit['exact_text_duplicate_overlap'].values()) == 0


def test_corrected_rbf_svm_uses_authoritative_class_predictions():
    cfg = load_config('configs/stage6d_replication_base.yaml')
    bundle = load_dataset(cfg, seed=11)
    prep = fit_preprocessor(bundle); arrays = make_arrays(bundle, prep)
    model = build_model({'name':'svm','type':'rbf_svm','C':10.0,'gamma':'scale','seed':11}, len(bundle.feature_columns), 2, bundle.feature_columns)
    model.fit_features(*arrays['train'])
    X, _ = arrays['test']
    assert np.array_equal(model.predict_labels_np(X), predict_labels(model, X, torch.device('cpu')))
    assert predict_proba(model, X, torch.device('cpu')).shape == (len(X), 2)


def test_selected_topologies_build_on_primitive_inputs():
    cfg = load_config('configs/stage6d_replication_base.yaml')
    bundle = load_dataset(cfg, seed=11)
    patterns = ['random_fixed_01', 'ring', 'all_to_all', 'generic_linear', 'grammar', 'none']
    for pattern in patterns:
        model = build_model({'name': pattern, 'type': 'grammar_structured_quantum_feature', 'n_qubits': 6, 'n_layers': 2, 'structured_entanglement': pattern}, len(bundle.feature_columns), 2, bundle.feature_columns)
        assert model.model_summary()['primitive_role_encoding_active'] is True
