from __future__ import annotations
import numpy as np
from qnlpbench_r.config import load_config
from qnlpbench_r.data.datasets import load_dataset
from qnlpbench_r.data.latent import generate_latent_compositional_dataset
from qnlpbench_r.models import build_model


def test_latent_dataset_exposes_primitive_inputs_only_and_strict_splits():
    cfg = load_config('configs/stage6d_replication_base.yaml')
    bundle = load_dataset(cfg, seed=11)
    visible = set(bundle.feature_columns)
    assert visible
    assert all(c.startswith(('ps_', 'pv_', 'po_', 'pn_', 'pd_')) for c in visible)
    assert not {'audit_latent_score','audit_latent_margin','clean_label'}.intersection(visible)
    audit = bundle.manifest['split_audit']
    assert audit['heldout_compositions_in_train'] == 0
    assert audit['heldout_compositions_in_val'] == 0
    assert audit['heldout_compositions_in_test'] == 0
    assert audit['heldout_compositions_in_ood_composition'] > 0
    assert audit['ood_composition_pair_overlap_with_train'] == 0
    assert audit['ood_composition_pair_overlap_with_val'] == 0
    assert audit['ood_composition_pair_overlap_with_test'] == 0
    assert max(audit['exact_text_duplicate_overlap'].values()) == 0


def test_latent_generator_is_deterministic_for_fixed_seeds():
    df1, report1 = generate_latent_compositional_dataset(240, seed=17, semantic_seed=2026, latent_dim=5, min_abs_margin=0.05)
    df2, report2 = generate_latent_compositional_dataset(240, seed=17, semantic_seed=2026, latent_dim=5, min_abs_margin=0.05)
    assert report1.content_hash == report2.content_hash
    assert df1.equals(df2)
    assert 'audit_latent_score' in df1.columns


def test_grammar_qfc_constructs_on_latent_primitive_inputs():
    cfg = load_config('configs/stage6d_replication_base.yaml')
    bundle = load_dataset(cfg, seed=11)
    model = build_model({'name':'qfc','type':'grammar_structured_quantum_feature','n_qubits':6,'n_layers':2,'structured_entanglement':'grammar'}, len(bundle.feature_columns), 2, bundle.feature_columns)
    summary = model.model_summary()
    assert summary['primitive_role_encoding_active'] is True
    assert len(summary['grammar_edges']) > 0


def test_stage6_poly_svm_is_available():
    cfg = load_config('configs/stage6d_replication_base.yaml')
    bundle = load_dataset(cfg, seed=11)
    model = build_model({'name':'poly','type':'poly_svm','degree':3,'C':1.0,'seed':11}, len(bundle.feature_columns), 2, bundle.feature_columns)
    assert model.model_summary()['model_type'] == 'poly_svm'
