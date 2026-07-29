from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_readme_freezes_negative_result_and_no_scaleup():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert 'no new model training' in text.lower()
    assert 'does not survive' in text.lower()
    assert 'quantum advantage' in text.lower()


def test_safe_extraction_rejects_path_traversal(tmp_path):
    mod=load('safe_zip_utils.py')
    malicious=tmp_path/'malicious.zip'
    with zipfile.ZipFile(malicious, 'w') as zf:
        zf.writestr('../../escape.txt', 'no')
    with pytest.raises(mod.UnsafeZipError):
        mod.safe_extract_zip(malicious, tmp_path/'out')
    assert not (tmp_path.parent/'escape.txt').exists()


def test_safe_extraction_accepts_clean_zip(tmp_path):
    mod=load('safe_zip_utils.py')
    clean=tmp_path/'clean.zip'
    with zipfile.ZipFile(clean, 'w') as zf:
        zf.writestr('folder/file.txt', 'ok')
    mod.safe_extract_zip(clean, tmp_path/'out')
    assert (tmp_path/'out/folder/file.txt').read_text() == 'ok'


def test_claim_matrix_generation(tmp_path):
    mod=load('generate_stage6g_claim_matrix.py')
    old=sys.argv; sys.argv=['x', '--output-dir', str(tmp_path)]
    try:
        mod.main()
    finally:
        sys.argv=old
    df=pd.read_csv(tmp_path/'final_claim_matrix.csv')
    assert df.loc[df.claim == 'Quantum advantage.', 'status'].iloc[0] == 'forbidden'


def test_stage6e_recompute_helper_preserves_exact_mean(tmp_path):
    mod=load('verify_stage6e_from_package.py')
    seed=pd.DataFrame([
        {'model_name':'m', 'split':'test', 'seed':1, 'qfc_balanced_accuracy':.6, 'classical_balanced_accuracy':.7, 'untrained_balanced_accuracy':.5, 'delta_qfc_minus_learned_classical':-.1, 'delta_trained_minus_untrained':.1},
        {'model_name':'m', 'split':'test', 'seed':2, 'qfc_balanced_accuracy':.8, 'classical_balanced_accuracy':.7, 'untrained_balanced_accuracy':.5, 'delta_qfc_minus_learned_classical':.1, 'delta_trained_minus_untrained':.3},
    ])
    submitted=pd.DataFrame([{'model_name':'m', 'split':'test', 'n_seeds':2, 'qfc_balanced_accuracy_mean':.7, 'learned_classical_balanced_accuracy_mean':.7, 'untrained_qfc_balanced_accuracy_mean':.5, 'delta_qfc_minus_learned_classical_mean':0.0, 'positive_seed_count':1, 'delta_trained_minus_untrained_mean':.2}])
    seed.to_csv(tmp_path/'seed.csv', index=False); submitted.to_csv(tmp_path/'summary.csv', index=False)
    _, _, error=mod.reproduce_summary(tmp_path/'seed.csv', tmp_path/'summary.csv')
    assert error < 1e-12


def test_internal_manifest_maps_source_and_evidence(tmp_path, monkeypatch):
    mod=load('build_internal_artifact_manifest.py')
    evidence=tmp_path/'evidence'; evidence.mkdir()
    (evidence/'QNLPBench_R_Stage6D_Replication_Output.zip').write_bytes(b'zip')
    source=tmp_path/'qnlpbench_r_code_refined_v19_stage6g_public_reproducibility_hardening.zip'; source.write_bytes(b'source')
    out=tmp_path/'results/stage6g/manifests'
    monkeypatch.chdir(tmp_path)
    old=sys.argv; sys.argv=['x', '--evidence-dir', str(evidence), '--source-zip', str(source), '--output-dir', str(out)]
    try:
        mod.main()
    finally:
        sys.argv=old
    manifest=json.loads((out/'internal_artifact_manifest.json').read_text())
    archive_paths={row['archive_path'] for row in manifest['files']}
    assert 'Output/QNLPBench_R_Stage6D_Replication_Output.zip' in archive_paths
    assert 'Source/qnlpbench_r_code_refined_v19_stage6g_public_reproducibility_hardening.zip' in archive_paths



def test_clean_source_check_rejects_cache_entries(tmp_path):
    mod=load('check_clean_source_release.py')
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad, 'w') as zf:
        zf.writestr('.pytest_cache/README.md', 'cache')
    old=sys.argv; sys.argv=['x', '--zip', str(bad)]
    try:
        with pytest.raises(SystemExit):
            mod.main()
    finally:
        sys.argv=old
