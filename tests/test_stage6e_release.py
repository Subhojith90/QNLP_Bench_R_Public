from pathlib import Path
import importlib.util
import numpy as np


def load_script():
    p = Path(__file__).resolve().parents[1] / 'scripts' / 'run_stage6e_learned_kernel_challenge.py'
    spec = importlib.util.spec_from_file_location('stage6e_script', p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_stage6e_config_and_pipeline_files_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in ['configs/stage6e_learned_kernel.yaml', 'scripts/run_stage6e_learned_kernel_challenge.py', 'scripts/run_stage6e_pipeline.sh', 'scripts/package_stage6e_artifacts.py', 'scripts/validate_stage6e_package.py']:
        assert (root / rel).exists()


def test_kernel_functions_are_finite_and_symmetric():
    mod = load_script()
    X = np.asarray([[1., 0.], [0., 1.], [1., 1.]])
    K = mod.kernel_matrix(X, X, 'rbf', {'gamma': 1.0})
    assert K.shape == (3, 3)
    assert np.isfinite(K).all()
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)


def test_tree_proximity_kernel_shape():
    mod = load_script()
    from sklearn.ensemble import ExtraTreesClassifier
    X = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.asarray([0, 1, 1, 0])
    est = ExtraTreesClassifier(n_estimators=5, random_state=1).fit(X, y)
    K = mod.tree_proximity(est, X, X)
    assert K.shape == (4, 4)
    assert np.allclose(np.diag(K), 1.0)
