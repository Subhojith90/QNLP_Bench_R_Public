import numpy as np
from qnlpbench_r.evaluation.metrics import classification_metrics, expected_calibration_error, kernel_target_alignment


def test_classification_metrics_perfect():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    m = classification_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["brier"] < 0.1


def test_calibration_bounds():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    ece = expected_calibration_error(y, p)
    assert 0.0 <= ece <= 1.0


def test_kernel_alignment_known():
    y = np.array([0, 0, 1, 1])
    signed = 2 * y - 1
    K = np.outer(signed, signed)
    assert abs(kernel_target_alignment(K, y) - 1.0) < 1e-8

from qnlpbench_r.evaluation.evaluate import select_kernel_sample_indices


def test_kernel_sample_indices_are_stratified_and_reproducible():
    y = np.array([0] * 80 + [1] * 20)
    idx1 = select_kernel_sample_indices(y, sample_cap=20, seed=7, strategy="stratified")
    idx2 = select_kernel_sample_indices(y, sample_cap=20, seed=7, strategy="stratified")
    assert np.array_equal(idx1, idx2)
    assert len(idx1) == 20
    assert set(np.unique(y[idx1])) == {0, 1}
    assert np.sum(y[idx1] == 1) > 1


def test_kernel_sample_indices_do_not_default_to_first_examples():
    y = np.array([0] * 50 + [1] * 50)
    idx = select_kernel_sample_indices(y, sample_cap=20, seed=13, strategy="stratified")
    assert not np.array_equal(idx, np.arange(20))
    assert np.any(idx >= 50)
