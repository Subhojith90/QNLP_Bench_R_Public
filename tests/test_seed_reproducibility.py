import numpy as np
import torch
from qnlpbench_r.seed import set_global_seed


def test_seed_reproducibility_numpy_torch():
    set_global_seed(123)
    a = np.random.rand(5)
    t = torch.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    u = torch.rand(5)
    assert np.allclose(a, b)
    assert torch.allclose(t, u)
