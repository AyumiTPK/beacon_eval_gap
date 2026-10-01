import numpy as np
from tsad_transfer.protocols import _threshold


def test_threshold_is_training_quantile():
    x = np.arange(100, dtype=float)
    assert _threshold(x, .95) == np.quantile(x, .95)
