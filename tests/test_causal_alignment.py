import numpy as np
import pytest

pytest.importorskip("hurst")
from TSB_AD.models.feature import align_trailing_scores


def test_trailing_alignment_maps_window_scores_to_window_end():
    scores = np.array([1.0, 2.0, 3.0])
    aligned = align_trailing_scores(scores, n_samples=6, window_size=4)
    np.testing.assert_allclose(aligned, [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0], equal_nan=True)
