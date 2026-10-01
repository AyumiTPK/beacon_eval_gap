# Adapted from the TSB-AD implementation for compatibility with the evaluation protocol used in this study.
"""
This function is adapted from [TimeEval-algorithms] by [CodeLionX&wenig]
Original source: [https://github.com/TimeEval/TimeEval-algorithms]
"""

from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import KMeans
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from ..utils.utility import zscore

class KMeansAD(BaseEstimator, OutlierMixin):
    def __init__(self, k, window_size, stride, n_jobs=1, normalize=True):
        self.k = k
        self.window_size = window_size
        self.stride = stride
        self.model = KMeans(n_clusters=k)
        self.padding_length = 0
        self.normalize = normalize

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        flat_shape = (X.shape[0] - (self.window_size - 1), -1)  # in case we have a multivariate TS
        slides = sliding_window_view(X, window_shape=self.window_size, axis=0).reshape(flat_shape)[::self.stride, :]
        self.padding_length = X.shape[0] - (slides.shape[0] * self.stride + self.window_size - self.stride)
        print(f"Required padding_length={self.padding_length}")
        if self.normalize: slides = zscore(slides, axis=1, ddof=1)
        return slides

    def _custom_reverse_windowing(self, scores: np.ndarray) -> np.ndarray:
        unwindowed_length = self.stride * (scores.shape[0] - 1) + self.window_size + self.padding_length
        mapped = np.full(unwindowed_length, np.nan)
        indices = np.arange(len(scores)) * self.stride + self.window_size - 1
        valid = indices < unwindowed_length
        mapped[indices[valid]] = scores[valid]
        return mapped

    def fit(self, X: np.ndarray, y=None, preprocess=True) -> 'KMeansAD':
        if preprocess:
            X = self._preprocess_data(X)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray, preprocess=True) -> np.ndarray:
        if preprocess:
            X = self._preprocess_data(X)
        clusters = self.model.predict(X)
        diffs = np.linalg.norm(X - self.model.cluster_centers_[clusters], axis=1)
        return self._custom_reverse_windowing(diffs)

    def fit_predict(self, X, y=None) -> np.ndarray:
        X = self._preprocess_data(X)
        self.fit(X, y, preprocess=False)
        return self.predict(X, preprocess=False)