from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict
    mode: str = "fit"
    family: str = "unspecified"

class DetectorPipeline:
    """Thin adapter around the vendored TSB-AD detector implementations."""
    def __init__(self, spec: ModelSpec, n_features: int):
        self.spec = spec
        self.model = self._build(n_features)

    def _build(self, n_features: int):
        p = dict(self.spec.params)
        name = self.spec.name.lower()
        if name == "pca":
            from TSB_AD.models.PCA import PCA
            return PCA(**p)
        if name == "robustpca":
            from TSB_AD.models.RobustPCA import RobustPCA
            return RobustPCA(**p)
        if name == "mcd":
            from TSB_AD.models.MCD import MCD
            return MCD(**p)
        if name == "kmeansad":
            from TSB_AD.models.KMeansAD import KMeansAD
            if "n_clusters" in p: p["k"] = p.pop("n_clusters")
            p.setdefault("stride", 1)
            return KMeansAD(**p)
        if name == "cblof":
            from TSB_AD.models.CBLOF import CBLOF
            return CBLOF(**p)
        if name == "ocsvm":
            from TSB_AD.models.OCSVM import OCSVM
            return OCSVM(**p)
        if name == "iforest":
            from TSB_AD.models.IForest import IForest
            return IForest(**p)
        if name == "eif":
            from TSB_AD.models.EIF import EIF
            return EIF(**p)
        if name == "knn":
            from TSB_AD.models.KNN import KNN
            return KNN(**p)
        if name == "lof":
            from TSB_AD.models.LOF import LOF
            return LOF(**p)
        if name == "autoencoder":
            from TSB_AD.models.AE import AutoEncoder
            return AutoEncoder(**p)
        if name == "usad":
            from TSB_AD.models.USAD import USAD
            p.setdefault("feats", n_features)
            return USAD(**p)
        if name == "lstmad":
            from TSB_AD.models.LSTMAD import LSTMAD
            p.setdefault("feats", n_features)
            return LSTMAD(**p)
        if name == "cnn":
            from TSB_AD.models.CNN import CNN
            p.setdefault("feats", n_features)
            return CNN(**p)
        if name == "chronos2":
            from TSB_AD.models.Chronos2 import Chronos2
            return Chronos2(**p)
        raise ValueError(f"Unknown model: {self.spec.name}")

    def fit_reference(self, X: np.ndarray) -> None:
        if self.spec.mode == "zero_shot":
            return
        result = self.model.fit(np.asarray(X, float))
        if result is not None:
            self.model = result

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        name = self.spec.name.lower()
        if name == "kmeansad":
            s = self.model.predict(X)
        elif name == "chronos2" and self.spec.mode == "zero_shot":
            s = self.model.decision_function(X, use_pretrained=True)
        else:
            s = self.model.decision_function(X)
        s = np.asarray(s, float).reshape(-1)
        if len(s) != len(X):
            raise ValueError(f"{self.spec.name}: score length {len(s)} != input length {len(X)}")
        return s
