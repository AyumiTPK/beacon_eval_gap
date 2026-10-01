from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class Batch:
    batch_id: str
    X: np.ndarray
    y: np.ndarray
    time: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    @property
    def is_labeled_normal(self) -> bool:
        return not np.any(self.y == 1)
