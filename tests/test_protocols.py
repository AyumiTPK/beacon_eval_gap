import numpy as np
import pytest

from tsad_transfer.protocols import resolve_vus_window
from tsad_transfer.types import Batch


def _batch(batch_id, labels):
    labels = np.asarray(labels, dtype=int)
    return Batch(batch_id=batch_id, X=np.zeros((len(labels), 1)), y=labels)


def test_vus_window_resolves_to_median_event_duration():
    evaluation = [
        _batch("a", [0, 1, 1, 0, 1, 1, 1, 1]),
        _batch("b", [1, 1, 1, 0, 0, 1, 1, 1, 1, 1]),
    ]
    assert resolve_vus_window(evaluation) == 3


def test_vus_window_respects_explicit_configuration():
    assert resolve_vus_window([], configured_window=100) == 100


def test_vus_window_requires_anomaly_events_when_automatic():
    with pytest.raises(ValueError, match="no anomaly events"):
        resolve_vus_window([_batch("normal", [0, 0, 0])])
