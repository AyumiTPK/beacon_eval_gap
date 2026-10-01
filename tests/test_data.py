import numpy as np
from tsad_transfer.types import Batch
from tsad_transfer.data import select_reference

def test_reference_is_removed_from_evaluation():
    a=Batch("clean", np.zeros((4,2)), np.zeros(4,dtype=int)); b=Batch("fault", np.zeros((4,2)), np.ones(4,dtype=int))
    ref, ev=select_reference([a,b], "clean")
    assert ref.batch_id == "clean" and [x.batch_id for x in ev] == ["fault"]
