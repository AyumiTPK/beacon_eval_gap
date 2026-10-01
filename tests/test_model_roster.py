from tsad_transfer.config import load_config

EXPECTED = {
    "chronos2", "autoencoder", "usad", "lstmad", "cnn",
    "pca", "robustpca", "iforest", "eif", "mcd", "ocsvm",
    "kmeansad", "cblof", "knn", "lof",
}

def test_publication_roster_has_15_models():
    _, specs = load_config("configs/study.yaml")
    assert len(specs) == 15
    assert {s.name.lower() for s in specs} == EXPECTED
    assert all(s.family != "unspecified" for s in specs)
