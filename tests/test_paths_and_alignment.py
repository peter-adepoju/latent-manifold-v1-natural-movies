from pathlib import Path
import numpy as np

from v1_manifold.utils import save_json, ensure_dir
from v1_manifold.dnn_alignment import linear_cka


def test_save_json_creates_parent(tmp_path):
    path = save_json({"a": 1}, tmp_path / "nested" / "file.json")
    assert Path(path).exists()


def test_linear_cka_identical_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 10))
    score = linear_cka(X, X)
    assert score > 0.99
