"""Unit tests for the pure helpers in experiments/embeddinggemma2_dims.py (no model, no downloads)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from embeddinggemma2_dims import spearman_rows, truncate  # noqa: E402


def test_truncate_keeps_prefix_and_renormalizes():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5, 768)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    t = truncate(v, 128)
    assert t.shape == (5, 128)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-6)
    # direction of the kept prefix is unchanged
    ref = v[:, :128] / np.linalg.norm(v[:, :128], axis=1, keepdims=True)
    assert np.allclose(t, ref, atol=1e-6)


def test_truncate_single_vector():
    v = np.ones(768, dtype=np.float32) / np.sqrt(768)
    assert abs(np.linalg.norm(truncate(v, 256)) - 1.0) < 1e-6


def test_spearman_rows_identity_reverse_and_monotone():
    a = np.array([[0.1, 0.5, 0.3, 0.9, 0.7]])
    assert np.allclose(spearman_rows(a, a), 1.0)
    assert np.allclose(spearman_rows(a, -a), -1.0)
    assert np.allclose(spearman_rows(a, a * 2 + 5), 1.0)  # a level shift and rescale keep the order
