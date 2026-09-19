"""Unit tests for src/eval/geometry.py against hand-constructed vectors."""

import numpy as np
import pytest

from eval.geometry import centroid, centroid_distance, cosine_gap, pca_project

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
NEG_EX = np.array([-1.0, 0.0, 0.0])


def test_centroid_is_mean():
    result = centroid([EX, EY])
    np.testing.assert_allclose(result, [0.5, 0.5, 0.0])


def test_centroid_distance_identical_sets_is_zero():
    vectors = [EX, EY, EX]
    assert centroid_distance(vectors, vectors, metric="cosine") == pytest.approx(0.0, abs=1e-9)


def test_centroid_distance_cosine_orthogonal_centroids():
    # centroid of [EX] is EX; centroid of [EY] is EY; cosine dist = 1 - 0 = 1
    assert centroid_distance([EX], [EY], metric="cosine") == pytest.approx(1.0)


def test_centroid_distance_euclidean_hand_value():
    assert centroid_distance([EX], [NEG_EX], metric="euclidean") == pytest.approx(2.0)


def test_centroid_distance_unknown_metric_raises():
    with pytest.raises(ValueError):
        centroid_distance([EX], [EY], metric="manhattan")


def test_cosine_gap_identical_vectors_is_zero():
    gaps = cosine_gap([EX, EY], [EX, EY])
    np.testing.assert_allclose(gaps, [0.0, 0.0], atol=1e-9)


def test_cosine_gap_orthogonal_vectors_is_one():
    gaps = cosine_gap([EX], [EY])
    np.testing.assert_allclose(gaps, [1.0])


def test_cosine_gap_opposite_vectors_is_two():
    gaps = cosine_gap([EX], [NEG_EX])
    np.testing.assert_allclose(gaps, [2.0])


def test_cosine_gap_mismatched_shape_raises():
    with pytest.raises(ValueError):
        cosine_gap([EX, EY], [EX])


def test_pca_project_recovers_dominant_axis():
    # points scattered along the x-axis with tiny y-noise -- PC1 should
    # explain almost all the variance
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(0, 10, n)
    y = rng.normal(0, 0.01, n)
    points = np.stack([x, y], axis=1)
    projected, explained = pca_project(points, n_components=2)
    assert projected.shape == (n, 2)
    assert explained[0] > 0.99
    assert sum(explained) == pytest.approx(1.0, abs=1e-6)
