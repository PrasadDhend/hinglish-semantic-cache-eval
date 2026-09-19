"""Embedding-space geometry diagnostics.

Responsibility: centroid distance between conditions, cosine gap
distributions, and PCA projections used to visualise how romanization and
code-mixing move query embeddings relative to their English anchors. Feeds
exp03 (attribution) and the results/discussion figures. Built in Phase 5.

All functions here assume the vectors they're handed are L2-normalized (as
embed/registry.py's Encoder guarantees) EXCEPT where a function explicitly
re-normalizes something derived (a centroid is a mean of unit vectors, which
is generally not itself unit-norm).
"""

from __future__ import annotations

import numpy as np


def centroid(vectors) -> np.ndarray:
    """Mean vector of `vectors`. NOT re-normalized -- a centroid of unit
    vectors is generally shorter than 1 (its length reflects how spread out
    the set is), and that length matters, so callers who need cosine
    similarity to a centroid should let centroid_distance() handle the
    renormalization rather than doing it here."""
    return np.asarray(vectors, dtype=np.float64).mean(axis=0)


def centroid_distance(vectors_a, vectors_b, metric: str = "cosine") -> float:
    """Distance between the centroids of two sets of embeddings.

    metric="cosine" (default): 1 - cosine_similarity(centroid_a, centroid_b).
        Centroids are re-normalized before comparing, since a raw centroid
        of unit vectors is not itself unit-norm.
    metric="euclidean": plain L2 distance between the raw centroids (their
        shrinkage is part of the signal here, so no renormalization).
    """
    ca = centroid(vectors_a)
    cb = centroid(vectors_b)
    if metric == "cosine":
        na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
        if na == 0 or nb == 0:
            return float("nan")
        return float(1.0 - (ca @ cb) / (na * nb))
    if metric == "euclidean":
        return float(np.linalg.norm(ca - cb))
    raise ValueError(f"unknown metric {metric!r}, expected 'cosine' or 'euclidean'")


def cosine_gap(vectors_a, vectors_b) -> np.ndarray:
    """Per-pair cosine gap (1 - cosine similarity) between matched rows of
    `vectors_a` and `vectors_b` -- e.g. the same paraphrase's English vs
    Hinglish embedding, row i in both being the same underlying probe.
    Vectors assumed L2-normalized already, so this is a plain per-row dot
    product. Returns one gap per matched pair, same order as the inputs.
    """
    a = np.asarray(vectors_a, dtype=np.float64)
    b = np.asarray(vectors_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("vectors_a and vectors_b must have matching shape (paired rows)")
    sims = np.sum(a * b, axis=1)
    return 1.0 - sims


def pca_project(vectors, n_components: int = 2):
    """Project `vectors` onto their top `n_components` principal components
    via SVD (no sklearn dependency). Returns (projected, explained_variance_ratio):
      projected                -- (n_samples, n_components) array
      explained_variance_ratio -- list of length n_components
    Deterministic (SVD has no randomness), unlike the bootstrap tools in
    eval.stats.
    """
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("vectors must be a 2D (n_samples, dim) array")
    n_components = min(n_components, X.shape[0], X.shape[1])
    Xc = X - X.mean(axis=0)
    # economy SVD: enough for any n_components <= min(n_samples, dim)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    projected = Xc @ components.T
    total_var = float(np.sum(S ** 2))
    explained = (S[:n_components] ** 2) / total_var if total_var > 0 else np.zeros(n_components)
    return projected, explained.tolist()
