"""Batch encoding with on-disk vector caching.

Responsibility: encode a list of texts with a given model, caching results to
disk keyed by (model_id, text_hash) so re-runs are cheap and reproducible.
Must not silently re-encode when the cache is warm. Built in Phase 4.

Cache layout: results/embeddings/<model_key>/<sha256(text)[:16]>.npy -- one
file per (model, text). Individual files (rather than one big per-model
archive) mean a crash mid-run can never corrupt vectors already cached, and a
partial run resumes for free: encode_texts() only calls the model for texts
whose cache file is missing. Writes go through a temp file + os.replace so a
crash mid-write can't leave a half-written .npy behind either.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from config import Config


def text_hash(text: str) -> str:
    """Stable content hash for a text, used as the cache filename stem."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, model_key: str, text: str) -> Path:
    return cache_dir / model_key / f"{text_hash(text)}.npy"


def _atomic_save(path: Path, vector: np.ndarray) -> None:
    tmp_path = path.with_suffix(".npy.tmp")
    # np.save silently appends ".npy" to a bare path/string argument that
    # doesn't already end in ".npy" -- tmp_path ends in ".tmp", so passing it
    # directly would write to "<...>.npy.tmp.npy" and the rename below would
    # find nothing there. Passing an open file object bypasses that
    # extension-guessing entirely: numpy writes exactly to what we opened.
    with open(tmp_path, "wb") as f:
        np.save(f, vector)
    os.replace(tmp_path, path)


def encode_texts(
    model_key: str,
    texts: list[str],
    config: Config | None = None,
    encoder=None,
) -> np.ndarray:
    """Encode `texts` with the model at `model_key`, caching each vector to
    disk keyed by (model_key, sha256(text)).

    Cache hits are read straight from disk. The underlying model is loaded
    (via embed.registry.get_encoder) only if there is at least one cache
    miss, and is called at most once per *unique* text even if `texts`
    repeats one -- so a fully warm cache costs zero model calls and zero
    model loads.

    `encoder` is a dependency-injection point for tests: anything exposing
    .encode(list[str]) -> array-like works. Production callers should leave
    it None.

    Returns an (len(texts), dim) float32 array, in the same order as `texts`.
    """
    config = config or Config()
    cache_dir = config.embed_cache_dir
    (cache_dir / model_key).mkdir(parents=True, exist_ok=True)

    n = len(texts)
    results: list[np.ndarray | None] = [None] * n
    dim: int | None = None
    miss_positions: dict[str, list[int]] = {}

    for i, text in enumerate(texts):
        path = _cache_path(cache_dir, model_key, text)
        if path.exists():
            vec = np.load(path)
            results[i] = vec
            dim = dim or vec.shape[0]
        else:
            miss_positions.setdefault(text, []).append(i)

    if miss_positions:
        if encoder is None:
            from embed import registry

            encoder = registry.get_encoder(model_key, config)
        unique_misses = list(miss_positions)
        vectors = np.asarray(encoder.encode(unique_misses), dtype=np.float32)
        if vectors.shape[0] != len(unique_misses):
            raise ValueError(
                f"encoder returned {vectors.shape[0]} vectors for "
                f"{len(unique_misses)} texts"
            )
        for text, vec in zip(unique_misses, vectors):
            path = _cache_path(cache_dir, model_key, text)
            _atomic_save(path, vec)
            for i in miss_positions[text]:
                results[i] = vec
        dim = dim or (vectors.shape[1] if vectors.ndim == 2 else None)

    if n == 0:
        return np.zeros((0, dim or 0), dtype=np.float32)
    return np.stack(results).astype(np.float32)
