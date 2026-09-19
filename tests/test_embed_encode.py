"""Unit tests for src/embed/encode.py's on-disk caching, using an injected
fake encoder so these never load a real model or touch the network -- that's
what scripts/phase4_sanity_check.py (real model, real data) is for.
"""

import numpy as np
import pytest

from config import Config
from embed.encode import encode_texts, text_hash


class _FakeEncoder:
    """Deterministic stand-in for embed.registry.Encoder: each text maps to
    a fixed-but-distinct vector, and every call is recorded so tests can
    assert the cache actually prevented re-encoding."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        # deterministic per-text vector: seed a tiny RNG from the text hash
        # so identical text -> identical vector, distinct text -> distinct.
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(int(text_hash(t), 16) % (2**32))
            vecs[i] = rng.random(self.dim)
        return vecs

    @property
    def n_texts_encoded(self) -> int:
        return sum(len(c) for c in self.calls)


def _cfg(tmp_path):
    return Config(embed_cache_dir=tmp_path / "embeddings")


def test_cold_cache_calls_encoder_and_writes_files(tmp_path):
    cfg = _cfg(tmp_path)
    encoder = _FakeEncoder()
    texts = ["hello", "world"]
    vecs = encode_texts("model_x", texts, config=cfg, encoder=encoder)

    assert vecs.shape == (2, 4)
    assert encoder.n_texts_encoded == 2
    for t in texts:
        cached_file = cfg.embed_cache_dir / "model_x" / f"{text_hash(t)}.npy"
        assert cached_file.exists()


def test_warm_cache_never_calls_encoder(tmp_path):
    cfg = _cfg(tmp_path)
    encoder1 = _FakeEncoder()
    texts = ["hello", "world"]
    first = encode_texts("model_x", texts, config=cfg, encoder=encoder1)

    encoder2 = _FakeEncoder()  # a fresh encoder that would raise if it were ever called incorrectly
    second = encode_texts("model_x", texts, config=cfg, encoder=encoder2)

    assert encoder2.n_texts_encoded == 0  # cache was fully warm -- no calls at all
    np.testing.assert_array_equal(first, second)


def test_partial_cache_only_encodes_the_miss(tmp_path):
    cfg = _cfg(tmp_path)
    encoder1 = _FakeEncoder()
    encode_texts("model_x", ["hello"], config=cfg, encoder=encoder1)

    encoder2 = _FakeEncoder()
    encode_texts("model_x", ["hello", "new text"], config=cfg, encoder=encoder2)

    assert encoder2.calls == [["new text"]]


def test_duplicate_text_in_one_call_encoded_once(tmp_path):
    cfg = _cfg(tmp_path)
    encoder = _FakeEncoder()
    vecs = encode_texts("model_x", ["same", "same", "same"], config=cfg, encoder=encoder)

    assert encoder.n_texts_encoded == 1  # only one unique text went through the model
    np.testing.assert_array_equal(vecs[0], vecs[1])
    np.testing.assert_array_equal(vecs[1], vecs[2])


def test_different_model_keys_do_not_share_cache(tmp_path):
    cfg = _cfg(tmp_path)
    v1 = encode_texts("model_a", ["hello"], config=cfg, encoder=_FakeEncoder())
    v2 = encode_texts("model_b", ["hello"], config=cfg, encoder=_FakeEncoder())
    # different fake encoders -> different RNG streams per model dir, but the
    # real guarantee under test is that model_b's cache miss still happened
    # (i.e. model_a's cache file didn't satisfy model_b's lookup)
    assert (cfg.embed_cache_dir / "model_a" / f"{text_hash('hello')}.npy").exists()
    assert (cfg.embed_cache_dir / "model_b" / f"{text_hash('hello')}.npy").exists()


def test_empty_input_returns_empty_array(tmp_path):
    cfg = _cfg(tmp_path)
    vecs = encode_texts("model_x", [], config=cfg, encoder=_FakeEncoder())
    assert vecs.shape[0] == 0


def test_output_order_matches_input_order_even_with_mixed_hits_and_misses(tmp_path):
    cfg = _cfg(tmp_path)
    encoder = _FakeEncoder()
    encode_texts("model_x", ["a", "b"], config=cfg, encoder=encoder)  # warm a, b

    combined = encode_texts("model_x", ["b", "c", "a"], config=cfg, encoder=_FakeEncoder())
    solo_b = encode_texts("model_x", ["b"], config=cfg, encoder=_FakeEncoder())
    solo_c = encode_texts("model_x", ["c"], config=cfg, encoder=_FakeEncoder())
    solo_a = encode_texts("model_x", ["a"], config=cfg, encoder=_FakeEncoder())

    np.testing.assert_array_equal(combined[0], solo_b[0])
    np.testing.assert_array_equal(combined[1], solo_c[0])
    np.testing.assert_array_equal(combined[2], solo_a[0])
