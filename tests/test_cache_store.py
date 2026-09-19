"""Unit tests for src/cache/store.py, against hand-constructed vectors."""

from dataclasses import dataclass

import numpy as np
import pytest

from cache.store import CacheEntry, SemanticCache, populate_from_records

EX = np.array([1.0, 0.0, 0.0], dtype=np.float32)
EY = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def test_query_returns_exact_nearest_match():
    cache = SemanticCache([
        CacheEntry(key="c1", equivalence_class_id="a", vector=EX),
        CacheEntry(key="c2", equivalence_class_id="b", vector=EY),
    ])
    result = cache.query(EX)
    assert result.entry.equivalence_class_id == "a"
    assert result.similarity == pytest.approx(1.0)


def test_query_picks_closer_of_two_entries():
    # probe closer to EY than EX by cosine similarity
    probe = np.array([0.1, 0.9, 0.0], dtype=np.float32)
    cache = SemanticCache([
        CacheEntry(key="c1", equivalence_class_id="a", vector=EX),
        CacheEntry(key="c2", equivalence_class_id="b", vector=EY),
    ])
    result = cache.query(probe)
    assert result.entry.equivalence_class_id == "b"


def test_empty_cache_raises():
    with pytest.raises(ValueError):
        SemanticCache([])


def test_duplicate_equivalence_class_id_raises():
    with pytest.raises(ValueError, match="duplicate equivalence_class_id"):
        SemanticCache([
            CacheEntry(key="c1", equivalence_class_id="a", vector=EX),
            CacheEntry(key="c2", equivalence_class_id="a", vector=EY),
        ])


def test_len_matches_entry_count():
    cache = SemanticCache([
        CacheEntry(key="c1", equivalence_class_id="a", vector=EX),
        CacheEntry(key="c2", equivalence_class_id="b", vector=EY),
    ])
    assert len(cache) == 2


# --- populate_from_records ------------------------------------------------

@dataclass(frozen=True)
class _FakeRecord:
    """Minimal stand-in for dataset.schema.Record -- only the attributes
    populate_from_records actually reads, so these tests don't depend on
    schema.py's stricter construction/validation rules."""
    record_id: str
    equivalence_class_id: str
    role: str


def test_populate_from_records_filters_by_role_and_dedups_correctly():
    records = [
        _FakeRecord("intent_a__canonical__en__001", "intent_a", "canonical"),
        _FakeRecord("intent_a__paraphrase__en__001", "intent_a", "paraphrase"),
        _FakeRecord("intent_b__canonical__en__001", "intent_b", "canonical"),
    ]
    vectors = {
        "intent_a__canonical__en__001": EX,
        "intent_a__paraphrase__en__001": EX,
        "intent_b__canonical__en__001": EY,
    }
    cache = populate_from_records(records, vectors, role="canonical")
    assert len(cache) == 2
    assert {e.equivalence_class_id for e in cache.entries} == {"intent_a", "intent_b"}


def test_populate_from_records_no_matching_role_raises():
    records = [_FakeRecord("intent_a__paraphrase__en__001", "intent_a", "paraphrase")]
    with pytest.raises(ValueError, match="no records with role"):
        populate_from_records(records, {"intent_a__paraphrase__en__001": EX}, role="canonical")


def test_populate_from_records_missing_vector_raises():
    records = [_FakeRecord("intent_a__canonical__en__001", "intent_a", "canonical")]
    with pytest.raises(KeyError):
        populate_from_records(records, {}, role="canonical")
