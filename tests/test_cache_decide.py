"""Unit tests for src/cache/decide.py -- the Phase 4 gate requires these to
exist and pass *before* running the decision logic on any real data.

All vectors here are hand-constructed, not model output: unit basis vectors
in a toy 3D space, chosen so cosine similarity works out to a clean 1.0, 0.0,
or -1.0 by construction, and threshold boundaries can be tested exactly.
"""

import numpy as np
import pytest

from cache.decide import Decision, classify, evaluate_probe, is_hit
from cache.store import CacheEntry, SemanticCache

EX = np.array([1.0, 0.0, 0.0], dtype=np.float32)
EY = np.array([0.0, 1.0, 0.0], dtype=np.float32)
NEG_EX = np.array([-1.0, 0.0, 0.0], dtype=np.float32)


# --- is_hit ------------------------------------------------------------

def test_is_hit_above_threshold():
    assert is_hit(0.9, 0.8) is True


def test_is_hit_below_threshold():
    assert is_hit(0.7, 0.8) is False


def test_is_hit_boundary_is_inclusive():
    assert is_hit(0.8, 0.8) is True


# --- classify: the four decision quadrants ------------------------------

def test_classify_true_hit():
    # hit, and the nearest entry really is the probe's own class
    d = classify(similarity=0.95, threshold=0.8,
                 nearest_equivalence_class_id="account.reset_password",
                 true_equivalence_class_id="account.reset_password")
    assert d == Decision.TRUE_HIT


def test_classify_false_hit():
    # hit, but the nearest entry is the WRONG class -- the costly failure
    # mode: cache serves a wrong answer with high confidence
    d = classify(similarity=0.95, threshold=0.8,
                 nearest_equivalence_class_id="billing.refund_status",
                 true_equivalence_class_id="account.reset_password")
    assert d == Decision.FALSE_HIT


def test_classify_true_miss():
    # miss, and correctly so -- nearest entry is genuinely a different class
    d = classify(similarity=0.5, threshold=0.8,
                 nearest_equivalence_class_id="billing.refund_status",
                 true_equivalence_class_id="account.reset_password")
    assert d == Decision.TRUE_MISS


def test_classify_false_miss():
    # miss, but the nearest entry WAS the right class -- a correct reuse
    # existed and got missed on similarity alone (threshold/drift cost)
    d = classify(similarity=0.5, threshold=0.8,
                 nearest_equivalence_class_id="account.reset_password",
                 true_equivalence_class_id="account.reset_password")
    assert d == Decision.FALSE_MISS


@pytest.mark.parametrize("threshold,expected", [
    (0.5, Decision.TRUE_HIT),   # boundary itself counts as a hit
    (0.5001, Decision.FALSE_MISS),
])
def test_classify_boundary_exact_threshold(threshold, expected):
    d = classify(similarity=0.5, threshold=threshold,
                 nearest_equivalence_class_id="account.reset_password",
                 true_equivalence_class_id="account.reset_password")
    assert d == expected


# --- evaluate_probe: full lookup + classify pipeline --------------------

def _toy_cache():
    return SemanticCache([
        CacheEntry(key="c1", equivalence_class_id="account.reset_password", vector=EX),
        CacheEntry(key="c2", equivalence_class_id="billing.refund_status", vector=EY),
    ])


def test_evaluate_probe_identical_vector_is_true_hit():
    cache = _toy_cache()
    result = evaluate_probe(cache, EX, true_equivalence_class_id="account.reset_password", threshold=0.8)
    assert result.decision == Decision.TRUE_HIT
    assert result.similarity == pytest.approx(1.0)
    assert result.nearest_equivalence_class_id == "account.reset_password"


def test_evaluate_probe_orthogonal_vector_is_true_miss():
    # probe is equidistant-orthogonal to both cache entries at sim=0 --
    # nearest neighbour still resolves to one of them (numpy argmax picks the
    # first on ties), but 0.0 similarity is well below any sane threshold
    orthogonal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cache = _toy_cache()
    result = evaluate_probe(cache, orthogonal, true_equivalence_class_id="billing.refund_status", threshold=0.5)
    assert result.similarity == pytest.approx(0.0)
    assert result.decision == Decision.TRUE_MISS


def test_evaluate_probe_negated_vector_is_false_miss_of_correct_class():
    # unit vector at ~39 degrees from EX (closer to EX than to EY, so EX is
    # unambiguously nearest) but with cosine similarity to EX below the
    # threshold -- a correct match exists but isn't confident enough to hit.
    # [0.55, 0.45, 0] normalized: cos(EX)=0.774, cos(EY)=0.633.
    near_ex_but_below_threshold = np.array([0.77392, 0.63320, 0.0], dtype=np.float32)
    cache = _toy_cache()
    result = evaluate_probe(cache, near_ex_but_below_threshold, true_equivalence_class_id="account.reset_password", threshold=0.8)
    assert result.nearest_equivalence_class_id == "account.reset_password"
    assert result.similarity < 0.8
    assert result.decision == Decision.FALSE_MISS


def test_evaluate_probe_false_hit_wrong_nearest_class_above_threshold():
    # probe sits close to the WRONG cache entry
    near_ey = np.array([0.05, 0.9987, 0.0], dtype=np.float32)
    cache = _toy_cache()
    result = evaluate_probe(cache, near_ey, true_equivalence_class_id="account.reset_password", threshold=0.8)
    assert result.nearest_equivalence_class_id == "billing.refund_status"
    assert result.similarity >= 0.8
    assert result.decision == Decision.FALSE_HIT
