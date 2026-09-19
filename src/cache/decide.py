"""Threshold decision logic for cache hit/miss.

Responsibility: given a nearest-neighbour similarity and a threshold, decide
hit or miss; given the true equivalence-class labels, classify the decision as
a true hit, false hit, true miss, or false miss. This is the module unit tests
must cover against hand-constructed vectors before any real run (Phase 4 gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cache.store import SemanticCache


class Decision(str, Enum):
    """The four outcomes of one cache lookup, crossing (hit vs miss) with
    (nearest entry's class actually correct vs not):

      TRUE_HIT   -- hit,  nearest entry's class matches the probe's true
                    class: correct reuse, no cost.
      FALSE_HIT  -- hit,  nearest entry's class does NOT match: the cache
                    serves a WRONG cached answer to a real query. This is the
                    costly failure mode the whole paper measures (feeds FHR).
      TRUE_MISS  -- miss, nearest entry's class does NOT match: correctly
                    declined to reuse; falls through to a fresh LLM call.
      FALSE_MISS -- miss, nearest entry's class DOES match: a correct reuse
                    was available but threshold/embedding drift missed it --
                    a latency/cost loss, not a correctness loss.
    """

    TRUE_HIT = "true_hit"
    FALSE_HIT = "false_hit"
    TRUE_MISS = "true_miss"
    FALSE_MISS = "false_miss"


def is_hit(similarity: float, threshold: float) -> bool:
    """similarity >= threshold counts as a hit -- boundary is inclusive, so a
    probe landing exactly on the threshold is served from cache."""
    return similarity >= threshold


def classify(
    similarity: float,
    threshold: float,
    nearest_equivalence_class_id: str,
    true_equivalence_class_id: str,
) -> Decision:
    """Classify one cache lookup outcome. See `Decision` for the four cases."""
    hit = is_hit(similarity, threshold)
    correct_match = nearest_equivalence_class_id == true_equivalence_class_id
    if hit and correct_match:
        return Decision.TRUE_HIT
    if hit and not correct_match:
        return Decision.FALSE_HIT
    if not hit and correct_match:
        return Decision.FALSE_MISS
    return Decision.TRUE_MISS


@dataclass(frozen=True)
class ProbeResult:
    decision: Decision
    similarity: float
    nearest_equivalence_class_id: str
    true_equivalence_class_id: str


def evaluate_probe(
    cache: SemanticCache,
    probe_vector,
    true_equivalence_class_id: str,
    threshold: float,
) -> ProbeResult:
    """Run one probe through the full lookup-then-decide pipeline: find the
    nearest entry in `cache`, then classify it against
    `true_equivalence_class_id` at `threshold`."""
    nearest = cache.query(probe_vector)
    decision = classify(
        similarity=nearest.similarity,
        threshold=threshold,
        nearest_equivalence_class_id=nearest.entry.equivalence_class_id,
        true_equivalence_class_id=true_equivalence_class_id,
    )
    return ProbeResult(
        decision=decision,
        similarity=nearest.similarity,
        nearest_equivalence_class_id=nearest.entry.equivalence_class_id,
        true_equivalence_class_id=true_equivalence_class_id,
    )
