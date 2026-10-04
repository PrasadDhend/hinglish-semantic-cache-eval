"""Unit tests for experiments/rank_metrics.py, against hand-constructed vectors
and ranks, plus a regression guard that its lookups agree with
probes_and_charts.py's (results/para_only_summary.json)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cache.store import CacheEntry, SemanticCache
from eval.metrics import confusion_counts, false_hit_rate, hit_rate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from rank_metrics import (  # noqa: E402
    cache_decision, mrr, ndcg_at_k, rank_buckets, rank_of_correct, ranked_similarities,
    recall_at_k, relative_drop,
)

EX = np.array([1.0, 0.0, 0.0], dtype=np.float32)
EY = np.array([0.0, 1.0, 0.0], dtype=np.float32)
EZ = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _cache(*pairs):
    return SemanticCache([CacheEntry(key=f"c{i}", equivalence_class_id=c, vector=v) for i, (c, v) in enumerate(pairs)])


# --- ranked lookup --------------------------------------------------------

def test_rank_of_correct_hand_values():
    cache = _cache(("a", EX), ("b", EY), ("c", EZ))
    probe = np.array([0.1, 0.9, 0.3], dtype=np.float32)  # b > c > a
    assert rank_of_correct(cache, probe, "b")[0] == 1
    assert rank_of_correct(cache, probe, "c")[0] == 2
    assert rank_of_correct(cache, probe, "a")[0] == 3


def test_rank_of_correct_returns_top1_and_correct_similarity():
    cache = _cache(("a", EX), ("b", EY))
    probe = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    rank, top1, correct = rank_of_correct(cache, probe, "a")
    assert rank == 2
    assert top1 == pytest.approx(0.8)
    assert correct == pytest.approx(0.6)


def test_first_ranked_entry_is_what_query_returns():
    cache = _cache(("a", EX), ("b", EY), ("c", EZ))
    probe = np.array([0.2, 0.5, 0.7], dtype=np.float32)
    sims, order = ranked_similarities(cache, probe)
    nearest = cache.query(probe)
    assert cache.entries[order[0]].equivalence_class_id == nearest.entry.equivalence_class_id
    assert float(sims[order[0]]) == nearest.similarity


def test_tie_breaks_the_same_way_as_query():
    # two identical cached vectors: query() (np.argmax) returns the first, so
    # rank 1 must go to the first as well or Recall@1 != top1_correct_rate
    cache = _cache(("a", EX), ("b", EX))
    assert cache.query(EX).entry.equivalence_class_id == "a"
    assert rank_of_correct(cache, EX, "a")[0] == 1
    assert rank_of_correct(cache, EX, "b")[0] == 2


def test_unknown_equivalence_class_raises():
    with pytest.raises(KeyError):
        rank_of_correct(_cache(("a", EX)), EX, "zzz")


# --- ranking metrics ------------------------------------------------------

def test_recall_at_k_hand_values():
    ranks = [1, 2, 6, 1]
    assert recall_at_k(ranks, 1) == pytest.approx(0.5)
    assert recall_at_k(ranks, 5) == pytest.approx(0.75)
    assert recall_at_k(ranks, 10) == pytest.approx(1.0)


def test_mrr_hand_values():
    assert mrr([1, 2, 4]) == pytest.approx((1 + 0.5 + 0.25) / 3)


def test_ndcg_one_relevant_item_hand_values():
    # rank 1 -> 1, rank 3 -> 1/log2(4) = 0.5, rank 6 -> outside top 5 -> 0
    assert ndcg_at_k([1, 3, 6], 5) == pytest.approx(0.5)


def test_ranking_metrics_ignore_similarity_magnitude():
    # the post's point: same ranks, any scores -> same ranking metrics
    ranks = [1, 1, 2, 7]
    assert recall_at_k(ranks, 1) == recall_at_k(np.array(ranks), 1)
    high = cache_decision(ranks, [0.95, 0.95, 0.95, 0.95], theta=0.80)
    low = cache_decision(ranks, [0.50, 0.50, 0.50, 0.50], theta=0.80)
    assert high["hr"] == 1.0 and low["hr"] == 0.0


def test_rank_buckets_partition_all_probes():
    b = rank_buckets([1, 1, 2, 5, 6, 20, 21, 166])
    assert b == {"1": 0.25, "2-5": 0.25, "6-20": 0.25, ">20": 0.25}
    assert sum(b.values()) == pytest.approx(1.0)


# --- cache decision -------------------------------------------------------

def test_cache_decision_hand_values():
    d = cache_decision([1, 1, 2, 3], [0.9, 0.7, 0.85, 0.5], theta=0.8)
    assert d["hits"] == 2
    assert d["true_hits"] == 1
    assert d["hr"] == pytest.approx(0.5)
    assert d["fhr"] == pytest.approx(0.5)
    assert d["rank1_not_served"] == 1


def test_cache_decision_boundary_threshold_is_served():
    assert cache_decision([1], [0.8], theta=0.8)["hits"] == 1


def test_cache_decision_nothing_served_is_nan_not_zero():
    assert np.isnan(cache_decision([1, 2], [0.1, 0.2], theta=0.8)["fhr"])


def test_cache_decision_agrees_with_eval_metrics():
    rng = np.random.default_rng(0)
    ranks = rng.integers(1, 6, 200)
    sims = rng.uniform(0.5, 1.0, 200)
    d = cache_decision(ranks, sims, theta=0.8)
    counts = confusion_counts(sims, ranks == 1, threshold=0.8)
    assert d["hr"] == pytest.approx(hit_rate(counts))
    assert d["fhr"] == pytest.approx(false_hit_rate(counts))


def test_relative_drop():
    assert relative_drop(0.8, 0.2) == pytest.approx(0.75)
    assert np.isnan(relative_drop(0.0, 0.0))


# --- regression guard against post 1's numbers ----------------------------

RANK_JSON = ROOT / "results" / "rank_metrics.json"
SUMMARY_JSON = ROOT / "results" / "para_only_summary.json"
needs_run = pytest.mark.skipif(not (RANK_JSON.exists() and SUMMARY_JSON.exists()),
                               reason="run experiments/rank_metrics.py first")


@needs_run
def test_recall_at_1_equals_post1_top1_correct_rate():
    new = json.loads(RANK_JSON.read_text(encoding="utf-8"))["models"]
    old = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    for model, by_cond in old.items():
        for cond in ("en", "hi_rom"):
            assert new[model][cond]["n_rank1"] == by_cond[cond]["nearest_is_correct"], (model, cond)
            assert new[model][cond]["recall_at_1"] == pytest.approx(by_cond[cond]["top1_correct_rate"]), (model, cond)
            assert new[model][cond]["hits"] == by_cond[cond]["hits"], (model, cond)
            assert new[model][cond]["hr"] == pytest.approx(by_cond[cond]["hr"]), (model, cond)


@needs_run
def test_minilm_hi_rom_ranked_301_first_and_served_none():
    m = json.loads(RANK_JSON.read_text(encoding="utf-8"))["models"]["english_centric"]["hi_rom"]
    assert m["n_rank1"] == 301
    assert m["n_probes"] == 664
    assert m["hits"] == 0
    assert m["rank1_not_served"] == 301
