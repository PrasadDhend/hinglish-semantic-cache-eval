"""Shared helpers for experiments/exp01-03 -- not itself an experiment.

Loading + encoding + cache-probing boilerplate that all three scripts need.
Kept in experiments/ rather than src/eval/ since this is experiment-script
glue specific to how exp01-03 are structured, not a reusable library module
with its own phase-gated responsibility in project_setup_and_prompts.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cache.store import populate_from_records  # noqa: E402
from config import Config  # noqa: E402
from dataset.build import load_three_way  # noqa: E402
from embed.encode import encode_texts  # noqa: E402

MODEL_KEYS: list[str] = list(Config().embedding_models)


def load_by_condition(config: Config | None = None):
    """Load + validate the three-way parallel dataset (dataset.build), then
    split it back into per-condition, per-role record lists:

        {condition: {"canonical": [...], "paraphrase": [...], "hard_negative": [...]}}

    IMPORTANT alignment guarantee callers rely on throughout exp01-03: for a
    fixed role, by_condition["en"][role][k], by_condition["hi_deva"][role][k],
    and by_condition["hi_rom"][role][k] are always the SAME underlying triple
    (built by iterating the same `triples` list once, in order, appending
    each condition's record to the same role bucket) -- so two conditions'
    lists concatenated in the same role order (e.g. canonical+paraphrase+
    hard_negative) stay row-for-row paired with no explicit record_id
    matching needed. This is what makes the paired bootstrap comparisons in
    exp02/exp03 simple index-aligned array operations.

    Returns (triples, by_condition).
    """
    config = config or Config()
    triples = load_three_way(config)
    by_condition = {
        "en": {"canonical": [], "paraphrase": [], "hard_negative": []},
        "hi_deva": {"canonical": [], "paraphrase": [], "hard_negative": []},
        "hi_rom": {"canonical": [], "paraphrase": [], "hard_negative": []},
    }
    for t in triples:
        by_condition["en"][t.role].append(t.en)
        by_condition["hi_deva"][t.role].append(t.hi_deva)
        by_condition["hi_rom"][t.role].append(t.hi_rom)
    return triples, by_condition


def all_records_aligned(by_condition: dict, condition: str) -> list:
    """canonical + paraphrase + hard_negative, in that fixed order -- the
    order every caller must use consistently across conditions to keep rows
    paired (see load_by_condition's docstring)."""
    bucket = by_condition[condition]
    return bucket["canonical"] + bucket["paraphrase"] + bucket["hard_negative"]


def encode_records(model_key: str, records: list, config: Config | None = None) -> dict:
    """Encode `records` (any mix of conditions/roles) with `model_key`,
    returning {record_id: vector}. Entirely backed by embed.encode's on-disk
    cache, so re-running any experiment script -- or running a second one
    over overlapping texts -- costs nothing for already-encoded texts."""
    config = config or Config()
    texts = [r.text for r in records]
    vectors = encode_texts(model_key, texts, config=config)
    return {r.record_id: v for r, v in zip(records, vectors)}


def build_cache(canonical_records: list, vec_by_id: dict):
    return populate_from_records(canonical_records, vec_by_id, role="canonical")


def probe_similarities_and_correctness(cache, probe_records: list, vec_by_id: dict):
    """One nearest-neighbour lookup per probe. Returns three parallel lists
    (similarities, correct_match, record_ids), in the same order as
    `probe_records`. No threshold applied here -- eval.metrics.sweep() /
    eval.stats's bootstrap tools do that."""
    similarities, correct_match, record_ids = [], [], []
    for r in probe_records:
        nearest = cache.query(vec_by_id[r.record_id])
        similarities.append(nearest.similarity)
        correct_match.append(nearest.entry.equivalence_class_id == r.equivalence_class_id)
        record_ids.append(r.record_id)
    return similarities, correct_match, record_ids
