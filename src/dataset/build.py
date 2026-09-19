"""Assembles the three-way parallel dataset (en / hi_deva / hi_rom) into a
single validated collection, ready for embedding.

Responsibility: read data/annotated/{en,hi_deva,hi_rom}.jsonl, join records by
intent ID and equivalence class ID, validate cross-condition completeness, and
emit the assembled set consumed by src/embed and the experiments. Built in
Phase 4.

Not needed until Phase 5's exp02/exp03, which are the first things that
actually operate on all three conditions of the SAME underlying probe at
once (the three-way degradation and attribution experiments) -- everything
before that (Phase 3 validation, Phase 4's sanity check) only ever needed one
condition at a time via dataset.schema.load_records_jsonl directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import Config
from dataset.schema import Record, load_records_jsonl

CONDITIONS = ("en", "hi_deva", "hi_rom")


@dataclass(frozen=True)
class AlignedTriple:
    """The same underlying (intent, role, sequence) rendered in all three
    conditions -- the unit exp02/exp03 actually operate on, since the
    research question is about the SAME query surviving translation, not
    about any one condition in isolation."""

    intent_id: str
    role: str
    equivalence_class_id: str
    en: Record
    hi_deva: Record
    hi_rom: Record


def _alignment_key(record_id: str) -> str:
    """record_id = '<intent_id>__<role>__<condition>__<seq>' (schema.py's
    make_record_id). The alignment key across conditions is everything
    except the condition segment, which is always third."""
    parts = record_id.split("__")
    if len(parts) != 4:
        raise ValueError(f"unexpected record_id format: {record_id!r}")
    intent_id, role, _condition, seq = parts
    return f"{intent_id}__{role}__{seq}"


def load_three_way(config: Config | None = None) -> list[AlignedTriple]:
    """Load data/annotated/{en,hi_deva,hi_rom}.jsonl and join them into
    AlignedTriples keyed by (intent_id, role, seq).

    Raises ValueError if the "perfectly parallel" invariant PROJECT.md
    requires doesn't hold: any alignment key present in one condition but
    missing from another, or present in all three but disagreeing on
    intent_id/role/equivalence_class_id across conditions. Both would mean
    something upstream (Phase 2/3 drafting) silently broke parallelism --
    this must fail loudly, not join partial/mismatched data.
    """
    config = config or Config()
    by_condition: dict[str, dict[str, Record]] = {}
    for condition in CONDITIONS:
        path = config.data_annotated_dir / f"{condition}.jsonl"
        keyed: dict[str, Record] = {}
        for r in load_records_jsonl(path):
            key = _alignment_key(r.record_id)
            if key in keyed:
                raise ValueError(f"duplicate alignment key {key!r} within {path}")
            keyed[key] = r
        by_condition[condition] = keyed

    all_keys = set().union(*(set(d) for d in by_condition.values()))
    missing = []
    mismatched = []
    triples = []
    for key in sorted(all_keys):
        recs = {c: by_condition[c].get(key) for c in CONDITIONS}
        if any(r is None for r in recs.values()):
            missing.append({"key": key, "missing_in": [c for c in CONDITIONS if recs[c] is None]})
            continue
        eq_ids = {r.equivalence_class_id for r in recs.values()}
        intent_ids = {r.intent_id for r in recs.values()}
        roles = {r.role for r in recs.values()}
        if len(eq_ids) > 1 or len(intent_ids) > 1 or len(roles) > 1:
            mismatched.append({
                "key": key,
                "equivalence_class_ids": sorted(eq_ids),
                "intent_ids": sorted(intent_ids),
                "roles": sorted(roles),
            })
            continue
        triples.append(AlignedTriple(
            intent_id=recs["en"].intent_id,
            role=recs["en"].role,
            equivalence_class_id=recs["en"].equivalence_class_id,
            en=recs["en"], hi_deva=recs["hi_deva"], hi_rom=recs["hi_rom"],
        ))

    if missing or mismatched:
        raise ValueError(
            f"three-way dataset is not perfectly parallel: {len(missing)} alignment key(s) "
            f"missing a counterpart in at least one condition, {len(mismatched)} key(s) have "
            f"mismatched intent_id/role/equivalence_class_id across conditions.\n"
            f"First few missing: {missing[:5]}\nFirst few mismatched: {mismatched[:5]}"
        )
    return triples
