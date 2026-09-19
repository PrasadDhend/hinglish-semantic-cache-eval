"""Dataclasses and validation for the three-way parallel dataset records.

Fixed in Phase 1, alongside data/guidelines/annotation_guidelines.md.
Changing this after annotation starts is expensive — see PROJECT.md working
rule 1 and the Phase 1 gate in project_setup_and_prompts.md.

Two record types:

- `SeedIntent`: one row per canonical English intent (data/raw/seed_intents.jsonl).
  The master list; everything else derives from it.
- `Record`: one row per rendered query — paraphrase or hard negative, in one
  of the three conditions (data/annotated/{en,hi_deva,hi_rom}.jsonl).

Read data/guidelines/annotation_guidelines.md for what these fields mean in
practice, especially the intent_id vs. equivalence_class_id distinction
(guidelines §3) and the role semantics (guidelines §8).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Condition = Literal["en", "hi_deva", "hi_rom"]
Role = Literal["canonical", "paraphrase", "hard_negative"]
ProvenanceSource = Literal["manual", "llm_assisted", "adapted"]

VALID_CONDITIONS: tuple[Condition, ...] = ("en", "hi_deva", "hi_rom")
VALID_ROLES: tuple[Role, ...] = ("canonical", "paraphrase", "hard_negative")
VALID_PROVENANCE_SOURCES: tuple[ProvenanceSource, ...] = (
    "manual", "llm_assisted", "adapted",
)

# Slug rule for intent_id: "<category>.<short_name>", lowercase, underscores
# within segments, one dot separating category from name. Category is
# derivable from intent_id without a separate field (split on the first ".").
_INTENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

# Rough script-consistency check, not a strict validator: Devanagari block.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# Latin letters, used to sanity-check hi_rom / en are not accidentally
# Devanagari-only text.
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class Provenance:
    """Who made this record and where it came from — see guidelines §6."""

    source: ProvenanceSource
    author: str  # annotator initials/id, e.g. "PD" or "PD+llm"
    created_at: str  # ISO 8601 date, e.g. "2026-08-15"
    license_note: str = ""  # required (non-empty) if source == "adapted"
    needs_review: bool = False  # true if licence/origin is unconfirmed

    def validate(self) -> list[str]:
        errors = []
        if self.source not in VALID_PROVENANCE_SOURCES:
            errors.append(f"provenance.source invalid: {self.source!r}")
        if not self.author.strip():
            errors.append("provenance.author is empty")
        if self.source == "adapted" and not self.license_note.strip():
            errors.append("provenance.license_note required when source='adapted'")
        return errors


@dataclass(frozen=True)
class SeedIntent:
    """One canonical English intent — data/raw/seed_intents.jsonl.

    intent_id also encodes the category (segment before the first '.'),
    matching Phase 2's "4-6 categories x ~150-250 intents" structure without
    a redundant category field.
    """

    intent_id: str
    canonical_text: str
    provenance: Provenance

    def validate(self) -> list[str]:
        errors = []
        if not _INTENT_ID_RE.match(self.intent_id):
            errors.append(
                f"intent_id {self.intent_id!r} must match '<category>.<name>' "
                "(lowercase, underscores, one dot)"
            )
        if not self.canonical_text.strip():
            errors.append("canonical_text is empty")
        elif not _LATIN_RE.search(self.canonical_text):
            errors.append("canonical_text (English) contains no Latin letters")
        errors.extend(self.provenance.validate())
        return errors

    def to_json_dict(self) -> dict:
        d = _asdict_shallow(self)
        d["provenance"] = _asdict_shallow(self.provenance)
        return d


@dataclass(frozen=True)
class Record:
    """One rendered query — data/annotated/{en,hi_deva,hi_rom}.jsonl.

    record_id format: "<intent_id>__<role>__<condition>__<seq>", e.g.
    "billing.download_invoice__paraphrase__hi_rom__002". Once assigned,
    never reassign a record_id even if the record is later dropped — leave
    the gap. Sequence numbers are per (intent_id, role, condition).

    equivalence_class_id: the TRUE grouping key for "shares a cached
    response" — this is what eval code uses to score hits as true/false.
    For role in {"canonical", "paraphrase"}, this always equals intent_id.
    For role == "hard_negative", this must NOT equal intent_id: use another
    existing intent_id if the hard negative genuinely coincides with a
    different intent's meaning, otherwise mint a singleton
    "<intent_id>__hardneg__<seq>" (see guidelines §3).
    """

    record_id: str
    intent_id: str
    equivalence_class_id: str
    condition: Condition
    role: Role
    text: str
    provenance: Provenance

    def validate(self) -> list[str]:
        errors = []

        if not _INTENT_ID_RE.match(self.intent_id):
            errors.append(f"intent_id {self.intent_id!r} malformed")

        if self.condition not in VALID_CONDITIONS:
            errors.append(f"condition invalid: {self.condition!r}")

        if self.role not in VALID_ROLES:
            errors.append(f"role invalid: {self.role!r}")

        if not self.text.strip():
            errors.append("text is empty")
        if self.text != self.text.strip():
            errors.append("text has leading/trailing whitespace")

        # equivalence-class rules (guidelines §2-3)
        if self.role in ("canonical", "paraphrase"):
            if self.equivalence_class_id != self.intent_id:
                errors.append(
                    "canonical/paraphrase records must have "
                    "equivalence_class_id == intent_id"
                )
        elif self.role == "hard_negative":
            if self.equivalence_class_id == self.intent_id:
                errors.append(
                    "hard_negative must NOT share its paired intent's "
                    "equivalence_class_id (guidelines §3) — use a different "
                    "intent_id or a singleton '<intent_id>__hardneg__<seq>' id"
                )

        # record_id format + internal consistency
        expected_prefix = f"{self.intent_id}__{self.role}__{self.condition}__"
        if not self.record_id.startswith(expected_prefix):
            errors.append(
                f"record_id {self.record_id!r} inconsistent with "
                f"intent_id/role/condition (expected prefix {expected_prefix!r})"
            )

        # rough script-consistency heuristics — catches copy-paste mistakes,
        # not a linguistic validator
        has_deva = bool(_DEVANAGARI_RE.search(self.text))
        has_latin = bool(_LATIN_RE.search(self.text))
        if self.condition == "en" and has_deva:
            errors.append("condition='en' but text contains Devanagari characters")
        if self.condition == "hi_deva" and not has_deva:
            errors.append("condition='hi_deva' but text contains no Devanagari characters")
        if self.condition == "hi_rom" and has_deva:
            errors.append(
                "condition='hi_rom' but text contains Devanagari characters "
                "(romanized Hinglish should be Latin script)"
            )
        if self.condition == "hi_rom" and not has_latin:
            errors.append("condition='hi_rom' but text contains no Latin characters")

        errors.extend(self.provenance.validate())
        return errors

    def to_json_dict(self) -> dict:
        d = _asdict_shallow(self)
        d["provenance"] = _asdict_shallow(self.provenance)
        return d


def make_record_id(intent_id: str, role: Role, condition: Condition, seq: int) -> str:
    """Build a record_id in the fixed format. seq is 1-based, zero-padded to 3 digits."""
    return f"{intent_id}__{role}__{condition}__{seq:03d}"


def _asdict_shallow(obj) -> dict:
    """dataclasses.asdict() recurses and breaks on our nested Provenance
    (we want it as a dict of primitives, not deep-copied dataclasses)."""
    return {f: getattr(obj, f) for f in obj.__dataclass_fields__}


# --- loading & validation over files ---------------------------------------

def _record_from_dict(d: dict) -> Record:
    prov = d["provenance"]
    return Record(
        record_id=d["record_id"],
        intent_id=d["intent_id"],
        equivalence_class_id=d["equivalence_class_id"],
        condition=d["condition"],
        role=d["role"],
        text=d["text"],
        provenance=Provenance(**prov),
    )


def _seed_intent_from_dict(d: dict) -> SeedIntent:
    prov = d["provenance"]
    return SeedIntent(
        intent_id=d["intent_id"],
        canonical_text=d["canonical_text"],
        provenance=Provenance(**prov),
    )


def load_records_jsonl(path: Path) -> list[Record]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(_record_from_dict(json.loads(line)))
    return records


def load_seed_intents_jsonl(path: Path) -> list[SeedIntent]:
    intents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            intents.append(_seed_intent_from_dict(json.loads(line)))
    return intents


def validate_records_file(path: Path) -> list[tuple[int, list[str]]]:
    """Returns [(line_number, [error, ...]), ...] for every line with >=1 error.
    Also checks cross-record invariants: unique record_id, and every
    canonical/paraphrase intent_id has exactly one equivalence_class_id
    (itself) while every hard_negative's equivalence_class_id differs from
    its own intent_id.
    """
    problems: list[tuple[int, list[str]]] = []
    seen_ids: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rec = _record_from_dict(d)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                problems.append((line_no, [f"parse error: {e}"]))
                continue
            errors = rec.validate()
            if rec.record_id in seen_ids:
                errors.append(
                    f"duplicate record_id (first seen at line {seen_ids[rec.record_id]})"
                )
            else:
                seen_ids[rec.record_id] = line_no
            if errors:
                problems.append((line_no, errors))
    return problems
