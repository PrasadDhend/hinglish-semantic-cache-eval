"""In-memory semantic cache with nearest-neighbour lookup.

Responsibility: hold a set of (query embedding, response/equivalence-class
label) pairs and, given a probe embedding, return the nearest cached entry and
its cosine similarity. Must support both cross-condition population (cache
built from English, probed with Hinglish/Hindi) and same-condition population
modes. Built in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CacheEntry:
    """One populated cache slot: an embedded query standing in for a cached
    response, and the equivalence class that response belongs to. `key` is
    just a human-readable label (a record_id) for debugging -- lookup logic
    only ever compares `equivalence_class_id`.
    """

    key: str
    equivalence_class_id: str
    vector: np.ndarray


@dataclass(frozen=True)
class NearestResult:
    entry: CacheEntry
    similarity: float


class SemanticCache:
    """A fixed set of cache entries with cosine-similarity nearest-neighbour
    lookup.

    Vectors are assumed L2-normalized on the way in (embed/registry.py and
    embed/encode.py already normalize) so cosine similarity reduces to a
    plain dot product -- SemanticCache does not re-normalize or defend
    against un-normalized input, to keep the hot path a single matmul.
    """

    def __init__(self, entries: list[CacheEntry]):
        if not entries:
            raise ValueError("SemanticCache needs at least one entry")
        seen: set[str] = set()
        for e in entries:
            if e.equivalence_class_id in seen:
                raise ValueError(
                    f"duplicate equivalence_class_id in cache population: "
                    f"{e.equivalence_class_id!r} -- a deployed cache should "
                    "hold exactly one entry per equivalence class"
                )
            seen.add(e.equivalence_class_id)
        self.entries = entries
        self._matrix = np.stack([e.vector for e in entries]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def query(self, probe_vector: np.ndarray) -> NearestResult:
        """Return the single nearest cache entry to `probe_vector`."""
        sims = self._matrix @ np.asarray(probe_vector, dtype=np.float32)
        idx = int(np.argmax(sims))
        return NearestResult(entry=self.entries[idx], similarity=float(sims[idx]))


def populate_from_records(records, vectors: dict[str, np.ndarray], role: str = "canonical") -> SemanticCache:
    """Build a SemanticCache from `records` restricted to `role` (default
    "canonical": a deployed cache holds one already-answered query per
    equivalence class, not every paraphrase of it).

    `vectors` maps record_id -> embedding, e.g. built by zipping
    embed.encode.encode_texts()'s output back onto the record_ids it was
    called with.

    Cross-condition vs. same-condition population is entirely a matter of
    which `records` the caller passes in here -- this function does not know
    or care about `condition`:

        cross-condition (headline): cache from en canonicals, probe with
            hi_deva/hi_rom paraphrases and hard negatives
                populate_from_records([r for r in en_records if r.role == "canonical"], en_vectors)

        same-condition (control): cache and probes from the same condition
                populate_from_records([r for r in hi_rom_records if r.role == "canonical"], hi_rom_vectors)
    """
    selected = [r for r in records if r.role == role]
    if not selected:
        raise ValueError(f"no records with role={role!r} to populate the cache from")
    entries = []
    for r in selected:
        if r.record_id not in vectors:
            raise KeyError(f"no vector provided for record_id {r.record_id!r}")
        entries.append(
            CacheEntry(
                key=r.record_id,
                equivalence_class_id=r.equivalence_class_id,
                vector=vectors[r.record_id],
            )
        )
    return SemanticCache(entries)
