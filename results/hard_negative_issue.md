# Hard-negative issue found 2026-09-19 (while making blog chart 2)

**What:** 328 of 332 English hard negatives are character-identical to the English canonical of the sibling class they are labelled with. In hi_rom, 277 of 332 hard negatives are identical to that sibling's hi_rom canonical.

**Effect:** the cache holds that exact English canonical. So in the English baseline, each of these probes is a guaranteed correct hit at similarity 1.0. That is 328 of the 996 probes. They never test a near-miss. They add free true hits, which raises English HR and lowers English FHR. In the cross-condition hi_rom runs they act like extra paraphrases of the sibling intent.

**Same θ=0.80 numbers, all probes vs paraphrases only (FHR / HR %):**

| model | en all (996) | en para-only (664) | hi_rom all (996) | hi_rom para-only (664) |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 2.1 / 52.0 | 5.9 / 28.0 | – / 0.0 | – / 0.0 |
| multilingual-e5-large | 12.2 / 100.0 | 18.4 / 100.0 | 22.5 / 99.9 | 31.7 / 99.8 |
| BGE-M3 | 5.7 / 86.5 | 9.2 / 79.8 | 0.0 / 6.0 | 0.0 / 2.7 |
| LaBSE | 6.1 / 46.3 | 19.4 / 19.4 | 2.6 / 3.9 | 20.0 / 0.8 (≈5 hits) |
| Indic-SBERT | 6.9 / 51.2 | 18.0 / 26.8 | 8.0 / 35.0 | 22.8 / 13.9 |
| HingRoBERTa | 33.8 / 100.0 | 50.3 / 100.0 | 42.3 / 100.0 | 59.8 / 100.0 |
| Granite R2 97M | 11.1 / 100.0 | 16.7 / 100.0 | 26.3 / 98.9 | 38.1 / 98.5 |

**What survives:** all three failure shapes hold (MiniLM silent; e5/Granite serve more wrong answers; BGE-M3 collapses from 79.8% to 2.7% HR). No CIs or paired tests were run on the para-only numbers yet.

**What doesn't:** every absolute number, and the paper's description of hard negatives as near-miss tests (Section 3.1 / 4.1). LaBSE's "deceptive safety" can't be read from para-only data: its FHR rests on about 5 hits.

Not fixed in the paper repo. Author's decision.
