# Annotation Guidelines — Hinglish Semantic Cache Dataset

Read this before annotating anything. If something here is ambiguous on a
real example, don't guess — flag it and add the resolution to §7 so the next
person doesn't hit the same fork. This document should stay short enough
that you'll actually re-read it, not just skim it once.

## 1. What we're building, in one paragraph

For each of ~150–250 enterprise-support **intents**, we need: the canonical
English phrasing, 4–6 **paraphrases** that a semantic cache should treat as
interchangeable (same cached response), and 2–3 **hard negatives** that look
lexically similar but need a *different* answer. Every paraphrase and hard
negative gets rendered in three **conditions**: English (`en`), Devanagari
Hindi (`hi_deva`), and romanized Hinglish (`hi_rom`). That's the three-way
parallel set the whole paper depends on.

## 2. The core unit: response-equivalence class

An **equivalence class** is the set of all queries — across all conditions —
that should retrieve the *same* cached response. Two queries belong to the
same class if and only if a human support agent would give them the exact
same answer, not just a similar-sounding one.

- "How do I reset my password?" and "I forgot my login password, help" →
  same class (same answer: the reset flow).
- "How do I reset my password?" and "How do I reset my payment PIN?" →
  **different** classes, even though they share most of their words. The
  answer is different, so the class is different.

Every intent defines exactly one equivalence class. All of its paraphrases
(in all three conditions) belong to that class. Hard negatives, by
definition, do **not** belong to it (§3).

**Test before you write a paraphrase:** if I typed this to a support agent,
would they hand me literally the same response as the canonical intent? If
you have to add "well, basically" or "close enough," it isn't a paraphrase —
either tighten it or make it a hard negative instead.

## 3. What makes a valid hard negative

A hard negative is a query that:
1. Shares significant surface form with the intent (same topic, overlapping
   vocabulary, same entity type), so a model relying on lexical overlap or
   loose topical similarity would be tempted to treat it as a match, **and**
2. Genuinely requires a different response — not a nuance of the same
   answer, a categorically different one.

Bad hard negative (too easy — no lexical pull): "How do I reset my
password?" vs. "What are your business hours?" — no embedding model will
confuse these; it tells us nothing.

Bad hard negative (actually a paraphrase in disguise): "How do I reset my
password?" vs. "I need to change my password" — these usually get the same
answer. That's a paraphrase, not a negative.

Good hard negative: "How do I reset my password?" vs. "How do I reset my
security questions?" — same domain, same verb, adjacent feature, different
flow and different answer.

A hard negative does not need to correspond to another intent already in the
dataset. Most won't. Its only job is to be a plausible near-miss for *this*
intent's equivalence class — it gets its own equivalence class ID, distinct
from the intent it's paired with (see `src/dataset/schema.py`). If a hard
negative happens to coincide with another intent's canonical meaning, use
that intent's equivalence class ID instead of minting a new one.

## 4. Meaning-preservation and naturalness for Hinglish rendering

This is the part most likely to quietly wreck the paper (see the standing
caution in `project_setup_and_prompts.md`), so read it twice.

**Meaning-preservation:** a `hi_rom` or `hi_deva` rendering must belong to
the *same equivalence class* as its English source — same intent, same
expected answer. Don't "improve" or reword the intent while translating;
translate the meaning, not a paraphrase of your own invention. If you notice
yourself drifting, go back to the English source and re-render.

**Naturalness — the harder rule:** a `hi_rom` rendering must read like
something a real bilingual speaker would actually type, not like an English
sentence run through a transliterator. Concretely:
- Prefer natural code-mixing over rigid full-sentence translation where
  that's how people actually write it. "Mera password reset kaise karu"
  beats a stiffly "correct" full-Hindi rendering typed in Roman script if
  the former is what real usage looks like.
- Vary formality and phrasing the way real users do (some queries terse,
  some polite/full-sentence, some with typos or dropped words) — but don't
  inject noise that changes meaning.
- If you (the annotator) wouldn't plausibly type a sentence yourself when
  messaging support, don't submit it as a "natural" rendering.
- When in doubt, say the sentence out loud. If it sounds like reading a
  translated document rather than a message, rewrite it.

**Why this matters:** if our Hinglish renderings are systematically more
different from their English sources than real user phrasing would be, the
experiments measure our own annotation habits, not a real property of the
embedding models. This is the single biggest validity risk in the dataset.

## 5. Orthographic-variant policy

Romanized Hindi has no standard spelling. "Kaise" also appears as "kaisay,"
"kese," "kaise"; "hai" as "h," "hein," "hai." This variation is real and
part of what we're testing — don't normalize it away, but don't manufacture
noise either.

- Within a single rendering, spell consistently (don't mix "h" and "hai" for
  the same word in one sentence unless that's genuinely how it'd be typed).
- Across the 4–6 paraphrases for one intent, deliberately include at least
  one or two orthographic variants of common words (e.g., one paraphrase
  uses "kaise," another uses "kese") where it's natural to do so. Don't
  force it onto every item.
- Do not invent exotic spellings nobody uses. If you're not confident a
  spelling is genuinely common, use the more standard one.
- Numerals, English loanwords ("password," "account," "OTP"), and brand/
  product names stay as-is — don't transliterate loanwords that are always
  typed in Latin script even mid-Hindi-sentence.

## 6. Provenance and licensing

Every record's `provenance` field must be filled honestly:
- `source`: `"manual"` (you wrote it from scratch), `"llm_assisted"` (drafted
  with LLM help, then edited/verified by a human), or `"adapted"` (based on
  an external example).
- If `"adapted"`, record the origin and its licence in `license_note`. If
  you're not sure the licence permits redistribution, set
  `needs_review: true` and flag it — don't guess. Per Phase 2 instructions,
  anything of uncertain licence gets excluded until resolved, not included
  "just in case."

## 7. Disagreement resolution procedure

Phase 3 requires a second annotator to independently validate a sample and
compute Cohen's κ. When the second annotator disagrees with the first on
equivalence-class membership, hard-negative validity, or naturalness:

1. Both annotators state their reasoning in one sentence each, referencing
   the specific rule in §2–5 they're applying.
2. If the disagreement is a rule *application* error (one annotator missed
   a rule), fix it — no need to escalate.
3. If the disagreement is a genuine judgment call (the rule is ambiguous on
   this example), the item is **dropped** from the dataset rather than
   arbitrated by fiat. We are not trying to force high agreement by
   argument; low agreement on an item means the item is a bad example.
4. Log the resolution as one row in `data/annotated/agreement.csv`
   (columns: item ID, annotator 1 judgment, annotator 2 judgment,
   resolution, rule-ambiguity flag).
5. If the same rule keeps causing disagreement across multiple items, that's
   a guideline bug — raise it for a guideline revision (append the change
   and rationale to `notes/decisions.md`), don't keep resolving the same
   ambiguity item-by-item.

Disagreement is expected and useful signal, not a failure — record it
honestly. Cohen's κ is computed over the independently-annotated sample
*before* any resolution discussion, not after.

## 8. Worked example

**Intent:** `billing.download_invoice` — "How do I download my last
invoice?"

| Role | Condition | Text |
|---|---|---|
| canonical | en | How do I download my last invoice? |
| paraphrase | en | Where can I get a copy of my most recent invoice? |
| paraphrase | hi_rom | Mera last invoice kaise download karu? |
| paraphrase | hi_rom | Recent invoice ki copy kahan se milegi? |
| paraphrase | hi_deva | मुझे अपना पिछला इनवॉइस कहाँ से मिलेगा? |
| hard_negative | en | How do I update my billing address? |
| hard_negative | hi_rom | Mera billing address kaise update karu? |

Note the hard negative shares "billing" vocabulary and structure with the
intent but needs a completely different answer (address-update flow, not
invoice retrieval) — that's what makes it a *good* hard negative, per §3.

## 9. Checklist before submitting a batch

- [ ] Every paraphrase passes the §2 "same answer" test.
- [ ] Every hard negative passes both §3 conditions (lexical pull + genuinely
      different answer).
- [ ] Every `hi_rom` rendering passes the §4 naturalness test — read it
      aloud.
- [ ] Orthographic variation appears across paraphrases where natural, not
      forced (§5).
- [ ] `provenance` filled for every record; anything uncertain flagged, not
      guessed (§6).
- [ ] IDs follow the schema in `src/dataset/schema.py` — don't hand-invent
      an ID format.
