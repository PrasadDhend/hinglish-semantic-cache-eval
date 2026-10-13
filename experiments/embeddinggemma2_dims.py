"""Blog companion (bridge post): EmbeddingGemma 2 on the Hinglish cache benchmark.

Adds google/embeddinggemma-2 (text-only, float32) to the 7-model lineup as an
8th model, at 768/512/256/128 dims, with and without its SentenceSimilarity
prompt. Same instrument as posts 1 and 2: English cache of 166 canonicals,
664 paraphrase probes per condition, theta fixed at 0.80, nothing tuned.

Also runs the direct test post 2 left open ("romanization lowers similarity
levels without reordering them"): for each probe, the similarity to all 166
cache entries under English vs romanized wording, compared as a level (mean
similarity) and as an order (Spearman across the 166 entries). Run for all 8
models from cached vectors.

Decisions fixed before the run (drafts/embeddinggemma-2-hinglish/outline.md):
symmetric SentenceSimilarity prompt on both sides, one no-prompt run as a check,
float32, text-only load, no threshold tuning.

Needs a newer sentence-transformers than the pinned 2.7.0, so run it in a
separate environment. Vectors land in results/embeddings/<model_key>/ like the
other models; the old seven are only read.

    python experiments/embeddinggemma2_dims.py [--skip-encode]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import rank_metrics as rm  # noqa: E402
from _common import build_cache, encode_records, load_by_condition  # noqa: E402

MODEL_ID = "google/embeddinggemma-2"
DIMS = (768, 512, 256, 128)
PROMPTS = {"ss": "SentenceSimilarity", "np": None}
COND = ("en", "hi_deva", "hi_rom")
OLD = rm.ORDER


def truncate(v, d):
    """Matryoshka truncation: keep the first d dims, re-normalize."""
    t = np.asarray(v, dtype=np.float32)[..., :d]
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


class EG2Encoder:
    """Full-768 float32 vectors for one prompt setting; truncation happens later."""

    def __init__(self, prompt_name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(MODEL_ID, config_kwargs={"vision_config": None, "audio_config": None})
        self.m = self.m.float()  # card: bf16 or fp32 only; fp32 chosen before the run
        self.prompt_name = prompt_name

    def encode(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            out.append(self.m.encode(texts[i:i + 64], prompt_name=self.prompt_name, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False))
        return np.vstack(out).astype(np.float32)


def get_vecs(key, records, encoder_factory):
    """encode_records, but with an injected encoder so the pinned registry isn't needed."""
    from config import Config
    from embed.encode import encode_texts
    texts = [r.text for r in records]
    cfg = Config()
    holder = {}

    class Lazy:
        def encode(self, t):
            if "e" not in holder:
                holder["e"] = encoder_factory()
            return holder["e"].encode(t)

    vecs = encode_texts(key, texts, config=cfg, encoder=Lazy())
    return {r.record_id: v for r, v in zip(records, vecs)}


def lookups_for(vec_by_id, canon, probes):
    cache = build_cache(canon, vec_by_id)
    out = {}
    for c in COND:
        rows = [rm.rank_of_correct(cache, vec_by_id[r.record_id], r.equivalence_class_id) for r in probes[c]]
        out[c] = (np.array([x[0] for x in rows]), np.array([x[1] for x in rows]), np.array([x[2] for x in rows]))
    return cache, out


def spearman_rows(a, b):
    """Row-wise Spearman between two (n, m) matrices (no ties handling needed for float sims)."""
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def level_vs_order(cache, vec_by_id, probes):
    """Paired probe k in en vs hi_rom (index-aligned paraphrases): similarity to all 166 entries."""
    en = np.stack([cache._matrix @ vec_by_id[r.record_id] for r in probes["en"]])
    hr = np.stack([cache._matrix @ vec_by_id[r.record_id] for r in probes["hi_rom"]])
    rho = spearman_rows(en, hr)
    return {
        "mean_sim_en": float(en.mean()), "mean_sim_hi_rom": float(hr.mean()),
        "mean_level_drop": float((en.mean(1) - hr.mean(1)).mean()),
        "std_of_sims_en": float(en.std(1).mean()), "std_of_sims_hi_rom": float(hr.std(1).mean()),
        "spearman_median": float(np.median(rho)), "spearman_p10": float(np.percentile(rho, 10)),
        "spearman_mean": float(rho.mean()),
        "top1_en": float(en.max(1).mean()), "top1_hi_rom": float(hr.max(1).mean()),
    }


def main():
    skip = "--skip-encode" in sys.argv
    _, by_cond = load_by_condition()
    canon = by_cond["en"]["canonical"]
    probes = {c: by_cond[c]["paraphrase"] for c in COND}
    all_recs = canon + [r for c in COND for r in probes[c]]
    result = {"theta": rm.THETA, "model": MODEL_ID, "n_probes": 664, "runs": {}, "old": {}, "level_vs_order": {}}

    # --- EmbeddingGemma 2: encode once per prompt setting at full 768, derive the rest ---
    t0 = time.time()
    for tag, pname in PROMPTS.items():
        full = get_vecs(f"eg2_{tag}_768", all_recs, lambda p=pname: EG2Encoder(p)) if not skip else None
        if skip:
            full = encode_records(f"eg2_{tag}_768", all_recs)
        for d in DIMS:
            vecs = {k: truncate(v, d) for k, v in full.items()} if d != 768 else full
            cache, lk = lookups_for(vecs, canon, probes)
            name = f"eg2_{tag}_{d}"
            result["runs"][name] = {c: rm.summarize(*lk[c][:2]) for c in COND}
            result["level_vs_order"][name] = level_vs_order(cache, vecs, probes)
            hr = result["runs"][name]["hi_rom"]
            print(f"{name}: en R@1 {result['runs'][name]['en']['recall_at_1']:.3f} HR {result['runs'][name]['en']['hr']:.3f} | "
                  f"hi_rom R@1 {hr['recall_at_1']:.3f} HR {hr['hr']:.3f} FHR {hr['fhr']:.3f} "
                  f"medTop1 {hr['median_top1_similarity']:.3f}")
    result["encode_and_eval_seconds"] = round(time.time() - t0, 1)

    # --- the 7 old models, from cached vectors (also feeds the level-vs-order test) ---
    for k in OLD:
        vecs = encode_records(k, all_recs)
        cache, lk = lookups_for(vecs, canon, probes)
        result["old"][k] = {c: rm.summarize(*lk[c][:2]) for c in COND}
        result["level_vs_order"][k] = level_vs_order(cache, vecs, probes)

    # regression guard: the old numbers must equal post 2's published rank_metrics.json
    ref = json.loads((ROOT / "results" / "rank_metrics.json").read_text(encoding="utf-8"))["models"]
    bad = []
    for k in OLD:
        for c in ("en", "hi_rom"):
            a, b = result["old"][k][c], ref[k][c]
            for f in ("n_rank1", "hits"):
                if a[f] != b[f]:
                    bad.append(f"{k}/{c}/{f}: {a[f]} vs {b[f]}")
    if bad:
        sys.exit("Old-model numbers disagree with results/rank_metrics.json:\n  " + "\n  ".join(bad))
    print("old 7 models reproduce results/rank_metrics.json (rank-1 counts, hits)")

    out = ROOT / "results" / "embeddinggemma2.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
