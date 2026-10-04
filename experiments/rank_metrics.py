"""Blog companion (series 2/4): ranking metrics vs cache decisions, on the same lookups.

Would the retrieval-benchmark view of an embedding model (Recall@k, MRR, nDCG)
have picked the right model for a semantic cache serving romanized Hinglish?
Same dataset, same seven models, same cached vectors and the same
paraphrase-only probes as experiments/probes_and_charts.py: an English cache of
166 canonicals, probed with the 664 paraphrases per condition.

This is blog-side analysis. src/eval/metrics.py declares ranking metrics out of
scope for the paper, and src/cache/store.py's query() returns only the nearest
entry by design, so the ranked lookup lives here and reads the cache's existing
matrix. Neither file is changed.

The cache holds exactly one entry per equivalence class, so every probe has one
relevant item. Recall@1 == Precision@1 == top-1 accuracy, and every ranking
metric below is a function of the rank of the correct entry alone. The cache
decision reads only the top-1 similarity. main() checks the rank-1 counts, hits
and median top-1 similarity against results/para_only_summary.json and stops if
the two scripts disagree.

Writes results/rank_metrics.json, results/rank_probes.json and
results/figures/chart3-6_*.png. Run from the repo root, after
probes_and_charts.py:

    python experiments/rank_metrics.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
THETA, SEED = 0.80, 42
KS = (1, 5, 10)
NDCG_K = 5
CONDITIONS = ("en", "hi_deva", "hi_rom")
SCATTER_MODEL = "bge_m3"  # chart 6: the model a reader would actually deploy
MIN_HITS = 20  # below this, an FHR is a count, not a rate (same cutoff as probes_and_charts.py chart 1)
ORDER = ["english_centric", "granite_r2", "multilingual_e5", "bge_m3", "labse", "indic_sbert", "hingroberta"]
LABELS = {
    "english_centric": "all-MiniLM-L6-v2", "granite_r2": "Granite R2 97M",
    "multilingual_e5": "multilingual-e5-large", "bge_m3": "BGE-M3",
    "labse": "LaBSE", "indic_sbert": "Indic-SBERT", "hingroberta": "HingRoBERTa*",
}
BUCKETS = (("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), (">20", 21, None))

# Same palette as probes_and_charts.py (dataviz skill, light mode).
EN_C, HI_C = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- metrics (pure, unit-tested)

def ranked_similarities(cache, probe_vector):
    """Similarity of `probe_vector` to every cache entry, plus entry indices best-first.

    Same float32 matvec as SemanticCache.query(). The stable sort keeps the
    lower index first on ties, which is what np.argmax does, so order[0] is
    always the entry query() returns."""
    sims = cache._matrix @ np.asarray(probe_vector, dtype=np.float32)
    return sims, np.argsort(-sims, kind="stable")


def rank_of_correct(cache, probe_vector, equivalence_class_id):
    """(1-based rank of the correct entry, top-1 similarity, correct entry's similarity).

    The top-1 similarity is the number the cache compares against θ; the rank
    is the only thing the ranking metrics read."""
    idx = next((i for i, e in enumerate(cache.entries) if e.equivalence_class_id == equivalence_class_id), None)
    if idx is None:
        raise KeyError(f"no cache entry for equivalence class {equivalence_class_id!r}")
    sims, order = ranked_similarities(cache, probe_vector)
    rank = int(np.flatnonzero(order == idx)[0]) + 1
    return rank, float(sims[order[0]]), float(sims[idx])


def recall_at_k(ranks, k):
    return float(np.mean(np.asarray(ranks) <= k))


def mrr(ranks):
    return float(np.mean(1.0 / np.asarray(ranks, dtype=float)))


def ndcg_at_k(ranks, k):
    """One relevant item per probe, so the ideal DCG is 1 and nDCG@k is
    1/log2(rank+1) when the correct entry is in the top k, else 0."""
    r = np.asarray(ranks, dtype=float)
    return float(np.mean(np.where(r <= k, 1.0 / np.log2(r + 1), 0.0)))


def rank_buckets(ranks):
    r = np.asarray(ranks)
    return {name: float(np.mean((r >= lo) & (r <= (hi if hi is not None else np.inf)))) for name, lo, hi in BUCKETS}


def cache_decision(ranks, top1_sims, theta):
    """Hit rate and false-hit rate at θ from the same lookups. A served probe
    is a false hit when its correct entry was not ranked first."""
    r, s = np.asarray(ranks), np.asarray(top1_sims)
    served = s >= theta
    hits = int(served.sum())
    return {
        "hits": hits,
        "true_hits": int((served & (r == 1)).sum()),
        "hr": hits / len(r),
        "fhr": float((served & (r > 1)).sum() / hits) if hits else float("nan"),
        "rank1_not_served": int(((r == 1) & ~served).sum()),
    }


def relative_drop(a, b):
    """(a - b) / a: the share of the English value lost under Hinglish."""
    return float((a - b) / a) if a else float("nan")


def summarize(ranks, top1_sims, theta=THETA):
    r = np.asarray(ranks)
    out = {"n_probes": len(r), "n_rank1": int((r == 1).sum())}
    out.update({f"recall_at_{k}": recall_at_k(r, k) for k in KS})
    out["mrr"] = mrr(r)
    out[f"ndcg_at_{NDCG_K}"] = ndcg_at_k(r, NDCG_K)
    out["median_rank"] = float(np.median(r))
    out["rank_buckets"] = rank_buckets(r)
    out["median_top1_similarity"] = float(np.median(top1_sims))
    out.update(cache_decision(r, top1_sims, theta))
    return out


# ---------------------------------------------------------------- run

def run_lookups():
    """{model: {condition: (record_ids, ranks, top1_sims, correct_sims)}}, English cache throughout."""
    sys.path.insert(0, str(ROOT / "experiments"))
    sys.path.insert(0, str(ROOT / "src"))
    from _common import build_cache, encode_records, load_by_condition

    _, by_cond = load_by_condition()
    canon_en = by_cond["en"]["canonical"]
    out = {}
    for k in ORDER:
        probes = {c: by_cond[c]["paraphrase"] for c in CONDITIONS}
        vecs = encode_records(k, canon_en + [r for c in CONDITIONS for r in probes[c]])
        cache = build_cache(canon_en, vecs)
        out[k] = {}
        for c in CONDITIONS:
            rows = [rank_of_correct(cache, vecs[r.record_id], r.equivalence_class_id) for r in probes[c]]
            out[k][c] = ([r.record_id for r in probes[c]],
                         np.array([x[0] for x in rows]), np.array([x[1] for x in rows]), np.array([x[2] for x in rows]))
        print(f"  {k}: done")
    return out


def check_against_post1(metrics):
    """Stop if this script's lookups disagree with probes_and_charts.py's."""
    path = RESULTS / "para_only_summary.json"
    if not path.exists():
        sys.exit(f"{path} not found: run experiments/probes_and_charts.py first")
    old = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for k in ORDER:
        for c in ("en", "hi_rom"):
            new, ref = metrics[k][c], old[k][c]
            if new["n_rank1"] != ref["nearest_is_correct"]:
                bad.append(f"{k}/{c}: rank-1 count {new['n_rank1']} vs nearest_is_correct {ref['nearest_is_correct']}")
            if new["hits"] != ref["hits"]:
                bad.append(f"{k}/{c}: hits {new['hits']} vs {ref['hits']}")
            if abs(new["median_top1_similarity"] - ref["median_similarity"]) > 1e-6:
                bad.append(f"{k}/{c}: median similarity {new['median_top1_similarity']} vs {ref['median_similarity']}")
    if bad:
        sys.exit("Disagrees with para_only_summary.json:\n  " + "\n  ".join(bad))
    print("  matches para_only_summary.json (rank-1 counts, hits, median top-1 similarity)")


# ---------------------------------------------------------------- charts

def _style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
    })


def chart3_relative_drop(plt, metrics):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    y = np.arange(len(ORDER))[::-1]
    h = 0.36
    for yi, k in zip(y, ORDER):
        d = metrics[k]["relative_drop_hi_rom_vs_en"]
        for off, key, col in ((h / 2, "recall_at_1", EN_C), (-h / 2, "hr", HI_C)):
            v = d[key]
            if np.isnan(v):
                continue
            ax.barh(yi + off, 100 * v, height=h, color=col)
            ax.text(100 * v + (1.5 if v >= 0 else -1.5), yi + off, f"{100*v:.0f}%", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [LABELS[k] for k in ORDER], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-10, 112)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Relative drop, English → romanized Hinglish (share of the English value lost)")
    ax.set_title("Same lookups: the ranking metric slides, the cache decision falls off", loc="left", fontsize=11.5, color=INK)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=EN_C), plt.Rectangle((0, 0), 1, 1, color=HI_C)],
              ["Recall@1 (what a retrieval leaderboard reports)", f"Hit rate at θ={THETA:.2f} (what the cache does)"],
              frameon=False, loc="lower right", fontsize=9.5)
    fig.text(0.01, 0.01, "English cache of 166 canonicals, 664 paraphrase probes per condition. *HingRoBERTa: masked LM, a control.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "chart3_relative_drop_recall_vs_hitrate.png", facecolor="white")
    plt.close(fig)


def chart4_recall_vs_hitrate(plt, metrics):
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    cmap = plt.get_cmap("Oranges")
    norm = plt.Normalize(0, 0.6)
    # hand-placed so neighbouring labels don't collide: (dx, dy, ha) in points
    offsets = {"multilingual_e5": (0, 11, "center"), "granite_r2": (-8, -18, "right"), "hingroberta": (9, 7, "left"),
               "bge_m3": (10, -4, "left"), "indic_sbert": (9, 7, "left"), "labse": (0, 11, "center"),
               "english_centric": (0, -20, "center")}
    for k in ORDER:
        m = metrics[k]["hi_rom"]
        x, yv, f = 100 * m["recall_at_1"], 100 * m["hr"], m["fhr"]
        if m["hits"] < MIN_HITS:
            ax.scatter(x, yv, s=110, facecolor="white", edgecolor=INK, linewidth=1.4, zorder=3)
            wrong = m["hits"] - m["true_hits"]
            note = " (nothing served)" if not m["hits"] else f"\n{m['hits']} served, {wrong} wrong: too few to rate"
        else:
            ax.scatter(x, yv, s=110, color=cmap(norm(f)), edgecolor=INK, linewidth=1, zorder=3)
            note = f" (FHR {100*f:.0f}%)"
        dx, dy, ha = offsets[k]
        ax.annotate(LABELS[k] + note, (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK, ha=ha)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="False-hit rate (share of served answers that are wrong)", format=lambda v, _: f"{100*v:.0f}%")
    ax.set_xlabel("Recall@1 on romanized Hinglish (retrieval leaderboard axis)")
    ax.set_ylabel(f"Hit rate at θ={THETA:.2f} (share of queries served)")
    ax.set_xlim(30, 80)
    ax.set_ylim(-12, 110)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Romanized Hinglish: the leaderboard axis says little about the deploy axis", loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "chart4_recall_at_1_vs_hit_rate_hi_rom.png", facecolor="white")
    plt.close(fig)


def chart5_rank_buckets(plt, metrics):
    order = sorted(ORDER, key=lambda k: metrics[k]["hi_rom"]["recall_at_1"])  # best Recall@1 at the top
    fig, ax = plt.subplots(figsize=(10.5, 5))
    y = np.arange(len(order))
    alphas = {"1": 1.0, "2-5": 0.55, "6-20": 0.3}
    for yi, k in zip(y, order):
        m = metrics[k]["hi_rom"]
        left = 0.0
        for name, _, _ in BUCKETS:
            w = 100 * m["rank_buckets"][name]
            ax.barh(yi, w, left=left, height=0.62, color=HI_C if name in alphas else GRID,
                    alpha=alphas.get(name, 1.0), edgecolor="white", linewidth=0.8)
            left += w
        ax.barh(yi, 100 * m["true_hits"] / m["n_probes"], height=0.62, fill=False, hatch="////",
                edgecolor=INK, linewidth=0)
        if not m["hits"]:
            fhr = ""
        elif m["hits"] < MIN_HITS:
            fhr = f" ({m['hits']} probes, {m['hits'] - m['true_hits']} wrong)"
        else:
            fhr = f", {100*m['fhr']:.0f}% of those wrong"
        ax.text(101.5, yi, f"served {100*m['hr']:.1f}%{fhr}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [LABELS[k] for k in order], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 140)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("Share of the 664 romanized-Hinglish paraphrase probes")
    ax.set_title("Where the correct cached question ranked, and how much of rank 1 was served", loc="left", fontsize=11.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HI_C, alpha=alphas.get(n, 1.0)) if n in alphas
               else plt.Rectangle((0, 0), 1, 1, color=GRID) for n, _, _ in BUCKETS]
    handles.append(plt.Rectangle((0, 0), 1, 1, fill=False, hatch="////", edgecolor=INK, linewidth=0))
    ax.legend(handles, [f"rank {n}" for n, _, _ in BUCKETS] + [f"rank 1 and served at θ={THETA:.2f}"],
              frameon=False, ncol=5, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.13))
    fig.tight_layout()
    fig.savefig(FIGURES / "chart5_rank_buckets_hi_rom.png", facecolor="white")
    plt.close(fig)


def chart6_rank_vs_similarity(plt, lookups):
    _, ranks, top1, _ = lookups[SCATTER_MODEL]["hi_rom"]
    rng = np.random.default_rng(SEED)
    x = ranks * np.exp(rng.uniform(-0.12, 0.12, len(ranks)))  # jitter so rank 1 isn't one vertical line
    first = ranks == 1
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(x[~first], top1[~first], s=12, color=INK2, alpha=0.45, linewidth=0, label="correct entry ranked 2nd or lower")
    ax.scatter(x[first], top1[first], s=12, color=HI_C, alpha=0.6, linewidth=0, label="correct entry ranked 1st")
    ax.axhline(THETA, color=INK, lw=1.2, ls=(0, (4, 3)))
    below = int((first & (top1 < THETA)).sum())
    ax.text(0.99, THETA - 0.008, f"θ={THETA:.2f}: served above, not below. {below} of {int(first.sum())} rank-1 probes sit below it.",
            fontsize=9.5, color=INK, va="top", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100) if t <= ranks.max()]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Rank of the correct cached question (log scale, jittered)")
    ax.set_ylabel("Top-1 similarity (what the cache compares to θ)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1.0), fontsize=9.5, markerscale=2)
    ax.set_title(f"{LABELS[SCATTER_MODEL]}, romanized Hinglish: ranked right, scored too low to serve",
                 loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / f"chart6_{SCATTER_MODEL}_rank_vs_similarity.png", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Ranked lookups (English cache, paraphrase probes) ...")
    lookups = run_lookups()
    metrics = {}
    for k in ORDER:
        metrics[k] = {c: summarize(lookups[k][c][1], lookups[k][c][2]) for c in CONDITIONS}
        en, hi = metrics[k]["en"], metrics[k]["hi_rom"]
        metrics[k]["relative_drop_hi_rom_vs_en"] = {
            key: relative_drop(en[key], hi[key]) for key in ("recall_at_1", "recall_at_5", "mrr", f"ndcg_at_{NDCG_K}", "hr")
        }
    check_against_post1(metrics)

    out = {"theta": THETA, "cache": "en canonicals (166)", "probes": "paraphrases only (664 per condition)",
           "note": "one relevant item per probe: Recall@1 == Precision@1 == top-1 accuracy", "models": metrics}
    (RESULTS / "rank_metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    per_probe = {k: {c: {"record_id": lookups[k][c][0], "rank": lookups[k][c][1].tolist(),
                         "top1_similarity": np.round(lookups[k][c][2], 5).tolist(),
                         "correct_similarity": np.round(lookups[k][c][3], 5).tolist()} for c in CONDITIONS} for k in ORDER}
    (RESULTS / "rank_probes.json").write_text(json.dumps(per_probe), encoding="utf-8")

    chart3_relative_drop(plt, metrics)
    chart4_recall_vs_hitrate(plt, metrics)
    chart5_rank_buckets(plt, metrics)
    chart6_rank_vs_similarity(plt, lookups)

    pct = lambda v: "  n/a" if np.isnan(v) else f"{100*v:5.1f}"
    for c in ("en", "hi_rom"):
        print(f"\n{c}: sorted by Recall@1 (the leaderboard view), cache decision at theta={THETA:.2f} alongside")
        print(f"  {'model':22s}   R@1   R@5  R@10   MRR nDCG@5 |    HR   FHR  rank1-unserved")
        for k in sorted(ORDER, key=lambda k: -metrics[k][c]["recall_at_1"]):
            m = metrics[k][c]
            print(f"  {LABELS[k]:22s} {pct(m['recall_at_1'])} {pct(m['recall_at_5'])} {pct(m['recall_at_10'])} "
                  f"{m['mrr']:5.3f}  {m[f'ndcg_at_{NDCG_K}']:5.3f} | {pct(m['hr'])} {pct(m['fhr'])}  {m['rank1_not_served']:3d}/{m['n_probes']}")
    print("\nRelative drop en -> hi_rom:")
    for k in ORDER:
        d = metrics[k]["relative_drop_hi_rom_vs_en"]
        print(f"  {LABELS[k]:22s} " + "  ".join(f"{key} {pct(v)}%" for key, v in d.items()))
    print(f"\nWrote {RESULTS / 'rank_metrics.json'}, {RESULTS / 'rank_probes.json'} and charts 3-6 in {FIGURES}")


if __name__ == "__main__":
    main()
