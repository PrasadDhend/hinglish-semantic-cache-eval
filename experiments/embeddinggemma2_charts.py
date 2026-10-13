"""Charts + paired bootstrap + probe examples for the EmbeddingGemma 2 post.

Reads results/embeddinggemma2.json (from embeddinggemma2_dims.py) and the cached
vectors. No model is loaded. Writes results/embeddinggemma2_extras.json and
results/figures/chart7-9_*.png.

    python experiments/embeddinggemma2_charts.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import rank_metrics as rm  # noqa: E402
from _common import build_cache, encode_records, load_by_condition  # noqa: E402
from embeddinggemma2_dims import DIMS, truncate  # noqa: E402

FIG = ROOT / "results" / "figures"
EN_C, HI_C, GREY = rm.EN_C, rm.HI_C, "#9a9993"
INK, INK2, GRID = rm.INK, rm.INK2, rm.GRID
THETA = rm.THETA
rng = np.random.default_rng(42)


def per_probe(vec_by_id, canon, probes_c):
    cache = build_cache(canon, vec_by_id)
    rows = [rm.rank_of_correct(cache, vec_by_id[r.record_id], r.equivalence_class_id) for r in probes_c]
    return cache, np.array([x[0] for x in rows]), np.array([x[1] for x in rows]), np.array([x[2] for x in rows])


def paired_boot(a, b, n=10000):
    """CI of mean(a) - mean(b) over paired probes, percentile bootstrap."""
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rm._style(plt)

    _, by = load_by_condition()
    canon = by["en"]["canonical"]
    probes = {c: by[c]["paraphrase"] for c in ("en", "hi_deva", "hi_rom")}
    recs = canon + [r for c in probes for r in probes[c]]
    full = {t: encode_records(f"eg2_{t}_768", recs) for t in ("ss", "np")}
    res = json.loads((ROOT / "results" / "embeddinggemma2.json").read_text(encoding="utf-8"))

    # ---- per-probe arrays on hi_rom
    pp = {}
    pp["ss768"] = per_probe(full["ss"], canon, probes["hi_rom"])
    pp["np768"] = per_probe(full["np"], canon, probes["hi_rom"])
    for k in ("multilingual_e5", "bge_m3", "granite_r2"):
        pp[k] = per_probe(encode_records(k, recs), canon, probes["hi_rom"])
    pp["np128"] = per_probe({i: truncate(v, 128) for i, v in full["np"].items()}, canon, probes["hi_rom"])

    extras = {"bootstrap_recall1_hi_rom": {}, "examples": {}, "prompt_flip": {}}
    r1 = lambda k: (pp[k][1] == 1).astype(float)
    for other in ("multilingual_e5", "bge_m3", "granite_r2"):
        extras["bootstrap_recall1_hi_rom"][f"eg2_ss768_minus_{other}"] = paired_boot(r1("ss768"), r1(other))
    extras["bootstrap_recall1_hi_rom"]["eg2_np768_minus_eg2_ss768"] = paired_boot(r1("np768"), r1("ss768"))
    # false hits among served for EG2 ss768 where correct entry sat at rank 2-5
    ranks, top1 = pp["ss768"][1], pp["ss768"][2]
    served = top1 >= THETA
    wrong = served & (ranks > 1)
    extras["eg2_ss768"] = {"served": int(served.sum()), "wrong": int(wrong.sum()),
                           "wrong_with_correct_at_rank_2_to_5": int((wrong & (ranks <= 5)).sum())}
    # no-prompt vs prompt: probes served under ss but blocked under np
    s_np = pp["np768"][2] >= THETA
    extras["prompt_flip"] = {"served_ss": int(served.sum()), "served_np": int(s_np.sum()),
                             "blocked_under_np_but_rank1_correct": int(((pp["np768"][1] == 1) & ~s_np).sum()),
                             "blocked_under_np_and_rank1_wrong": int(((pp["np768"][1] > 1) & ~s_np).sum())}
    # dims on np: how many probes served at each dim
    for d in DIMS:
        v = {i: truncate(x, d) for i, x in full["np"].items()} if d != 768 else full["np"]
        _, rk, t1, cs = per_probe(v, canon, probes["hi_rom"])
        extras.setdefault("np_by_dim", {})[d] = {"served": int((t1 >= THETA).sum()), "rank1": int((rk == 1).sum()),
                                                  "median_top1": float(np.median(t1)),
                                                  "median_gap_top1_minus_correct_when_wrong": float(np.median((t1 - cs)[rk > 1]))}
    # examples: served-wrong with small gap, ss768
    cache = pp["ss768"][0]
    ex = []
    gap = pp["ss768"][2] - pp["ss768"][3]
    order = np.argsort(gap + np.where(wrong, 0, 9))  # smallest gap among wrong+served
    by_id = {e.equivalence_class_id: e for e in cache.entries}
    for i in order[:60]:
        if not wrong[i]:
            continue
        p = probes["hi_rom"][i]
        sims = cache._matrix @ full["ss"][p.record_id]
        o = np.argsort(-sims)
        canon_text = {c.equivalence_class_id: c.text for c in canon}
        ex.append({"probe": p.text, "served_entry": canon_text[cache.entries[o[0]].equivalence_class_id],
                   "served_sim": float(sims[o[0]]), "correct_rank": int(ranks[i]),
                   "correct_sim": float(pp["ss768"][3][i]), "correct_class": p.equivalence_class_id,
                   "correct_entry": canon_text[p.equivalence_class_id]})
        if len(ex) >= 6:
            break
    extras["examples"]["ss768_served_wrong_smallest_gap"] = ex
    (ROOT / "results" / "embeddinggemma2_extras.json").write_text(json.dumps(extras, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- chart 7: Recall@1 vs hit rate on hi_rom, 8 models
    pts = []
    for k in rm.ORDER:
        s = res["old"][k]["hi_rom"]
        pts.append((rm.LABELS[k], s["recall_at_1"], s["hr"], s["fhr"], False))
    s = res["runs"]["eg2_ss_768"]["hi_rom"]
    pts.append(("EmbeddingGemma 2 (768, SentenceSimilarity prompt)", s["recall_at_1"], s["hr"], s["fhr"], True))
    s = res["runs"]["eg2_np_768"]["hi_rom"]
    pts.append(("EmbeddingGemma 2 (768, no prompt)", s["recall_at_1"], s["hr"], s["fhr"], True))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for name, x, y, f, new in pts:
        ax.scatter(100 * x, 100 * y, s=110, color=HI_C if new else EN_C, edgecolor=INK, linewidth=0.6, zorder=3)
        lab = f"{name}" + ("" if np.isnan(f) else f"  FHR {100*f:.0f}%")
        offs = {"all-MiniLM": (-1.0, 0, "right"), "LaBSE": (1.0, 4.5, "left"),
                "Granite": (-1.0, -5.0, "right"), "multilingual": (1.0, -5.0, "left"),
                "EmbeddingGemma 2 (768, Sent": (1.0, 5.0, "right"),
                "EmbeddingGemma 2 (768, no": (1.0, -4.5, "left")}
        dx, dy, ha = next((v for p, v in offs.items() if name.startswith(p)), (1.0, 0, "left"))
        ax.annotate(lab, (100 * x + dx, 100 * y + dy), fontsize=8.5, color=INK2, ha=ha, va="center")
    ax.set_xlabel("Recall@1 on romanized Hinglish (what a leaderboard reports)")
    ax.set_ylabel("Hit rate at θ=0.80 (what the cache does)")
    ax.set_xlim(35, 95)
    ax.set_ylim(-6, 108)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("EmbeddingGemma 2 has the best Recall@1 of eight models and serves everything", loc="left", fontsize=11.5, color=INK)
    fig.text(0.01, 0.01, "English cache of 166 canonicals, 664 paraphrase probes. Orange: EmbeddingGemma 2 (text-only, float32). *HingRoBERTa: masked LM, a control.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "chart7_eg2_recall1_vs_hitrate.png")
    plt.close(fig)

    # ---- chart 8: dims
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)
    xs = np.arange(len(DIMS))
    for tag, ls, lab in (("ss", "-", "SentenceSimilarity prompt"), ("np", "--", "no prompt")):
        hr = [100 * res["runs"][f"eg2_{tag}_{d}"]["hi_rom"]["hr"] for d in DIMS]
        r1v = [100 * res["runs"][f"eg2_{tag}_{d}"]["hi_rom"]["recall_at_1"] for d in DIMS]
        axs[0].plot(xs, hr, ls, color=HI_C, marker="o", label=lab)
        axs[1].plot(xs, r1v, ls, color=EN_C, marker="o", label=lab)
        for x_, y_ in zip(xs, hr):
            axs[0].annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=INK2)
        for x_, y_ in zip(xs, r1v):
            axs[1].annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=INK2)
    axs[0].set_title("Hit rate at θ=0.80", loc="left", fontsize=11, color=INK)
    axs[1].set_title("Recall@1", loc="left", fontsize=11, color=INK)
    for a in axs:
        a.set_xticks(xs, [str(d) for d in DIMS])
        a.set_xlabel("Vector size (Matryoshka truncation, re-normalized)")
        a.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)
    axs[0].set_ylim(70, 106)
    axs[1].set_ylim(60, 84)
    axs[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.01, 0.01, "Romanized Hinglish, 664 probes, English cache of 166. Same model, same threshold.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "chart8_eg2_dims.png")
    plt.close(fig)

    # ---- chart 9: top-1 similarity distribution, np 768 vs np 128
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(0.5, 1.0, 51)
    ax.hist(pp["np768"][2], bins=bins, color=EN_C, alpha=0.75, label="768 dims")
    ax.hist(pp["np128"][2], bins=bins, color=HI_C, alpha=0.65, label="128 dims")
    ax.axvline(THETA, color=INK, lw=1.2, ls="--")
    ax.text(THETA - 0.004, ax.get_ylim()[1] * 0.93, "θ = 0.80", ha="right", fontsize=9.5, color=INK)
    ax.set_xlabel("Top-1 cosine similarity, 664 romanized Hinglish probes (no prompt)")
    ax.set_ylabel("Probes")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("Truncating the vector slides the whole distribution across the threshold", loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "chart9_eg2_similarity_768_vs_128.png")
    plt.close(fig)
    print(json.dumps(extras, indent=1, ensure_ascii=False)[:3500])


if __name__ == "__main__":
    main()
