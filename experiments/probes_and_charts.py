"""Blog companion: paraphrase-only re-analysis, BGE-M3 near-miss probes, and the post's two charts.

Why paraphrase-only: 328/332 English hard negatives are exact copies of a cached
canonical (see results/hard_negative_issue.md), so they add free hits. This
script scores only the 664 paraphrase probes per condition.

Writes results/para_only_summary.json, results/near_misses.json and
results/figures/*.png. Run from the repo root:

    python experiments/probes_and_charts.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from _common import all_records_aligned, build_cache, encode_records, load_by_condition  # noqa: E402
from eval.stats import bootstrap_confusion_metric_ci, paired_bootstrap_confusion_metric_diff  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
THETA, SEED, BUDGET = 0.80, 42, 0.02
SWEEP = np.round(np.arange(0.60, 0.981, 0.01), 2)
ORDER = ["english_centric", "granite_r2", "multilingual_e5", "bge_m3", "labse", "indic_sbert", "hingroberta"]
LABELS = {
    "english_centric": "all-MiniLM-L6-v2", "granite_r2": "Granite R2 97M",
    "multilingual_e5": "multilingual-e5-large", "bge_m3": "BGE-M3",
    "labse": "LaBSE", "indic_sbert": "Indic-SBERT", "hingroberta": "HingRoBERTa*",
}

# Reference palette (dataviz skill), light mode: slot 1 blue, slot 2 orange.
EN_C, HI_C = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
})

triples, by_cond = load_by_condition()
canon_en = by_cond["en"]["canonical"]
text_by_id = {r.record_id: r.text for c in ("en", "hi_deva", "hi_rom") for r in all_records_aligned(by_cond, c)}


def lookups(model_key, probe_cond):
    """English cache; paraphrase probes in `probe_cond`. Index-aligned across conditions."""
    probes = by_cond[probe_cond]["paraphrase"]
    vecs = encode_records(model_key, canon_en + probes)
    cache = build_cache(canon_en, vecs)
    out = []
    for r in probes:
        n = cache.query(vecs[r.record_id])
        out.append((r, n, n.entry.equivalence_class_id == r.equivalence_class_id))
    return out


def rates(sims, correct, t):
    served = sims >= t
    h = served.sum()
    return (float((served & ~correct).sum() / h) if h else float("nan")), float(h / len(sims))


def best_under_budget(sims, correct):
    ok = [(t, *rates(sims, correct, t)) for t in SWEEP]
    ok = [x for x in ok if not np.isnan(x[1]) and x[1] <= BUDGET]
    return max(ok, key=lambda x: x[2]) if ok else None


summary, cached = {}, {}
for k in ORDER:
    en, rom = lookups(k, "en"), lookups(k, "hi_rom")
    cached[k] = (en, rom)
    s_en = np.array([n.similarity for _, n, _ in en]); c_en = np.array([c for _, _, c in en])
    s_hi = np.array([n.similarity for _, n, _ in rom]); c_hi = np.array([c for _, _, c in rom])
    row = {"n_probes": len(en)}
    for tag, s, c in (("en", s_en, c_en), ("hi_rom", s_hi, c_hi)):
        fhr, hr = rates(s, c, THETA)
        row[tag] = {
            "fhr": fhr, "hr": hr, "hits": int((s >= THETA).sum()),
            "fhr_ci": bootstrap_confusion_metric_ci(s, c, threshold=THETA, metric="false_hit_rate", seed=SEED),
            "hr_ci": bootstrap_confusion_metric_ci(s, c, threshold=THETA, metric="hit_rate", seed=SEED),
            "best_theta_under_2pct_fhr": best_under_budget(s, c),
            "nearest_is_correct": int(c.sum()),
            "median_similarity": float(np.median(s)),
        }
    for metric in ("false_hit_rate", "hit_rate"):
        row[f"paired_{metric}_hi_rom_minus_en"] = paired_bootstrap_confusion_metric_diff(
            s_hi, c_hi, s_en, c_en, threshold=THETA, metric=metric, seed=SEED)
    summary[k] = row

(RESULTS / "para_only_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")

# ---------------------------------------------------------------- near misses (BGE-M3)
en_b, rom_b = cached["bge_m3"]
cands = []
for (r, n, ok), (r_en, n_en, ok_en) in zip(rom_b, en_b):
    if ok and n.similarity < THETA:
        cands.append({
            "record_id": r.record_id, "hinglish_probe": r.text, "same_probe_in_english": r_en.text,
            "nearest_cached_english": text_by_id[n.entry.key],
            "similarity_hinglish": round(n.similarity, 4), "similarity_english_version": round(n_en.similarity, 4),
            "english_version_would_hit": bool(ok_en and n_en.similarity >= THETA),
        })
cands.sort(key=lambda d: -d["similarity_hinglish"])
picked, seen = [], set()
for c in cands:
    cat = c["record_id"].split(".")[0]
    if cat not in seen:
        picked.append(c); seen.add(cat)
    if len(picked) == 3:
        break
nm = {"model": "bge_m3", "theta": THETA, "n_probes": len(rom_b),
      "nearest_entry_correct_hinglish": sum(ok for _, _, ok in rom_b),
      "correct_but_below_theta": len(cands), "picked": picked, "next_10_candidates": cands[:10]}
(RESULTS / "near_misses.json").write_text(json.dumps(nm, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- chart 1
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
y = np.arange(len(ORDER))[::-1]
for ax, (title, key) in zip(axes, [("Hit rate: share of queries served from cache", "hr"),
                                   ("False-hit rate: share of served answers that are wrong", "fhr")]):
    for yi, k in zip(y, ORDER):
        a, b = 100 * summary[k]["en"][key], 100 * summary[k]["hi_rom"][key]
        few = key == "fhr" and summary[k]["hi_rom"]["hits"] < 20
        if not (np.isnan(a) or np.isnan(b) or few):
            ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        if not np.isnan(a):
            ax.scatter(a, yi, s=70, color=EN_C, edgecolor="white", linewidth=2, zorder=3)
        if key == "fhr" and np.isnan(b):
            ax.text(a + 4, yi, "Hinglish: 0 hits, nothing served", va="center", fontsize=9.5, color=INK2, style="italic")
        elif few:
            h = summary[k]["hi_rom"]["hits"]
            ax.text(a + 4, yi, f"Hinglish: {round(summary[k]['hi_rom']['fhr'] * h)} wrong of {h} served, too few to rate",
                    va="center", fontsize=9.5, color=INK2, style="italic")
        elif not np.isnan(b):
            ax.scatter(b, yi, s=70, color=HI_C, edgecolor="white", linewidth=2, zorder=4)
    ax.set_xlim(-3, 103)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(y, [LABELS[k] for k in ORDER], color=INK)
bi = ORDER.index("bge_m3")
b = summary["bge_m3"]
axes[0].annotate(f"{100*b['en']['hr']:.1f}% → {100*b['hi_rom']['hr']:.1f}%", (100 * b["hi_rom"]["hr"], y[bi]),
                 xytext=(8, 9), textcoords="offset points", fontsize=9.5, color=INK)
handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=EN_C, label="English query, English cache"),
           plt.Line2D([], [], marker="o", ls="", ms=8, color=HI_C, label="Romanized Hinglish query, English cache")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2, frameon=False, fontsize=10)
fig.text(0.01, 0.01, "Threshold 0.80, 664 paraphrase probes per model. Where the dots overlap near 100%, both conditions are served in full. "
         "*HingRoBERTa: masked LM, mean-pooled, a control.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(FIGURES / "chart1_fhr_hr_by_model.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- chart 2
sim_en = np.array([n.similarity for _, n, _ in en_b])
sim_rom = np.array([n.similarity for _, n, _ in rom_b])
fig, ax = plt.subplots(figsize=(9, 4.6))
bins = np.arange(0.40, 1.001, 0.02)
c_en_h, _, _ = ax.hist(sim_en, bins=bins, histtype="step", lw=2, color=EN_C)
c_hi_h, _, _ = ax.hist(sim_rom, bins=bins, histtype="step", lw=2, color=HI_C)
top = max(c_en_h.max(), c_hi_h.max())
ax.set_ylim(0, top * 1.35)
ax.axvline(THETA, color=INK, lw=1.2, ls=(0, (4, 3)))
ax.text(THETA - 0.006, top * 1.28, "threshold 0.80 →\nserved only to the right", fontsize=9.5, color=INK, va="top", ha="right")
ax.text(np.median(sim_en), c_en_h.max() + top * 0.03, f"English, median {np.median(sim_en):.2f}", color=INK, fontsize=9.5, ha="center", va="bottom")
ax.text(np.median(sim_rom), c_hi_h.max() + top * 0.03, f"Hinglish, median {np.median(sim_rom):.2f}", color=INK, fontsize=9.5, ha="center", va="bottom")
ax.plot([], [], color=EN_C, lw=2, label="English query → English cache")
ax.plot([], [], color=HI_C, lw=2, label="Hinglish query → English cache")
ax.legend(frameon=False, loc="upper left", fontsize=10)
ax.set_xlabel("Cosine similarity to the nearest cached query (BGE-M3)")
ax.set_ylabel("Paraphrase probes")
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("BGE-M3: the Hinglish scores slide left of the threshold", loc="left", fontsize=11.5, color=INK)
fig.tight_layout()
fig.savefig(FIGURES / "chart2_bge_m3_similarity_hist.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- print
pct = lambda v: "  n/a" if np.isnan(v) else f"{100*v:5.1f}"
ci = lambda d: "" if np.isnan(d["ci_low"]) else f"[{100*d['ci_low']:.1f}-{100*d['ci_high']:.1f}]"
for k in ORDER:
    s = summary[k]
    d_f, d_h = s["paired_false_hit_rate_hi_rom_minus_en"], s["paired_hit_rate_hi_rom_minus_en"]
    print(f"{LABELS[k]:22s} en FHR {pct(s['en']['fhr'])} {ci(s['en']['fhr_ci']):13s} HR {pct(s['en']['hr'])} | "
          f"hi_rom FHR {pct(s['hi_rom']['fhr'])} {ci(s['hi_rom']['fhr_ci']):13s} HR {pct(s['hi_rom']['hr'])} hits={s['hi_rom']['hits']:3d} | "
          f"dFHR {pct(d_f['observed_diff'])}pp p={d_f['p_value']:.4f} | dHR {pct(d_h['observed_diff'])}pp p={d_h['p_value']:.4f} | "
          f"2% budget en {s['en']['best_theta_under_2pct_fhr']} hi_rom {s['hi_rom']['best_theta_under_2pct_fhr']}")
print("BGE-M3 hi_rom: nearest correct", nm["nearest_entry_correct_hinglish"], "/", nm["n_probes"], "; correct but below θ", nm["correct_but_below_theta"])
print("BGE-M3 median sim en/hi_rom", round(float(np.median(sim_en)), 3), round(float(np.median(sim_rom)), 3))
for p in picked:
    print(json.dumps(p, ensure_ascii=False))
