"""Experiment 2 — three-way degradation, all models, both population modes.

For each model in the lineup:

  same-condition population (control): cache populated from condition X's
    own canonicals, probed with condition X's own paraphrases + hard
    negatives, for X in {en, hi_deva, hi_rom}. No cross-lingual mismatch --
    establishes each condition's own ceiling.

  cross-condition population (the HEADLINE -- matches real deployments,
    where the cache holds English-answered queries): cache populated from
    English canonicals only, probed separately with hi_deva and hi_rom
    paraphrases + hard negatives. (English probed against an English cache
    is identical to the same-condition en run, so it's not duplicated here.)

Also runs a paired bootstrap significance test (eval.stats) on
false_hit_rate between each cross-condition run and the same-condition
English baseline, at SIGNIFICANCE_THRESHOLD. This is a genuinely paired
test, not an approximation: dataset.build's three-way join guarantees probe
i under hi_rom and probe i under en are the same underlying
paraphrase/hard negative, just rendered differently (see _common.py's
load_by_condition docstring for exactly how that alignment is preserved).

Answers RQ1 ("how much does cache-hit reliability degrade under romanized
Hinglish"). Reports numbers only -- no interpretation (PROJECT.md working
rule 5; this phase's explicit "report what you find without interpreting it
yet").

Writes results/raw/exp02_threeway_degradation.json.
"""

import json
import time

from _common import MODEL_KEYS, build_cache, encode_records, load_by_condition, probe_similarities_and_correctness
from config import Config
from eval.metrics import default_thresholds, pr_auc, roc_auc, sweep
from eval.stats import bootstrap_confusion_metric_ci, paired_bootstrap_confusion_metric_diff

REFERENCE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]
SIGNIFICANCE_THRESHOLD = 0.80  # fixed operating point for the paired gap test


def _population_run(cache_canonical_records, probe_records, vec_by_id, thresholds, seed):
    cache = build_cache(cache_canonical_records, vec_by_id)
    sims, correct, record_ids = probe_similarities_and_correctness(cache, probe_records, vec_by_id)
    sweep_rows = sweep(sims, correct, thresholds=thresholds)
    summary = {
        "n_canonical": len(cache_canonical_records),
        "n_probes": len(probe_records),
        "roc_auc": roc_auc(sweep_rows),
        "pr_auc": pr_auc(sweep_rows),
        "threshold_sweep": sweep_rows,
        "false_hit_rate_bootstrap_ci": {
            str(t): bootstrap_confusion_metric_ci(sims, correct, threshold=t, metric="false_hit_rate", seed=seed)
            for t in REFERENCE_THRESHOLDS
        },
    }
    return sims, correct, record_ids, summary


def main() -> None:
    cfg = Config()
    thresholds = default_thresholds(cfg)

    print("Loading + validating the three-way parallel dataset ...")
    triples, by_condition = load_by_condition(cfg)
    print(f"  {len(triples)} aligned triples across en/hi_deva/hi_rom")

    results = {}
    for model_key in MODEL_KEYS:
        print(f"\n=== {model_key} ({cfg.embedding_models[model_key]}) ===")
        t0 = time.time()
        all_records = []
        for cond in ("en", "hi_deva", "hi_rom"):
            all_records += (
                by_condition[cond]["canonical"]
                + by_condition[cond]["paraphrase"]
                + by_condition[cond]["hard_negative"]
            )
        vec_by_id = encode_records(model_key, all_records, cfg)
        encode_s = time.time() - t0
        print(f"  encoded {len(all_records)} texts (all 3 conditions) in {encode_s:.1f}s")

        model_result = {
            "model_id": cfg.embedding_models[model_key],
            "encode_seconds": round(encode_s, 3),
            "same_condition": {},
            "cross_condition": {},
        }
        probe_data = {}  # (mode, condition) -> (sims, correct, record_ids)

        for cond in ("en", "hi_deva", "hi_rom"):
            canon = by_condition[cond]["canonical"]
            probes = by_condition[cond]["paraphrase"] + by_condition[cond]["hard_negative"]
            sims, correct, record_ids, summary = _population_run(canon, probes, vec_by_id, thresholds, cfg.random_seed)
            model_result["same_condition"][cond] = summary
            probe_data[("same", cond)] = (sims, correct, record_ids)
            row80 = next(r for r in summary["threshold_sweep"] if abs(r["threshold"] - 0.80) < 1e-9)
            print(f"  same-condition  {cond:8s}: @0.80 hit_rate={row80['hit_rate']:.1%}  "
                  f"false_hit_rate={row80['false_hit_rate']:.1%}")

        en_canon = by_condition["en"]["canonical"]
        for probe_cond in ("hi_deva", "hi_rom"):
            probes = by_condition[probe_cond]["paraphrase"] + by_condition[probe_cond]["hard_negative"]
            sims, correct, record_ids, summary = _population_run(en_canon, probes, vec_by_id, thresholds, cfg.random_seed)
            summary["cache_condition"] = "en"
            summary["probe_condition"] = probe_cond
            model_result["cross_condition"][probe_cond] = summary
            probe_data[("cross", probe_cond)] = (sims, correct, record_ids)
            row80 = next(r for r in summary["threshold_sweep"] if abs(r["threshold"] - 0.80) < 1e-9)
            print(f"  cross-condition en-cache <- {probe_cond:8s}: @0.80 hit_rate={row80['hit_rate']:.1%}  "
                  f"false_hit_rate={row80['false_hit_rate']:.1%}")

        # paired significance: cross-condition Hindi/Hinglish FHR vs the
        # same-condition English baseline, at a fixed reference threshold.
        # probe_data lists are already row-aligned across conditions (see
        # _common.load_by_condition) -- no record_id remapping needed.
        sims_en, correct_en, _ = probe_data[("same", "en")]
        significance = {}
        for probe_cond in ("hi_deva", "hi_rom"):
            sims_x, correct_x, _ = probe_data[("cross", probe_cond)]
            significance[probe_cond] = paired_bootstrap_confusion_metric_diff(
                sims_x, correct_x, sims_en, correct_en,
                threshold=SIGNIFICANCE_THRESHOLD, metric="false_hit_rate", seed=cfg.random_seed,
            )
        model_result["paired_significance_vs_english_same_condition"] = {
            "note": "point_a = cross-condition (hi_deva/hi_rom probing an EN cache), "
                    "point_b = same-condition English baseline. observed_diff = point_a - point_b.",
            "threshold": SIGNIFICANCE_THRESHOLD,
            **significance,
        }
        results[model_key] = model_result

    out = {
        "experiment": "exp02_threeway_degradation",
        "reference_thresholds": REFERENCE_THRESHOLDS,
        "significance_threshold": SIGNIFICANCE_THRESHOLD,
        "random_seed": cfg.random_seed,
        "models": results,
    }
    cfg.results_raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_raw_dir / "exp02_threeway_degradation.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
