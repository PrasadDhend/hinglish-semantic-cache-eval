"""Cache-decision metrics: false-hit rate, hit rate, precision, ROC/PR curves.

Responsibility: compute these across the threshold sweep defined in
config.Config (0.60-0.98 step 0.01), per model, per condition, per population
mode. Ranking metrics (e.g. nDCG) are explicitly out of scope per PROJECT.md.
Built in Phase 5.

Operates on two parallel per-probe arrays, not on cache/decide objects
directly, so a full threshold sweep never re-runs nearest-neighbour search
(that lookup doesn't depend on the threshold at all -- only hit/no-hit does):

  similarities[i]   -- cosine similarity of probe i to its single nearest
                        cache entry (cache.store.SemanticCache.query)
  correct_match[i]  -- whether that nearest entry's equivalence_class_id
                        equals probe i's true equivalence_class_id

This maps directly onto the four cache.decide.Decision outcomes at any
threshold t:
  TRUE_HIT   (TP): correct_match=True,  similarity >= t
  FALSE_MISS (FN): correct_match=True,  similarity <  t
  FALSE_HIT  (FP): correct_match=False, similarity >= t
  TRUE_MISS  (TN): correct_match=False, similarity <  t

so precision/recall/FPR/TPR at threshold t are the standard binary
classification metrics for "will the nearest cache entry, if served, be the
right one", with "will it be served" as the threshold-gated prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    true_hit: int
    false_hit: int
    true_miss: int
    false_miss: int

    @property
    def n(self) -> int:
        return self.true_hit + self.false_hit + self.true_miss + self.false_miss


def confusion_counts(similarities, correct_match, threshold: float) -> ConfusionCounts:
    sims = np.asarray(similarities, dtype=float)
    correct = np.asarray(correct_match, dtype=bool)
    if sims.shape != correct.shape:
        raise ValueError("similarities and correct_match must be the same length")
    hit = sims >= threshold
    return ConfusionCounts(
        true_hit=int(np.sum(hit & correct)),
        false_hit=int(np.sum(hit & ~correct)),
        true_miss=int(np.sum(~hit & ~correct)),
        false_miss=int(np.sum(~hit & correct)),
    )


def false_hit_rate(counts: ConfusionCounts) -> float:
    """FP / (TP+FP) -- of everything served from cache, the fraction that
    was WRONG. This is the paper's headline reliability metric (RQ1). NaN
    when nothing was served (undefined, not zero -- a threshold so strict
    nothing hits says nothing about correctness of what would have hit)."""
    served = counts.true_hit + counts.false_hit
    return counts.false_hit / served if served else float("nan")


def hit_rate(counts: ConfusionCounts) -> float:
    """(TP+FP) / N -- fraction of probes served from cache at all,
    correctly or not. Cache utilization/coverage, distinct from correctness
    (false_hit_rate)."""
    return (counts.true_hit + counts.false_hit) / counts.n if counts.n else float("nan")


def precision(counts: ConfusionCounts) -> float:
    """TP / (TP+FP) == 1 - false_hit_rate. Kept as its own named function
    since PROJECT.md's Phase 5 prompt and the literature both name it
    separately from FHR."""
    served = counts.true_hit + counts.false_hit
    return counts.true_hit / served if served else float("nan")


def recall(counts: ConfusionCounts) -> float:
    """TP / (TP+FN) -- of probes that DO have a correct cache entry
    available (nearest neighbour is genuinely the right class), the
    fraction actually served. AKA true positive rate; ROC's y-axis."""
    correctly_matchable = counts.true_hit + counts.false_miss
    return counts.true_hit / correctly_matchable if correctly_matchable else float("nan")


def false_positive_rate(counts: ConfusionCounts) -> float:
    """FP / (FP+TN) -- of probes whose nearest entry is WRONG, the fraction
    still served anyway. ROC's x-axis."""
    incorrectly_matched = counts.false_hit + counts.true_miss
    return counts.false_hit / incorrectly_matched if incorrectly_matched else float("nan")


def default_thresholds(config=None) -> list[float]:
    """The project-wide threshold sweep from config.Config (0.60-0.98, step
    0.01 by default), materialized as a list. Rounds each step to avoid
    float-accumulation drift (0.60, 0.61, ..., 0.9800000000000001, ...)."""
    if config is None:
        from config import Config
        config = Config()
    n_steps = round((config.threshold_max - config.threshold_min) / config.threshold_step) + 1
    return [round(config.threshold_min + i * config.threshold_step, 10) for i in range(n_steps)]


def sweep(similarities, correct_match, thresholds=None, config=None) -> list[dict]:
    """Compute confusion counts + every derived metric at each threshold in
    `thresholds` (defaults to default_thresholds(config)). One dict per
    threshold, in the same order as `thresholds`."""
    if thresholds is None:
        thresholds = default_thresholds(config)
    rows = []
    for t in thresholds:
        counts = confusion_counts(similarities, correct_match, t)
        rows.append({
            "threshold": t,
            "true_hit": counts.true_hit,
            "false_hit": counts.false_hit,
            "true_miss": counts.true_miss,
            "false_miss": counts.false_miss,
            "n": counts.n,
            "false_hit_rate": false_hit_rate(counts),
            "hit_rate": hit_rate(counts),
            "precision": precision(counts),
            "recall": recall(counts),
            "false_positive_rate": false_positive_rate(counts),
        })
    return rows


def _trapezoidal_auc(xs: list[float], ys: list[float]) -> float:
    """Trapezoidal area under y(x), after dropping NaNs and collapsing
    duplicate x-values to their upper envelope (max y). The envelope step
    matters: a threshold sweep can genuinely produce several rows sharing
    the same x (e.g. false_positive_rate pinned at 1.0 once the threshold
    is loose enough to hit everything) -- picking the max y rather than
    whatever a plain sort happens to put first is what keeps this the
    correct/conventional ROC-AUC regardless of row order.
    """
    grouped: dict[float, float] = {}
    for x, y in zip(xs, ys):
        if np.isnan(x) or np.isnan(y):
            continue
        prev = grouped.get(x)
        grouped[x] = y if prev is None else max(prev, y)
    if len(grouped) < 2:
        return float("nan")
    xs_sorted = sorted(grouped)
    ys_sorted = [grouped[x] for x in xs_sorted]
    return float(np.trapezoid(ys_sorted, xs_sorted))


def roc_auc(sweep_rows: list[dict]) -> float:
    """Area under the ROC curve (FPR vs TPR/recall), trapezoidal, no
    sklearn dependency. Linear interpolation between two ROC points is a
    real, achievable operating point (a randomized mix of the two
    thresholds), which is what justifies trapezoidal integration here."""
    return _trapezoidal_auc(
        [r["false_positive_rate"] for r in sweep_rows],
        [r["recall"] for r in sweep_rows],
    )


def pr_auc(sweep_rows: list[dict]) -> float:
    """Average precision: sum over successive recall increments (processing
    thresholds from strict to loose) of (recall_k - recall_{k-1}) * precision_k.

    Deliberately NOT trapezoidal-on-resorted-points like roc_auc: linear
    interpolation between two PR points is generally NOT achievable by any
    real classifier (unlike ROC), so naively trapezoidal-integrating a
    Precision-Recall curve is a known-optimistic mistake. This is the
    standard "average precision" definition instead (matches, e.g.,
    sklearn's average_precision_score): recall is non-decreasing as the
    threshold relaxes, so this only credits precision at the moment each
    additional true positive is actually captured, not at thresholds that
    keep recall pinned while precision degrades further (e.g. once every
    correct match has already been captured, further-loosened thresholds
    just add false hits -- that shouldn't retroactively improve the score).
    """
    rows = sorted(
        (r for r in sweep_rows if not np.isnan(r["recall"])),
        key=lambda r: -r["threshold"],
    )
    ap = 0.0
    prev_recall = 0.0
    for r in rows:
        precision_k = 0.0 if np.isnan(r["precision"]) else r["precision"]
        ap += (r["recall"] - prev_recall) * precision_k
        prev_recall = r["recall"]
    return float(ap)
