"""Statistical inference over experiment results.

Responsibility: bootstrap confidence intervals for metrics in metrics.py, and
paired significance testing for the English-vs-Hinglish gap (and other
condition-pair comparisons). Built in Phase 5.

Everything here is a resampling method (percentile bootstrap), not a
parametric test -- the underlying quantities (false-hit rate, hit rate) are
proportions over a small, non-random-sample probe set (a designed benchmark,
not a natural query stream), so there's no traffic-model assumption to lean
on for a closed-form CI. All resampling is done with config.Config's
random_seed by default, so a re-run reproduces bit-for-bit (Phase 5 gate).
"""

from __future__ import annotations

import numpy as np

from eval.metrics import ConfusionCounts, confusion_counts

_METRIC_FNS = {
    "false_hit_rate": lambda tp, fp, tn, fn, n: fp / (tp + fp),
    "hit_rate": lambda tp, fp, tn, fn, n: (tp + fp) / n,
    "precision": lambda tp, fp, tn, fn, n: tp / (tp + fp),
    "recall": lambda tp, fp, tn, fn, n: tp / (tp + fn),
    "false_positive_rate": lambda tp, fp, tn, fn, n: fp / (fp + tn),
}


def _default_seed() -> int:
    from config import Config
    return Config().random_seed


def bootstrap_ci(values, statistic=np.mean, n_resamples: int = 10000, ci: float = 0.95, seed=None) -> dict:
    """Percentile bootstrap CI for `statistic` applied to a plain array of
    scalar values (e.g. per-probe cosine similarities or cosine gaps from
    eval.geometry). NaNs are dropped before resampling."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"point_estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": 0, "n_resamples": n_resamples, "ci_level": ci}
    rng = np.random.default_rng(seed if seed is not None else _default_seed())
    point = float(statistic(values))
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_stats = statistic(values[idx], axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(resample_stats, [alpha, 1 - alpha])
    return {"point_estimate": point, "ci_low": float(lo), "ci_high": float(hi),
            "n": n, "n_resamples": n_resamples, "ci_level": ci}


def bootstrap_confusion_metric_ci(
    similarities, correct_match, threshold: float, metric: str = "false_hit_rate",
    n_resamples: int = 10000, ci: float = 0.95, seed=None,
) -> dict:
    """Bootstrap CI for one of eval.metrics's confusion-count-derived rates
    (false_hit_rate, hit_rate, precision, recall, false_positive_rate) at a
    fixed threshold. Resamples PROBES (with replacement) and recomputes the
    full confusion matrix from scratch each time, so both the numerator and
    the denominator (e.g. how many probes get served at all) carry their
    resampling uncertainty -- not just the numerator conditional on a fixed
    denominator.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"unknown metric {metric!r}. Valid: {sorted(_METRIC_FNS)}")
    sims = np.asarray(similarities, dtype=float)
    correct = np.asarray(correct_match, dtype=bool)
    if sims.shape != correct.shape:
        raise ValueError("similarities and correct_match must be the same length")
    n = len(sims)
    if n == 0:
        return {"metric": metric, "threshold": threshold, "point_estimate": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": 0,
                "n_resamples": n_resamples, "n_valid_resamples": 0, "ci_level": ci}

    point_counts = confusion_counts(sims, correct, threshold)
    point = _point_metric(point_counts, metric)

    rng = np.random.default_rng(seed if seed is not None else _default_seed())
    idx = rng.integers(0, n, size=(n_resamples, n))
    sims_r = sims[idx]
    correct_r = correct[idx]
    hit_r = sims_r >= threshold
    tp = np.sum(hit_r & correct_r, axis=1).astype(float)
    fp = np.sum(hit_r & ~correct_r, axis=1).astype(float)
    tn = np.sum(~hit_r & ~correct_r, axis=1).astype(float)
    fn = np.sum(~hit_r & correct_r, axis=1).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        vals = _METRIC_FNS[metric](tp, fp, tn, fn, n)
    vals = vals[~np.isnan(vals)]

    alpha = (1 - ci) / 2
    if len(vals):
        lo, hi = np.quantile(vals, [alpha, 1 - alpha])
    else:
        lo, hi = float("nan"), float("nan")
    return {
        "metric": metric, "threshold": threshold, "point_estimate": point,
        "ci_low": float(lo), "ci_high": float(hi), "n": n,
        "n_resamples": n_resamples, "n_valid_resamples": int(len(vals)), "ci_level": ci,
    }


def _point_metric(counts: ConfusionCounts, metric: str) -> float:
    n = counts.n
    if n == 0:
        return float("nan")
    # np.float64, not plain Python float: a 0/0 case (e.g. no probe reaches
    # a strict reference threshold, so nothing was served) must come back
    # as nan like the vectorized bootstrap path does, not raise
    # ZeroDivisionError -- Python float division has no such graceful path.
    with np.errstate(invalid="ignore", divide="ignore"):
        val = _METRIC_FNS[metric](
            np.float64(counts.true_hit), np.float64(counts.false_hit),
            np.float64(counts.true_miss), np.float64(counts.false_miss), np.float64(n),
        )
    return float(val)


def paired_bootstrap_diff(values_a, values_b, statistic=np.mean, n_resamples: int = 10000,
                           ci: float = 0.95, seed=None) -> dict:
    """Two-sided paired bootstrap test for statistic(a) - statistic(b),
    where a[i] and b[i] are the SAME underlying unit measured under two
    conditions (e.g. the same paraphrase's cosine gap under two different
    condition pairs, or a per-probe 0/1 array). Resamples matched pair
    indices jointly so the pairing is preserved every resample.

    Returns the observed difference, a CI on the difference, and a two-sided
    bootstrap p-value (twice the smaller tail's proportion crossing zero,
    capped at 1.0).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("values_a and values_b must be paired (same length)")
    n = len(a)
    if n == 0:
        return {"observed_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "n_pairs": 0, "n_resamples": n_resamples}
    rng = np.random.default_rng(seed if seed is not None else _default_seed())
    observed_diff = float(statistic(a) - statistic(b))
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = statistic(a[idx], axis=1) - statistic(b[idx], axis=1)
    alpha = (1 - ci) / 2
    ci_low, ci_high = np.quantile(diffs, [alpha, 1 - alpha])
    p_left = float(np.mean(diffs <= 0))
    p_right = float(np.mean(diffs >= 0))
    p_value = min(1.0, 2 * min(p_left, p_right))
    return {
        "observed_diff": observed_diff, "ci_low": float(ci_low), "ci_high": float(ci_high),
        "p_value": p_value, "n_pairs": n, "n_resamples": n_resamples, "ci_level": ci,
    }


def paired_bootstrap_confusion_metric_diff(
    similarities_a, correct_match_a, similarities_b, correct_match_b,
    threshold: float, metric: str = "false_hit_rate",
    n_resamples: int = 10000, ci: float = 0.95, seed=None,
) -> dict:
    """Paired bootstrap test for metric(a) - metric(b) at a fixed threshold,
    where probe i under condition a and probe i under condition b are the
    SAME underlying paraphrase/hard negative (e.g. hi_rom vs en) -- each
    resample draws the same pair-index set for both conditions, preserving
    the pairing, and recomputes both confusion matrices from scratch.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"unknown metric {metric!r}. Valid: {sorted(_METRIC_FNS)}")
    sims_a = np.asarray(similarities_a, dtype=float)
    correct_a = np.asarray(correct_match_a, dtype=bool)
    sims_b = np.asarray(similarities_b, dtype=float)
    correct_b = np.asarray(correct_match_b, dtype=bool)
    if not (sims_a.shape == correct_a.shape == sims_b.shape == correct_b.shape):
        raise ValueError("all four arrays must be paired (same length)")
    n = len(sims_a)
    if n == 0:
        return {"metric": metric, "threshold": threshold, "observed_diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"),
                "n_pairs": 0, "n_resamples": n_resamples}

    point_a = _point_metric(confusion_counts(sims_a, correct_a, threshold), metric)
    point_b = _point_metric(confusion_counts(sims_b, correct_b, threshold), metric)
    observed_diff = point_a - point_b

    rng = np.random.default_rng(seed if seed is not None else _default_seed())
    idx = rng.integers(0, n, size=(n_resamples, n))

    def _resampled_metric(sims, correct):
        sims_r = sims[idx]
        correct_r = correct[idx]
        hit_r = sims_r >= threshold
        tp = np.sum(hit_r & correct_r, axis=1).astype(float)
        fp = np.sum(hit_r & ~correct_r, axis=1).astype(float)
        tn = np.sum(~hit_r & ~correct_r, axis=1).astype(float)
        fn = np.sum(~hit_r & correct_r, axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return _METRIC_FNS[metric](tp, fp, tn, fn, n)

    vals_a = _resampled_metric(sims_a, correct_a)
    vals_b = _resampled_metric(sims_b, correct_b)
    diffs = vals_a - vals_b
    valid = ~(np.isnan(diffs))
    diffs = diffs[valid]

    alpha = (1 - ci) / 2
    if len(diffs):
        ci_low, ci_high = np.quantile(diffs, [alpha, 1 - alpha])
        p_left = float(np.mean(diffs <= 0))
        p_right = float(np.mean(diffs >= 0))
        p_value = min(1.0, 2 * min(p_left, p_right))
    else:
        ci_low = ci_high = p_value = float("nan")

    return {
        "metric": metric, "threshold": threshold,
        "point_a": float(point_a) if not np.isnan(point_a) else float("nan"),
        "point_b": float(point_b) if not np.isnan(point_b) else float("nan"),
        "observed_diff": float(observed_diff) if not np.isnan(observed_diff) else float("nan"),
        "ci_low": float(ci_low), "ci_high": float(ci_high), "p_value": p_value,
        "n_pairs": n, "n_resamples": n_resamples, "n_valid_resamples": int(len(diffs)), "ci_level": ci,
    }
