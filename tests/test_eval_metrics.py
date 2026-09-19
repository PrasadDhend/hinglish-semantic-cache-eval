"""Unit tests for src/eval/metrics.py against hand-constructed confusion
data, so a metrics bug can't silently corrupt every downstream experiment
result before it's caught."""

import numpy as np
import pytest

from config import Config
from eval.metrics import (
    confusion_counts, false_hit_rate, hit_rate, precision, recall,
    false_positive_rate, default_thresholds, sweep, roc_auc, pr_auc,
)


def test_confusion_counts_hand_values():
    # 5 probes: sim, correct_match
    sims = [0.9, 0.9, 0.5, 0.5, 0.5]
    correct = [True, False, True, False, False]
    counts = confusion_counts(sims, correct, threshold=0.8)
    assert counts.true_hit == 1    # sim=0.9, correct=True
    assert counts.false_hit == 1   # sim=0.9, correct=False
    assert counts.false_miss == 1  # sim=0.5, correct=True
    assert counts.true_miss == 2   # sim=0.5, correct=False (x2)
    assert counts.n == 5


def test_mismatched_length_raises():
    with pytest.raises(ValueError):
        confusion_counts([0.9, 0.5], [True], threshold=0.8)


def test_boundary_threshold_is_a_hit():
    counts = confusion_counts([0.8], [True], threshold=0.8)
    assert counts.true_hit == 1
    assert counts.false_miss == 0


@pytest.mark.parametrize("counts_kwargs,expected", [
    ({"true_hit": 3, "false_hit": 1, "true_miss": 0, "false_miss": 0}, 0.25),
    ({"true_hit": 0, "false_hit": 0, "true_miss": 5, "false_miss": 5}, float("nan")),  # nothing served
])
def test_false_hit_rate(counts_kwargs, expected):
    from eval.metrics import ConfusionCounts
    result = false_hit_rate(ConfusionCounts(**counts_kwargs))
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_precision_is_complement_of_false_hit_rate():
    from eval.metrics import ConfusionCounts
    counts = ConfusionCounts(true_hit=7, false_hit=3, true_miss=4, false_miss=6)
    assert precision(counts) == pytest.approx(1 - false_hit_rate(counts))


def test_hit_rate_counts_all_served_regardless_of_correctness():
    from eval.metrics import ConfusionCounts
    counts = ConfusionCounts(true_hit=3, false_hit=2, true_miss=1, false_miss=4)
    assert hit_rate(counts) == pytest.approx(5 / 10)


def test_recall_and_fpr_hand_values():
    from eval.metrics import ConfusionCounts
    counts = ConfusionCounts(true_hit=8, false_hit=2, true_miss=18, false_miss=2)
    assert recall(counts) == pytest.approx(8 / 10)          # TP / (TP+FN)
    assert false_positive_rate(counts) == pytest.approx(2 / 20)  # FP / (FP+TN)


def test_default_thresholds_matches_config_range():
    cfg = Config()
    thresholds = default_thresholds(cfg)
    assert thresholds[0] == pytest.approx(cfg.threshold_min)
    assert thresholds[-1] == pytest.approx(cfg.threshold_max)
    assert len(thresholds) == round((cfg.threshold_max - cfg.threshold_min) / cfg.threshold_step) + 1
    # no float drift: every consecutive gap is exactly threshold_step
    for a, b in zip(thresholds, thresholds[1:]):
        assert (b - a) == pytest.approx(cfg.threshold_step)


def test_sweep_length_matches_thresholds():
    sims = [0.5, 0.6, 0.7, 0.8, 0.9]
    correct = [True, True, False, True, False]
    thresholds = [0.5, 0.7, 0.9]
    rows = sweep(sims, correct, thresholds=thresholds)
    assert [r["threshold"] for r in rows] == thresholds
    assert all(r["n"] == 5 for r in rows)


def test_sweep_lower_threshold_never_decreases_hit_rate():
    rng = np.random.default_rng(0)
    sims = rng.random(200).tolist()
    correct = (rng.random(200) > 0.5).tolist()
    thresholds = sorted(default_thresholds(Config()))
    rows = sweep(sims, correct, thresholds=thresholds)
    hit_rates = [r["hit_rate"] for r in rows]
    # hit_rate must be monotonically non-increasing as threshold rises
    assert all(a >= b - 1e-12 for a, b in zip(hit_rates, hit_rates[1:]))


def test_roc_auc_perfect_separation_is_one():
    # every correct-match probe has higher similarity than every incorrect
    # one; sweep a dense, full-range threshold grid (not the project's
    # restricted 0.60-0.98 operating range) so both FPR and TPR actually
    # trace out their full [0, 1] span -- a coarse or narrow-range sweep
    # under-integrates the curve without metrics.py itself being wrong
    # (see test_roc_auc_random_guess_is_near_half's comment for the same
    # point from the other direction).
    sims = [0.95, 0.90, 0.85, 0.30, 0.20, 0.10]
    correct = [True, True, True, False, False, False]
    thresholds = np.linspace(0.0, 1.0, 201).tolist()
    rows = sweep(sims, correct, thresholds=thresholds)
    assert roc_auc(rows) == pytest.approx(1.0, abs=0.02)


def test_pr_auc_perfect_separation_is_one():
    sims = [0.95, 0.90, 0.85, 0.30, 0.20, 0.10]
    correct = [True, True, True, False, False, False]
    thresholds = np.linspace(0.0, 1.0, 201).tolist()
    rows = sweep(sims, correct, thresholds=thresholds)
    assert pr_auc(rows) == pytest.approx(1.0, abs=0.02)


def test_roc_auc_random_guess_is_near_half():
    # AUC=0.5 for random guessing is a full-[0,1]-range property of the ROC
    # curve. config.Config's default threshold sweep (0.60-0.98) is
    # deliberately restricted to plausible cache-operating points, not the
    # full similarity range, so integrating over it gives the area under
    # only a narrow slice of the curve, not 0.5 -- that's expected, and
    # exercised separately by test_sweep_lower_threshold_never_decreases_hit_rate.
    # This test checks the full-range AUC=0.5 property directly instead.
    rng = np.random.default_rng(42)
    sims = rng.random(2000)
    correct = rng.random(2000) > 0.5  # correctness independent of similarity
    thresholds = np.linspace(0.0, 1.0, 201).tolist()
    rows = sweep(sims.tolist(), correct.tolist(), thresholds=thresholds)
    assert roc_auc(rows) == pytest.approx(0.5, abs=0.05)
