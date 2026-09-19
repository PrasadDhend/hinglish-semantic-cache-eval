"""Unit tests for src/eval/stats.py -- bootstrap CIs and paired significance
testing, checked against hand-constructed data with known properties."""

import numpy as np
import pytest

from eval.stats import (
    bootstrap_ci, bootstrap_confusion_metric_ci, paired_bootstrap_diff,
    paired_bootstrap_confusion_metric_diff,
)


def test_bootstrap_ci_constant_array_collapses_to_a_point():
    result = bootstrap_ci([0.7] * 50, n_resamples=500, seed=0)
    assert result["point_estimate"] == pytest.approx(0.7)
    assert result["ci_low"] == pytest.approx(0.7)
    assert result["ci_high"] == pytest.approx(0.7)


def test_bootstrap_ci_point_estimate_matches_true_mean():
    values = np.linspace(0, 1, 200)
    result = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert result["point_estimate"] == pytest.approx(float(np.mean(values)))
    assert result["ci_low"] < result["point_estimate"] < result["ci_high"]


def test_bootstrap_ci_is_reproducible_with_fixed_seed():
    values = list(range(100))
    r1 = bootstrap_ci(values, n_resamples=500, seed=123)
    r2 = bootstrap_ci(values, n_resamples=500, seed=123)
    assert r1 == r2


def test_bootstrap_ci_empty_input_returns_nan():
    result = bootstrap_ci([], n_resamples=100, seed=0)
    assert result["n"] == 0
    assert np.isnan(result["point_estimate"])


def test_bootstrap_confusion_metric_ci_matches_point_estimate():
    rng = np.random.default_rng(1)
    n = 300
    sims = rng.random(n)
    correct = rng.random(n) > 0.4
    result = bootstrap_confusion_metric_ci(sims, correct, threshold=0.5, metric="false_hit_rate",
                                            n_resamples=1000, seed=0)
    from eval.metrics import confusion_counts, false_hit_rate
    expected_point = false_hit_rate(confusion_counts(sims, correct, 0.5))
    assert result["point_estimate"] == pytest.approx(expected_point)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


def test_bootstrap_confusion_metric_ci_unknown_metric_raises():
    with pytest.raises(ValueError):
        bootstrap_confusion_metric_ci([0.9], [True], threshold=0.5, metric="not_a_metric")


def test_bootstrap_confusion_metric_ci_nothing_served_at_threshold_is_nan_not_crash():
    # regression test: a threshold nothing clears (0 served) used to raise
    # ZeroDivisionError from the plain-Python point-estimate path instead of
    # returning nan like the vectorized bootstrap path already did -- this
    # is exactly the case a strict reference threshold hits for a poorly
    # separated (e.g. degraded cross-lingual) model.
    sims = [0.1, 0.2, 0.3]
    correct = [True, False, True]
    result = bootstrap_confusion_metric_ci(sims, correct, threshold=0.99, metric="false_hit_rate",
                                            n_resamples=200, seed=0)
    assert np.isnan(result["point_estimate"])


def test_paired_bootstrap_diff_identical_arrays_is_zero_and_not_significant():
    values = list(range(50))
    result = paired_bootstrap_diff(values, values, n_resamples=1000, seed=0)
    assert result["observed_diff"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_paired_bootstrap_diff_detects_clear_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(loc=1.0, scale=0.1, size=200)
    b = rng.normal(loc=0.0, scale=0.1, size=200)
    result = paired_bootstrap_diff(a, b, n_resamples=2000, seed=0)
    assert result["observed_diff"] == pytest.approx(1.0, abs=0.1)
    assert result["p_value"] < 0.01
    assert result["ci_low"] > 0  # CI on the difference excludes zero


def test_paired_bootstrap_diff_mismatched_length_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2, 3], [1, 2])


def test_paired_bootstrap_confusion_metric_diff_identical_conditions_is_zero():
    rng = np.random.default_rng(3)
    n = 200
    sims = rng.random(n)
    correct = rng.random(n) > 0.5
    result = paired_bootstrap_confusion_metric_diff(
        sims, correct, sims, correct, threshold=0.5, metric="false_hit_rate",
        n_resamples=1000, seed=0,
    )
    assert result["observed_diff"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_paired_bootstrap_confusion_metric_diff_detects_degradation():
    rng = np.random.default_rng(4)
    n = 400
    # condition A: mostly correct matches, high similarity -> low FHR
    correct_a = np.ones(n, dtype=bool)
    sims_a = rng.uniform(0.8, 0.99, n)
    # condition B: same probes, but now half are wrongly matched -> high FHR
    correct_b = rng.random(n) > 0.5
    sims_b = rng.uniform(0.8, 0.99, n)
    result = paired_bootstrap_confusion_metric_diff(
        sims_a, correct_a, sims_b, correct_b, threshold=0.7, metric="false_hit_rate",
        n_resamples=2000, seed=0,
    )
    assert result["point_a"] == pytest.approx(0.0)
    assert result["point_b"] > 0.3
    assert result["observed_diff"] < 0  # A's FHR - B's FHR is negative (A is better)
    assert result["p_value"] < 0.01
