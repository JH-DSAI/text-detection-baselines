"""Tests for metrics registry and metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from text_detection_baselines.metrics import (
    METRIC_REGISTRY,
    compute_abstention_curve,
    list_registered_metrics,
    register_metric,
    run_all_metrics,
    safe_round,
)
from text_detection_baselines.metrics.calibration import brier_metric, ece_metric, expected_calibration_error
from text_detection_baselines.metrics.detection import (
    auroc_at_1pct_metric,
    auroc_metric,
    average_precision_metric,
    calibration_gap_metric,
    fpr_at_tau_metric,
    ood_percent_metric,
    tpr_at_tau_metric,
)

_RANKING_METRICS = [auroc_metric, auroc_at_1pct_metric, average_precision_metric]
_CALIBRATION_METRICS = [brier_metric, ece_metric]


@pytest.fixture
def metric_inputs():
    """Non-OOD samples are perfectly separable, so all ranking metrics are 1.0."""
    labels = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.6, 0.7, 0.9], dtype=float)
    ood_flags = np.array([False, False, False, True], dtype=bool)
    tau = 0.65
    flags = (scores >= tau) & (~ood_flags)
    target_alpha = 0.05
    return labels, scores, ood_flags, flags, target_alpha, tau


@pytest.fixture
def imperfect_metric_inputs():
    """One inverted pair, so ranking metrics take non-degenerate values."""
    labels = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.8, 0.7, 0.9], dtype=float)
    ood_flags = np.zeros(4, dtype=bool)
    tau = 0.75
    flags = (scores >= tau) & (~ood_flags)
    target_alpha = 0.05
    return labels, scores, ood_flags, flags, target_alpha, tau


def test_detection_metrics(metric_inputs):
    labels, scores, ood_flags, flags, target_alpha, tau = metric_inputs
    assert auroc_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 1.0
    assert auroc_at_1pct_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 1.0
    assert average_precision_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 1.0
    assert fpr_at_tau_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 0.0
    assert tpr_at_tau_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 0.5
    assert calibration_gap_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 0.05
    assert ood_percent_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 25.0


def test_ranking_metrics_imperfect_separation(imperfect_metric_inputs):
    labels, scores, ood_flags, flags, target_alpha, tau = imperfect_metric_inputs
    assert auroc_metric(labels, scores, ood_flags, flags, target_alpha, tau) == pytest.approx(0.75)
    # McClish-standardized partial AUC, not the raw area over FPR <= 1%.
    assert auroc_at_1pct_metric(labels, scores, ood_flags, flags, target_alpha, tau) == pytest.approx(
        0.748744, abs=1e-6
    )
    # Average precision sums over achievable operating points; it does not
    # interpolate between them the way a trapezoidal PR-AUC would.
    assert average_precision_metric(labels, scores, ood_flags, flags, target_alpha, tau) == pytest.approx(
        0.833333, abs=1e-6
    )


@pytest.mark.parametrize("metric", _RANKING_METRICS)
def test_ranking_metrics_none_when_single_class_survives_ood(metric):
    labels = np.array([0, 0, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.9], dtype=float)
    # Dropping the only machine sample leaves one class among non-OOD samples.
    ood_flags = np.array([False, False, True], dtype=bool)
    assert metric(labels, scores, ood_flags, ood_flags, 0.05, 0.5) is None


@pytest.mark.parametrize("metric", _RANKING_METRICS)
def test_ranking_metrics_none_when_all_samples_ood(metric):
    labels = np.array([0, 1], dtype=int)
    scores = np.array([0.1, 0.9], dtype=float)
    ood_flags = np.ones(2, dtype=bool)
    assert metric(labels, scores, ood_flags, ood_flags, 0.05, 0.5) is None


def test_calibration_metrics(metric_inputs):
    labels, scores, ood_flags, flags, target_alpha, tau = metric_inputs
    # The OOD sample (0.9, label 1) is excluded, so the mean is over three samples.
    assert brier_metric(labels, scores, ood_flags, flags, target_alpha, tau) == pytest.approx(0.46 / 3)
    assert 0.0 <= ece_metric(labels, scores, ood_flags, flags, target_alpha, tau) <= 1.0


@pytest.mark.parametrize("metric", _CALIBRATION_METRICS)
def test_calibration_metrics_exclude_ood_samples(metric):
    """Calibration restricts to non-OOD samples, matching the ranking metrics."""
    labels = np.array([0, 1], dtype=int)
    scores = np.array([0.2, 0.2], dtype=float)
    ood_flags = np.array([False, True], dtype=bool)

    kept_only = metric(labels[:1], scores[:1], np.zeros(1, dtype=bool), np.zeros(1, dtype=bool), 0.05, 0.5)
    assert metric(labels, scores, ood_flags, ood_flags, 0.05, 0.5) == pytest.approx(kept_only)
    # The whole-population value differs, so the restriction is load-bearing here.
    assert metric(labels, scores, np.zeros(2, dtype=bool), ood_flags, 0.05, 0.5) != pytest.approx(kept_only)


@pytest.mark.parametrize("metric", _CALIBRATION_METRICS)
def test_calibration_metrics_none_when_all_samples_ood(metric):
    labels = np.array([0, 1], dtype=int)
    scores = np.array([0.1, 0.9], dtype=float)
    ood_flags = np.ones(2, dtype=bool)
    assert metric(labels, scores, ood_flags, ood_flags, 0.05, 0.5) is None


def test_average_precision_on_constant_scores_equals_prevalence():
    """AP floors at the base rate; a trapezoidal PR-AUC would read prev + (1-prev)/2."""
    labels = np.array([0, 1, 1, 1], dtype=int)
    scores = np.full(4, 0.5, dtype=float)
    ood_flags = np.zeros(4, dtype=bool)

    prevalence = float(labels.mean())
    assert average_precision_metric(labels, scores, ood_flags, ood_flags, 0.05, 0.5) == pytest.approx(prevalence)
    assert average_precision_metric(labels, scores, ood_flags, ood_flags, 0.05, 0.5) != pytest.approx(
        prevalence + (1.0 - prevalence) / 2.0
    )


def test_expected_calibration_error_exact():
    scores = np.array([0.0, 1.0], dtype=float)
    labels = np.array([0, 1], dtype=int)
    assert expected_calibration_error(scores=scores, labels=labels, bins=10) == 0.0


def test_metric_registration_defaults_present():
    names = list_registered_metrics()
    assert "auroc" in names
    assert "auroc_at_1pct" in names
    assert "average_precision" in names
    assert "fpr_at_tau" in names
    assert "tpr_at_tau" in names
    assert "calibration_gap" in names
    assert "ood_percent" in names
    assert "brier" in names
    assert "ece" in names


def test_register_metric_decorator_and_run_all(metric_inputs):
    labels, scores, ood_flags, flags, target_alpha, tau = metric_inputs

    @register_metric("dummy_metric")
    def dummy_metric(
        labels: np.ndarray,
        scores: np.ndarray,
        ood_flags: np.ndarray,
        flags: np.ndarray,
        target_alpha: float,
        tau: float,
    ) -> float:
        del labels, scores, ood_flags, flags, target_alpha, tau
        return 0.123456

    try:
        results = run_all_metrics(
            labels=labels,
            scores=scores,
            ood_flags=ood_flags,
            flags=flags,
            target_alpha=target_alpha,
            tau=tau,
            normalized_scores=True,
        )
        assert "dummy_metric" in METRIC_REGISTRY
        assert results["dummy_metric"] == 0.1235
    finally:
        METRIC_REGISTRY.pop("dummy_metric", None)


def test_run_all_metrics_skips_normalized_only_for_raw(metric_inputs):
    labels, scores, ood_flags, flags, target_alpha, tau = metric_inputs
    results = run_all_metrics(
        labels=labels,
        scores=scores,
        ood_flags=ood_flags,
        flags=flags,
        target_alpha=target_alpha,
        tau=tau,
        normalized_scores=False,
    )
    assert "brier" not in results
    assert "ece" not in results
    assert "auroc" in results


# ---------------------------------------------------------------------------
# safe_round / expected_calibration_error
# ---------------------------------------------------------------------------


def test_safe_round_none():
    assert safe_round(None) is None


def test_safe_round_nan():
    assert safe_round(float("nan")) is None


def test_safe_round_value():
    assert safe_round(0.123456, 4) == 0.1235


# ---------------------------------------------------------------------------
# compute_abstention_curve
# ---------------------------------------------------------------------------

_CURVE_KEYS = ("coverage", "accuracy", "roc_auc", "roc_auc_at_1pct", "thresholds", "num_kept")


@pytest.fixture
def abstention_inputs():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 20 + [1] * 20, dtype=int)
    # Machine samples score higher on average, with overlap.
    scores = np.concatenate([rng.normal(0.3, 0.15, 20), rng.normal(0.7, 0.15, 20)])
    # Confidence degrades monotonically with index within each class.
    variances = np.concatenate([np.linspace(0.01, 0.5, 20), np.linspace(0.01, 0.5, 20)])
    return scores, labels, variances


def test_abstention_curve_shape_and_coverage(abstention_inputs):
    scores, labels, variances = abstention_inputs
    curve = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=0.5)

    assert set(curve) == set(_CURVE_KEYS)
    lengths = {len(curve[key]) for key in _CURVE_KEYS}
    assert len(lengths) == 1, f"curve series have mismatched lengths: {lengths}"
    assert lengths.pop() > 1

    coverage = curve["coverage"]
    assert all(a <= b for a, b in zip(coverage, coverage[1:], strict=False)), "coverage must be non-decreasing"
    assert coverage[-1] == pytest.approx(1.0)
    assert curve["num_kept"][-1] == len(scores)
    assert all(0.0 <= value <= 1.0 for value in curve["accuracy"])
    assert all(0.0 <= value <= 1.0 for value in curve["roc_auc"])


def test_abstention_curve_respects_num_points(abstention_inputs):
    scores, labels, variances = abstention_inputs
    curve = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=0.5, num_points=5)
    assert len(curve["coverage"]) <= 5


@pytest.mark.parametrize("tau", [0.5, 0.0, -1.5])
def test_abstention_curve_uses_inclusive_tau(tau):
    # Every score sits exactly on tau, so `score >= tau` predicts all machine and
    # `score > tau` would predict all human. The class imbalance separates the two.
    # Parametrized over tau=0, where raw (unnormalized) logits make an exact tie
    # a realistic rather than measure-zero event.
    labels = np.array([0] * 10 + [1] * 30, dtype=int)
    scores = np.full(len(labels), tau)
    variances = np.linspace(0.01, 0.5, len(labels))
    curve = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=tau)
    assert curve["accuracy"][-1] == pytest.approx(0.75)


def test_abstention_curve_full_coverage_accuracy_matches_evaluate_convention():
    # The full-coverage point must agree with the `scores >= tau` flags that
    # evaluate_predictions derives FPR@tau / TPR@tau from.
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 2, 60)
    scores = rng.normal(labels * 0.8, 1.0)
    variances = rng.random(60)
    tau = 0.0

    curve = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=tau)
    expected = float(np.mean((scores >= tau).astype(int) == labels))
    assert curve["coverage"][-1] == pytest.approx(1.0)
    assert curve["accuracy"][-1] == pytest.approx(expected)


def test_abstention_curve_deduplicates_tied_variances():
    # Identical variances make every quantile collapse onto the same subset;
    # the sweep should report that subset once, not num_points times.
    labels = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    variances = np.full(4, 0.3)
    curve = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=0.5)
    assert curve["num_kept"] == [4]
    assert curve["coverage"] == [pytest.approx(1.0)]


def test_abstention_curve_min_quantile_widens_sweep(abstention_inputs):
    scores, labels, variances = abstention_inputs
    narrow = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=0.5, min_quantile=0.5)
    wide = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=0.5, min_quantile=0.1)
    assert min(wide["coverage"]) < min(narrow["coverage"])
    assert min(narrow["coverage"]) >= 0.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_points": 1}, "num_points must be at least 2"),
        ({"min_quantile": 1.0}, r"min_quantile must lie in \[0, 1\)"),
        ({"min_quantile": -0.1}, r"min_quantile must lie in \[0, 1\)"),
    ],
)
def test_abstention_curve_rejects_invalid_sweep_parameters(abstention_inputs, kwargs, match):
    scores, labels, variances = abstention_inputs
    with pytest.raises(ValueError, match=match):
        compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=0.5, **kwargs)


def test_abstention_curve_drops_single_class_points():
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=int)
    # Tight thresholds keep only the two human samples, which cannot be scored.
    variances = np.array([0.01, 0.02, 0.9, 1.0], dtype=float)
    curve = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=0.5, num_points=3)
    assert curve["coverage"], "at least the full-coverage point should survive"
    assert all(count >= 2 for count in curve["num_kept"])
    assert curve["coverage"][-1] == pytest.approx(1.0)


def test_abstention_curve_drops_points_keeping_fewer_than_two_samples():
    # With two samples, tight thresholds admit only one, which cannot be scored.
    scores = np.array([0.2, 0.8], dtype=float)
    labels = np.array([0, 1], dtype=int)
    variances = np.array([0.1, 0.9], dtype=float)
    curve = compute_abstention_curve(scores=scores, labels=labels, variances=variances, tau=0.5)
    assert curve["num_kept"] == [2]
    assert curve["coverage"] == [pytest.approx(1.0)]


def test_abstention_curve_empty_input():
    empty = np.array([], dtype=float)
    curve = compute_abstention_curve(scores=empty, labels=empty, variances=empty, tau=0.5)
    assert all(curve[key] == [] for key in _CURVE_KEYS)


def test_abstention_curve_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        compute_abstention_curve(
            scores=np.array([0.1, 0.2]),
            labels=np.array([0, 1, 1]),
            variances=np.array([0.1, 0.2]),
            tau=0.5,
        )
