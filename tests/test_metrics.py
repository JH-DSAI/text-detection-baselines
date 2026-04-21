"""Tests for metrics registry and metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from text_detection_baselines.metrics import METRIC_REGISTRY, list_registered_metrics, register_metric, run_all_metrics
from text_detection_baselines.metrics.calibration import brier_metric, ece_metric, expected_calibration_error
from text_detection_baselines.metrics.detection import (
    auroc_metric,
    calibration_gap_metric,
    fpr_at_tau_metric,
    ood_percent_metric,
    tpr_at_tau_metric,
)


@pytest.fixture
def metric_inputs():
    labels = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.6, 0.7, 0.9], dtype=float)
    ood_flags = np.array([False, False, False, True], dtype=bool)
    tau = 0.65
    flags = (scores >= tau) & (~ood_flags)
    target_alpha = 0.05
    return labels, scores, ood_flags, flags, target_alpha, tau


def test_detection_metrics(metric_inputs):
    labels, scores, ood_flags, flags, target_alpha, tau = metric_inputs
    assert auroc_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 1.0
    assert fpr_at_tau_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 0.0
    assert tpr_at_tau_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 0.5
    assert calibration_gap_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 0.05
    assert ood_percent_metric(labels, scores, ood_flags, flags, target_alpha, tau) == 25.0


def test_calibration_metrics(metric_inputs):
    labels, scores, ood_flags, flags, target_alpha, tau = metric_inputs
    assert brier_metric(labels, scores, ood_flags, flags, target_alpha, tau) == pytest.approx(0.1175)
    assert 0.0 <= ece_metric(labels, scores, ood_flags, flags, target_alpha, tau) <= 1.0


def test_expected_calibration_error_exact():
    scores = np.array([0.0, 1.0], dtype=float)
    labels = np.array([0, 1], dtype=int)
    assert expected_calibration_error(scores=scores, labels=labels, bins=10) == 0.0


def test_metric_registration_defaults_present():
    names = list_registered_metrics()
    assert "auroc" in names
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
