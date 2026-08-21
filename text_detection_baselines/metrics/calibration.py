"""Calibration metrics for normalized model scores."""

from __future__ import annotations

import numpy as np

from . import register_metric


def _non_ood(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Restrict labels and scores to non-OOD samples.

    Mirrors :func:`~text_detection_baselines.metrics.detection._non_ood_binary` so
    that the calibration columns of a results row describe the same sample set as
    the ranking columns.  Unlike the ranking metrics, calibration is well defined
    when only one class survives, so no class check is applied here.

    Returns:
        ``(labels, scores)`` for the non-OOD samples, or ``None`` when no sample
        survives.
    """
    non_ood = ~ood_flags
    if non_ood.sum() == 0:
        return None
    return labels[non_ood], scores[non_ood]


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Compute expected calibration error using equal-width bins in [0, 1]."""
    scores = np.clip(scores, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores <= hi if i == bins - 1 else scores < hi)
        if not np.any(mask):
            continue

        conf = float(np.mean(scores[mask]))
        acc = float(np.mean(labels[mask]))
        frac = float(np.mean(mask))
        ece += abs(acc - conf) * frac

    return ece


@register_metric("brier", requires_normalized_scores=True)
def brier_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """Brier score for probabilistic outputs in [0, 1], on non-OOD samples.

    Returns:
        The Brier score, or ``None`` when every sample is flagged OOD.
    """
    del flags, target_alpha, tau
    kept = _non_ood(labels, scores, ood_flags)
    if kept is None:
        return None
    kept_labels, kept_scores = kept
    return float(np.mean((kept_scores - kept_labels) ** 2))


@register_metric("ece", requires_normalized_scores=True)
def ece_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """Expected calibration error for probabilistic outputs in [0, 1], on non-OOD samples.

    Returns:
        The expected calibration error, or ``None`` when every sample is flagged OOD.
    """
    del flags, target_alpha, tau
    kept = _non_ood(labels, scores, ood_flags)
    if kept is None:
        return None
    kept_labels, kept_scores = kept
    return expected_calibration_error(scores=kept_scores, labels=kept_labels, bins=10)
