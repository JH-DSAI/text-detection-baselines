"""Calibration metrics for normalized model scores."""

from __future__ import annotations

import numpy as np

from . import register_metric


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
) -> float:
    """Brier score for probabilistic outputs in [0, 1]."""
    del ood_flags, flags, target_alpha, tau
    return float(np.mean((scores - labels) ** 2))


@register_metric("ece", requires_normalized_scores=True)
def ece_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float:
    """Expected calibration error for probabilistic outputs in [0, 1]."""
    del ood_flags, flags, target_alpha, tau
    return expected_calibration_error(scores=scores, labels=labels, bins=10)
