"""Detection metrics for binary text-detection tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from . import register_metric


@register_metric("auroc")
def auroc_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """Compute AUROC on non-OOD samples, returning None when undefined."""
    del flags, target_alpha, tau
    non_ood = ~ood_flags
    if non_ood.sum() == 0 or len(np.unique(labels[non_ood])) != 2:
        return None
    return float(roc_auc_score(labels[non_ood], scores[non_ood]))


@register_metric("fpr_at_tau")
def fpr_at_tau_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float:
    """False positive rate at threshold tau among human samples."""
    del scores, ood_flags, target_alpha, tau
    human_mask = labels == 0
    n_human = int(human_mask.sum())
    return float((flags & human_mask).sum() / max(n_human, 1))


@register_metric("tpr_at_tau")
def tpr_at_tau_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float:
    """True positive rate at threshold tau among machine samples."""
    del scores, ood_flags, target_alpha, tau
    machine_mask = labels == 1
    n_machine = int(machine_mask.sum())
    return float((flags & machine_mask).sum() / max(n_machine, 1))


@register_metric("calibration_gap")
def calibration_gap_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float:
    """Absolute distance between FPR@tau and target_alpha."""
    del scores, ood_flags, tau
    human_mask = labels == 0
    n_human = int(human_mask.sum())
    fpr = float((flags & human_mask).sum() / max(n_human, 1))
    return abs(fpr - target_alpha)


@register_metric("ood_percent")
def ood_percent_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float:
    """Percentage of samples flagged as out-of-distribution."""
    del labels, scores, flags, target_alpha, tau
    return float(ood_flags.mean() * 100.0)
