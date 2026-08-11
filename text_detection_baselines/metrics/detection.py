"""Detection metrics for binary text-detection tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from . import register_metric


def _non_ood_binary(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Restrict labels and scores to non-OOD samples.

    Args:
        labels: Binary array (0=human, 1=machine).
        scores: Continuous detection scores; higher → more likely machine.
        ood_flags: True where the sample is flagged out-of-distribution.

    Returns:
        ``(labels, scores)`` for the non-OOD samples, or ``None`` when no sample
        survives or the surviving samples do not contain both classes.  Ranking
        metrics are undefined in those cases.
    """
    non_ood = ~ood_flags
    if non_ood.sum() == 0 or len(np.unique(labels[non_ood])) != 2:
        return None
    return labels[non_ood], scores[non_ood]


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
    kept = _non_ood_binary(labels, scores, ood_flags)
    if kept is None:
        return None
    kept_labels, kept_scores = kept
    return float(roc_auc_score(kept_labels, kept_scores))


@register_metric("auroc_at_1pct")
def auroc_at_1pct_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """AUROC restricted to the FPR <= 1% regime, on non-OOD samples.

    This is the *McClish-standardized* partial AUC that
    :func:`sklearn.metrics.roc_auc_score` returns for ``max_fpr=0.01``: the raw
    area over ``FPR <= 0.01`` rescaled so that a random ranker scores ``0.5``
    and a perfect ranker scores ``1.0``.  It is therefore comparable in range to
    ``auroc``, and is *not* the un-normalized partial area.

    Returns:
        The standardized partial AUC, or ``None`` when undefined (see
        :func:`_non_ood_binary`).
    """
    del flags, target_alpha, tau
    kept = _non_ood_binary(labels, scores, ood_flags)
    if kept is None:
        return None
    kept_labels, kept_scores = kept
    return float(roc_auc_score(kept_labels, kept_scores, max_fpr=0.01))


@register_metric("average_precision")
def average_precision_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """Average precision on non-OOD samples: the preferred PR summary.

    Average precision sums ``(R_n - R_{n-1}) * P_n`` over the achievable
    precision-recall operating points, so unlike :func:`pr_auc_metric` it does
    not interpolate between them.  This is the estimator scikit-learn recommends
    for summarizing a PR curve, and the one reported in the console tables.

    Returns:
        The average precision, or ``None`` when undefined (see
        :func:`_non_ood_binary`).
    """
    del flags, target_alpha, tau
    kept = _non_ood_binary(labels, scores, ood_flags)
    if kept is None:
        return None
    kept_labels, kept_scores = kept
    return float(average_precision_score(kept_labels, kept_scores))


@register_metric("pr_auc")
def pr_auc_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """Trapezoidal area under the precision-recall curve, on non-OOD samples.

    Computed as ``auc(recall, precision)``.  Linear interpolation between
    adjacent PR points describes operating points that are not actually
    achievable, so scikit-learn documents this estimator as misleading and
    recommends average precision instead.  It is retained here for continuity
    with previously reported numbers; prefer
    :func:`average_precision_metric` for new analysis.

    Returns:
        The trapezoidal PR-AUC, or ``None`` when undefined (see
        :func:`_non_ood_binary`).
    """
    del flags, target_alpha, tau
    kept = _non_ood_binary(labels, scores, ood_flags)
    if kept is None:
        return None
    kept_labels, kept_scores = kept
    precision, recall, _ = precision_recall_curve(kept_labels, kept_scores)
    return float(auc(recall, precision))


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
