"""Detection metrics for binary text-detection tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
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
    """Average precision on non-OOD samples: the PR-curve summary.

    Average precision sums ``(R_n - R_{n-1}) * P_n`` over the achievable
    precision-recall operating points rather than interpolating between them, as
    trapezoidal integration of the curve would.  This is the estimator
    scikit-learn recommends for summarizing a PR curve, and the one reported in
    the console tables.

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


def _rate_at_tau(
    labels: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    label_value: int,
) -> float | None:
    """Flagged fraction of one class among the non-OOD samples, or None if empty.

    ``flags`` is already ``False`` for every OOD sample, so the denominator is
    restricted to ``~ood_flags`` as well: without that, an OOD sample could only
    ever land in the denominator, and a slice whose entire denominator class is
    OOD would read as a measured rate of zero.

    Args:
        labels: Binary array (0=human, 1=machine).
        ood_flags: True where the sample is flagged out-of-distribution.
        flags: True where the sample was called machine at the threshold.
        label_value: The class forming the denominator — 0 for FPR, 1 for TPR.

    Returns:
        The flagged fraction of that class, or ``None`` when the class has no
        non-OOD sample.  A ``max(n, 1)`` guard would return ``0.0`` instead,
        which renders as an ordinary measurement on a slice that cannot support
        one — single-label category slices are the common case.
    """
    denominator_mask = (labels == label_value) & ~ood_flags
    n_denominator = int(denominator_mask.sum())
    if n_denominator == 0:
        return None
    return float((flags & denominator_mask).sum() / n_denominator)


@register_metric("fpr_at_tau")
def fpr_at_tau_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """False positive rate at threshold tau among non-OOD human samples.

    Returns:
        The false positive rate, or ``None`` when the slice holds no non-OOD
        human sample (see :func:`_rate_at_tau`).
    """
    del scores, target_alpha, tau
    return _rate_at_tau(labels, ood_flags, flags, label_value=0)


@register_metric("tpr_at_tau")
def tpr_at_tau_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """True positive rate at threshold tau among non-OOD machine samples.

    Returns:
        The true positive rate, or ``None`` when the slice holds no non-OOD
        machine sample (see :func:`_rate_at_tau`).
    """
    del scores, target_alpha, tau
    return _rate_at_tau(labels, ood_flags, flags, label_value=1)


@register_metric("calibration_gap")
def calibration_gap_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
) -> float | None:
    """Absolute distance between FPR@tau and target_alpha.

    Returns:
        ``abs(fpr_at_tau - target_alpha)``, or ``None`` when ``fpr_at_tau`` is
        undefined.  Substituting ``0.0`` for the missing rate would report
        exactly ``target_alpha`` — a plausible-looking gap that measures the
        guard rather than the threshold.
    """
    del scores, tau
    fpr = _rate_at_tau(labels, ood_flags, flags, label_value=0)
    if fpr is None:
        return None
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
