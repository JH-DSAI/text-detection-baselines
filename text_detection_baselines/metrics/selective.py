"""Selective-prediction (abstention) curves.

Unlike the metrics in :mod:`~text_detection_baselines.metrics.detection` and
:mod:`~text_detection_baselines.metrics.calibration`, the function here returns a
*curve* rather than a scalar, so it is not registered in ``METRIC_REGISTRY``
(:func:`~text_detection_baselines.metrics.run_all_metrics` reduces every metric
through :func:`~text_detection_baselines.metrics.safe_round`, which expects a
scalar).  Call it directly instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]


def compute_abstention_curve(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    variances: np.ndarray,
    tau: float,
    num_points: int = 20,
    min_quantile: float = 0.1,
) -> dict[str, list[float]]:
    """Sweep an uncertainty threshold to trace coverage against detection quality.

    Variance thresholds are swept from tight (low coverage, only the most
    confident samples) to loose (full coverage).  At each threshold, samples
    whose variance exceeds it are abstained on and the remaining samples are
    scored.

    A kept sample is predicted machine when ``score >= tau``, matching the
    threshold convention used by
    :func:`~text_detection_baselines.evaluate.evaluate_predictions` and by the
    detectors themselves, so that the full-coverage point of this curve is
    consistent with the ``fpr_at_tau`` / ``tpr_at_tau`` metrics reported for the
    same run.  The comparison is inclusive, so scores lying exactly on ``tau``
    (including ``tau = 0`` against a score of exactly zero) count as machine.

    Threshold points are dropped when fewer than two samples survive, when the
    survivors carry only one label, or when a threshold duplicates the coverage
    of the preceding point (which happens whenever ``variances`` contains ties).
    The returned lists are parallel to each other but may therefore be shorter
    than ``num_points``.

    Args:
        scores: Detection scores per sample; higher → more likely machine.
        labels: Binary labels (0=human, 1=machine).
        variances: Per-sample uncertainty used to rank abstention candidates.
            No model in this package currently exposes such a quantity, so this
            must be supplied by the caller.
        tau: Decision threshold; a kept sample is predicted machine when
            ``score >= tau``.  Held fixed across the sweep rather than
            recalibrated on each retained subset.
        num_points: Number of variance thresholds to sweep.
        min_quantile: Variance quantile of the tightest threshold, which bounds
            the smallest coverage examined.  Defaults to ``0.1`` so that the
            high-confidence regime is visible; raise it to restrict the sweep to
            higher coverage.

    Returns:
        A dict of parallel lists, ordered from tightest to full coverage:

        - ``coverage`` — fraction of all samples kept at each threshold
        - ``accuracy`` — accuracy at ``tau`` on the kept samples
        - ``roc_auc`` — AUROC on the kept samples
        - ``roc_auc_at_1pct`` — standardized partial AUROC (``max_fpr=0.01``)
        - ``thresholds`` — the variance threshold used
        - ``num_kept`` — number of samples kept

    Raises:
        ValueError: If ``scores``, ``labels``, and ``variances`` differ in
            length, if ``num_points`` is below 2, or if ``min_quantile`` is
            outside ``[0, 1)``.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    variances = np.asarray(variances)

    n_total = scores.shape[0]
    if labels.shape[0] != n_total or variances.shape[0] != n_total:
        raise ValueError(
            f"scores, labels, and variances must have equal length; "
            f"got {n_total}, {labels.shape[0]}, {variances.shape[0]}",
        )
    if num_points < 2:
        raise ValueError(f"num_points must be at least 2; got {num_points}")
    if not 0.0 <= min_quantile < 1.0:
        raise ValueError(f"min_quantile must lie in [0, 1); got {min_quantile}")

    curve: dict[str, list[float]] = {
        "coverage": [],
        "accuracy": [],
        "roc_auc": [],
        "roc_auc_at_1pct": [],
        "thresholds": [],
        "num_kept": [],
    }

    if n_total == 0:
        return curve

    # The final quantile is the maximum variance, which `variances <= threshold`
    # (below) admits.
    thresholds = np.quantile(variances, np.linspace(min_quantile, 1.0, num_points))

    last_n_kept = 0
    for threshold in thresholds:
        mask = variances <= threshold
        n_kept = int(mask.sum())
        # Tied variances make consecutive quantiles collapse onto the same subset.
        if n_kept < 2 or n_kept == last_n_kept:
            continue

        kept_scores = scores[mask]
        kept_labels = labels[mask]
        if len(np.unique(kept_labels)) < 2:
            continue

        predictions = (kept_scores >= tau).astype(int)

        curve["coverage"].append(n_kept / n_total)
        curve["accuracy"].append(float(np.mean(predictions == kept_labels)))
        curve["roc_auc"].append(float(roc_auc_score(kept_labels, kept_scores)))
        curve["roc_auc_at_1pct"].append(float(roc_auc_score(kept_labels, kept_scores, max_fpr=0.01)))
        curve["thresholds"].append(float(threshold))
        curve["num_kept"].append(n_kept)
        last_n_kept = n_kept

    return curve
