"""Core evaluation logic for text detection baselines.

Loads datasets, runs detectors, and computes metrics.

Metrics computed per (dataset, model) pair
- AUROC (on non-OOD samples)
- FPR@tau  false positive rate at the learned threshold
- TPR@tau  true positive rate at the learned threshold
- CalGap   |FPR@tau - target_alpha|
- OOD%     percentage of samples flagged out-of-distribution

For models with normalized [0, 1] scores two extra calibration metrics are
also computed:
- Brier   mean squared error between score and binary label
- ECE     expected calibration error (10 equal-width bins)

All of the above are also computed per contribution_level category:
- AUROC by category (only when both labels appear in the category)
- TPR, FPR, CalGap, OOD% by category
- Brier, ECE by category (normalized-score models only)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import load_dataset as load_dataset_batch
from .datasets.file import normalize_label as normalize_label_value
from .metrics import run_all_metrics, safe_round
from .models.base import StubModelOutput, StubTextDetector

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetRecord:
    """Single evaluation sample."""

    text: str
    label: int
    category: str


def load_dataset(path: Path, text_key: str, label_key: str, category_key: str) -> list[DatasetRecord]:
    """Compatibility wrapper returning record objects from file-based datasets."""
    batch = load_dataset_batch(
        dataset_type="file",
        path=path,
        text_key=text_key,
        label_key=label_key,
        category_key=category_key,
    )
    return [
        DatasetRecord(text=text, label=int(label), category=str(category))
        for text, label, category in zip(batch.texts, batch.labels, batch.categories)
    ]


def normalize_label(raw_label: Any) -> int:
    """Compatibility wrapper around the file dataset label normalizer."""
    return normalize_label_value(raw_label)


def _base_counts(labels: np.ndarray, tau: float, target_alpha: float) -> dict[str, Any]:
    """Compute non-derived counts and threshold metadata for one data slice."""
    return {
        "n_samples": int(labels.size),
        "n_human": int((labels == 0).sum()),
        "n_machine": int((labels == 1).sum()),
        "tau": safe_round(tau),
        "target_alpha": target_alpha,
    }


# ---------------------------------------------------------------------------
# Core evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_predictions(
    labels: np.ndarray,
    categories: np.ndarray,
    output: StubModelOutput,
    target_alpha: float,
    normalized_scores: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compute overall and per-category metrics for one model/dataset pair.

    Args:
        labels:           Binary array (0=human, 1=machine).
        categories:       String array of contribution_level per sample.
        output:           Raw model output from :meth:`StubTextDetector.predict`.
        target_alpha:     Target FPR used to learn the threshold ``tau``.
        normalized_scores: Whether scores are in ``[0, 1]``.

    Returns:
        A tuple of (overall_metrics_dict, {category: per_category_metrics_dict}).
    """
    scores = output.scores
    ood_flags = output.ood_flags

    human_mask = labels == 0
    machine_mask = labels == 1

    if human_mask.sum() == 0 or machine_mask.sum() == 0:
        raise ValueError("Dataset must contain both human and machine labels")

    non_ood = ~ood_flags
    human_scores_non_ood = scores[human_mask & non_ood]
    if human_scores_non_ood.size == 0:
        tau = float(np.quantile(scores[human_mask], 1 - target_alpha))
    else:
        tau = float(np.quantile(human_scores_non_ood, 1 - target_alpha))

    flags = (scores >= tau) & non_ood

    overall = _base_counts(labels=labels, tau=tau, target_alpha=target_alpha)
    overall.update(
        run_all_metrics(
            labels=labels,
            scores=scores,
            ood_flags=ood_flags,
            flags=flags,
            target_alpha=target_alpha,
            tau=tau,
            normalized_scores=normalized_scores,
        ),
    )

    per_category: dict[str, dict[str, Any]] = {}
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        cat_entry = _base_counts(labels=labels[mask], tau=tau, target_alpha=target_alpha)
        cat_entry.update(
            run_all_metrics(
                labels=labels[mask],
                scores=scores[mask],
                ood_flags=ood_flags[mask],
                flags=flags[mask],
                target_alpha=target_alpha,
                tau=tau,
                normalized_scores=normalized_scores,
            ),
        )
        per_category[category] = cat_entry

    return overall, per_category


def evaluate_model_on_dataset(
    dataset_path: Path,
    model: StubTextDetector,
    target_alpha: float,
    text_key: str,
    label_key: str,
    category_key: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Run one model against one dataset and return metric dicts.

    Args:
        dataset_path:  Path to the JSONL (or JSON array) dataset file.
        model:         Instantiated stub detector.
        target_alpha:  Target FPR for threshold learning.
        text_key:      Field name for the text to score.
        label_key:     Field name for the ground-truth label.
        category_key:  Field name for the per-category grouping variable.

    Returns:
        ``(overall_metrics, {category: category_metrics})``.  Neither dict
        contains ``dataset`` or ``model`` keys; those are tracked by the
        caller in the results tree.
    """
    batch = load_dataset_batch(
        dataset_type="file",
        path=dataset_path,
        text_key=text_key,
        label_key=label_key,
        category_key=category_key,
    )

    texts = batch.texts
    labels = batch.labels
    categories = batch.categories

    model_output = model.predict(texts)

    overall, per_category = evaluate_predictions(
        labels=labels,
        categories=categories,
        output=model_output,
        target_alpha=target_alpha,
        normalized_scores=model.normalized_scores,
    )

    overall["dataset_path"] = str(dataset_path)
    overall["normalized_scores"] = model.normalized_scores

    return overall, per_category


# ---------------------------------------------------------------------------
# Results tree builder
# ---------------------------------------------------------------------------


def build_results_tree(
    results: list[tuple[str, str, dict[str, Any], dict[str, dict[str, Any]]]],
) -> dict[str, Any]:
    """Assemble the nested results structure from per-run tuples.

    Args:
        results: List of ``(dataset_name, model_name, overall_metrics,
            per_category_metrics)`` tuples collected across all runs.

    Returns:
        A dict with the following shape::

            {
              "overall": {
                "<dataset>": {"<model>": {<metrics>}, ...},
                ...
              },
              "per-category": {
                "<dataset>": {
                  "<category>": {"<model>": {<metrics>}, ...},
                  ...
                },
                ...
              },
            }
    """
    tree: dict[str, Any] = {"overall": {}, "per-category": {}}

    for dataset_name, model_name, overall, per_cat in results:
        tree["overall"].setdefault(dataset_name, {})[model_name] = overall

        for category, cat_metrics in per_cat.items():
            tree["per-category"].setdefault(dataset_name, {}).setdefault(category, {})[model_name] = cat_metrics

    return tree
