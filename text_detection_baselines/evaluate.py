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

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .models.base import StubModelOutput, StubTextDetector

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class DatasetRecord:
    """Single evaluation sample."""

    text: str
    label: int
    category: str


def load_dataset(path: Path, text_key: str, label_key: str, category_key: str) -> list[DatasetRecord]:
    """Load records from a JSONL file; JSON arrays are also accepted."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset is empty: {path}")

    records: list[dict[str, Any]]
    if raw[0] == "[":
        loaded = json.loads(raw)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected JSON array in {path}")
        records = [r for r in loaded if isinstance(r, dict)]
    else:
        records = []
        for idx, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {path}") from exc
            if isinstance(row, dict):
                records.append(row)

    parsed: list[DatasetRecord] = []
    for row in records:
        if text_key not in row or label_key not in row:
            continue
        parsed.append(
            DatasetRecord(
                text=str(row[text_key]),
                label=normalize_label(row[label_key]),
                category=str(row.get(category_key, "unknown")),
            ),
        )

    if not parsed:
        raise ValueError(f"No valid samples with required keys in {path}")
    return parsed


def normalize_label(raw_label: Any) -> int:
    """Map common label representations to 0 (human) or 1 (machine)."""
    if isinstance(raw_label, bool):
        return int(raw_label)
    if isinstance(raw_label, (int, np.integer)):
        return int(raw_label)

    val = str(raw_label).strip().lower()
    if val in {"1", "machine", "fake", "ai", "generated"}:
        return 1
    if val in {"0", "human", "real", "organic"}:
        return 0
    raise ValueError(f"Unsupported label value: {raw_label}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def safe_round(value: float | None, ndigits: int = 4) -> float | None:
    """Round a numeric value while preserving None / NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), ndigits)


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Compute ECE for predicted probabilities in [0, 1].

    Uses equal-width binning over the score range.
    """
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


def _compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
    normalized_scores: bool,
) -> dict[str, Any]:
    """Compute the standard metric bundle for a slice of the data."""
    human_mask = labels == 0
    machine_mask = labels == 1
    non_ood = ~ood_flags

    n_human = int(human_mask.sum())
    n_machine = int(machine_mask.sum())

    fpr = float((flags & human_mask).sum() / max(n_human, 1))
    tpr = float((flags & machine_mask).sum() / max(n_machine, 1))
    cal_gap = abs(fpr - target_alpha)
    ood_pct = float(ood_flags.mean() * 100.0)

    auroc: float | None = None
    if non_ood.sum() > 0 and len(np.unique(labels[non_ood])) == 2:
        auroc = float(roc_auc_score(labels[non_ood], scores[non_ood]))

    entry: dict[str, Any] = {
        "n_samples": int(labels.size),
        "n_human": n_human,
        "n_machine": n_machine,
        "tau": safe_round(tau),
        "target_alpha": target_alpha,
        "auroc": safe_round(auroc),
        "fpr_at_tau": safe_round(fpr),
        "tpr_at_tau": safe_round(tpr),
        "calibration_gap": safe_round(cal_gap),
        "ood_percent": safe_round(ood_pct),
    }

    if normalized_scores:
        entry["brier"] = safe_round(float(np.mean((scores - labels) ** 2)))
        entry["ece"] = safe_round(expected_calibration_error(scores=scores, labels=labels, bins=10))

    return entry


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

    overall = _compute_metrics(
        labels=labels,
        scores=scores,
        ood_flags=ood_flags,
        flags=flags,
        target_alpha=target_alpha,
        tau=tau,
        normalized_scores=normalized_scores,
    )

    per_category: dict[str, dict[str, Any]] = {}
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        per_category[category] = _compute_metrics(
            labels=labels[mask],
            scores=scores[mask],
            ood_flags=ood_flags[mask],
            flags=flags[mask],
            target_alpha=target_alpha,
            tau=tau,
            normalized_scores=normalized_scores,
        )

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
    records = load_dataset(dataset_path, text_key=text_key, label_key=label_key, category_key=category_key)

    texts = [r.text for r in records]
    labels = np.array([r.label for r in records], dtype=int)
    categories = np.array([r.category for r in records], dtype=object)

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
              "per_category": {
                "<dataset>": {
                  "<category>": {"<model>": {<metrics>}, ...},
                  ...
                },
                ...
              },
            }
    """
    tree: dict[str, Any] = {"overall": {}, "per_category": {}}

    for dataset_name, model_name, overall, per_cat in results:
        tree["overall"].setdefault(dataset_name, {})[model_name] = overall

        for category, cat_metrics in per_cat.items():
            tree["per_category"].setdefault(dataset_name, {}).setdefault(category, {})[model_name] = cat_metrics

    return tree
