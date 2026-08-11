"""Tests for core evaluation logic in text_detection_baselines.evaluate."""

from __future__ import annotations

import json

import numpy as np
import pytest

from text_detection_baselines.evaluate import (
    DatasetRecord,
    build_results_tree,
    evaluate_model_on_dataset,
    evaluate_predictions,
    load_dataset,
    normalize_label,
)
from text_detection_baselines.models import build_stub_model
from text_detection_baselines.models.base import StubModelOutput


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


_SAMPLE_ROWS = [
    {"answer": "short human text", "label": "real", "contribution_level": "Human"},
    {
        "answer": "this human sample is much longer and varied in style to look natural indeed",
        "label": "real",
        "contribution_level": "Human",
    },
    {
        "answer": "machine generated response with repeated repeated repeated patterns patterns",
        "label": "fake",
        "contribution_level": "Summary",
    },
    {
        "answer": "another synthetic answer that has a different shape and sentence cadence entirely",
        "label": "fake",
        "contribution_level": "Task",
    },
]


# ---------------------------------------------------------------------------
# normalize_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("real", 0),
        ("fake", 1),
        ("human", 0),
        ("machine", 1),
        ("Real", 0),
        ("FAKE", 1),
        (0, 0),
        (1, 1),
        (True, 1),
        (False, 0),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_label_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported label value"):
        normalize_label("unknown_label_xyz")


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, _SAMPLE_ROWS)
    records = load_dataset(p, text_key="answer", label_key="label", category_key="contribution_level")
    assert len(records) == 4
    assert all(isinstance(r, DatasetRecord) for r in records)
    assert records[0].label == 0
    assert records[2].label == 1


def test_load_dataset_json_array(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(_SAMPLE_ROWS), encoding="utf-8")
    records = load_dataset(p, text_key="answer", label_key="label", category_key="contribution_level")
    assert len(records) == 4


def test_load_dataset_skips_rows_missing_keys(tmp_path):
    rows = [{"answer": "text", "label": "real", "contribution_level": "Human"}, {"no_answer": "x", "label": "fake"}]
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, rows)
    records = load_dataset(p, text_key="answer", label_key="label", category_key="contribution_level")
    assert len(records) == 1


def test_load_dataset_empty_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(p, text_key="answer", label_key="label", category_key="contribution_level")


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------


def _make_output(scores, ood_flags=None):
    scores = np.array(scores, dtype=float)
    ood_flags = np.zeros(len(scores), dtype=bool) if ood_flags is None else np.array(ood_flags, dtype=bool)
    preds = (scores >= 0.5).astype(int)
    return StubModelOutput(scores=scores, predictions=preds, ood_flags=ood_flags)


def test_evaluate_predictions_overall_keys():
    labels = np.array([0, 0, 1, 1])
    categories = np.array(["Human", "Human", "Summary", "Task"])
    output = _make_output([0.2, 0.3, 0.7, 0.8])
    overall, per_cat = evaluate_predictions(labels, categories, output, target_alpha=0.05, normalized_scores=True)
    for key in (
        "auroc",
        "auroc_at_1pct",
        "pr_auc",
        "average_precision",
        "fpr_at_tau",
        "tpr_at_tau",
        "calibration_gap",
        "ood_percent",
        "tau",
        "brier",
        "ece",
    ):
        assert key in overall, f"missing key: {key}"


def test_evaluate_predictions_per_category_keys():
    labels = np.array([0, 0, 1, 1])
    categories = np.array(["Human", "Human", "Summary", "Task"])
    output = _make_output([0.2, 0.3, 0.7, 0.8])
    _, per_cat = evaluate_predictions(labels, categories, output, target_alpha=0.05, normalized_scores=True)
    assert set(per_cat.keys()) == {"Human", "Summary", "Task"}
    for cat_metrics in per_cat.values():
        for key in (
            "auroc",
            "auroc_at_1pct",
            "pr_auc",
            "fpr_at_tau",
            "tpr_at_tau",
            "calibration_gap",
            "ood_percent",
            "brier",
            "ece",
        ):
            assert key in cat_metrics, f"missing per-category key: {key}"


def test_evaluate_predictions_unnormalized_no_brier_ece():
    labels = np.array([0, 0, 1, 1])
    categories = np.array(["A", "A", "B", "B"])
    output = _make_output([-2.0, -1.0, 1.0, 2.0])
    overall, per_cat = evaluate_predictions(labels, categories, output, target_alpha=0.05, normalized_scores=False)
    assert "brier" not in overall
    assert "ece" not in overall
    for cat_metrics in per_cat.values():
        assert "brier" not in cat_metrics
        assert "ece" not in cat_metrics


def test_evaluate_predictions_missing_class_raises():
    labels = np.array([0, 0, 0])
    categories = np.array(["Human", "Human", "Human"])
    output = _make_output([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="both human and machine"):
        evaluate_predictions(labels, categories, output, target_alpha=0.05, normalized_scores=False)


# ---------------------------------------------------------------------------
# evaluate_model_on_dataset
# ---------------------------------------------------------------------------


def test_evaluate_model_on_dataset_produces_required_metrics(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    model = build_stub_model("length-normalized", ood_margin=0.01, seed=7)
    overall, per_cat = evaluate_model_on_dataset(
        dataset_path=dataset_path,
        model=model,
        target_alpha=0.1,
        text_key="answer",
        label_key="label",
        category_key="contribution_level",
    )

    for key in (
        "auroc",
        "auroc_at_1pct",
        "pr_auc",
        "fpr_at_tau",
        "tpr_at_tau",
        "calibration_gap",
        "ood_percent",
        "brier",
        "ece",
    ):
        assert key in overall

    assert set(per_cat.keys()) == {"Human", "Summary", "Task"}
    for cat_metrics in per_cat.values():
        for key in (
            "auroc_at_1pct",
            "pr_auc",
            "fpr_at_tau",
            "tpr_at_tau",
            "calibration_gap",
            "ood_percent",
            "brier",
            "ece",
        ):
            assert key in cat_metrics


# ---------------------------------------------------------------------------
# build_results_tree
# ---------------------------------------------------------------------------


def test_build_results_tree_structure():
    results = [
        ("ds1", "model-a", {"auroc": 0.8}, {"Cat1": {"tpr_at_tau": 0.9}, "Cat2": {"tpr_at_tau": 0.5}}),
        ("ds1", "model-b", {"auroc": 0.75}, {"Cat1": {"tpr_at_tau": 0.6}}),
    ]
    tree = build_results_tree(results)

    assert "overall" in tree
    assert "per-category" in tree
    assert tree["overall"]["ds1"]["model-a"]["auroc"] == 0.8
    assert tree["overall"]["ds1"]["model-b"]["auroc"] == 0.75
    assert tree["per-category"]["ds1"]["Cat1"]["model-a"]["tpr_at_tau"] == 0.9
    assert tree["per-category"]["ds1"]["Cat2"]["model-a"]["tpr_at_tau"] == 0.5
    assert tree["per-category"]["ds1"]["Cat1"]["model-b"]["tpr_at_tau"] == 0.6
