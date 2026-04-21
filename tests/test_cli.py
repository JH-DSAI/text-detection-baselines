"""Tests for the CLI in text_detection_baselines.cli."""

from __future__ import annotations

import json

from click.testing import CliRunner

from text_detection_baselines.cli import main, _flatten_overall, _flatten_per_category


def _write_jsonl(path, rows):
    import json as _json
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(_json.dumps(row) + "\n")


_SAMPLE_ROWS = [
    {"answer": "human text one with varied words and structure", "label": "real", "contribution_level": "Human"},
    {"answer": "human text two with punctuation, grammar, and detail!", "label": "real", "contribution_level": "Human"},
    {"answer": "machine answer one has repeated repeated words in this sentence here", "label": "fake", "contribution_level": "Summary"},
    {"answer": "machine answer two has repeated repeated patterns and extra length added", "label": "fake", "contribution_level": "Task"},
]


def test_cli_does_not_write_summary_txt(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(main, [
        "--dataset", str(dataset_path),
        "--model", "length-normalized",
        "--output-dir", str(output_dir),
    ])
    assert result.exit_code == 0, result.output
    assert not (output_dir / "summary.txt").exists()


def test_cli_csv_export(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(main, [
        "--dataset", str(dataset_path),
        "--model", "length-normalized",
        "--model", "torch-raw",
        "--output-dir", str(output_dir),
        "--export", "csv",
    ])
    assert result.exit_code == 0, result.output
    assert (output_dir / "overall-metrics.csv").exists()
    assert (output_dir / "per-category-metrics.csv").exists()


def test_cli_json_export_structure(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(main, [
        "--dataset", str(dataset_path),
        "--model", "length-normalized",
        "--output-dir", str(output_dir),
        "--export", "json",
    ])
    assert result.exit_code == 0, result.output
    metrics_file = output_dir / "metrics.json"
    assert metrics_file.exists()
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "overall" in data
    assert "per_category" in data
    assert "toy" in data["overall"]
    assert "length-normalized" in data["overall"]["toy"]
    assert "toy" in data["per_category"]


def test_cli_yaml_export(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(main, [
        "--dataset", str(dataset_path),
        "--model", "length-normalized",
        "--output-dir", str(output_dir),
        "--export", "yaml",
    ])
    assert result.exit_code == 0, result.output
    assert (output_dir / "metrics.yaml").exists()


def test_cli_multiple_datasets_and_models(tmp_path):
    ds1 = tmp_path / "ds1.jsonl"
    ds2 = tmp_path / "ds2.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(ds1, _SAMPLE_ROWS)
    _write_jsonl(ds2, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(main, [
        "--dataset", str(ds1),
        "--dataset", str(ds2),
        "--model", "length-normalized",
        "--model", "torch-raw",
        "--output-dir", str(output_dir),
        "--export", "json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "ds1" in data["overall"]
    assert "ds2" in data["overall"]
    assert "length-normalized" in data["overall"]["ds1"]
    assert "torch-raw" in data["overall"]["ds1"]


def test_flatten_helpers():
    tree = {
        "overall": {"ds1": {"model-a": {"auroc": 0.8}}},
        "per_category": {"ds1": {"Cat1": {"model-a": {"tpr_at_tau": 0.9}}}},
    }
    overall_rows = _flatten_overall(tree)
    assert len(overall_rows) == 1
    assert overall_rows[0]["dataset"] == "ds1"
    assert overall_rows[0]["model"] == "model-a"
    assert overall_rows[0]["auroc"] == 0.8

    per_cat_rows = _flatten_per_category(tree)
    assert len(per_cat_rows) == 1
    assert per_cat_rows[0]["category"] == "Cat1"
    assert per_cat_rows[0]["tpr_at_tau"] == 0.9
