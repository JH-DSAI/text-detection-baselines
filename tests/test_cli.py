"""Tests for the CLI in text_detection_baselines.cli."""

from __future__ import annotations

import json

from click.testing import CliRunner

import text_detection_baselines.cli as cli_module
from text_detection_baselines.cli import _flatten_overall, _flatten_per_category, main
from text_detection_baselines.models import build_model as registry_build_model
from text_detection_baselines.models.length_heuristic import LengthHeuristicStubDetector


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


_SAMPLE_ROWS = [
    {"answer": "human text one with varied words and structure", "label": "real", "contribution_level": "Human"},
    {"answer": "human text two with punctuation, grammar, and detail!", "label": "real", "contribution_level": "Human"},
    {
        "answer": "machine answer one has repeated repeated words in this sentence here",
        "label": "fake",
        "contribution_level": "Summary",
    },
    {
        "answer": "machine answer two has repeated repeated patterns and extra length added",
        "label": "fake",
        "contribution_level": "Task",
    },
]


def test_cli_csv_export(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "--dataset",
            "toy",
            "--model",
            "length-normalized",
            "--model",
            "torch-raw",
            "--output-dir",
            str(output_dir),
            "--export",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "overall-metrics.csv").exists()
    assert (output_dir / "per-category-metrics.csv").exists()


def test_cli_json_export_structure(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "--dataset",
            "toy",
            "--model",
            "length-normalized",
            "--output-dir",
            str(output_dir),
            "--export",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    metrics_file = output_dir / "metrics.json"
    assert metrics_file.exists()
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "overall" in data
    assert "per-category" in data
    assert "toy" in data["overall"]
    assert "length-normalized" in data["overall"]["toy"]
    assert "toy" in data["per-category"]


def test_cli_yaml_export(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "--dataset",
            "toy",
            "--model",
            "length-normalized",
            "--output-dir",
            str(output_dir),
            "--export",
            "yaml",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "metrics.yaml").exists()


def test_cli_multiple_datasets_and_models(tmp_path):
    ds1 = tmp_path / "ds1.jsonl"
    ds2 = tmp_path / "ds2.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(ds1, _SAMPLE_ROWS)
    _write_jsonl(ds2, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"ds1={ds1}",
            "--register-file-dataset",
            f"ds2={ds2}",
            "--dataset",
            "ds1",
            "--dataset",
            "ds2",
            "--model",
            "length-normalized",
            "--model",
            "torch-raw",
            "--output-dir",
            str(output_dir),
            "--export",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "ds1" in data["overall"]
    assert "ds2" in data["overall"]
    assert "length-normalized" in data["overall"]["ds1"]
    assert "torch-raw" in data["overall"]["ds1"]


def test_cli_runtime_dataset_registration_uses_default_models(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "--dataset",
            "toy",
            "--output-dir",
            str(output_dir),
            "--export",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(data["overall"]["toy"].keys()) == {"torch-normalized", "torch-raw", "length-normalized"}


def test_cli_runtime_dataset_without_preregistered_defaults(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "--no-default-datasets",
            "--model",
            "length-normalized",
            "--output-dir",
            str(output_dir),
            "--export",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(data["overall"].keys()) == {"toy"}


def test_cli_no_default_models_requires_explicit_selection(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "--dataset",
            "toy",
            "--no-default-models",
        ],
    )

    assert result.exit_code != 0
    assert "No models selected" in result.output


def test_cli_no_default_datasets_requires_some_selection():
    runner = CliRunner()
    result = runner.invoke(main, ["--no-default-datasets", "--model", "length-normalized"])

    assert result.exit_code != 0
    assert "No datasets selected" in result.output


def test_cli_help_lists_registered_datasets_and_models():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "gede" in result.output
    assert "torch-normalized" in result.output
    assert "torch-raw" in result.output
    assert "length-normalized" in result.output
    assert "smollm2-prompting" in result.output


def test_cli_all_models_includes_non_default_model(tmp_path, monkeypatch):
    def fake_build_model(name: str, ood_margin: float, seed: int):
        if name == "smollm2-prompting":
            return LengthHeuristicStubDetector(name, normalized_scores=True, ood_margin=ood_margin, seed=seed)
        return registry_build_model(name, ood_margin=ood_margin, seed=seed)

    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    monkeypatch.setattr(cli_module, "build_model", fake_build_model)
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "--dataset",
            "toy",
            "--all-models",
            "--export",
            "json",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(data["overall"]["toy"].keys()) == {
        "torch-normalized",
        "torch-raw",
        "length-normalized",
        "smollm2-prompting",
    }


def test_cli_all_datasets_includes_runtime_registered_datasets(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "--all-datasets",
            "--model",
            "length-normalized",
            "--export",
            "json",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(data["overall"].keys()) == {"gede", "toy"}


def test_cli_all_shortcut_enables_all_datasets_and_models(tmp_path, monkeypatch):
    def fake_build_model(name: str, ood_margin: float, seed: int):
        if name == "smollm2-prompting":
            return LengthHeuristicStubDetector(name, normalized_scores=True, ood_margin=ood_margin, seed=seed)
        return registry_build_model(name, ood_margin=ood_margin, seed=seed)

    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(dataset_path, _SAMPLE_ROWS)

    runner = CliRunner()
    monkeypatch.setattr(cli_module, "build_model", fake_build_model)
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"toy={dataset_path}",
            "-a",
            "--export",
            "json",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(data["overall"].keys()) == {"gede", "toy"}
    assert set(data["overall"]["toy"].keys()) == {
        "torch-normalized",
        "torch-raw",
        "length-normalized",
        "smollm2-prompting",
    }


def test_flatten_helpers():
    tree = {
        "overall": {"ds1": {"model-a": {"auroc": 0.8}}},
        "per-category": {"ds1": {"Cat1": {"model-a": {"tpr_at_tau": 0.9}}}},
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
