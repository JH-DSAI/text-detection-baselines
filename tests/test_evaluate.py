import json

from click.testing import CliRunner

from text_detection_baselines.evaluate import evaluate_model_on_dataset, main


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_evaluate_model_on_dataset_produces_required_metrics(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"answer": "short human text", "label": "real", "contribution_level": "Human"},
            {
                "answer": "this human sample is much longer and varied in style to look natural",
                "label": "real",
                "contribution_level": "Human",
            },
            {
                "answer": "machine generated response with repeated repeated repeated patterns",
                "label": "fake",
                "contribution_level": "Summary",
            },
            {
                "answer": "another synthetic answer that has a different shape and sentence cadence",
                "label": "fake",
                "contribution_level": "Task",
            },
        ],
    )

    model_name = "length-normalized"
    from text_detection_baselines.evaluate import build_stub_model

    model = build_stub_model(model_name, ood_margin=0.01, seed=7)
    metrics, per_category = evaluate_model_on_dataset(
        dataset_path=dataset_path,
        model=model,
        target_alpha=0.1,
        text_key="answer",
        label_key="label",
        category_key="contribution_level",
    )

    assert "auroc" in metrics
    assert "fpr_at_tau" in metrics
    assert "tpr_at_tau" in metrics
    assert "calibration_gap" in metrics
    assert "ood_percent" in metrics
    assert "brier" in metrics
    assert "ece" in metrics
    assert metrics["model"] == model_name
    assert metrics["dataset"] == "toy"

    by_category = {row["category"] for row in per_category}
    assert {"Human", "Summary", "Task"}.issubset(by_category)


def test_cli_writes_summary_and_export_files(tmp_path):
    dataset_path = tmp_path / "toy.jsonl"
    output_dir = tmp_path / "out"

    _write_jsonl(
        dataset_path,
        [
            {"answer": "human text one with varied words", "label": "real", "contribution_level": "Human"},
            {"answer": "human text two with punctuation!", "label": "real", "contribution_level": "Human"},
            {
                "answer": "machine answer one has repeated repeated words in this sentence",
                "label": "fake",
                "contribution_level": "Summary",
            },
            {
                "answer": "machine answer two has repeated repeated patterns and extra length",
                "label": "fake",
                "contribution_level": "Task",
            },
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--dataset",
            str(dataset_path),
            "--model",
            "length-normalized",
            "--model",
            "torch-raw",
            "--output-dir",
            str(output_dir),
            "--export",
            "csv",
            "--export",
            "json",
            "--export",
            "yaml",
            "--target-alpha",
            "0.1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "summary.txt").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "per_category.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "per_category.json").exists()
    assert (output_dir / "metrics.yaml").exists()
    assert (output_dir / "per_category.yaml").exists()
