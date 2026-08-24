"""End-to-end checks of the evaluation CLI.

Most CLI behaviour is covered in-process in ``test_cli.py``. What lives here is the
part in-process invocation cannot reach: the dataset registry is populated at import
(``datasets/__init__.py``), so the effect of ``TDB_GEDE_PATH`` on dataset resolution is
only observable in a fresh interpreter. That test doubles as the check that
``python -m text_detection_baselines.cli`` is wired up at all, covering the
``__main__`` guard, its logging setup, and real process exit codes.

Assertions are on exit status and exported data, not on console formatting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from text_detection_baselines.cli import main


def test_cli_evaluates_the_default_dataset_and_exports(runner, tmp_path):
    result = runner.invoke(main, ["--export", "json", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert set(metrics) == {"overall", "per-category"}
    assert set(metrics["overall"]) == {"demo"}
    for model_metrics in metrics["overall"]["demo"].values():
        assert model_metrics["n_samples"] == 200
        assert model_metrics["n_human"] + model_metrics["n_machine"] == 200
        assert model_metrics["ood_percent"] > 0

    # Both slice shapes reach the per-category table: a two-class category where the
    # ranking metrics are defined, and single-label ones where they are not.
    categories = metrics["per-category"]["demo"]
    assert categories["Mixed"]["dummy-norm"]["auroc"] is not None
    assert categories["Human"]["dummy-norm"]["auroc"] is None


def test_cli_reports_an_unprepared_dataset_without_a_traceback(tmp_path):
    env = dict(os.environ)
    env["TDB_GEDE_PATH"] = str(tmp_path / "never-prepared.jsonl")

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "text_detection_baselines.cli", "--dataset", "gede"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "prepare-gede" in combined
    # A ClickException, not an unhandled FileNotFoundError out of read_text.
    assert "Traceback" not in combined


def _write_runtime_dataset(path, text_field):
    """Write a small two-class JSONL whose text field is named ``text_field``."""
    rows = [
        {text_field: f"sample text number {idx} written for the key-override test", "label": idx % 2}
        for idx in range(20)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_text_key_applies_to_a_runtime_registered_dataset(tmp_path):
    """``--text-key`` must reach the loader for --register-file-dataset entries."""
    dataset = _write_runtime_dataset(tmp_path / "mine.jsonl", "text")

    result = _run(
        [
            "--register-file-dataset",
            f"mine={dataset}",
            # Runtime registrations are always selected; drop the default so this
            # case exercises the registered file on its own.
            "--exclude-dataset",
            "demo",
            "--text-key",
            "text",
            "--export",
            "json",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["overall"]) == {"mine"}
    for model_metrics in metrics["overall"]["mine"].values():
        assert model_metrics["n_samples"] == 20


def test_text_key_leaves_built_in_dataset_schemas_alone(tmp_path):
    """The flag describes runtime files only; ``demo`` keeps its own ``answer`` field."""
    dataset = _write_runtime_dataset(tmp_path / "mine.jsonl", "text")

    result = _run(
        [
            "--register-file-dataset",
            f"mine={dataset}",
            "--text-key",
            "text",
            "--export",
            "json",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["overall"]) == {"demo", "mine"}
    for model_metrics in metrics["overall"]["demo"].values():
        assert model_metrics["n_samples"] == 200
