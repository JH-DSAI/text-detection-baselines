"""End-to-end CLI checks run as subprocesses.

``AGENTS.md`` forbids ``click.testing.CliRunner``; these run the real entry point
instead and assert only on exit status and the presence of expected keys, not on
console formatting.
"""

from __future__ import annotations

import json
import subprocess
import sys


def _run(args, env=None):
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "text_detection_baselines.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_evaluates_the_default_dataset_and_exports(tmp_path):
    result = _run(["--export", "json", "--output-dir", str(tmp_path)])

    assert result.returncode == 0, result.stderr
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


def test_cli_reports_an_unprepared_dataset_without_a_traceback(tmp_path, monkeypatch):
    import os

    env = dict(os.environ)
    env["TDB_GEDE_PATH"] = str(tmp_path / "never-prepared.jsonl")

    result = _run(["--dataset", "gede"], env=env)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "prepare-gede" in combined
    # A ClickException, not an unhandled FileNotFoundError out of read_text.
    assert "Traceback" not in combined
