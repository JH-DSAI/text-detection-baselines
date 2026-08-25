"""End-to-end check of the evaluation CLI in a fresh interpreter.

Everything the in-process :class:`click.testing.CliRunner` can reach lives in
``test_cli.py``. What is left here is the part it cannot: the dataset registry is
populated at import (``datasets/__init__.py``), so the effect of ``TDB_GEDE_PATH`` on
dataset resolution is only observable in a new process. That test doubles as the check
that ``python -m text_detection_baselines.cli`` is wired up at all, covering the
``__main__`` guard, its logging setup, and real process exit codes.

Assertions are on exit status and exported data, not on console formatting.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_cli_reports_an_unprepared_dataset_without_a_traceback(tmp_path):
    env = dict(os.environ)
    env["TDB_GEDE_PATH"] = str(tmp_path / "never-prepared.jsonl")

    result = subprocess.run(
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
