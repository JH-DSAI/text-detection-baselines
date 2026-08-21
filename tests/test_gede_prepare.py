"""Tests for the upstream GEDE database converter.

Builds a miniature database with upstream's schema rather than requiring the real
corpus, which is not redistributable and cannot be checked in.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import click
import pytest

from text_detection_baselines.datasets import GEDE_PATH_ENV_VAR
from text_detection_baselines.datasets.file import load_file_dataset
from text_detection_baselines.datasets.gede import (
    GedePrepReport,
    GedeSourceError,
    metadata_path,
    normalize_contribution_level,
    prepare_gede,
)
from text_detection_baselines.prepare_gede import _summarize, main

# Mirrors the columns of upstream's database that the converter reads. Only those
# columns are declared; the real schema carries timestamps and other fields the
# query never touches.
_SCHEMA = """
CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE questions (id INTEGER PRIMARY KEY, dataset_id INTEGER, question TEXT);
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    model TEXT,
    prompt_mode TEXT,
    temperature REAL,
    max_new_tokens INTEGER
);
CREATE TABLE answers (
    id INTEGER PRIMARY KEY,
    question_id INTEGER,
    job_id INTEGER,
    is_human INTEGER,
    answer TEXT,
    rewrite_of TEXT
);
"""

# (job_id, model, prompt_mode) covering every category value upstream maps, plus
# one it does not: task+resource appears in the published corpus unmapped.
_JOBS = [
    (1, "gpt-4o-mini-2024-07-18", "task"),
    (2, "gpt-4o-mini-2024-07-18", "summary"),
    (3, "gpt-4o-mini-2024-07-18", "task+summary"),
    (4, "gpt-4o-mini-2024-07-18", "improve-human"),
    (5, "gpt-4o-mini-2024-07-18", "rewrite-human"),
    (6, "meta-llama/Llama-3.3-70B-Instruct", "rewrite-3"),
    (7, "dipper", "dipper-2"),
    (8, "gpt-4o-mini-2024-07-18", "task+resource"),
]


def _build_database(path, *, blank_aae: bool = False):
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript(_SCHEMA)
        connection.executemany(
            "INSERT INTO datasets (id, name) VALUES (?, ?)",
            [(1, "argument-annotated-essays"), (2, "BAWE")],
        )
        connection.executemany(
            "INSERT INTO questions (id, dataset_id, question) VALUES (?, ?, ?)",
            [(1, 1, "Prompt one?"), (2, 2, "Prompt two?")],
        )
        connection.executemany(
            "INSERT INTO jobs (id, model, prompt_mode, temperature, max_new_tokens) VALUES (?, ?, ?, 1.0, 512)",
            _JOBS,
        )

        # One human row: no job, so every job-derived column is NULL.
        rows = [(1, 2, None, 1, "A human essay about something.", None)]
        # One machine row per job. Row 2 belongs to the AAE dataset (question 1) and
        # is the one blanked to simulate a skipped add_aae_to_database.py.
        for offset, (job_id, _model, _mode) in enumerate(_JOBS):
            answer_id = 2 + offset
            question_id = 1 if answer_id == 2 else 2
            answer = "" if (blank_aae and answer_id == 2) else f"Machine text {answer_id}."
            rows.append((answer_id, question_id, job_id, 0, answer, "1"))

        connection.executemany(
            "INSERT INTO answers (id, question_id, job_id, is_human, answer, rewrite_of) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    connection.close()


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "Human"),
        ("human", "Human"),
        ("task", "Task"),
        ("summary", "Summary"),
        ("task+summary", "Task+Summary"),
        ("improve-human", "Improved-Human"),
        ("rewrite-human", "Rewrite-Human"),
        ("rewrite-3", "Rewrite-LLM"),
        ("rewrite-17", "Rewrite-LLM"),
        ("dipper-2", "Humanize"),
    ],
)
def test_normalize_contribution_level_matches_upstream(raw, expected):
    category, was_mapped = normalize_contribution_level(raw)
    assert category == expected
    assert was_mapped


def test_normalize_contribution_level_reports_unmapped_values():
    # Upstream's export script does not map this value, which is why the published
    # corpus carries it in lowercase. It must pass through, not be swallowed.
    category, was_mapped = normalize_contribution_level("task+resource")
    assert category == "task+resource"
    assert not was_mapped

    # "rewrite-human" must not be captured by the ^rewrite-\d+$ rule.
    assert normalize_contribution_level("rewrite-human") == ("Rewrite-Human", True)


def test_prepare_gede_emits_loadable_records(tmp_path):
    source = tmp_path / "database.db"
    _build_database(source)
    out = tmp_path / "gede.jsonl"

    report = prepare_gede(source, out)

    assert report.n_records == 9
    assert report.counts_by_label == {"fake": 8, "real": 1}
    assert report.counts_by_dataset == {"BAWE": 8, "argument-annotated-essays": 1}
    assert report.counts_by_category == {
        "Humanize": 1,
        "Human": 1,
        "Improved-Human": 1,
        "Rewrite-Human": 1,
        "Rewrite-LLM": 1,
        "Summary": 1,
        "Task": 1,
        "Task+Summary": 1,
        "task+resource": 1,
    }
    assert report.unmapped_categories == {"task+resource": 1}
    assert report.dropped_blank_answers == 0
    assert report.label_disagreements == 0

    # The emitted keys must be the loader's defaults, so no key overrides are needed.
    batch = load_file_dataset(out, text_key="answer", label_key="label", category_key="contribution_level")
    assert len(batch) == 9
    assert sorted(batch.labels.tolist()) == [0] + [1] * 8

    human = next(row for row in _read_jsonl(out) if row["label"] == "real")
    assert human["text_author"] == "human"
    assert human["contribution_level"] == "Human"
    assert human["temperature"] == "n/a"
    assert human["max_tokens"] == "n/a"
    assert human["rewrite_of"] == "human"
    assert human["question"] == "Prompt two?"


def test_prepare_gede_writes_provenance_sidecar(tmp_path):
    source = tmp_path / "database.db"
    _build_database(source)
    out = tmp_path / "gede.jsonl"

    report = prepare_gede(source, out)
    meta = json.loads(metadata_path(out).read_text(encoding="utf-8"))

    assert meta["n_records"] == report.n_records
    assert meta["counts_by_category"] == report.counts_by_category
    assert len(meta["source_sha256"]) == 64
    assert meta["prepared_at"]


def test_prepare_gede_rejects_blank_text_by_default(tmp_path):
    source = tmp_path / "database.db"
    _build_database(source, blank_aae=True)
    out = tmp_path / "gede.jsonl"

    with pytest.raises(GedeSourceError, match="add_aae_to_database"):
        prepare_gede(source, out)

    # Nothing partially written: a silently truncated corpus is worse than none.
    assert not out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_prepare_gede_can_drop_blank_text(tmp_path):
    source = tmp_path / "database.db"
    _build_database(source, blank_aae=True)
    out = tmp_path / "gede.jsonl"

    report = prepare_gede(source, out, allow_missing_aae=True)

    assert report.n_records == 8
    assert report.dropped_blank_answers == 1
    assert report.dropped_by_dataset == {"argument-annotated-essays": 1}
    assert all(row["answer"].strip() for row in _read_jsonl(out))


def test_prepare_gede_rejects_the_csv_export(tmp_path):
    # essays.csv has no label column at all, so the failure has to name the reason.
    source = tmp_path / "essays.csv"
    source.write_text("id,answer\n1,text\n", encoding="utf-8")

    with pytest.raises(GedeSourceError, match="is_human"):
        prepare_gede(source, tmp_path / "gede.jsonl")


def test_prepare_gede_rejects_a_zip_archive(tmp_path):
    source = tmp_path / "database.db.zip"
    source.write_bytes(b"PK\x03\x04not-really-a-zip")

    with pytest.raises(GedeSourceError, match="zip archive"):
        prepare_gede(source, tmp_path / "gede.jsonl")


def test_prepare_gede_rejects_a_missing_source(tmp_path):
    with pytest.raises(GedeSourceError, match="not found"):
        prepare_gede(tmp_path / "absent.db", tmp_path / "gede.jsonl")


def test_prepare_gede_rejects_a_non_sqlite_file(tmp_path):
    source = tmp_path / "database.db"
    source.write_text("this is not a database", encoding="utf-8")

    with pytest.raises(GedeSourceError, match="not an SQLite database"):
        prepare_gede(source, tmp_path / "gede.jsonl")


def test_prepare_gede_rejects_a_database_without_the_gede_schema(tmp_path):
    source = tmp_path / "database.db"
    connection = sqlite3.connect(source)
    with connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
    connection.close()

    with pytest.raises(GedeSourceError, match="GEDE schema"):
        prepare_gede(source, tmp_path / "gede.jsonl")


def _run_cli(args):
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "text_detection_baselines.prepare_gede", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_prepare_gede_cli_converts_and_prints_attribution(tmp_path):
    source = tmp_path / "database.db"
    _build_database(source)
    out = tmp_path / "gede.jsonl"

    result = _run_cli(["--source", str(source), "--out", str(out)])

    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert metadata_path(out).is_file()
    # The licence terms are the point of routing acquisition through the user.
    assert "CC BY-NC-SA 4.0" in result.stdout
    assert "BAWE" in result.stdout
    # Values upstream does not normalize must be surfaced, not silently accepted.
    assert "task+resource" in result.stdout


def test_prepare_gede_cli_reports_blank_text_without_a_traceback(tmp_path):
    source = tmp_path / "database.db"
    _build_database(source, blank_aae=True)
    out = tmp_path / "gede.jsonl"

    result = _run_cli(["--source", str(source), "--out", str(out)])

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "add_aae_to_database" in combined
    assert "Traceback" not in combined
    assert not out.exists()


def test_prepare_gede_cli_summary_reports_drops_and_disagreements(tmp_path):
    report = GedePrepReport(source=tmp_path / "db", source_sha256="0" * 64, output=tmp_path / "out.jsonl")
    report.n_records = 3
    report.counts_by_label = {"fake": 2, "real": 1}
    report.dropped_blank_answers = 2
    report.dropped_by_dataset = {"argument-annotated-essays": 2}
    report.label_disagreements = 1

    summary = _summarize(report)

    assert "Dropped 2 records" in summary
    assert "argument-annotated-essays: 2" in summary
    assert "1 records where is_human disagrees" in summary


def test_prepare_gede_command_callback_succeeds(tmp_path, capsys):
    # Calling the command's underlying function rather than click's CliRunner,
    # which AGENTS.md forbids.
    source = tmp_path / "database.db"
    _build_database(source)
    out = tmp_path / "gede.jsonl"

    main.callback(source=source, out=out, allow_missing_aae=False)

    captured = capsys.readouterr().out
    assert f"Wrote 9 records to {out}" in captured
    assert "CC BY-NC-SA 4.0" in captured


def test_prepare_gede_command_callback_raises_click_exception(tmp_path):
    with pytest.raises(click.ClickException, match="not found"):
        main.callback(source=tmp_path / "absent.db", out=tmp_path / "gede.jsonl", allow_missing_aae=False)


def test_prepare_gede_command_callback_defaults_the_output_path(tmp_path, monkeypatch, capsys):
    source = tmp_path / "database.db"
    _build_database(source)
    monkeypatch.setenv(GEDE_PATH_ENV_VAR, str(tmp_path / "resolved" / "gede_essays.jsonl"))

    main.callback(source=source, out=None, allow_missing_aae=False)

    assert (tmp_path / "resolved" / "gede_essays.jsonl").is_file()
    assert "resolved" in capsys.readouterr().out
