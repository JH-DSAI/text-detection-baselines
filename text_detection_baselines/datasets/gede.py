"""Convert the upstream GEDE SQLite database into the JSONL this package loads.

The GEDE corpus is **not** redistributed with this package. Users obtain it
themselves from the upstream project, which makes them the party accepting the
CC BY-NC-SA 4.0 terms and the terms of the three source corpora. See
``datasets/README.md`` for the acquisition steps, the licence chain, and the
citations that use of the corpus requires.

This module reads the upstream database directly rather than the CSV that
upstream's ``database/export_to_csv.py`` produces, because that export drops the
``is_human`` column -- so ``essays.csv`` carries no label at all -- and does not
carry the question text or the generation parameters. Reading the database needs
only :mod:`sqlite3` from the standard library and is the only source with every
field this package's dataset schema expects.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .. import __version__

#: Bumped whenever the emitted schema or a normalization rule changes, so a
#: prepared file can be traced to the logic that produced it.
ADAPTER_VERSION = 1

_QUERY = """
SELECT
    answers.id            AS id,
    answers.question_id   AS question_id,
    answers.answer        AS answer,
    answers.rewrite_of    AS rewrite_of,
    answers.is_human      AS is_human,
    questions.question    AS question,
    datasets.name         AS dataset,
    jobs.prompt_mode      AS contribution_level,
    jobs.model            AS text_author,
    jobs.temperature      AS temperature,
    jobs.max_new_tokens   AS max_tokens
FROM answers
JOIN questions ON questions.id = answers.question_id
JOIN datasets  ON datasets.id  = questions.dataset_id
LEFT JOIN jobs ON jobs.id = answers.job_id
ORDER BY answers.id
"""

# Reproduces the ``contribution_level`` normalization in upstream's
# database/export_to_csv.py. Kept as data rather than a chain of replacements so
# that a value upstream does not map -- ``task+resource`` is one such value in the
# published corpus -- is detectable rather than silently passed through unnoticed.
_EXACT_CATEGORY_MAP = {
    "improve-human": "Improved-Human",
    "rewrite-human": "Rewrite-Human",
    "task": "Task",
    "summary": "Summary",
    "task+summary": "Task+Summary",
    "human": "Human",
}

_REGEX_CATEGORY_MAP = (
    (re.compile(r"^rewrite-\d+$"), "Rewrite-LLM"),
    (re.compile(r"^dipper-\d+$"), "Humanize"),
)

#: Value upstream substitutes for the generation parameters of human-written rows.
_NOT_APPLICABLE = "n/a"

_HUMAN_LABEL = "real"
_MACHINE_LABEL = "fake"

#: Field order of the emitted records. Matches the keys this package's dataset
#: loader defaults to (``answer`` / ``label`` / ``contribution_level``), so a
#: prepared file needs no key overrides.
FIELD_ORDER = (
    "id",
    "rewrite_of",
    "dataset",
    "contribution_level",
    "text_author",
    "question_id",
    "question",
    "answer",
    "temperature",
    "max_tokens",
    "label",
)


class GedeSourceError(ValueError):
    """Raised when the source database is missing, unreadable, or incomplete."""


@dataclass
class GedePrepReport:
    """Provenance and composition record for one prepare run."""

    source: Path
    source_sha256: str
    output: Path
    adapter_version: int = ADAPTER_VERSION
    package_version: str = __version__
    prepared_at: str = ""
    n_records: int = 0
    counts_by_label: dict[str, int] = field(default_factory=dict)
    counts_by_dataset: dict[str, int] = field(default_factory=dict)
    counts_by_category: dict[str, int] = field(default_factory=dict)
    unmapped_categories: dict[str, int] = field(default_factory=dict)
    dropped_blank_answers: int = 0
    dropped_by_dataset: dict[str, int] = field(default_factory=dict)
    label_disagreements: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the report."""
        return {
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "output": str(self.output),
            "adapter_version": self.adapter_version,
            "package_version": self.package_version,
            "prepared_at": self.prepared_at,
            "n_records": self.n_records,
            "counts_by_label": self.counts_by_label,
            "counts_by_dataset": self.counts_by_dataset,
            "counts_by_category": self.counts_by_category,
            "unmapped_categories": self.unmapped_categories,
            "dropped_blank_answers": self.dropped_blank_answers,
            "dropped_by_dataset": self.dropped_by_dataset,
            "label_disagreements": self.label_disagreements,
        }


def normalize_contribution_level(raw: Any) -> tuple[str, bool]:
    """Map an upstream ``prompt_mode`` to a GEDE category name.

    Args:
        raw: Raw ``jobs.prompt_mode`` value, or ``None`` for human-written rows.

    Returns:
        ``(category, was_mapped)``. Unrecognized values are returned unchanged
        with ``was_mapped`` false so the caller can report them.
    """
    if raw is None:
        return "Human", True

    value = str(raw).strip()
    if value in _EXACT_CATEGORY_MAP:
        return _EXACT_CATEGORY_MAP[value], True
    for pattern, replacement in _REGEX_CATEGORY_MAP:
        if pattern.match(value):
            return replacement, True
    return value, False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(source: Path) -> None:
    if not source.exists():
        raise GedeSourceError(
            f"GEDE source database not found: {source}\n"
            "Obtain it from the upstream project as described in datasets/README.md.",
        )
    suffix = source.suffix.lower()
    if suffix == ".zip":
        raise GedeSourceError(
            f"{source} is still a zip archive. Unpack it first (upstream ships "
            "database/database.db.zip), then run database/add_aae_to_database.py "
            "against the extracted database.db and pass that.",
        )
    if suffix == ".csv":
        raise GedeSourceError(
            f"{source} looks like upstream's CSV export, which cannot be used: "
            "export_to_csv.py drops the 'is_human' column, so the file carries no "
            "label. Pass the SQLite database.db instead.",
        )
    with source.open("rb") as handle:
        if handle.read(16) != b"SQLite format 3\x00":
            raise GedeSourceError(f"{source} is not an SQLite database file.")


def _connect_readonly(source: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _iter_rows(connection: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    try:
        cursor = connection.execute(_QUERY)
    except sqlite3.DatabaseError as exc:
        raise GedeSourceError(
            f"Could not query the GEDE schema (answers/questions/datasets/jobs): {exc}",
        ) from exc
    yield from cursor


def _build_record(row: sqlite3.Row, report: GedePrepReport) -> dict[str, Any]:
    category, was_mapped = normalize_contribution_level(row["contribution_level"])
    if not was_mapped:
        report.unmapped_categories[category] = report.unmapped_categories.get(category, 0) + 1

    text_author = "human" if row["text_author"] is None else str(row["text_author"])

    # Prefer the explicit is_human column; text_author is the fallback for rows
    # where it is absent. Disagreement between the two is reported rather than
    # silently resolved, because it would mean the corpus is not what we assume.
    by_author = text_author == "human"
    if row["is_human"] is None:
        is_human = by_author
    else:
        is_human = bool(row["is_human"])
        if is_human != by_author:
            report.label_disagreements += 1

    return {
        "id": row["id"],
        "rewrite_of": "human" if row["rewrite_of"] is None else str(row["rewrite_of"]),
        "dataset": str(row["dataset"]),
        "contribution_level": category,
        "text_author": text_author,
        "question_id": row["question_id"],
        "question": "" if row["question"] is None else str(row["question"]),
        "answer": str(row["answer"]),
        "temperature": _NOT_APPLICABLE if row["temperature"] is None else row["temperature"],
        "max_tokens": _NOT_APPLICABLE if row["max_tokens"] is None else row["max_tokens"],
        "label": _HUMAN_LABEL if is_human else _MACHINE_LABEL,
    }


def _blank_answer_message(dropped_by_dataset: dict[str, int]) -> str:
    breakdown = ", ".join(f"{name}: {count}" for name, count in sorted(dropped_by_dataset.items()))
    total = sum(dropped_by_dataset.values())
    return (
        f"{total} records have an empty 'answer' ({breakdown}).\n"
        "This almost always means database/add_aae_to_database.py has not been run "
        "against this database, which leaves the argument-annotated-essays texts "
        "blank. Run it and prepare again.\n"
        "Loading these records would score empty strings as real samples and "
        "silently corrupt every metric.\n"
        "Pass --allow-missing-aae to drop them instead and evaluate on the rest."
    )


def prepare_gede(source: Path, out: Path, *, allow_missing_aae: bool = False) -> GedePrepReport:
    """Convert the upstream GEDE database into a JSONL dataset file.

    Args:
        source: Path to the upstream ``database.db``, with the
            argument-annotated-essays texts already added.
        out: Path of the JSONL file to write. A ``<out>.meta.json`` provenance
            sidecar is written alongside it.
        allow_missing_aae: When true, records with an empty ``answer`` are
            dropped and counted instead of raising.

    Returns:
        The composition and provenance report for the run.

    Raises:
        GedeSourceError: If the source is missing, is the wrong kind of file,
            does not carry the GEDE schema, has records with empty text (unless
            *allow_missing_aae*), or yields no records at all.
    """
    source = Path(source)
    out = Path(out)
    _validate_source(source)

    report = GedePrepReport(
        source=source,
        source_sha256=_sha256(source),
        output=out,
        prepared_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    labels: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    dropped: Counter[str] = Counter()

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(out.name + ".partial")
    connection = _connect_readonly(source)
    try:
        with tmp_out.open("w", encoding="utf-8") as handle:
            for row in _iter_rows(connection):
                record = _build_record(row, report)
                if not record["answer"].strip():
                    dropped[record["dataset"]] += 1
                    continue
                labels[record["label"]] += 1
                datasets[record["dataset"]] += 1
                categories[record["contribution_level"]] += 1
                handle.write(json.dumps({key: record[key] for key in FIELD_ORDER}, ensure_ascii=False) + "\n")
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    if dropped and not allow_missing_aae:
        tmp_out.unlink(missing_ok=True)
        raise GedeSourceError(_blank_answer_message(dict(dropped)))

    if not labels:
        tmp_out.unlink(missing_ok=True)
        raise GedeSourceError(f"No usable records found in {source}.")

    report.n_records = sum(labels.values())
    report.counts_by_label = dict(sorted(labels.items()))
    report.counts_by_dataset = dict(sorted(datasets.items()))
    report.counts_by_category = dict(sorted(categories.items()))
    report.dropped_blank_answers = sum(dropped.values())
    report.dropped_by_dataset = dict(sorted(dropped.items()))

    tmp_out.replace(out)
    meta_path = metadata_path(out)
    meta_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report


def metadata_path(out: Path) -> Path:
    """Return the provenance sidecar path for a prepared dataset file."""
    return Path(out).with_name(Path(out).name + ".meta.json")


#: Printed on every successful prepare run. The corpus is licensed
#: CC BY-NC-SA 4.0, which obliges attribution and share-alike downstream, and the
#: three source corpora keep their own terms on top of that.
ATTRIBUTION_NOTICE = """
The GEDE corpus is licensed CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en):
attribution required, non-commercial use only, share-alike on derivatives.

It is derived from three corpora that retain their own terms, all of which apply
to your use of the prepared file:
  * BAWE (British Academic Written English)
  * Argument Annotated Essays (TU Darmstadt)
  * PERSUADE 2.0

Cite GEDE and all three source corpora. See datasets/README.md for the
citations and licence details. This package redistributes none of this text.
""".strip()
