"""File-based dataset loader (GEDE JSON-lines / JSON-array)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Record field names of the GEDE schema, which the bundled datasets follow.
#: Single source of truth for the :class:`~text_detection_baselines.datasets.DatasetSpec`
#: defaults, the ``register_file_dataset`` defaults, and the CLI ``--*-key`` defaults;
#: re-exported from the package root, which is where callers should import them from.
DEFAULT_TEXT_KEY = "answer"
DEFAULT_LABEL_KEY = "label"
DEFAULT_CATEGORY_KEY = "contribution_level"
DEFAULT_QUESTION_KEY = "question"


class DatasetError(ValueError):
    """Raised when a dataset file cannot be loaded as given.

    A ``ValueError``, so callers that catch that still do. The message names the
    file, and the record where there is one, so it can be shown to the user as is.
    """


@dataclass(frozen=True)
class FileDatasetBatch:
    """Reusable in-memory dataset representation.

    This shape is convenient for transformer pipelines that need parallel arrays
    of text inputs, labels, and optional category metadata.
    """

    texts: list[str]
    labels: np.ndarray
    categories: np.ndarray
    questions: np.ndarray
    """Question per sample; empty string where the dataset has none."""

    def __len__(self) -> int:
        return len(self.texts)


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


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise DatasetError(f"Not UTF-8 text: {path}") from exc
    if not raw:
        raise DatasetError(f"Dataset is empty: {path}")

    if raw[0] == "[":
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(loaded, list):
            raise DatasetError(f"Expected JSON array in {path}")
        return [row for row in loaded if isinstance(row, dict)]

    records: list[dict[str, Any]] = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON on line {idx} in {path}") from exc
        if isinstance(row, dict):
            records.append(row)
    return records


def _optional_field(row: dict[str, Any], key: str, default: str) -> str:
    """Read an optional field as text, treating JSON ``null`` as missing.

    ``str(None)`` would otherwise turn a null into the literal text ``"None"``.
    """
    value = row.get(key)
    return default if value is None else str(value)


def load_file_dataset(
    path: Path,
    text_key: str,
    label_key: str,
    category_key: str,
    question_key: str = DEFAULT_QUESTION_KEY,
) -> FileDatasetBatch:
    """Load GEDE-style file datasets from JSONL or JSON-array files.

    Unlike *text_key* and *label_key*, a missing *question_key* does not skip the
    row: a dataset with no question is still evaluable, and loads with
    an empty question throughout. A ``null`` question or category reads the same
    as a missing one.

    Raises:
        DatasetError: If the file is not valid JSON or JSON lines, holds no
            usable rows, or has a row whose label is unrecognized or whose text
            is ``null``, which would otherwise be scored as the text ``"None"``.
    """
    records = _read_json_records(path)

    texts: list[str] = []
    labels: list[int] = []
    categories: list[str] = []
    questions: list[str] = []

    for number, row in enumerate(records, start=1):
        if text_key not in row or label_key not in row:
            continue
        if row[text_key] is None:
            raise DatasetError(f"Null {text_key!r} in record {number} of {path}")
        try:
            label = normalize_label(row[label_key])
        except ValueError as exc:
            raise DatasetError(
                f"Unsupported {label_key!r} value {row[label_key]!r} in record {number} of {path}"
            ) from exc
        texts.append(str(row[text_key]))
        labels.append(label)
        categories.append(_optional_field(row, category_key, "unknown"))
        questions.append(_optional_field(row, question_key, ""))

    if not texts:
        raise DatasetError(f"No records in {path} have both {text_key!r} and {label_key!r}")

    return FileDatasetBatch(
        texts=texts,
        labels=np.array(labels, dtype=int),
        categories=np.array(categories, dtype=object),
        questions=np.array(questions, dtype=object),
    )
