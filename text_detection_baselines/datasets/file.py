"""File-based dataset loader (GEDE JSON-lines / JSON-array)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FileDatasetBatch:
    """Reusable in-memory dataset representation.

    This shape is convenient for transformer pipelines that need parallel arrays
    of text inputs, labels, and optional category metadata.
    """

    texts: list[str]
    labels: np.ndarray
    categories: np.ndarray

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
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset is empty: {path}")

    if raw[0] == "[":
        loaded = json.loads(raw)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected JSON array in {path}")
        return [row for row in loaded if isinstance(row, dict)]

    records: list[dict[str, Any]] = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {idx} in {path}") from exc
        if isinstance(row, dict):
            records.append(row)
    return records


def load_file_dataset(path: Path, text_key: str, label_key: str, category_key: str) -> FileDatasetBatch:
    """Load GEDE-style file datasets from JSONL or JSON-array files."""
    records = _read_json_records(path)

    texts: list[str] = []
    labels: list[int] = []
    categories: list[str] = []

    for row in records:
        if text_key not in row or label_key not in row:
            continue
        texts.append(str(row[text_key]))
        labels.append(normalize_label(row[label_key]))
        categories.append(str(row.get(category_key, "unknown")))

    if not texts:
        raise ValueError(f"No valid samples with required keys in {path}")

    return FileDatasetBatch(
        texts=texts,
        labels=np.array(labels, dtype=int),
        categories=np.array(categories, dtype=object),
    )
