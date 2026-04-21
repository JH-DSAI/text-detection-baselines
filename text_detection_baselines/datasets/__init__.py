"""Dataset loader registry and dispatch helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .file import FileDatasetBatch, load_file_dataset


DatasetLoader = Callable[[Path, str, str, str], FileDatasetBatch]
DATASET_LOADERS: dict[str, DatasetLoader] = {"file": load_file_dataset}


def load_dataset(
    *,
    dataset_type: str,
    path: Path,
    text_key: str,
    label_key: str,
    category_key: str,
) -> FileDatasetBatch:
    """Load a dataset by type into a reusable batch format."""
    if dataset_type not in DATASET_LOADERS:
        valid = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"Unknown dataset type '{dataset_type}'. Expected one of: {valid}")
    loader = DATASET_LOADERS[dataset_type]
    return loader(path, text_key, label_key, category_key)


__all__ = ["FileDatasetBatch", "DATASET_LOADERS", "load_dataset", "load_file_dataset"]
