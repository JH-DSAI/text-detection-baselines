"""Dataset loader registry and dispatch helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .file import FileDatasetBatch, load_file_dataset

DatasetLoader = Callable[[Path, str, str, str], FileDatasetBatch]
DATASET_LOADERS: dict[str, DatasetLoader] = {"file": load_file_dataset}


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for one named dataset instance."""

    name: str
    dataset_type: str
    path: Path
    text_key: str = "answer"
    label_key: str = "label"
    category_key: str = "contribution_level"


DATASET_REGISTRY: dict[str, DatasetSpec] = {}


def _canon(name: str) -> str:
    return name.strip().lower()


def register_dataset(spec: DatasetSpec) -> None:
    """Register or replace a named dataset instance."""
    DATASET_REGISTRY[_canon(spec.name)] = DatasetSpec(
        name=_canon(spec.name),
        dataset_type=spec.dataset_type,
        path=Path(spec.path),
        text_key=spec.text_key,
        label_key=spec.label_key,
        category_key=spec.category_key,
    )


def register_file_dataset(
    *,
    name: str,
    path: Path,
    text_key: str = "answer",
    label_key: str = "label",
    category_key: str = "contribution_level",
) -> None:
    """Register a file-backed dataset instance."""
    register_dataset(
        DatasetSpec(
            name=name,
            dataset_type="file",
            path=path,
            text_key=text_key,
            label_key=label_key,
            category_key=category_key,
        ),
    )


def get_dataset_spec(name: str) -> DatasetSpec:
    """Resolve a dataset name into a registered dataset spec."""
    key = _canon(name)
    if key not in DATASET_REGISTRY:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{name}'. Valid options: {valid}")
    return DATASET_REGISTRY[key]


def list_registered_datasets() -> list[str]:
    """Return registered dataset names in registration order."""
    return list(DATASET_REGISTRY.keys())


def get_default_dataset_names() -> tuple[str, ...]:
    """Return the default dataset set for CLI evaluation."""
    return ("gede",)


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


_DEFAULT_GEDE_PATH = Path(__file__).resolve().parents[2] / "datasets" / "gede_essays.json"
register_file_dataset(name="gede", path=_DEFAULT_GEDE_PATH)


__all__ = [
    "DATASET_LOADERS",
    "DATASET_REGISTRY",
    "DatasetSpec",
    "FileDatasetBatch",
    "get_dataset_spec",
    "get_default_dataset_names",
    "list_registered_datasets",
    "load_dataset",
    "load_file_dataset",
    "register_dataset",
    "register_file_dataset",
]
