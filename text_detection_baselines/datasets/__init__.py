"""Dataset loader registry and dispatch helpers."""

from __future__ import annotations

import os
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
    """Return the default dataset set for CLI evaluation.

    ``demo`` rather than ``gede``: ``gede`` is not redistributed with this
    package and has to be prepared first, so it cannot be a working default.
    """
    return ("demo",)


def dataset_available(spec: DatasetSpec) -> bool:
    """Report whether a registered dataset's file is actually present.

    Registration deliberately does not require the file to exist -- ``gede`` must
    stay selectable and discoverable in ``--help`` before it is prepared -- so
    callers check availability separately and can explain what is missing.
    """
    try:
        return spec.path.is_file() and spec.path.stat().st_size > 0
    except OSError:
        return False


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


#: Environment variable that overrides where the prepared GEDE file is looked for.
GEDE_PATH_ENV_VAR = "TDB_GEDE_PATH"

#: File name ``prepare-gede`` writes and the resolver looks for.
GEDE_FILENAME = "gede_essays.jsonl"

#: Command that produces the prepared file, quoted in error messages.
GEDE_PREPARE_HINT = "pixi run prepare-gede --source /path/to/database.db"

# The demo data lives *inside* the package, so it resolves identically from a
# source checkout and from an installed wheel. The previous ``parents[2]`` walk
# escaped the package and landed in site-packages once installed, where nothing
# was ever shipped.
_PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"


def demo_dataset_path() -> Path:
    """Return the path of the bundled synthetic ``demo`` dataset."""
    return _PACKAGE_DATA_DIR / "demo.jsonl"


def gede_cache_dir() -> Path:
    """Return the user cache directory prepared datasets default to."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return root / "text-detection-baselines"


def _checkout_datasets_dir() -> Path:
    """Return the source checkout's ``datasets/`` directory (may not exist)."""
    return Path(__file__).resolve().parents[2] / "datasets"


def _gede_path_override() -> Path | None:
    override = os.environ.get(GEDE_PATH_ENV_VAR)
    return Path(override).expanduser() if override else None


def resolve_gede_path() -> Path:
    """Resolve where a prepared GEDE dataset is expected to live.

    Checked in order: the :data:`GEDE_PATH_ENV_VAR` override, a prepared file in
    the source checkout's ``datasets/`` directory, then the user cache directory.
    The returned path is not required to exist -- use :func:`dataset_available`.
    """
    override = _gede_path_override()
    if override is not None:
        return override

    checkout_candidate = _checkout_datasets_dir() / GEDE_FILENAME
    if checkout_candidate.is_file():
        return checkout_candidate

    return gede_cache_dir() / GEDE_FILENAME


def default_gede_output_path() -> Path:
    """Return where ``prepare-gede`` writes when no output path is given.

    Chosen so that a first run lands somewhere :func:`resolve_gede_path` will
    subsequently find: the override if set, the source checkout's ``datasets/``
    directory when working from a clone, otherwise the user cache directory.
    """
    override = _gede_path_override()
    if override is not None:
        return override

    checkout_datasets = _checkout_datasets_dir()
    if checkout_datasets.is_dir():
        return checkout_datasets / GEDE_FILENAME

    return gede_cache_dir() / GEDE_FILENAME


register_file_dataset(name="demo", path=demo_dataset_path())
register_file_dataset(name="gede", path=resolve_gede_path())


__all__ = [
    "DATASET_LOADERS",
    "DATASET_REGISTRY",
    "GEDE_FILENAME",
    "GEDE_PATH_ENV_VAR",
    "GEDE_PREPARE_HINT",
    "DatasetSpec",
    "FileDatasetBatch",
    "dataset_available",
    "default_gede_output_path",
    "demo_dataset_path",
    "gede_cache_dir",
    "get_dataset_spec",
    "get_default_dataset_names",
    "list_registered_datasets",
    "load_dataset",
    "load_file_dataset",
    "register_dataset",
    "register_file_dataset",
    "resolve_gede_path",
]
