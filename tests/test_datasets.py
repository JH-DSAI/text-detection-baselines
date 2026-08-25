"""Tests for dataset loading subpackage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from text_detection_baselines.datasets import (
    GEDE_PATH_ENV_VAR,
    dataset_available,
    get_dataset_spec,
    get_default_dataset_names,
    list_registered_datasets,
    load_dataset,
    register_file_dataset,
    resolve_gede_path,
)
from text_detection_baselines.datasets.file import FileDatasetBatch, load_file_dataset

_ROWS = [
    {"answer": "A short human answer.", "label": "real", "contribution_level": "Human"},
    {"answer": "Another human answer, longer than the first.", "label": "real", "contribution_level": "Human"},
    {"answer": "In conclusion, a balanced approach is essential.", "label": "fake", "contribution_level": "Task"},
    {"answer": "Furthermore it is important to consider stakeholders.", "label": "fake", "contribution_level": "Task"},
    {"answer": "A human answer in a mixed slice.", "label": "real", "contribution_level": "Mixed"},
    {"answer": "Moreover the evidence demonstrates a clear trend.", "label": "fake", "contribution_level": "Mixed"},
]


def _write_records(path: Path, *, as_array: bool) -> Path:
    """Write :data:`_ROWS` as a JSON array or as JSON lines."""
    if as_array:
        path.write_text(json.dumps(_ROWS), encoding="utf-8")
    else:
        path.write_text("".join(json.dumps(row) + "\n" for row in _ROWS), encoding="utf-8")
    return path


# Both encodings are named explicitly: _read_json_records branches on the first
# character, not the file extension, and the JSON-array branch has no other caller
# in the suite.
@pytest.mark.parametrize("as_array", [True, False], ids=["json-array", "json-lines"])
def test_load_file_dataset_reads_both_encodings(tmp_path, as_array):
    path = _write_records(tmp_path / "data.json", as_array=as_array)
    batch = load_file_dataset(path, text_key="answer", label_key="label", category_key="contribution_level")
    assert isinstance(batch, FileDatasetBatch)
    assert len(batch) == len(_ROWS)
    assert len(batch.texts) == len(batch.labels) == len(batch.categories)
    assert isinstance(batch.texts[0], str)
    assert batch.labels.dtype == np.int64 or batch.labels.dtype == np.int32
    assert set(np.unique(batch.labels).tolist()) == {0, 1}
    assert "Human" in set(batch.categories.tolist())


def test_dataset_dispatch_file_type(tmp_path):
    path = _write_records(tmp_path / "data.jsonl", as_array=False)
    batch = load_dataset(
        dataset_type="file",
        path=path,
        text_key="answer",
        label_key="label",
        category_key="contribution_level",
    )
    assert isinstance(batch, FileDatasetBatch)


def test_dataset_dispatch_unknown_type_raises(tmp_path):
    path = _write_records(tmp_path / "data.jsonl", as_array=False)
    with pytest.raises(ValueError, match="Unknown dataset type"):
        load_dataset(
            dataset_type="not-a-type",
            path=path,
            text_key="answer",
            label_key="label",
            category_key="contribution_level",
        )


def test_default_dataset_is_the_bundled_demo_set():
    # gede cannot be the default: it is not redistributed and has to be prepared.
    names = list_registered_datasets()
    assert "demo" in names
    assert get_default_dataset_names() == ("demo",)
    spec = get_dataset_spec("DEMO")
    assert spec.name == "demo"


def test_gede_stays_registered_when_not_prepared():
    # Registration must not depend on the file existing, so the dataset remains
    # discoverable in --help and selectable by name before it is prepared.
    spec = get_dataset_spec("GEDE")
    assert spec.name == "gede"
    assert spec.dataset_type == "file"


def test_dataset_available_reports_missing_and_empty_files(clean_registry, tmp_path):
    missing = tmp_path / "absent.jsonl"
    register_file_dataset(name="availability-missing", path=missing)
    assert not dataset_available(get_dataset_spec("availability-missing"))

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    register_file_dataset(name="availability-empty", path=empty)
    assert not dataset_available(get_dataset_spec("availability-empty"))

    present = tmp_path / "present.jsonl"
    present.write_text('{"answer":"a","label":"real","contribution_level":"Human"}\n', encoding="utf-8")
    register_file_dataset(name="availability-present", path=present)
    assert dataset_available(get_dataset_spec("availability-present"))


def test_resolve_gede_path_honours_the_environment_override(monkeypatch, tmp_path):
    override = tmp_path / "elsewhere" / "gede_essays.jsonl"
    monkeypatch.setenv(GEDE_PATH_ENV_VAR, str(override))
    assert resolve_gede_path() == override


def test_resolve_gede_path_falls_back_to_the_cache_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(GEDE_PATH_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    resolved = resolve_gede_path()
    # Only asserted when the checkout has no prepared file, which is the state of a
    # fresh clone and of CI.
    if not (Path(__file__).resolve().parents[1] / "datasets" / "gede_essays.jsonl").is_file():
        assert resolved == tmp_path / "text-detection-baselines" / "gede_essays.jsonl"


def test_register_file_dataset_runtime(clean_registry, tmp_path):
    path = tmp_path / "runtime.jsonl"
    path.write_text(
        '{"answer":"a","label":"real","contribution_level":"Human"}\n'
        '{"answer":"b","label":"fake","contribution_level":"Task"}\n',
        encoding="utf-8",
    )

    register_file_dataset(name="runtime-test", path=path)
    spec = get_dataset_spec("runtime-test")
    assert spec.path == path
    batch = load_dataset(
        dataset_type=spec.dataset_type,
        path=spec.path,
        text_key=spec.text_key,
        label_key=spec.label_key,
        category_key=spec.category_key,
    )
    assert len(batch) == 2
