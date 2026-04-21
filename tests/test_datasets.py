"""Tests for dataset loading subpackage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from text_detection_baselines.datasets import (
    get_dataset_spec,
    get_default_dataset_names,
    list_registered_datasets,
    load_dataset,
    register_file_dataset,
)
from text_detection_baselines.datasets.file import FileDatasetBatch, load_file_dataset


def test_load_file_dataset_single_doc_fixture():
    path = Path("tests/data/test_single_doc_per_author.json")
    batch = load_file_dataset(path, text_key="answer", label_key="label", category_key="contribution_level")
    assert isinstance(batch, FileDatasetBatch)
    assert len(batch) > 0
    assert isinstance(batch.texts[0], str)
    assert batch.labels.dtype == np.int64 or batch.labels.dtype == np.int32
    assert set(np.unique(batch.labels).tolist()) == {0, 1}
    assert "Human" in set(batch.categories.tolist())


def test_load_file_dataset_multi_doc_fixture():
    path = Path("tests/data/test_multi_docs_per_author.json")
    batch = load_file_dataset(path, text_key="answer", label_key="label", category_key="contribution_level")
    assert len(batch) > 0
    assert len(batch.texts) == len(batch.labels) == len(batch.categories)


def test_dataset_dispatch_file_type():
    path = Path("tests/data/test_single_doc_per_author.json")
    batch = load_dataset(
        dataset_type="file",
        path=path,
        text_key="answer",
        label_key="label",
        category_key="contribution_level",
    )
    assert isinstance(batch, FileDatasetBatch)


def test_dataset_dispatch_unknown_type_raises():
    path = Path("tests/data/test_single_doc_per_author.json")
    with pytest.raises(ValueError, match="Unknown dataset type"):
        load_dataset(
            dataset_type="not-a-type",
            path=path,
            text_key="answer",
            label_key="label",
            category_key="contribution_level",
        )


def test_default_dataset_registry_contains_gede():
    names = list_registered_datasets()
    assert "gede" in names
    assert get_default_dataset_names() == ("gede",)
    spec = get_dataset_spec("GEDE")
    assert spec.name == "gede"


def test_register_file_dataset_runtime(tmp_path):
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
