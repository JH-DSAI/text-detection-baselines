"""Tests for dataset loading subpackage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from text_detection_baselines.datasets import load_dataset
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
