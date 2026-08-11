"""Tests for the bundled synthetic ``demo`` dataset.

These assert the properties that make ``demo`` usable as the default dataset. Each
one is a proxy the generator enforces via a token budget; see the ``MAX_TOKENS``
comment in tools/make_demo_dataset.py.
"""

from __future__ import annotations

import collections

import numpy as np

from text_detection_baselines.datasets import (
    dataset_available,
    demo_dataset_path,
    get_dataset_spec,
    load_dataset,
)
from text_detection_baselines.models import build_model, get_default_model_names
from text_detection_baselines.util import find_package_location


def _demo_batch():
    spec = get_dataset_spec("demo")
    return load_dataset(
        dataset_type=spec.dataset_type,
        path=spec.path,
        text_key=spec.text_key,
        label_key=spec.label_key,
        category_key=spec.category_key,
    )


def test_demo_dataset_is_bundled_inside_the_package():
    # Must live inside the package rather than beside it: resolving via a path walk
    # out of the package is what made the previously bundled corpus vanish from an
    # installed wheel.
    assert demo_dataset_path().is_relative_to(find_package_location())
    assert dataset_available(get_dataset_spec("demo"))


def test_demo_dataset_has_both_classes_and_a_mixed_category():
    batch = _demo_batch()
    assert len(batch) == 200
    assert set(np.unique(batch.labels).tolist()) == {0, 1}

    by_category = collections.defaultdict(set)
    for label, category in zip(batch.labels, batch.categories):
        by_category[str(category)].add(int(label))

    # Both slice shapes must be present: single-label categories are where the
    # per-category ranking metrics are undefined, and a two-class category is
    # where they are defined.
    assert {0, 1} in by_category.values()
    assert any(len(labels) == 1 for labels in by_category.values())


def test_demo_dataset_scores_are_not_saturated():
    # dummy-norm keeps its sigmoid in float32, so long texts tie at exactly 1.0 and
    # the model reports a meaningless AUROC. Guarding the property directly rather
    # than trusting the generator's token budget to stay correct.
    batch = _demo_batch()
    for name in get_default_model_names():
        model = build_model(name, ood_margin=0.05, seed=7)
        output = model.predict(batch.texts)
        distinct = len(np.unique(output.scores))
        assert distinct > len(batch) // 2, f"{name} produced only {distinct} distinct scores"
        if model.normalized_scores:
            # Raw-score models are unbounded, so the ceiling only means anything here.
            assert output.scores.max() < 1.0, f"{name} saturated at 1.0"


def test_demo_dataset_exercises_ood_flagging():
    # With no OOD samples the restriction of every metric to ~ood_flags never runs
    # outside its own unit tests, and the OOD% column always reads 0.000.
    batch = _demo_batch()
    for name in get_default_model_names():
        output = build_model(name, ood_margin=0.05, seed=7).predict(batch.texts)
        n_ood = int(output.ood_flags.sum())
        assert 0 < n_ood < len(batch), f"{name} flagged {n_ood} of {len(batch)} samples OOD"
