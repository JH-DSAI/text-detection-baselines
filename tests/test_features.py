"""Tests for the shared surface-feature extractor."""

from __future__ import annotations

import numpy as np

from text_detection_baselines.models.features import FEATURE_NAMES, surface_features

_TEXTS = [
    "short",
    "a longer human-written text with varied vocabulary and diverse structure",
    "machine generated text with repeated repeated repeated repeated patterns",
]


def test_surface_features_shape():
    feats = surface_features(_TEXTS)
    assert feats.shape == (len(_TEXTS), len(FEATURE_NAMES))


def test_surface_features_column_order():
    # The stub detectors index these columns positionally, so the order is part
    # of the contract rather than an implementation detail.
    feats = surface_features(["Hi there, friend!"])
    char_length, token_count, punctuation, type_token_ratio = feats[0]

    assert char_length == 17
    assert token_count == 3
    assert punctuation == 2  # the comma and the exclamation mark
    assert type_token_ratio == 1.0  # three distinct tokens


def test_surface_features_ratio_is_defined_for_an_empty_text():
    # Token count is floored at 1 so the ratio does not divide by zero.
    feats = surface_features([""])
    assert feats[0][1] == 1.0
    assert np.isfinite(feats[0][3])
