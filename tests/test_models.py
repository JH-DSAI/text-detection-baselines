"""Tests for stub model implementations in text_detection_baselines.models."""

from __future__ import annotations

import numpy as np
import pytest

from text_detection_baselines.models import (
    MODEL_REGISTRY,
    build_model,
    build_stub_model,
    get_default_model_names,
    list_registered_models,
    register_model,
)
from text_detection_baselines.models.base import StubModelOutput, StubTextDetector
from text_detection_baselines.models.length_heuristic import LengthHeuristicStubDetector
from text_detection_baselines.models.torch_linear import TorchLinearStubDetector

_TEXTS = [
    "short",
    "a longer human-written text with varied vocabulary and diverse structure",
    "machine generated text with repeated repeated repeated repeated patterns",
]


def test_build_stub_model_length_normalized():
    model = build_stub_model("length-normalized", ood_margin=0.05, seed=1)
    assert isinstance(model, LengthHeuristicStubDetector)
    assert model.normalized_scores is True


def test_build_stub_model_torch_normalized():
    model = build_stub_model("torch-normalized", ood_margin=0.05, seed=1)
    assert isinstance(model, TorchLinearStubDetector)
    assert model.normalized_scores is True


def test_build_stub_model_torch_raw():
    model = build_stub_model("torch-raw", ood_margin=0.05, seed=1)
    assert isinstance(model, TorchLinearStubDetector)
    assert model.normalized_scores is False


def test_build_stub_model_invalid_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_stub_model("not-a-model", ood_margin=0.05, seed=1)


def test_length_heuristic_output_shape():
    model = LengthHeuristicStubDetector("length-normalized", normalized_scores=True, ood_margin=0.05, seed=1)
    output = model.predict(_TEXTS)
    assert isinstance(output, StubModelOutput)
    assert output.scores.shape == (3,)
    assert output.predictions.shape == (3,)
    assert output.ood_flags.shape == (3,)


def test_length_heuristic_scores_normalized():
    model = LengthHeuristicStubDetector("length-normalized", normalized_scores=True, ood_margin=0.0, seed=1)
    output = model.predict(_TEXTS)
    assert np.all(output.scores >= 0.0)
    assert np.all(output.scores <= 1.0)


def test_torch_raw_scores_unbounded():
    model = TorchLinearStubDetector("torch-raw", normalized_scores=False, ood_margin=0.0, seed=1)
    output = model.predict(_TEXTS)
    # Raw scores can be outside [0, 1]
    assert output.scores.shape == (3,)


def test_torch_normalized_scores_in_unit_interval():
    model = TorchLinearStubDetector("torch-normalized", normalized_scores=True, ood_margin=0.0, seed=1)
    output = model.predict(_TEXTS)
    assert np.all(output.scores >= 0.0)
    assert np.all(output.scores <= 1.0)


def test_determinism_same_seed():
    model1 = build_stub_model("torch-normalized", ood_margin=0.05, seed=42)
    model2 = build_stub_model("torch-normalized", ood_margin=0.05, seed=42)
    out1 = model1.predict(_TEXTS)
    out2 = model2.predict(_TEXTS)
    np.testing.assert_array_equal(out1.scores, out2.scores)


def test_feature_matrix_shape():
    feats = StubTextDetector._feature_matrix(_TEXTS)
    assert feats.shape == (3, 4)


def test_model_registry_defaults_present():
    names = list_registered_models()
    assert names == ["torch-normalized", "torch-raw", "length-normalized"]
    assert get_default_model_names() == tuple(names)


def test_build_model_from_registry():
    model = build_model("TORCH-RAW", ood_margin=0.05, seed=1)
    assert isinstance(model, TorchLinearStubDetector)
    assert model.normalized_scores is False


def test_register_model_runtime():
    def custom_factory(ood_margin: float, seed: int):
        return LengthHeuristicStubDetector("custom-length", normalized_scores=True, ood_margin=ood_margin, seed=seed)

    register_model("custom-length", custom_factory)
    try:
        model = build_model("custom-length", ood_margin=0.01, seed=2)
        assert isinstance(model, LengthHeuristicStubDetector)
        assert model.model_name == "custom-length"
    finally:
        MODEL_REGISTRY.pop("custom-length", None)
