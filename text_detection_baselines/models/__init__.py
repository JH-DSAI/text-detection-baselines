"""Stub detector models for text detection baselines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .base import StubModelOutput, StubTextDetector
from .length_heuristic import LengthHeuristicStubDetector
from .prompting_smol import SmolLMPromptingDetector
from .torch_linear import TorchLinearStubDetector

__all__ = [
    "build_model",
    "build_stub_model",
    "get_default_model_names",
    "LengthHeuristicStubDetector",
    "list_registered_models",
    "MODEL_REGISTRY",
    "ModelSpec",
    "register_model",
    "SmolLMPromptingDetector",
    "StubModelOutput",
    "StubTextDetector",
    "TorchLinearStubDetector",
]


ModelFactory = Callable[[float, int], StubTextDetector]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing one registered model factory."""

    name: str
    factory: ModelFactory
    is_default: bool


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def _canon(name: str) -> str:
    return name.strip().lower()


def register_model(name: str, factory: ModelFactory, *, is_default: bool = True) -> None:
    """Register or replace a model factory."""
    key = _canon(name)
    MODEL_REGISTRY[key] = ModelSpec(name=key, factory=factory, is_default=is_default)


def list_registered_models() -> list[str]:
    """Return registered model names in registration order."""
    return list(MODEL_REGISTRY.keys())


def get_default_model_names() -> tuple[str, ...]:
    """Return default model names for CLI evaluation."""
    return tuple(spec.name for spec in MODEL_REGISTRY.values() if spec.is_default)


def build_model(model_name: str, ood_margin: float, seed: int) -> StubTextDetector:
    """Instantiate a registered model by name."""
    key = _canon(model_name)
    if key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Valid options: {valid}")
    return MODEL_REGISTRY[key].factory(ood_margin, seed)


def _torch_normalized_factory(ood_margin: float, seed: int) -> StubTextDetector:
    return TorchLinearStubDetector(
        model_name="fixed-linear-normalized", normalized_scores=True, ood_margin=ood_margin, seed=seed
    )


def _torch_raw_factory(ood_margin: float, seed: int) -> StubTextDetector:
    return TorchLinearStubDetector(
        model_name="fixed-linear-raw", normalized_scores=False, ood_margin=ood_margin, seed=seed
    )


def _length_normalized_factory(ood_margin: float, seed: int) -> StubTextDetector:
    return LengthHeuristicStubDetector(
        model_name="length-normalized",
        normalized_scores=True,
        ood_margin=ood_margin,
        seed=seed,
    )


def _smollm2_prompting_factory(ood_margin: float, seed: int) -> StubTextDetector:
    return SmolLMPromptingDetector(
        model_name="smollm2-prompting",
        normalized_scores=True,
        ood_margin=ood_margin,
        seed=seed,
    )


register_model("fixed-linear-normalized", _torch_normalized_factory)
register_model("fixed-linear-raw", _torch_raw_factory)
register_model("length-normalized", _length_normalized_factory)
register_model("smollm2-prompting", _smollm2_prompting_factory, is_default=False)


def build_stub_model(model_name: str, ood_margin: float, seed: int) -> StubTextDetector:
    """Backward-compatible alias for :func:`build_model`."""
    return build_model(model_name=model_name, ood_margin=ood_margin, seed=seed)
