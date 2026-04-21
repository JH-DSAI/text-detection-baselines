"""Stub detector models for text detection baselines."""

from __future__ import annotations

from .base import StubModelOutput, StubTextDetector
from .length_heuristic import LengthHeuristicStubDetector
from .torch_linear import TorchLinearStubDetector

__all__ = [
    "LengthHeuristicStubDetector",
    "StubModelOutput",
    "StubTextDetector",
    "TorchLinearStubDetector",
    "build_stub_model",
]

_VALID_MODELS = ("torch-normalized", "torch-raw", "length-normalized")


def build_stub_model(model_name: str, ood_margin: float, seed: int) -> StubTextDetector:
    """Instantiate a named stub detector.

    Args:
        model_name: One of ``"torch-normalized"``, ``"torch-raw"``,
            ``"length-normalized"``.
        ood_margin: Uncertainty threshold below which samples are flagged OOD.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`StubTextDetector` instance ready to call ``.predict()``.
    """
    if model_name == "torch-normalized":
        return TorchLinearStubDetector(model_name=model_name, normalized_scores=True, ood_margin=ood_margin, seed=seed)
    if model_name == "torch-raw":
        return TorchLinearStubDetector(model_name=model_name, normalized_scores=False, ood_margin=ood_margin, seed=seed)
    if model_name == "length-normalized":
        return LengthHeuristicStubDetector(
            model_name=model_name,
            normalized_scores=True,
            ood_margin=ood_margin,
            seed=seed,
        )
    raise ValueError(f"Unknown model '{model_name}'. Valid options: {', '.join(_VALID_MODELS)}")
