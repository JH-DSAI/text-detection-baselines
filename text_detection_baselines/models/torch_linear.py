"""Torch-based linear stub detector."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import StubModelOutput, StubTextDetector

torch: Any
nn: Any
try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class TorchLinearStubDetector(StubTextDetector):
    """Linear detector stub backed by a PyTorch layer with fixed weights.

    Uses a fixed parameter initialisation so that scores are deterministic
    without any training loop.

    When *normalized_scores* is True the raw logit is passed through a sigmoid
    and scores are in ``[0, 1]``.  When False the raw logit is used directly as
    an unnormalized score.
    """

    def __init__(self, model_name: str, normalized_scores: bool, ood_margin: float, seed: int) -> None:
        super().__init__(model_name, normalized_scores, ood_margin, seed)
        weights = np.array([0.015, 0.09, -0.03, 1.3], dtype=np.float32)
        bias = np.float32(-2.1)

        self._np_weights = weights.astype(float)
        self._np_bias = float(bias)

        if torch is not None and nn is not None:
            torch.manual_seed(seed)
            self._layer: nn.Module | None = nn.Linear(4, 1, bias=True)
            with torch.no_grad():
                self._layer.weight[:] = torch.tensor(weights.reshape(1, -1))
                self._layer.bias[:] = torch.tensor([bias])
        else:
            self._layer = None

    def predict(self, texts: list[str]) -> StubModelOutput:
        feats = self._feature_matrix(texts)
        raw = self._forward(feats)

        if self.normalized_scores:
            scores = 1.0 / (1.0 + np.exp(-raw))
            preds = (scores >= 0.5).astype(int)
            uncertainty = np.abs(scores - 0.5)
        else:
            scores = raw
            preds = (scores >= 0.0).astype(int)
            scale = max(np.std(scores), 1e-6)
            uncertainty = np.abs(scores) / scale

        ood = (uncertainty < self.ood_margin) | (feats[:, 0] < 40)
        return StubModelOutput(scores=scores, predictions=preds, ood_flags=ood)

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        if self._layer is None:
            return feats @ self._np_weights + self._np_bias

        x = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            return self._layer(x).squeeze(1).cpu().numpy()
