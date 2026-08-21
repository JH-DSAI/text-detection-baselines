"""Torch-based linear stub detector."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import StubModelOutput, StubTextDetector


class TorchLinearStubDetector(StubTextDetector):
    """Dummy detector stub backed by a PyTorch layer with arbitrary weights.

    The weights are hard-coded constants chosen by hand and fit to no data, so
    this model carries no detection signal: it exists to produce deterministic,
    realistic-looking scores for exercising the evaluation pipeline. Any
    apparent skill on a dataset is an artifact of that dataset's feature
    distribution, not of the model.

    When *normalized_scores* is True the raw logit is passed through a sigmoid
    and scores are in ``[0, 1]``.  When False the raw logit is used directly as
    an unnormalized score.
    """

    def __init__(self, model_name: str, normalized_scores: bool, ood_margin: float, seed: int) -> None:
        super().__init__(model_name, normalized_scores, ood_margin, seed)
        weights = np.array([0.015, 0.09, -0.03, 1.3], dtype=np.float32)
        bias = np.float32(-2.1)

        self._layer = nn.Linear(4, 1, bias=True)
        with torch.no_grad():
            # reshape to the parameter's own shape so a size mismatch raises
            # instead of broadcasting silently.
            self._layer.weight[:] = torch.tensor(weights).reshape(self._layer.weight.shape)
            self._layer.bias[:] = torch.tensor([bias])

    def predict(self, texts: list[str]) -> StubModelOutput:
        feats = self._feature_matrix(texts)
        raw = self._forward(feats)

        # ``confidence`` is distance from the decision boundary, so it is *low*
        # for middling scores and high for extreme ones.
        if self.normalized_scores:
            scores = 1.0 / (1.0 + np.exp(-raw))
            preds = (scores >= 0.5).astype(int)
            confidence = np.abs(scores - 0.5)
        else:
            scores = raw
            preds = (scores >= 0.0).astype(int)
            scale = max(np.std(scores), 1e-6)
            confidence = np.abs(scores) / scale

        # Flagged when the model is unconfident, or the text is too short to judge.
        ood = (confidence < self.ood_margin) | (feats[:, 0] < 40)
        return StubModelOutput(scores=scores, predictions=preds, ood_flags=ood)

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        x = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            return self._layer(x).squeeze(1).cpu().numpy()
