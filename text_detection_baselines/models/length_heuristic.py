"""Length-based heuristic stub detector (no PyTorch dependency)."""

from __future__ import annotations

import numpy as np

from .base import StubModelOutput, StubTextDetector


class LengthHeuristicStubDetector(StubTextDetector):
    """Heuristic detector that scores texts with a hand-crafted formula.

    Scores are always normalized to ``[0, 1]`` via a sigmoid.

    The heuristic assumes that machine-generated texts tend to be longer,
    have lower type-token ratio (more repetition), and use fewer hard
    punctuation marks relative to their length.
    """

    def predict(self, texts: list[str]) -> StubModelOutput:
        feats = self._feature_matrix(texts)
        length = feats[:, 0]
        token_count = feats[:, 1]
        punct = feats[:, 2]
        unique_ratio = feats[:, 3]

        raw = (length / np.maximum(token_count, 1)) * 0.45 + (1 - unique_ratio) * 1.4 - punct * 0.02 - 2.0
        scores = 1.0 / (1.0 + np.exp(-raw))
        preds = (scores >= 0.5).astype(int)
        ood = (np.abs(scores - 0.5) < self.ood_margin) | (length < 40)
        return StubModelOutput(scores=scores, predictions=preds, ood_flags=ood)
