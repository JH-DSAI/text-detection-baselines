"""Base classes shared by all stub model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class StubModelOutput:
    """Outputs produced by a stub detector for a batch of texts."""

    scores: np.ndarray
    """Continuous detection score; higher → more likely machine-generated."""

    predictions: np.ndarray
    """Binary prediction (1 = machine, 0 = human)."""

    ood_flags: np.ndarray
    """True where the sample is flagged as out-of-distribution."""


class StubTextDetector(ABC):
    """Abstract interface for all detector implementations."""

    def __init__(self, model_name: str, normalized_scores: bool, ood_margin: float, seed: int) -> None:
        self.model_name = model_name
        self.normalized_scores = normalized_scores
        self.ood_margin = ood_margin
        self.seed = seed

    @abstractmethod
    def predict(self, question: str, answers: list[str]) -> StubModelOutput:
        """Score one assignment's submissions.

        The unit of invocation is an assignment, not a single text: one question
        with the one-or-more answers written in response to it. This mirrors a
        real-world application in education in which we might plausibly receive
        all submissions for an assignment at once, allowing us to utilize batch
        statistics in prediction.

        Args:
            question: The assignment prompt the answers respond to. May be empty
                for datasets that carry no prompt; detectors that need one are
                responsible for saying so.
            answers: The submissions to score, one per returned array element.

        Returns:
            A :class:`StubModelOutput` whose three arrays are parallel to
            *answers* and in the same order.
        """

    @staticmethod
    def _feature_matrix(texts: list[str]) -> np.ndarray:
        """Deterministic numeric features: char-length, token count, punct count, type-token ratio."""
        lengths = np.array([len(t) for t in texts], dtype=float)
        token_counts = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
        punct = np.array([sum(c in ".,!?:;" for c in t) for t in texts], dtype=float)
        unique_ratio = np.array(
            [len(set(t.split())) / max(len(t.split()), 1) for t in texts],
            dtype=float,
        )
        return np.column_stack((lengths, token_counts, punct, unique_ratio))
