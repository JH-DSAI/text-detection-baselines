"""Surface text features shared by the stub detectors.

A free function rather than a method on the detector base class: it depends on
nothing but its argument, and a general detector interface has no business
shipping one particular feature extractor. The real detectors compute their own
features remotely and never call this.
"""

from __future__ import annotations

import numpy as np

#: Column order of :func:`surface_features`. Callers index positionally, so the
#: order is part of the contract.
FEATURE_NAMES = ("char_length", "token_count", "punctuation_count", "type_token_ratio")


def surface_features(texts: list[str]) -> np.ndarray:
    """Compute deterministic surface statistics for each text.

    Args:
        texts: The texts to featurize.

    Returns:
        An ``(len(texts), 4)`` array whose columns are, in order,
        :data:`FEATURE_NAMES`: character length, whitespace token count, count of
        ``.,!?:;`` characters, and the type-token ratio. Token count is floored
        at 1 so the ratio stays defined for an empty text.
    """
    lengths = np.array([len(t) for t in texts], dtype=float)
    token_counts = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
    punct = np.array([sum(c in ".,!?:;" for c in t) for t in texts], dtype=float)
    unique_ratio = np.array(
        [len(set(t.split())) / max(len(t.split()), 1) for t in texts],
        dtype=float,
    )
    return np.column_stack((lengths, token_counts, punct, unique_ratio))
