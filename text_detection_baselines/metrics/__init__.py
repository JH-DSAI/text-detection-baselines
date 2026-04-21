"""Metric registry and utility runner.

Metric functions are registered with :func:`register_metric` and discovered
automatically by :func:`run_all_metrics`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


MetricFunc = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float],
    float | None,
]


@dataclass(frozen=True)
class MetricSpec:
    """Metadata describing one registered metric function."""

    name: str
    func: MetricFunc
    requires_normalized_scores: bool


METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(
    name: str | None = None,
    *,
    requires_normalized_scores: bool = False,
) -> Callable[[MetricFunc], MetricFunc]:
    """Register a metric function for discovery by :func:`run_all_metrics`."""

    def decorator(func: MetricFunc) -> MetricFunc:
        metric_name = name or func.__name__
        METRIC_REGISTRY[metric_name] = MetricSpec(
            name=metric_name,
            func=func,
            requires_normalized_scores=requires_normalized_scores,
        )
        return func

    return decorator


def safe_round(value: float | None, ndigits: int = 4) -> float | None:
    """Round a numeric value while preserving None / NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), ndigits)


def run_all_metrics(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    ood_flags: np.ndarray,
    flags: np.ndarray,
    target_alpha: float,
    tau: float,
    normalized_scores: bool,
) -> dict[str, float | None]:
    """Run all registered metrics and return a metric-name to value mapping."""
    results: dict[str, float | None] = {}
    for name, spec in METRIC_REGISTRY.items():
        if spec.requires_normalized_scores and not normalized_scores:
            continue
        value = spec.func(labels, scores, ood_flags, flags, target_alpha, tau)
        results[name] = safe_round(value)
    return results


def list_registered_metrics() -> list[str]:
    """Return metric names in registration order."""
    return list(METRIC_REGISTRY.keys())


# Import side effects register default metrics.
from . import calibration as _calibration  # noqa: E402,F401
from . import detection as _detection  # noqa: E402,F401
