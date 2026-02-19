"""Input validation helpers for the LMM core pipeline.

Centralised checks so that every entry point validates early
and fails with clear error messages.
"""

from __future__ import annotations

import math
import warnings

import numpy as np


def validate_array_finite(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if *arr* contains NaN or Inf."""
    arr = np.asarray(arr)
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        flat = arr.flatten()
        n_nan = sum(1 for v in flat if math.isnan(float(v)))
        n_inf = sum(1 for v in flat if math.isinf(float(v)))
        raise ValueError(
            f"{name} contains non-finite values: {n_nan} NaN, {n_inf} Inf"
        )


def validate_k(k: int, context: str = "k") -> None:
    """Raise ValueError if *k* < 1."""
    if k < 1:
        raise ValueError(f"{context} must be >= 1, got {k}")


def validate_nonneg(value: float, name: str) -> None:
    """Raise ValueError if *value* is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def warn_k_clamped(k: int, n: int) -> int:
    """Clamp *k* to *n*, emitting a warning when ``k > n``.

    Returns the (possibly clamped) value.
    """
    if k > n:
        warnings.warn(
            f"k={k} exceeds number of candidates n={n}; clamping to k={n}",
            stacklevel=3,
        )
        return n
    return k
