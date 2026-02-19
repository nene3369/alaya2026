"""Information-theoretic surprise calculator.

Surprise = -log P(x): lower prior probability → higher surprise.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from lmm._validation import validate_array_finite

SurpriseMethod = Literal["kl", "entropy", "bayesian"]


class SurpriseCalculator:
    """Compute surprise (information content) for observations.

    Methods: "kl", "entropy" (both use -log P(x)), "bayesian" (KL posterior||prior).
    """

    def __init__(self, method: SurpriseMethod = "kl"):
        method_clean = method.strip().lower()
        if method_clean != method:
            warnings.warn(
                f"surprise method {method!r} was normalized to {method_clean!r}",
                stacklevel=2,
            )
        self.method = method_clean
        self._prior: np.ndarray | None = None
        self._bin_edges: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> SurpriseCalculator:
        """Learn prior distribution from reference data."""
        validate_array_finite(data, "reference data")
        self._prior, self._bin_edges = self._estimate_distribution(data)
        return self

    def compute(self, observations: np.ndarray) -> np.ndarray:
        """Compute surprise for each observation."""
        if len(observations) == 0:
            return np.array([])
        validate_array_finite(observations, "observations")
        if self.method in ("kl", "entropy"):
            return self._surprise_from_prior(observations)
        if self.method == "bayesian":
            return self._bayesian_surprise(observations)
        raise ValueError(f"Unknown method: {self.method}")

    # -- internal --------------------------------------------------------

    def _to_scalar(self, observations: np.ndarray) -> np.ndarray:
        if observations.ndim > 1:
            return np.linalg.norm(observations, axis=1)
        return np.asarray(observations, dtype=np.float64)

    def _vectorized_bin_indices(
        self, observations: np.ndarray, n_bins: int = 50,
    ) -> np.ndarray:
        scalars = self._to_scalar(observations)
        if self._bin_edges is not None:
            edges_inner = self._bin_edges[1:-1]
            raw = np.searchsorted(edges_inner, scalars)
            return np.minimum(raw, n_bins - 1)
        return np.minimum((scalars * n_bins).astype(int) % n_bins, n_bins - 1)

    def _surprise_from_prior(self, observations: np.ndarray) -> np.ndarray:
        if self._prior is None:
            raise RuntimeError("Call fit() first")
        eps = 1e-10
        prior = np.clip(self._prior, eps, None)
        bin_indices = self._vectorized_bin_indices(observations)
        surprises = -np.log(prior[bin_indices] + eps)
        return np.clip(surprises, 0.0, None)

    def _bayesian_surprise(self, observations: np.ndarray) -> np.ndarray:
        if self._prior is None:
            raise RuntimeError("Call fit() first")
        eps = 1e-10
        n = len(observations)
        n_bins = 50
        surprises = np.zeros(n)
        prior = np.clip(self._prior, eps, None)
        log_prior = np.log(prior)
        counts = np.zeros(n_bins, dtype=np.float64)
        obs_scalar = self._to_scalar(observations)
        all_bin_indices = np.minimum(
            (obs_scalar * n_bins).astype(int) % n_bins, n_bins - 1,
        )
        # Incremental Bayesian update — cache cumulative sum
        cumsum = 0.0
        for i in range(n):
            counts[all_bin_indices[i]] += 1.0
            cumsum += 1.0
            posterior = np.clip(counts / cumsum, eps, None)
            surprises[i] = np.sum(posterior * (np.log(posterior) - log_prior))
        return surprises

    def _estimate_distribution(
        self, data: np.ndarray, n_bins: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        if data.ndim > 1:
            data = np.linalg.norm(data, axis=1)
        counts, bin_edges = np.histogram(data, bins=n_bins, density=False)
        return counts / (counts.sum() + 1e-10), bin_edges

    def _get_bin(self, value: float, n_bins: int = 50) -> int:
        if isinstance(value, np.ndarray):
            value = float(np.linalg.norm(value))
        if self._bin_edges is not None:
            idx = int(np.searchsorted(self._bin_edges[1:-1], value))
            return min(idx, n_bins - 1)
        return min(int(value * n_bins) % n_bins, n_bins - 1)
