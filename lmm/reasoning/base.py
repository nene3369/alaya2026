"""Reasoning base classes — FEP-integrated reasoning framework.

Provides BaseReasoner ABC, ReasonerResult, ComplexityProfile, and
information-geometric metrics (Gini, CV, entropy).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import sparse


ReasoningMode = Literal["adaptive", "theoretical", "hyper"]


@dataclass
class ComplexityProfile:
    """Input data complexity profile — information-geometric metrics."""

    gini: float
    cv: float
    entropy: float
    complexity_score: float


@dataclass
class ReasonerResult:
    """Unified reasoning result."""

    selected_indices: np.ndarray
    energy: float
    solver_used: str
    reasoning_mode: str
    steps_used: int = 0
    power_history: list[float] = field(default_factory=list)
    complexity: ComplexityProfile | None = None
    diagnostics: dict = field(default_factory=dict)


def compute_gini(values: np.ndarray) -> float:
    """Gini coefficient — O(n log n)."""
    n = len(values)
    if n == 0:
        return 0.0
    v = np.array(sorted(float(x) for x in values))
    total = float(np.sum(v))
    if total < 1e-15:
        return 0.0
    weighted = sum((i + 1) * float(v[i]) for i in range(n))
    gini = (2.0 * weighted) / (n * total) - (n + 1.0) / n
    return max(0.0, min(1.0, gini))


def compute_cv(values: np.ndarray) -> float:
    """Coefficient of variation = std / mean."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(float(x) for x in values) / n
    if abs(mean) < 1e-15:
        return 0.0
    variance = sum((float(x) - mean) ** 2 for x in values) / n
    return variance ** 0.5 / abs(mean)


def compute_entropy(values: np.ndarray) -> float:
    """Normalized Shannon entropy [0, 1]."""
    n = len(values)
    if n <= 1:
        return 0.0
    v = np.array([max(float(x), 0.0) for x in values])
    total = float(np.sum(v))
    if total < 1e-15:
        return 1.0
    probs = v / total
    h = sum(-float(p) * math.log(float(p)) for p in probs if float(p) > 1e-15)
    max_h = math.log(n)
    return min(1.0, h / max_h) if max_h > 1e-15 else 0.0


def compute_complexity(
    values: np.ndarray,
    *,
    w_gini: float = 0.4,
    w_cv: float = 0.3,
    w_entropy: float = 0.3,
) -> ComplexityProfile:
    """Compute complexity profile from value distribution."""
    gini = compute_gini(values)
    cv = compute_cv(values)
    entropy = compute_entropy(values)
    cv_norm = min(1.0, cv / 2.0)
    score = w_gini * gini + w_cv * cv_norm + w_entropy * (1.0 - entropy)
    return ComplexityProfile(
        gini=gini, cv=cv, entropy=entropy,
        complexity_score=max(0.0, min(1.0, score)),
    )


class BaseReasoner(ABC):
    """Base class for all reasoning modes."""

    def __init__(
        self,
        n_variables: int,
        k: int,
        *,
        nirvana_threshold: float = 1e-4,
    ):
        if n_variables <= 0:
            raise ValueError(f"n_variables must be positive, got {n_variables}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.n = n_variables
        self.k = min(k, n_variables)
        self.nirvana_threshold = nirvana_threshold

    @property
    @abstractmethod
    def mode(self) -> str:
        ...

    @abstractmethod
    def reason(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        **kwargs,
    ) -> ReasonerResult:
        ...

    def _empty_csr(self, n: int) -> sparse.csr_matrix:
        return sparse.csr_matrix(([], ([], [])), shape=(n, n))

    def _topk_from_activations(self, x_final: np.ndarray, k: int) -> np.ndarray:
        top_k_idx = np.argsort(-x_final)[:k]
        return np.array(sorted(int(i) for i in top_k_idx))

    def _evaluate_energy(
        self,
        h: np.ndarray,
        J: sparse.csr_matrix,
        sila_gamma: float,
        selected: np.ndarray,
    ) -> float:
        k = len(selected)
        if k == 0:
            return 0.0
        energy = sum(float(h[int(i)]) for i in selected)
        if J.nnz > 0:
            sel_set = set(int(i) for i in selected)
            for i in selected:
                row = J.getrow(int(i)).tocoo()
                for c, v in zip(row.col, row.data):
                    if int(c) in sel_set:
                        energy += float(v)
        if sila_gamma > 0:
            energy += sila_gamma * (k * k - k)
        return energy
