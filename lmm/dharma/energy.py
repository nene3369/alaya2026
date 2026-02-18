"""Dharma Energy Terms — pluggable Buddhist-concept energy operators.

Each term declares its mathematical property (linear/submodular/supermodular/frustrated)
so the engine layer can auto-route to the optimal solver.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy import sparse

MathProperty = Literal["linear", "submodular", "supermodular", "frustrated"]


def _to_float_array(arr: np.ndarray) -> np.ndarray:
    return np.array([float(x) for x in arr])


def _pad_or_trim(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.zeros(n - len(arr))])


def _to_csr(graph) -> sparse.csr_matrix:
    if sparse.issparse(graph):
        return graph
    return sparse.csr_matrix(np.array(graph, dtype=float))


def _csr_scale_data(J: sparse.csr_matrix, factor: float) -> sparse.csr_matrix:
    coo = J.tocoo()
    return sparse.csr_matrix(
        (coo.data * factor, (coo.row, coo.col)), shape=J.shape,
    )


def _resize_sparse(g: sparse.csr_matrix, n: int) -> sparse.csr_matrix:
    """Resize sparse matrix to (n, n) by trimming or padding."""
    if g.shape[0] == n:
        return g
    coo = g.tocoo()
    if g.shape[0] > n:
        mask = (coo.row < n) & (coo.col < n)
        return sparse.csr_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=(n, n),
        )
    return sparse.csr_matrix((coo.data, (coo.row, coo.col)), shape=(n, n))


# ===================================================================
# Abstract base
# ===================================================================

class DharmaEnergyTerm(ABC):
    """Contract for all Buddhist energy terms."""

    def __init__(self, weight: float = 1.0):
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")
        self.weight = weight

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def math_property(self) -> MathProperty: ...

    @abstractmethod
    def build(self, n_variables: int) -> tuple[np.ndarray, sparse.csr_matrix | None]: ...

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name={self.name!r}, "
                f"property={self.math_property!r}, weight={self.weight})")


# ===================================================================
# Concrete terms
# ===================================================================

class DukkhaTerm(DharmaEnergyTerm):
    """Dukkha (Suffering/Surprise) — select high-surprise items. Linear."""

    def __init__(self, surprises: np.ndarray, weight: float = 1.0):
        super().__init__(weight)
        self.surprises = _to_float_array(surprises)

    @property
    def name(self) -> str:
        return "Dukkha (Surprise)"

    @property
    def math_property(self) -> MathProperty:
        return "linear"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        return -self.weight * _pad_or_trim(self.surprises, n), None


class PrajnaTerm(DharmaEnergyTerm):
    """Prajna (Wisdom) — maximise information diversity via entropy-weighted selection.

    Unlike Dukkha (raw surprise), Prajna favours items that contribute to
    a more *diverse* information set.  It scales surprises by how far each
    value deviates from the median — rewarding novel-but-not-extreme items.
    """

    def __init__(self, surprises: np.ndarray, weight: float = 1.0):
        super().__init__(weight)
        self.surprises = _to_float_array(surprises)

    @property
    def name(self) -> str:
        return "Prajna (Wisdom)"

    @property
    def math_property(self) -> MathProperty:
        return "linear"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        s = _pad_or_trim(self.surprises, n)
        # Manual median (compat with minimal numpy)
        sorted_s = sorted(float(x) for x in s) if len(s) > 0 else [0.0]
        mid = len(sorted_s) // 2
        median = sorted_s[mid] if len(sorted_s) % 2 == 1 else (sorted_s[mid - 1] + sorted_s[mid]) / 2
        # Favour items near but above median (diverse, not extreme)
        deviation = np.abs(s - median)
        max_dev = float(deviation.max()) if len(deviation) > 0 else 1.0
        diversity_weight = 1.0 - deviation / max(max_dev, 1e-10)
        return -self.weight * s * diversity_weight, None


class KarunaTerm(DharmaEnergyTerm):
    """Karuna (Compassion Synergy) — supermodular co-selection reward with temporal decay."""

    def __init__(
        self,
        impact_graph: sparse.csr_matrix | np.ndarray,
        weight: float = 0.5,
        decay_rate: float = 0.01,
    ):
        super().__init__(weight)
        self.graph = _to_csr(impact_graph)
        self.decay_rate = decay_rate
        self._age: int = 0

    @property
    def name(self) -> str:
        return "Karuna (Compassion Synergy)"

    @property
    def math_property(self) -> MathProperty:
        return "supermodular"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        # Temporal decay: reduce weight of old co-selection patterns
        effective_weight = self.weight * math.exp(-self.decay_rate * self._age)
        self._age += 1
        g = _resize_sparse(self.graph, n)
        row_sums = np.asarray(g.sum(axis=1)).flatten()
        h = effective_weight * row_sums
        J = _csr_scale_data(g, -2.0 * effective_weight)
        return h, J

    def reset_age(self) -> None:
        """Reset temporal decay counter (e.g. on new co-selection)."""
        self._age = 0


class MettaTerm(DharmaEnergyTerm):
    """Metta (Loving-kindness/Diversity) — submodular similarity penalty."""

    def __init__(self, similarity_graph: sparse.csr_matrix | np.ndarray, weight: float = 1.0):
        super().__init__(weight)
        self.graph = _to_csr(similarity_graph)

    @property
    def name(self) -> str:
        return "Metta (Diversity)"

    @property
    def math_property(self) -> MathProperty:
        return "submodular"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        g = _resize_sparse(self.graph, n)
        return np.zeros(n), _csr_scale_data(g, self.weight)


class KarmaTerm(DharmaEnergyTerm):
    """Karma (Attachment Penalty) — penalise historically over-selected items. Linear."""

    def __init__(self, history_counts: np.ndarray, weight: float = 1.0):
        super().__init__(weight)
        self.history = _to_float_array(history_counts)

    @property
    def name(self) -> str:
        return "Karma (Attachment Penalty)"

    @property
    def math_property(self) -> MathProperty:
        return "linear"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        return self.weight * _pad_or_trim(self.history, n), None


class SilaTerm(DharmaEnergyTerm):
    """Sila (Discipline) — cardinality constraint gamma*(sum(x)-k)^2. Frustrated."""

    def __init__(self, k: int, weight: float = 10.0):
        super().__init__(weight)
        self.k = k

    @property
    def name(self) -> str:
        return "Sila (Cardinality Constraint)"

    @property
    def math_property(self) -> MathProperty:
        return "frustrated"

    @property
    def sila_gamma(self) -> float:
        return self.weight

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        k = min(self.k, n)
        h = np.full(n, self.weight * (1.0 - 2.0 * k))
        return h, None

    def implicit_matvec(self, v: np.ndarray) -> np.ndarray:
        return self.weight * (float(np.sum(v)) * np.ones(len(v)) - v)

    def implicit_quadratic(self, x: np.ndarray) -> float:
        s = float(np.sum(x))
        return self.weight * (s * s - float(np.sum(x * x)))


class QueryDukkhaTerm(DharmaEnergyTerm):
    """QueryDukkha (RAG Surprise) — query-aware dynamic surprise. Linear."""

    def __init__(
        self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray,
        *, query_weight: float = 0.7, corpus_weight: float = 0.3, weight: float = 1.0,
    ):
        super().__init__(weight)
        self.query_embedding = np.array([float(x) for x in query_embedding])
        self.candidate_embeddings = candidate_embeddings
        self.query_weight = query_weight
        self.corpus_weight = corpus_weight

    @property
    def name(self) -> str:
        return "QueryDukkha (RAG Surprise)"

    @property
    def math_property(self) -> MathProperty:
        return "linear"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        emb = self.candidate_embeddings
        if len(emb) > n:
            emb = emb[:n]
        elif len(emb) < n:
            emb = np.concatenate([emb, np.zeros((n - len(emb), emb.shape[1]))])

        a_norms = np.clip(np.linalg.norm(emb, axis=1), 1e-10, None)
        b_norm = max(float(np.linalg.norm(self.query_embedding)), 1e-10)
        query_sim = np.clip((emb @ self.query_embedding) / (a_norms * b_norm), 0.0, None)

        norms = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10, None)
        normed = emb / norms
        try:
            sim_matrix = normed @ normed.T
        except (TypeError, ValueError):
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    d = float(np.dot(normed[i], normed[j]))
                    sim_matrix[i, j] = d
                    sim_matrix[j, i] = d
        np.fill_diagonal(sim_matrix, 0.0)
        corpus_sim = np.clip(np.sum(sim_matrix, axis=1) / max(n - 1, 1), 0.0, None)

        blended = self.query_weight * query_sim + self.corpus_weight * corpus_sim
        return -self.weight * blended, None


class UpekkhaTerm(DharmaEnergyTerm):
    """Upekkha (Equanimity) — penalise surprise deviation from mean. Linear."""

    def __init__(self, surprises: np.ndarray, weight: float = 0.5):
        super().__init__(weight)
        self.surprises = _to_float_array(surprises)

    @property
    def name(self) -> str:
        return "Upekkha (Equanimity)"

    @property
    def math_property(self) -> MathProperty:
        return "linear"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        s = _pad_or_trim(self.surprises, n)
        mean = float(s.mean()) if len(s) > 0 else 0.0
        return self.weight * (s - mean) ** 2, None
