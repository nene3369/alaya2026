"""AlayaMemory — Hebbian synaptic storage (Alaya-vijnana)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class MemoryTrace:
    """A stored memory trace."""

    pattern: np.ndarray
    strength: float
    access_count: int = 0


class AlayaMemory:
    """Alaya consciousness — Hebbian associative memory store.

    Implements a Hopfield-like associative memory using Hebbian learning:
      dJ[i,j] = eta * x[i] * x[j]

    Stored patterns can be recalled through energy minimization.
    """

    def __init__(
        self,
        n_variables: int,
        *,
        learning_rate: float = 0.01,
        decay_rate: float = 0.001,
        max_patterns: int = 100,
    ):
        self.n = n_variables
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_patterns = max_patterns

        self._J = sparse.lil_matrix((n_variables, n_variables))
        self._patterns: list[MemoryTrace] = []

    def store(self, pattern: np.ndarray) -> None:
        """Store a pattern via Hebbian learning."""
        x = np.asarray(pattern).flatten()[:self.n]
        n = len(x)

        # Hebbian update: J += eta * x * x^T (outer product)
        for i in range(n):
            xi = float(x[i])
            if abs(xi) < 1e-8:
                continue
            for j in range(i + 1, n):
                xj = float(x[j])
                if abs(xj) < 1e-8:
                    continue
                delta = self.learning_rate * xi * xj
                self._J[i, j] = float(self._J[i, j]) + delta
                self._J[j, i] = float(self._J[j, i]) + delta

        self._patterns.append(MemoryTrace(pattern=x.copy(), strength=1.0))

        # Prune oldest if over capacity
        if len(self._patterns) > self.max_patterns:
            self._patterns.pop(0)

    def record_and_learn(
        self,
        pattern: np.ndarray,
        converged: bool,
        effective_eta: float | None = None,
    ) -> None:
        """Record experience with QUBO-aware Hebbian learning.

        Unlike :meth:`store` which uses **positive** coupling for Hopfield
        recall, this method uses **negative** coupling suitable for QUBO
        co-selection: negative J values encourage co-selection (lower energy).

        Only learns from converged (successful) experiences.

        Parameters
        ----------
        pattern : array-like
            Binary activation pattern (1 = selected, 0 = not selected).
        converged : bool
            Whether the reasoning process converged successfully.
            No learning occurs when *converged* is False.
        effective_eta : float, optional
            Override learning rate for this update.
        """
        if not converged:
            return

        eta = effective_eta if effective_eta is not None else self.learning_rate
        x = np.asarray(pattern).flatten()[:self.n]
        n = len(x)

        for i in range(n):
            xi = float(x[i])
            if abs(xi) < 1e-8:
                continue
            for j in range(i + 1, n):
                xj = float(x[j])
                if abs(xj) < 1e-8:
                    continue
                delta = -eta * xi * xj  # NEGATIVE for QUBO co-selection
                self._J[i, j] = float(self._J[i, j]) + delta
                self._J[j, i] = float(self._J[j, i]) + delta

        self._patterns.append(MemoryTrace(pattern=x.copy(), strength=1.0))

        # Prune oldest if over capacity
        if len(self._patterns) > self.max_patterns:
            self._patterns.pop(0)

    def recall(self, cue: np.ndarray, n_steps: int = 10) -> np.ndarray:
        """Recall a pattern from a partial cue via energy descent."""
        x = np.array(cue).flatten()[:self.n]
        J_csr = self._J.tocsr()

        for _ in range(n_steps):
            # Compute local field: h_i = sum_j J[i,j] * x[j]
            if J_csr.nnz > 0:
                field = J_csr @ x
            else:
                field = np.zeros(self.n)

            # Update: x = sign(field) where field is strong enough
            for i in range(len(x)):
                if abs(float(field[i])) > 0.1:
                    x[i] = 1.0 if float(field[i]) > 0 else 0.0

        return x

    def decay(self) -> None:
        """Apply temporal decay to all connections."""
        self._J *= (1.0 - self.decay_rate)

        # Decay pattern strengths
        remaining = []
        for trace in self._patterns:
            trace.strength *= (1.0 - self.decay_rate)
            if trace.strength > 0.01:
                remaining.append(trace)
        self._patterns = remaining

    def get_association_matrix(self) -> sparse.csr_matrix:
        """Return the current association matrix."""
        return self._J.tocsr()

    @property
    def n_patterns(self) -> int:
        return len(self._patterns)

    @property
    def total_strength(self) -> float:
        return sum(t.strength for t in self._patterns)
