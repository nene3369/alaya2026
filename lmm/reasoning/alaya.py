"""AlayaMemory — Modern Hopfield Network with Hebbian learning (Alaya-vijnana).

Upgraded from classical Hopfield (capacity ~0.14N) to Modern Hopfield Network
(Ramsauer et al. 2020) with exponential capacity via softmax attention.

Recall mechanism:
  Classical: x_{t+1} = sign(J @ x_t)           capacity O(N)
  Modern:    x_{t+1} = X^T @ softmax(beta * X @ x_t)  capacity exp(N)
"""

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
    """Alaya consciousness — Modern Hopfield associative memory.

    Combines Hebbian learning (for J matrix compatibility) with
    Modern Hopfield recall (softmax attention over stored patterns).

    Parameters
    ----------
    n_variables : int
        Dimensionality of patterns.
    beta : float
        Inverse temperature for softmax attention. Higher = sharper recall.
        Default 8.0 gives good separation for N=8 dimensions.
    learning_rate : float
        Hebbian learning rate for J matrix updates.
    decay_rate : float
        Temporal decay rate for patterns and J matrix.
    max_patterns : int
        Maximum number of stored patterns (FIFO eviction).
    """

    def __init__(
        self,
        n_variables: int,
        *,
        beta: float = 8.0,
        learning_rate: float = 0.01,
        decay_rate: float = 0.001,
        max_patterns: int = 100,
        use_hebbian: bool = False,
    ):
        self.n = n_variables
        self.beta = beta
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_patterns = max_patterns
        self.use_hebbian = use_hebbian

        self._J = sparse.lil_matrix((n_variables, n_variables))
        self._patterns: list[MemoryTrace] = []

    def store(self, pattern: np.ndarray) -> None:
        """Store a pattern. Hebbian J update only when use_hebbian=True."""
        x = np.asarray(pattern).flatten()[:self.n]
        n = len(x)

        # Hebbian update: J += eta * x * x^T (only if opt-in)
        if self.use_hebbian:
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
        """Recall a pattern via Modern Hopfield softmax attention.

        Uses iterative softmax attention over stored patterns:
          alpha = softmax(beta * X @ xi)     (attention weights)
          xi    = X^T @ alpha                (weighted pattern retrieval)

        This gives exponential storage capacity compared to classical
        Hopfield's linear capacity (~0.14N).

        Falls back to classical Hopfield recall when J matrix has
        stored couplings but no explicit patterns exist.
        """
        xi = np.asarray(cue).flatten()[:self.n].copy()

        # Modern Hopfield: softmax attention over stored patterns
        if self._patterns:
            # Build pattern matrix X (M x N) weighted by strength
            X = np.array([t.pattern for t in self._patterns])  # (M, N)
            strengths = np.array([t.strength for t in self._patterns])  # (M,)

            for _ in range(n_steps):
                # Compute similarity: scores = X @ xi (M,)
                scores = X @ xi
                # Adaptive beta: scale by inverse std of scores
                _mean = float(np.sum(scores)) / max(len(scores), 1)
                _var = float(np.sum((scores - _mean) ** 2)) / max(len(scores), 1)
                std_scores = float(np.sqrt(_var)) if _var > 0 else 1.0
                adaptive_beta = 4.0 / max(std_scores, 1e-6)
                beta = min(adaptive_beta, self.beta * 2)  # cap at 2x base
                # Apply inverse temperature and strength weighting
                scores = beta * scores * strengths

                # Numerically stable softmax
                scores_shifted = scores - scores.max()
                exp_scores = np.exp(scores_shifted)
                alpha = exp_scores / (exp_scores.sum() + 1e-12)

                # Retrieve: weighted combination of stored patterns
                xi_new = X.T @ alpha  # (N,)

                # Check convergence
                if np.linalg.norm(xi_new - xi) < 1e-6:
                    xi = xi_new
                    break
                xi = xi_new

            # Increment access counts for top-contributing patterns
            top_idx = int(np.argmax(alpha))
            self._patterns[top_idx].access_count += 1

            return xi

        # Fallback: classical Hopfield if no explicit patterns stored
        J_csr = self._J.tocsr()
        for _ in range(n_steps):
            if J_csr.nnz > 0:
                field = J_csr @ xi
            else:
                field = np.zeros(self.n)

            for i in range(len(xi)):
                if abs(float(field[i])) > 0.1:
                    xi[i] = 1.0 if float(field[i]) > 0 else 0.0

        return xi

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
