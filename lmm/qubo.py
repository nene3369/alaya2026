"""QUBO matrix builder — sparse, implicit cardinality constraint.

Builds a Quadratic Unconstrained Binary Optimization problem without
constructing dense O(n^2) matrices. Cardinality constraint is stored
implicitly and evaluated in O(n).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm._compat import sparse_matvec


class QUBOBuilder:
    """Build and evaluate QUBO problems efficiently.

    Internal storage:
      _diag          : (n,) diagonal / linear coefficients
      _offdiag_{rows,cols,vals} : COO triplets for off-diagonal terms
      _cardinality_gamma, _k   : implicit cardinality penalty
      _csr_cache               : lazily-built CSR of off-diagonal part
    """

    def __init__(self, n_variables: int):
        self.n = n_variables
        self._diag = np.zeros(n_variables)
        self._offdiag_rows: list[int] = []
        self._offdiag_cols: list[int] = []
        self._offdiag_vals: list[float] = []
        self._cardinality_gamma = 0.0
        self._cardinality_k = 0
        self._csr_cache: sparse.csr_matrix | None = None

    def _invalidate_cache(self) -> None:
        self._csr_cache = None

    # -- term addition ---------------------------------------------------

    def add_linear(self, i: int, weight: float) -> None:
        """Add linear term w_i * x_i."""
        if not (0 <= i < self.n):
            raise IndexError(f"index {i} out of range [0, {self.n})")
        self._diag[i] += weight

    def add_quadratic(self, i: int, j: int, weight: float) -> None:
        """Add quadratic term w_ij * x_i * x_j."""
        if not (0 <= i < self.n) or not (0 <= j < self.n):
            raise IndexError(f"index ({i}, {j}) out of range [0, {self.n})")
        if i == j:
            self._diag[i] += weight
        else:
            self._offdiag_rows.extend([i, j])
            self._offdiag_cols.extend([j, i])
            self._offdiag_vals.extend([weight / 2, weight / 2])
            self._invalidate_cache()

    def add_surprise_objective(
        self, surprises: np.ndarray, alpha: float = 1.0,
    ) -> None:
        """Minimise -alpha * sum(s_i * x_i) to select high-surprise items."""
        surprises = np.asarray(surprises, dtype=np.float64)
        if len(surprises) != self.n:
            raise ValueError(
                f"surprises length {len(surprises)} != n_variables {self.n}"
            )
        self._diag -= alpha * surprises
        self._invalidate_cache()  # defensive: future-proof against refactors

    def add_cardinality_constraint(self, k: int, gamma: float = 10.0) -> None:
        """Add penalty gamma * (sum(x_i) - k)^2 implicitly in O(n)."""
        if k < 0 or k > self.n:
            raise ValueError(f"k={k} out of range [0, {self.n}]")
        self._diag += gamma * (1.0 - 2.0 * k)
        self._cardinality_gamma += gamma
        self._cardinality_k = k
        self._invalidate_cache()  # defensive: future-proof against refactors

    def add_diversity_penalty(
        self, similarity_matrix: np.ndarray, beta: float = 1.0,
    ) -> None:
        """Add beta * sum_ij sim(i,j) * x_i * x_j for diversity."""
        sim = np.array(similarity_matrix)
        if sim.shape != (self.n, self.n):
            raise ValueError(
                f"similarity_matrix shape {sim.shape} != ({self.n}, {self.n})"
            )
        np.fill_diagonal(sim, 0.0)
        mask = np.abs(sim) > 1e-15
        rows, cols = np.where(mask)
        vals = sim[rows, cols] * beta / 2.0
        self._offdiag_rows.extend(int(r) for r in rows)
        self._offdiag_cols.extend(int(c) for c in cols)
        self._offdiag_vals.extend(float(v) for v in vals)
        self._invalidate_cache()

    # -- matrix access ---------------------------------------------------

    def _build_offdiag_csr(self) -> sparse.csr_matrix:
        if self._csr_cache is not None:
            return self._csr_cache
        if not self._offdiag_vals:
            self._csr_cache = sparse.csr_matrix(
                ([], ([], [])), shape=(self.n, self.n),
            )
        else:
            self._csr_cache = sparse.csr_matrix(
                (self._offdiag_vals,
                 (self._offdiag_rows, self._offdiag_cols)),
                shape=(self.n, self.n),
            )
        return self._csr_cache

    @property
    def Q(self) -> np.ndarray:
        """Dense Q matrix (backward compat). Use evaluate() for large n."""
        csr = self._build_offdiag_csr()
        Q = csr.toarray()
        for i in range(self.n):
            Q[i, i] = float(self._diag[i])
        if self._cardinality_gamma > 0:
            gamma = self._cardinality_gamma
            Q += gamma
            for i in range(self.n):
                Q[i, i] -= gamma
        return Q

    def get_matrix(self) -> np.ndarray:
        return self.Q.copy()

    # -- evaluation ------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate energy x^T Q x in O(n + nnz) without dense Q."""
        x = np.asarray(x, dtype=np.float64)
        energy = float(np.dot(self._diag, x))
        csr = self._build_offdiag_csr()
        if csr.nnz > 0:
            Jx = sparse_matvec(csr, x)
            energy += float(np.dot(x, Jx))
        if self._cardinality_gamma > 0:
            sx = float(np.sum(x))
            sx2 = float(np.dot(x, x))
            energy += self._cardinality_gamma * (sx * sx - sx2)
        return energy
