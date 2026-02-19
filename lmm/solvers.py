"""Classical QUBO solvers and submodular selector — no quantum computer required.

Solvers:
  1. Continuous relaxation + rounding (SLSQP)
  2. Simulated annealing (SA) with warm-start, O(1) delta
  3. Ising SA — SIMD-friendly dense local-field updates
  4. Greedy — O(k*n) marginal gain

Submodular:
  Lazy greedy maximiser with (1-1/e) approximation guarantee.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from lmm._compat import HAS_ARGPARTITION, sparse_matvec, sparse_row_scatter
from lmm.qubo import QUBOBuilder

SolverMethod = Literal["sa", "ising_sa", "greedy", "relaxation"]


# ===================================================================
# Classical QUBO Solver
# ===================================================================

class ClassicalQUBOSolver:
    """Solve QUBO problems with classical methods."""

    def __init__(self, qubo: QUBOBuilder):
        self.qubo = qubo
        self.n = qubo.n

    def solve(self, method: SolverMethod = "sa", k: int | None = None, **kw) -> np.ndarray:
        method_clean = method.strip().lower()
        if method_clean != method:
            warnings.warn(
                f"solver method {method!r} was normalized to {method_clean!r}; "
                f"use the exact string to suppress this warning",
                stacklevel=2,
            )
            method = method_clean  # type: ignore[assignment]
        if method == "sa":
            return self.solve_sa(k=k, **kw)
        if method == "ising_sa":
            return self.solve_sa_ising(k=k, **kw)
        if method == "relaxation":
            return self.solve_relaxation(**kw)
        if method == "greedy":
            return self.solve_greedy(k=k)
        raise ValueError(f"Unknown solver method: {method}")

    # -- relaxation ------------------------------------------------------

    def solve_relaxation(self, n_restarts: int = 10) -> np.ndarray:
        best_x, best_energy = None, float("inf")
        for _ in range(n_restarts):
            x0 = np.random.rand(self.n)
            result = minimize(
                lambda x: self.qubo.evaluate(x), x0,
                method="SLSQP", bounds=[(0, 1)] * self.n,
            )
            x_bin = (result.x > 0.5).astype(float)
            energy = self.qubo.evaluate(x_bin)
            if energy < best_energy:
                best_energy, best_x = energy, x_bin
        return best_x

    # -- greedy ----------------------------------------------------------

    def solve_greedy(self, k: int | None = None) -> np.ndarray:
        x = np.zeros(self.n)
        diag = self.qubo._diag
        csr = self.qubo._build_offdiag_csr()
        gamma = self.qubo._cardinality_gamma
        max_steps = k if k is not None else self.n
        Jx = np.zeros(self.n)
        sx = 0.0
        for _ in range(max_steps):
            marginal = diag + 2.0 * Jx
            if gamma > 0:
                marginal = marginal + gamma * (2.0 * sx + 1.0 - 2.0 * x)
            marginal = np.where(x < 0.5, marginal, np.inf)
            best = int(np.argmin(marginal))
            if marginal[best] >= 0.0:
                break
            x[best] = 1.0
            sx += 1.0
            if csr.nnz > 0:
                sparse_row_scatter(csr, int(best), Jx, 1.0)
        return x

    # -- simulated annealing ---------------------------------------------

    def solve_sa(
        self,
        initial_state: np.ndarray | None = None,
        n_iterations: int = 5000,
        temp_start: float = 10.0,
        temp_end: float = 0.01,
        k: int | None = None,
    ) -> np.ndarray:
        """SA with O(1) delta computation and sparse incremental Jx updates."""
        rng = np.random.default_rng()
        x = (initial_state.copy().astype(float) if initial_state is not None
             else rng.integers(0, 2, size=self.n).astype(float))

        diag = self.qubo._diag
        csr = self.qubo._build_offdiag_csr()
        gamma = self.qubo._cardinality_gamma

        Jx = (np.asarray(sparse_matvec(csr, x)).flatten()
              if csr.nnz > 0 else np.zeros(self.n))
        sx = float(np.sum(x))
        energy = self.qubo.evaluate(x)
        best_x, best_energy = x.copy(), energy
        temp_ratio = temp_end / temp_start

        for step in range(n_iterations):
            temp = temp_start * temp_ratio ** (step / n_iterations)
            flip = rng.integers(0, self.n)
            sign = 1.0 - 2.0 * x[flip]

            delta = sign * diag[flip] + 2.0 * sign * Jx[flip]
            if gamma > 0:
                delta += gamma * 2.0 * sign * (sx - x[flip])

            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-15)):
                x[flip] += sign
                energy += delta
                sx += sign
                if csr.nnz > 0:
                    sparse_row_scatter(csr, int(flip), Jx, sign)
                if energy < best_energy:
                    best_energy, best_x = energy, x.copy()

        if k is not None:
            best_x = self._project_to_k(best_x, k)
        return best_x

    # -- Ising SA --------------------------------------------------------

    def solve_sa_ising(
        self,
        initial_state: np.ndarray | None = None,
        n_iterations: int = 5000,
        temp_start: float = 10.0,
        temp_end: float = 0.01,
        k: int | None = None,
    ) -> np.ndarray:
        """Ising-form SA with dense local-field updates."""
        rng = np.random.default_rng()
        n = self.n
        diag = self.qubo._diag
        csr = self.qubo._build_offdiag_csr()
        gamma = self.qubo._cardinality_gamma

        csr_rowsum = (np.asarray(csr.sum(axis=1)).flatten()
                      if csr.nnz > 0 else np.zeros(n))
        h = diag / 2.0 + (diag + csr_rowsum + (n - 1) * gamma) / 4.0

        s = ((2.0 * initial_state - 1.0).copy() if initial_state is not None
             else rng.choice([-1.0, 1.0], size=n))

        sum_s = float(np.sum(s))
        Js = diag * s
        if csr.nnz > 0:
            Js = Js + np.asarray(sparse_matvec(csr, s)).flatten()
        Js = Js + gamma * (sum_s - s)
        Js /= 4.0
        local_field = h + 2.0 * Js

        energy = float(np.dot(s, Js) + np.dot(h, s))
        best_s, best_energy = s.copy(), energy
        temp_ratio = temp_end / temp_start

        for step in range(n_iterations):
            temp = temp_start * temp_ratio ** (step / n_iterations)
            idx = rng.integers(0, n)
            delta_e = -2.0 * s[idx] * local_field[idx]

            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-15)):
                s[idx] *= -1.0
                energy += delta_e
                sum_s += 2.0 * s[idx]
                local_field += gamma * s[idx]
                local_field[idx] += (diag[idx] - gamma) * s[idx]
                if csr.nnz > 0:
                    sparse_row_scatter(csr, int(idx), local_field, float(s[idx]))
                if energy < best_energy:
                    best_energy, best_s = energy, s.copy()

        best_x = (best_s + 1.0) / 2.0
        if k is not None:
            best_x = self._project_to_k(best_x, k)
        return best_x

    # -- projection ------------------------------------------------------

    def _project_to_k(self, x: np.ndarray, k: int) -> np.ndarray:
        x = x.copy()
        diag = self.qubo._diag
        csr = self.qubo._build_offdiag_csr()
        gamma = self.qubo._cardinality_gamma
        has_offdiag = csr.nnz > 0
        # Cache Jx and update incrementally instead of recomputing each iteration
        Jx = (np.asarray(sparse_matvec(csr, x)).flatten()
              if has_offdiag else np.zeros(self.n))
        while True:
            n_sel = int((x > 0.5).sum())
            if n_sel == k:
                break
            Qx = diag * x + Jx
            if gamma > 0:
                sx = float(np.sum(x))
                Qx = Qx + gamma * (sx - x)
            if n_sel > k:
                sel = np.where(x > 0.5)[0]
                mc = -diag[sel] + 2.0 * Qx[sel]
                if gamma > 0:
                    mc = mc - gamma
                drop = int(sel[int(np.argmax(mc))])
                x[drop] = 0.0
                if has_offdiag:
                    sparse_row_scatter(csr, drop, Jx, -1.0)
            else:
                unsel = np.where(x <= 0.5)[0]
                if len(unsel) == 0:
                    break
                delta = diag[unsel] + 2.0 * Qx[unsel]
                if gamma > 0:
                    delta = delta + gamma
                add = int(unsel[int(np.argmin(delta))])
                x[add] = 1.0
                if has_offdiag:
                    sparse_row_scatter(csr, add, Jx, 1.0)
        return x


# ===================================================================
# Convenience function
# ===================================================================

def solve_classical(
    surprises: np.ndarray,
    k: int = 10,
    alpha: float = 1.0,
    gamma: float = 10.0,
    method: SolverMethod = "sa",
) -> np.ndarray:
    """One-shot: build QUBO from surprises and return selected indices."""
    n = len(surprises)
    builder = QUBOBuilder(n)
    builder.add_surprise_objective(surprises, alpha=alpha)
    builder.add_cardinality_constraint(k, gamma=gamma)
    solver = ClassicalQUBOSolver(builder)
    x = solver.solve(method=method, k=k)
    selected = np.where(x > 0.5)[0]
    return selected[:k]


# ===================================================================
# Submodular Selector
# ===================================================================

@dataclass
class SubmodularResult:
    selected_indices: np.ndarray
    objective_value: float
    marginal_gains: np.ndarray
    method: str = "submodular_greedy"


class SubmodularSelector:
    """Submodular greedy selector with (1-1/e) approximation guarantee.

    Objective: max f(S) = alpha * surprise(S) + beta * cut(S, W)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def select(
        self,
        surprises: np.ndarray,
        impact_graph: sparse.csr_matrix | np.ndarray | None = None,
        k: int = 10,
    ) -> SubmodularResult:
        """Greedy selection, O(n*k)."""
        n = len(surprises)
        k = min(k, n)
        is_sparse = impact_graph is not None and sparse.issparse(impact_graph)
        has_graph = impact_graph is not None

        marginal = self.alpha * surprises.copy()
        if has_graph:
            row_sums = (np.asarray(impact_graph.sum(axis=1)).flatten()
                        if is_sparse else impact_graph.sum(axis=1))
            marginal += self.beta * row_sums

        selected: list[int] = []
        mask = np.ones(n, dtype=bool)
        gains_history: list[float] = []
        total_value = 0.0
        INF = float('inf')
        masked = marginal.copy()

        for _ in range(k):
            best = int(np.argmax(masked))
            if float(masked[best]) <= 0 and selected:
                break
            if float(masked[best]) <= -INF:
                break
            gains_history.append(float(masked[best]))
            total_value += float(masked[best])
            selected.append(best)
            mask[best] = False
            masked[best] = -INF

            if has_graph:
                if is_sparse:
                    col = (impact_graph.T.getrow(best)
                           if hasattr(impact_graph.T, 'getrow') else None)
                    if col is not None:
                        for idx, w in zip(col.indices, col.data):
                            if mask[idx]:
                                marginal[idx] -= 2.0 * self.beta * w
                                masked[idx] -= 2.0 * self.beta * w
                    else:
                        row = impact_graph.getrow(best)
                        for idx, w in zip(row.indices, row.data):
                            if mask[idx]:
                                marginal[idx] -= 2.0 * self.beta * w
                                masked[idx] -= 2.0 * self.beta * w
                else:
                    d = 2.0 * self.beta * np.asarray(impact_graph[best]).flatten()
                    update = d * mask
                    marginal -= update
                    masked -= update

        return SubmodularResult(
            selected_indices=np.array(selected, dtype=int),
            objective_value=total_value,
            marginal_gains=np.array(gains_history),
        )

    def select_lazy(
        self,
        surprises: np.ndarray,
        normalized_data: np.ndarray,
        k: int = 10,
        sparse_k: int = 20,
    ) -> SubmodularResult:
        """Lazy graph evaluation: O(k*n*d) instead of O(n^2*d)."""
        n = len(surprises)
        k = min(k, n)
        sparse_k = min(sparse_k, n - 1)
        norm_T = normalized_data.T

        marginal = self.alpha * surprises.copy()
        selected: list[int] = []
        mask = np.ones(n, dtype=bool)
        gains_history: list[float] = []
        total_value = 0.0
        INF = float('inf')
        masked = marginal.copy()

        for _ in range(k):
            best = int(np.argmax(masked))
            if float(masked[best]) <= 0 and selected:
                break
            if float(masked[best]) <= -INF:
                break
            gains_history.append(float(masked[best]))
            total_value += float(masked[best])
            selected.append(best)
            mask[best] = False
            masked[best] = -INF

            row_sims = np.dot(normalized_data[best], norm_T)
            row_sims[best] = 0.0
            if HAS_ARGPARTITION and n > sparse_k + 1:
                top_idx = np.argpartition(row_sims, -sparse_k)[-sparse_k:]
            else:
                top_idx = np.argsort(row_sims)[-sparse_k:]
            for j_arr in top_idx:
                j = int(j_arr)
                w = float(row_sims[j])
                if w > 1e-6 and mask[j]:
                    marginal[j] -= 2.0 * self.beta * w
                    masked[j] -= 2.0 * self.beta * w

        return SubmodularResult(
            selected_indices=np.array(selected, dtype=int),
            objective_value=total_value,
            marginal_gains=np.array(gains_history),
            method="submodular_greedy_lazy",
        )

    def select_adaptive(
        self,
        surprises: np.ndarray,
        impact_graph: sparse.csr_matrix | np.ndarray | None = None,
        k_max: int = 50,
    ) -> SubmodularResult:
        """Auto-detect cardinality K via 1/e saturation rule."""
        n = len(surprises)
        k_max = min(k_max, n)
        result = self.select(surprises, impact_graph, k=k_max)
        gains = result.marginal_gains
        if len(gains) <= 1:
            return result
        initial_gain = float(gains[0])
        if initial_gain <= 0:
            return result
        threshold = 1.0 / math.e
        adaptive_k = len(gains)
        for i in range(1, len(gains)):
            if float(gains[i]) / initial_gain < threshold:
                adaptive_k = i
                break
        sel = result.selected_indices[:adaptive_k]
        mg = result.marginal_gains[:adaptive_k]
        return SubmodularResult(
            selected_indices=sel,
            objective_value=sum(float(g) for g in mg),
            marginal_gains=mg,
            method="submodular_greedy_adaptive",
        )

    def evaluate(
        self,
        selected: np.ndarray,
        surprises: np.ndarray,
        impact_graph: sparse.csr_matrix | np.ndarray | None = None,
    ) -> float:
        S = set(int(i) for i in selected)
        n = len(surprises)
        value = self.alpha * sum(surprises[i] for i in S)
        if impact_graph is not None:
            is_sp = sparse.issparse(impact_graph)
            for i in S:
                if is_sp:
                    row = impact_graph.getrow(i)
                    for j, w in zip(row.indices, row.data):
                        if j not in S:
                            value += self.beta * w
                else:
                    for j in range(n):
                        if j not in S:
                            value += self.beta * impact_graph[i, j]
        return value
