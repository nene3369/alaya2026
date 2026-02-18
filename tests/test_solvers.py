"""Tests for lmm.solvers — ClassicalQUBOSolver + SubmodularSelector."""

from __future__ import annotations

import numpy as np

from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver, SubmodularSelector, solve_classical


class TestClassicalQUBOSolver:
    def _build_simple_qubo(self, n: int = 10, k: int = 3) -> QUBOBuilder:
        builder = QUBOBuilder(n_variables=n)
        surprises = np.random.RandomState(42).rand(n)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=10.0)
        return builder

    def test_sa_solver(self):
        builder = self._build_simple_qubo()
        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="sa", k=3)
        assert isinstance(x, np.ndarray)
        assert int((x > 0.5).sum()) == 3

    def test_greedy_solver(self):
        builder = self._build_simple_qubo()
        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="greedy", k=3)
        assert isinstance(x, np.ndarray)

    def test_relaxation_solver(self):
        builder = self._build_simple_qubo()
        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="relaxation")
        assert isinstance(x, np.ndarray)

    def test_ising_sa_solver(self):
        builder = self._build_simple_qubo()
        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="ising_sa", k=3)
        assert isinstance(x, np.ndarray)
        assert int((x > 0.5).sum()) == 3

    def test_solve_classical_convenience(self):
        surprises = np.random.RandomState(42).rand(10)
        selected = solve_classical(surprises, k=3, method="greedy")
        assert isinstance(selected, np.ndarray)
        assert len(selected) <= 3

    def test_energy_decreases(self):
        builder = self._build_simple_qubo(n=20, k=5)
        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="sa", k=5)
        energy = builder.evaluate(x)
        # SA should find a solution with some energy
        assert np.isfinite(energy)


class TestSubmodularSelector:
    def test_select(self):
        rng = np.random.RandomState(42)
        n = 20
        relevance = rng.rand(n)
        impact_graph = rng.rand(n, n) * 0.5
        np.fill_diagonal(impact_graph, 0)

        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        result = selector.select(relevance, impact_graph, k=5)
        assert len(result.selected_indices) == 5
        assert len(set(result.selected_indices)) == 5  # no duplicates

    def test_select_lazy(self):
        rng = np.random.RandomState(42)
        n = 15
        relevance = rng.rand(n)
        # select_lazy takes normalized_data, not impact_graph
        normalized_data = rng.randn(n, 5)
        norms = np.linalg.norm(normalized_data, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        normalized_data = normalized_data / norms

        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        result = selector.select_lazy(relevance, normalized_data, k=4)
        assert len(result.selected_indices) == 4

    def test_select_adaptive(self):
        rng = np.random.RandomState(42)
        n = 20
        relevance = rng.rand(n)
        impact_graph = rng.rand(n, n) * 0.5
        np.fill_diagonal(impact_graph, 0)

        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        result = selector.select_adaptive(relevance, impact_graph, k_max=10)
        assert len(result.selected_indices) <= 10
        assert len(result.selected_indices) > 0

    def test_k_larger_than_n(self):
        rng = np.random.RandomState(42)
        n = 5
        relevance = rng.rand(n)
        impact_graph = rng.rand(n, n) * 0.5
        np.fill_diagonal(impact_graph, 0)

        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        result = selector.select(relevance, impact_graph, k=10)
        assert len(result.selected_indices) <= n
        assert len(result.selected_indices) > 0
