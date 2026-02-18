"""Tests for lmm.solvers — ClassicalQUBOSolver + SubmodularSelector."""

from __future__ import annotations

import pytest
import numpy as np
from scipy import sparse

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

    def test_unknown_method_raises(self):
        """Cover ValueError for unknown solver method (line 46)."""
        builder = self._build_simple_qubo()
        solver = ClassicalQUBOSolver(builder)
        with pytest.raises(ValueError, match="Unknown solver method"):
            solver.solve(method="quantum")

    def test_greedy_early_stop(self):
        """Cover greedy early break when marginal >= 0 (line 81)."""
        # Build QUBO where selecting anything has positive marginal (high gamma)
        builder = QUBOBuilder(n_variables=5)
        # Small surprises, huge cardinality penalty for k=0 effectively
        surprises = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        builder.add_surprise_objective(surprises, alpha=0.001)
        builder.add_cardinality_constraint(k=0, gamma=100.0)
        solver = ClassicalQUBOSolver(builder)
        x = solver.solve_greedy(k=5)
        # With k=0 constraint, greedy should select nothing or very few
        assert isinstance(x, np.ndarray)

    def test_project_to_k_add_items(self):
        """Cover _project_to_k adding items when n_sel < k (lines 222-231)."""
        builder = self._build_simple_qubo(n=10, k=5)
        solver = ClassicalQUBOSolver(builder)
        # Start with too few selected
        x = np.zeros(10)
        x[0] = 1.0
        x[1] = 1.0
        result = solver._project_to_k(x, k=5)
        assert int((result > 0.5).sum()) == 5

    def test_project_to_k_drop_items(self):
        """Cover _project_to_k dropping items when n_sel > k."""
        builder = self._build_simple_qubo(n=10, k=3)
        solver = ClassicalQUBOSolver(builder)
        # Start with too many selected
        x = np.ones(10)
        result = solver._project_to_k(x, k=3)
        assert int((result > 0.5).sum()) == 3

    def test_project_to_k_with_offdiag(self):
        """Cover _project_to_k with off-diagonal entries (lines 219-220, 230-231)."""
        builder = QUBOBuilder(n_variables=8)
        surprises = np.random.RandomState(42).rand(8)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=3, gamma=10.0)
        # Add diversity penalty to create off-diagonal entries
        sim = np.random.RandomState(42).rand(8, 8) * 0.1
        builder.add_diversity_penalty(sim, beta=0.5)
        solver = ClassicalQUBOSolver(builder)
        # Over-selected
        x = np.ones(8)
        result = solver._project_to_k(x, k=3)
        assert int((result > 0.5).sum()) == 3
        # Under-selected
        x2 = np.zeros(8)
        x2[0] = 1.0
        result2 = solver._project_to_k(x2, k=3)
        assert int((result2 > 0.5).sum()) == 3

    def test_sa_with_initial_state(self):
        """Cover initial_state branch in solve_sa (line 100)."""
        builder = self._build_simple_qubo(n=10, k=3)
        solver = ClassicalQUBOSolver(builder)
        init = np.zeros(10)
        init[:3] = 1.0
        x = solver.solve_sa(initial_state=init, k=3, n_iterations=100)
        assert int((x > 0.5).sum()) == 3

    def test_ising_sa_with_initial_state(self):
        """Cover initial_state branch in solve_sa_ising (line 157)."""
        builder = self._build_simple_qubo(n=10, k=3)
        solver = ClassicalQUBOSolver(builder)
        init = np.zeros(10)
        init[:3] = 1.0
        x = solver.solve_sa_ising(initial_state=init, k=3, n_iterations=100)
        assert int((x > 0.5).sum()) == 3


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

    def test_select_with_sparse_graph(self):
        """Cover sparse graph paths in select (lines 317-330)."""
        rng = np.random.RandomState(42)
        n = 20
        relevance = rng.rand(n)
        dense = rng.rand(n, n) * 0.5
        np.fill_diagonal(dense, 0)
        impact_graph = sparse.csr_matrix(dense)

        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        result = selector.select(relevance, impact_graph, k=5)
        assert len(result.selected_indices) == 5
        assert len(set(result.selected_indices)) == 5

    def test_select_no_graph(self):
        """Cover select with impact_graph=None (line 309 path skip)."""
        rng = np.random.RandomState(42)
        n = 20
        relevance = rng.rand(n)
        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        result = selector.select(relevance, impact_graph=None, k=5)
        assert len(result.selected_indices) == 5

    def test_select_adaptive_single_gain(self):
        """Cover select_adaptive with single gain (line 407-408)."""
        selector = SubmodularSelector(alpha=1.0, beta=0.0)
        # Only 1 item, so gains has length 1
        relevance = np.array([5.0])
        result = selector.select_adaptive(relevance, impact_graph=None, k_max=1)
        assert len(result.selected_indices) == 1

    def test_select_adaptive_zero_initial_gain(self):
        """Cover select_adaptive when initial gain <= 0 (line 410-411)."""
        selector = SubmodularSelector(alpha=1.0, beta=0.0)
        # All zero surprises -> gains are all 0
        relevance = np.zeros(10)
        result = selector.select_adaptive(relevance, impact_graph=None, k_max=5)
        assert isinstance(result.selected_indices, np.ndarray)

    def test_evaluate_with_dense_graph(self):
        """Cover evaluate method with dense graph (lines 433-448)."""
        rng = np.random.RandomState(42)
        n = 10
        surprises = rng.rand(n)
        graph = rng.rand(n, n) * 0.3
        np.fill_diagonal(graph, 0)

        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        selected = np.array([0, 2, 5])
        value = selector.evaluate(selected, surprises, impact_graph=graph)
        assert np.isfinite(value)
        assert value > 0

    def test_evaluate_with_sparse_graph(self):
        """Cover evaluate method with sparse graph (lines 438-443)."""
        rng = np.random.RandomState(42)
        n = 10
        surprises = rng.rand(n)
        dense = rng.rand(n, n) * 0.3
        np.fill_diagonal(dense, 0)
        graph = sparse.csr_matrix(dense)

        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        selected = np.array([0, 2, 5])
        value = selector.evaluate(selected, surprises, impact_graph=graph)
        assert np.isfinite(value)
        assert value > 0

    def test_evaluate_without_graph(self):
        """Cover evaluate method without graph (line 433-435)."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(10)
        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        selected = np.array([0, 2, 5])
        value = selector.evaluate(selected, surprises, impact_graph=None)
        expected = sum(surprises[i] for i in [0, 2, 5])
        assert abs(value - expected) < 1e-10

    def test_select_lazy_early_stop(self):
        """Cover select_lazy break on zero marginal (lines 366-369)."""
        selector = SubmodularSelector(alpha=1.0, beta=0.5)
        # Mix of positive and zero/negative values to trigger early stop
        relevance = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        normalized_data = np.eye(5)
        result = selector.select_lazy(relevance, normalized_data, k=5)
        assert len(result.selected_indices) >= 1
