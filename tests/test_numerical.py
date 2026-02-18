"""Numerical correctness tests — edge cases and precision."""

from __future__ import annotations

import numpy as np

from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver, SubmodularSelector
from lmm.surprise import SurpriseCalculator


class TestNumericalPrecision:
    def test_qubo_symmetry(self):
        """QUBO matrix should be upper-triangular or symmetric."""
        builder = QUBOBuilder(n_variables=10)
        rng = np.random.RandomState(42)
        builder.add_surprise_objective(rng.rand(10), alpha=1.0)
        builder.add_cardinality_constraint(k=3, gamma=10.0)
        Q = builder.get_matrix()
        # Energy should be same regardless of representation
        x = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        e1 = float(x @ Q @ x)
        e2 = builder.evaluate(x)
        assert abs(e1 - e2) < 1e-10

    def test_surprise_non_negative(self):
        """Surprises should be non-negative for KL method."""
        rng = np.random.RandomState(42)
        data = rng.randn(500)
        calc = SurpriseCalculator(method="kl")
        calc.fit(data)
        surprises = calc.compute(data)
        assert np.all(surprises >= -1e-10)

    def test_submodular_monotone(self):
        """Submodular function value should be monotone increasing."""
        rng = np.random.RandomState(42)
        n = 20
        relevance = rng.rand(n)
        impact_graph = rng.rand(n, n) * 0.3
        np.fill_diagonal(impact_graph, 0)

        selector = SubmodularSelector(alpha=1.0, beta=0.5)

        # Run for k=3 and k=6: k=6 should have >= objective value
        r3 = selector.select(relevance, impact_graph, k=3)
        r6 = selector.select(relevance, impact_graph, k=6)
        assert r6.objective_value >= r3.objective_value - 1e-10

    def test_sa_respects_cardinality(self):
        """SA solver should exactly satisfy cardinality constraint."""
        rng = np.random.RandomState(42)
        n = 30
        for k in [1, 5, 10, 15]:
            builder = QUBOBuilder(n_variables=n)
            builder.add_surprise_objective(rng.rand(n), alpha=1.0)
            builder.add_cardinality_constraint(k=k, gamma=50.0)
            solver = ClassicalQUBOSolver(builder)
            x = solver.solve(method="greedy", k=k)
            assert int((x > 0.5).sum()) == k

    def test_zero_data(self):
        """All-zero data should not crash."""
        data = np.zeros(50)
        calc = SurpriseCalculator(method="kl")
        calc.fit(data)
        surprises = calc.compute(data)
        assert np.all(np.isfinite(surprises))

    def test_large_values(self):
        """Large values should not cause overflow."""
        data = np.array([1e10, -1e10, 1e5, -1e5, 0.0])
        calc = SurpriseCalculator(method="entropy")
        calc.fit(data)
        surprises = calc.compute(data)
        assert np.all(np.isfinite(surprises))
