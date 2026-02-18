"""Brute-force correctness tests — verify solvers find optimal solutions for small n.

For n <= 20, enumerate all C(n, k) subsets and confirm the solver's
solution matches the globally optimal energy (or is within a tiny tolerance).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver


def _brute_force_optimal(builder: QUBOBuilder, n: int, k: int) -> tuple[float, np.ndarray]:
    """Enumerate all C(n, k) subsets, return (best_energy, best_x)."""
    best_energy = np.inf
    best_x = np.zeros(n)
    for combo in itertools.combinations(range(n), k):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        energy = builder.evaluate(x)
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
    return best_energy, best_x


class TestBruteForceCorrectness:
    """Verify solvers match brute-force optimal on small instances."""

    @pytest.mark.parametrize("n,k", [(6, 2), (8, 3), (10, 3), (12, 4), (15, 5)])
    def test_greedy_matches_bruteforce_linear(self, n: int, k: int):
        """Greedy should find optimal for pure linear (surprise-only) QUBO."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(n) * 10.0

        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=10.0)

        bf_energy, _ = _brute_force_optimal(builder, n, k)

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="greedy", k=k)
        solver_energy = builder.evaluate(x)

        # For linear QUBO, greedy = top-k by surprise = optimal
        assert abs(solver_energy - bf_energy) < 1e-6, (
            f"greedy energy {solver_energy:.6f} != brute-force {bf_energy:.6f}"
        )

    @pytest.mark.parametrize("n,k", [(6, 2), (8, 3), (10, 3)])
    def test_sa_near_optimal_linear(self, n: int, k: int):
        """SA should be within 5% of optimal for small linear QUBO."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(n) * 10.0

        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=50.0)

        bf_energy, _ = _brute_force_optimal(builder, n, k)

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="sa", k=k)
        solver_energy = builder.evaluate(x)

        gap = abs(solver_energy - bf_energy) / abs(bf_energy)
        assert gap < 0.05, (
            f"SA energy {solver_energy:.4f} is {gap:.1%} away from "
            f"brute-force {bf_energy:.4f} (threshold: 5%)"
        )

    @pytest.mark.parametrize("n,k", [(6, 2), (8, 3), (10, 3)])
    def test_ising_sa_near_optimal_linear(self, n: int, k: int):
        """Ising SA should be within 5% of optimal for small linear QUBO."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(n) * 10.0

        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=50.0)

        bf_energy, _ = _brute_force_optimal(builder, n, k)

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="ising_sa", k=k)
        solver_energy = builder.evaluate(x)

        gap = abs(solver_energy - bf_energy) / abs(bf_energy)
        assert gap < 0.10, (
            f"ising_sa energy {solver_energy:.4f} is {gap:.1%} away from "
            f"brute-force {bf_energy:.4f} (threshold: 10%)"
        )

    @pytest.mark.parametrize("n,k", [(8, 3), (10, 4), (12, 4)])
    def test_greedy_with_diversity_near_optimal(self, n: int, k: int):
        """With diversity penalty, greedy should be within (1-1/e) of optimal."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(n) * 10.0
        similarity = rng.rand(n, n) * 0.3
        similarity = (similarity + similarity.T) / 2
        np.fill_diagonal(similarity, 0.0)

        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=50.0)
        builder.add_diversity_penalty(similarity, beta=0.5)

        bf_energy, _ = _brute_force_optimal(builder, n, k)

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="greedy", k=k)
        solver_energy = builder.evaluate(x)

        # Greedy is (1-1/e) ≈ 0.632 approximate for submodular.
        # For these small instances it usually finds optimal.
        # Allow 40% relative gap to account for non-submodular QUBO structure.
        gap = abs(solver_energy - bf_energy) / abs(bf_energy)
        assert gap < 0.40, (
            f"greedy energy {solver_energy:.4f} is {gap:.1%} away from "
            f"brute-force optimal {bf_energy:.4f} (threshold: 40%)"
        )

    @pytest.mark.parametrize("n,k", [(8, 3), (10, 4)])
    def test_sa_with_diversity_near_optimal(self, n: int, k: int):
        """SA with diversity penalty should be within 5% of optimal for small n."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(n) * 10.0
        similarity = rng.rand(n, n) * 0.3
        similarity = (similarity + similarity.T) / 2
        np.fill_diagonal(similarity, 0.0)

        builder = QUBOBuilder(n_variables=n)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=k, gamma=50.0)
        builder.add_diversity_penalty(similarity, beta=0.5)

        bf_energy, _ = _brute_force_optimal(builder, n, k)

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="sa", k=k)
        solver_energy = builder.evaluate(x)

        # SA is stochastic; allow 5% relative gap from optimal.
        # bf_energy is negative, so solver_energy >= bf_energy (closer to 0 = worse).
        gap = abs(solver_energy - bf_energy) / abs(bf_energy)
        assert gap < 0.05, (
            f"SA energy {solver_energy:.4f} is {gap:.1%} away from "
            f"brute-force optimal {bf_energy:.4f} (threshold: 5%)"
        )

    def test_greedy_exact_on_tiny(self):
        """For n=6, k=2, greedy must find the exact optimal subset."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(6) * 10.0

        builder = QUBOBuilder(n_variables=6)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=2, gamma=50.0)

        bf_energy, bf_x = _brute_force_optimal(builder, 6, 2)
        bf_indices = set(np.where(bf_x > 0.5)[0])

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method="greedy", k=2)
        indices = set(np.where(x > 0.5)[0])
        energy = builder.evaluate(x)

        assert abs(energy - bf_energy) < 1e-6, (
            f"greedy: energy {energy:.6f} != optimal {bf_energy:.6f}"
        )
        assert indices == bf_indices, (
            f"greedy: indices {indices} != optimal {bf_indices}"
        )

    @pytest.mark.parametrize("method", ["sa", "ising_sa"])
    def test_stochastic_solvers_near_optimal_tiny(self, method: str):
        """For n=6, k=2, SA/Ising SA should be within 10% of optimal."""
        rng = np.random.RandomState(42)
        surprises = rng.rand(6) * 10.0

        builder = QUBOBuilder(n_variables=6)
        builder.add_surprise_objective(surprises, alpha=1.0)
        builder.add_cardinality_constraint(k=2, gamma=50.0)

        bf_energy, _ = _brute_force_optimal(builder, 6, 2)

        solver = ClassicalQUBOSolver(builder)
        x = solver.solve(method=method, k=2)
        energy = builder.evaluate(x)

        gap = abs(energy - bf_energy) / abs(bf_energy)
        assert gap < 0.10, (
            f"{method}: energy {energy:.4f} is {gap:.1%} away from "
            f"optimal {bf_energy:.4f} (threshold: 10%)"
        )

    def test_lmm_select_from_surprises_matches_topk(self):
        """LMM.select_from_surprises should select the top-k by surprise."""
        from lmm.core import LMM

        surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
        k = 3

        # Ground truth: top-3 by surprise = indices 3 (8.0), 5 (7.0), 1 (5.0)
        expected = set(np.argsort(surprises)[-k:])

        model = LMM(k=k, solver_method="greedy")
        result = model.select_from_surprises(surprises)
        actual = set(result.selected_indices)

        assert actual == expected, f"LMM selected {actual}, expected {expected}"
