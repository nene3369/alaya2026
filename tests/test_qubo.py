"""Tests for lmm.qubo — QUBOBuilder."""

from __future__ import annotations

import numpy as np
import pytest

from lmm.qubo import QUBOBuilder


class TestQUBOBuilder:
    def test_add_linear(self):
        builder = QUBOBuilder(n_variables=5)
        for i in range(5):
            builder.add_linear(i, float(i + 1))
        Q = builder.get_matrix()
        assert Q.shape == (5, 5)
        for i in range(5):
            assert Q[i, i] == pytest.approx(float(i + 1))

    def test_add_quadratic(self):
        builder = QUBOBuilder(n_variables=3)
        builder.add_quadratic(0, 1, 2.5)
        builder.add_quadratic(1, 2, -1.0)
        Q = builder.get_matrix()
        # add_quadratic symmetrizes: w/2 on each side
        assert Q[0, 1] == pytest.approx(1.25)
        assert Q[1, 0] == pytest.approx(1.25)
        assert Q[1, 2] == pytest.approx(-0.5)
        assert Q[2, 1] == pytest.approx(-0.5)

    def test_add_surprise_objective(self):
        builder = QUBOBuilder(n_variables=4)
        surprises = np.array([0.5, 1.5, 0.8, 2.0])
        builder.add_surprise_objective(surprises, alpha=1.0)
        Q = builder.get_matrix()
        # Diagonal should be negative (maximization)
        for i in range(4):
            assert Q[i, i] < 0

    def test_add_cardinality_constraint(self):
        builder = QUBOBuilder(n_variables=5)
        builder.add_cardinality_constraint(k=2, gamma=10.0)
        Q = builder.get_matrix()
        assert Q.shape == (5, 5)

    def test_add_diversity_penalty(self):
        rng = np.random.RandomState(42)
        sim = rng.rand(4, 4)
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 0)
        builder = QUBOBuilder(n_variables=4)
        builder.add_diversity_penalty(sim, beta=0.5)
        Q = builder.get_matrix()
        # Off-diagonal should have penalty for similar items
        assert Q.shape == (4, 4)

    def test_evaluate(self):
        builder = QUBOBuilder(n_variables=3)
        for i in range(3):
            builder.add_linear(i, -(i + 1.0))
        x = np.array([1, 0, 1])
        energy = builder.evaluate(x)
        assert energy == pytest.approx(-4.0)

    def test_empty_builder(self):
        builder = QUBOBuilder(n_variables=3)
        Q = builder.get_matrix()
        assert Q.shape == (3, 3)
        assert np.allclose(Q, 0)

    def test_get_matrix_idempotent(self):
        builder = QUBOBuilder(n_variables=3)
        for i in range(3):
            builder.add_linear(i, 1.0)
        Q1 = builder.get_matrix()
        Q2 = builder.get_matrix()
        np.testing.assert_array_equal(Q1, Q2)
