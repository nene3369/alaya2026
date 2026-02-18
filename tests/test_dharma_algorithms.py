"""Tests for lmm.dharma.algorithms — graph building, BodhisattvaQUBO, MadhyamakaBalancer."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.dharma.algorithms import (
    BodhisattvaQUBO,
    MadhyamakaBalancer,
    build_sparse_impact_graph,
    extract_subgraph,
    vectorized_greedy_initialize,
)


class TestBuildSparseImpactGraph:
    def test_basic(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 5)
        graph = build_sparse_impact_graph(embeddings, k=5, use_hnswlib=False)
        assert graph.shape == (20, 20)

    def test_with_hnswlib_fallback(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(15, 4)
        graph = build_sparse_impact_graph(embeddings, k=5, use_hnswlib=True)
        assert graph.shape == (15, 15)

    def test_small_n(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(3, 4)
        graph = build_sparse_impact_graph(embeddings, k=2, use_hnswlib=False)
        assert graph.shape == (3, 3)


class TestExtractSubgraph:
    def test_basic(self):
        rng = np.random.RandomState(42)
        # extract_subgraph expects a sparse matrix
        dense = rng.rand(10, 10)
        graph = sparse.csr_matrix(dense)
        indices = np.array([1, 3, 5])
        sub = extract_subgraph(graph, indices)
        assert sub.shape == (3, 3)


class TestVectorizedGreedyInitialize:
    def test_basic(self):
        rng = np.random.RandomState(42)
        n = 10
        surprises = rng.rand(n)
        impact_graph = rng.rand(n, n) * 0.3
        np.fill_diagonal(impact_graph, 0)
        x = vectorized_greedy_initialize(surprises, impact_graph, k=3, alpha=1.0, beta=0.5)
        assert int(x.sum()) == 3
        assert x.shape == (n,)


class TestBodhisattvaQUBO:
    def test_build_and_solve(self):
        rng = np.random.RandomState(42)
        n = 15
        surprises = rng.rand(n)

        bq = BodhisattvaQUBO(n)
        bq.add_prajna_term(surprises, alpha=1.0)
        bq.add_sila_term(k=5, gamma=10.0)
        builder = bq.get_builder()
        Q = builder.get_matrix()
        assert Q.shape == (n, n)


class TestMadhyamakaBalancer:
    def test_balance(self):
        rng = np.random.RandomState(42)
        surprises = rng.rand(10)

        balancer = MadhyamakaBalancer(target_cv=0.5)
        new_alpha, new_beta = balancer.balance(surprises, current_alpha=1.0, current_beta=0.5)
        assert isinstance(new_alpha, float)
        assert isinstance(new_beta, float)
        assert new_alpha > 0
        assert new_beta > 0
