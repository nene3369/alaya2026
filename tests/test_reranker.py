"""Tests for lmm.dharma.reranker — DharmaReranker + IntentAwareRouter."""

from __future__ import annotations

import numpy as np

from lmm.dharma.reranker import DharmaReranker, IntentAwareRouter


class TestDharmaReranker:
    def test_basic_rerank(self):
        rng = np.random.RandomState(42)
        query_emb = rng.randn(8)
        candidate_embs = rng.randn(20, 8)

        reranker = DharmaReranker(k=5)
        result = reranker.rerank(
            query_embedding=query_emb,
            candidate_embeddings=candidate_embs,
        )
        assert len(result.selected_indices) == 5

    def test_with_solver_modes(self):
        rng = np.random.RandomState(42)
        query_emb = rng.randn(8)
        candidate_embs = rng.randn(15, 8)

        for mode in ["auto", "fep_analog"]:
            reranker = DharmaReranker(k=4, solver_mode=mode)
            result = reranker.rerank(
                query_embedding=query_emb,
                candidate_embeddings=candidate_embs,
            )
            assert len(result.selected_indices) == 4

    def test_k_larger_than_n(self):
        rng = np.random.RandomState(42)
        query_emb = rng.randn(5)
        candidate_embs = rng.randn(3, 5)

        reranker = DharmaReranker(k=10)
        result = reranker.rerank(
            query_embedding=query_emb,
            candidate_embeddings=candidate_embs,
        )
        assert len(result.selected_indices) <= 3


class TestIntentAwareRouter:
    def test_route(self):
        rng = np.random.RandomState(42)
        n = 20
        corpus_embs = rng.randn(n, 8)
        query_vec = rng.randn(8)

        router = IntentAwareRouter(k=5)
        router.fit_offline(corpus_embs, graph_k=5)

        fetched = np.arange(n)
        result, diagnosis = router.route(query_vec, fetched)
        assert len(result.selected_indices) == 5
        assert diagnosis is not None
