"""Tests for lmm.integrations.llamaindex — no LlamaIndex dependency needed."""

from __future__ import annotations


from lmm.integrations.llamaindex import (
    rerank_nodes_by_dharma_engine,
    rerank_nodes_by_diversity,
    rerank_nodes_by_diversity_adaptive,
)


class _FakeNode:
    """Minimal node-like object for testing without LlamaIndex."""

    def __init__(self, text: str, score: float | None = None):
        self._text = text
        self.score = score

    def get_text(self) -> str:
        return self._text


class TestRerankNodesByDiversity:
    def test_basic(self):
        nodes = [_FakeNode(f"node {i} about topic {i % 3}", score=0.5 + i * 0.01)
                 for i in range(20)]
        result = rerank_nodes_by_diversity(nodes, query_str="topic 1", top_k=5)
        assert len(result) == 5
        assert all(isinstance(n, _FakeNode) for n in result)

    def test_without_query(self):
        nodes = [_FakeNode(f"node {i}", score=float(i) / 10) for i in range(15)]
        result = rerank_nodes_by_diversity(nodes, query_str=None, top_k=5)
        assert len(result) == 5

    def test_top_k_larger_than_n(self):
        nodes = [_FakeNode("short", score=0.5)]
        result = rerank_nodes_by_diversity(nodes, top_k=10)
        assert len(result) == 1


class TestRerankNodesByDiversityAdaptive:
    def test_basic(self):
        nodes = [_FakeNode(f"node {i} content {i % 5}", score=0.5)
                 for i in range(20)]
        result = rerank_nodes_by_diversity_adaptive(
            nodes, query_str="content 2", k_max=10,
        )
        assert len(result) <= 10
        assert len(result) > 0


class TestRerankNodesByDharmaEngine:
    def test_with_query(self):
        nodes = [_FakeNode(f"node {i} about topic {i % 4}", score=0.5)
                 for i in range(15)]
        result = rerank_nodes_by_dharma_engine(
            nodes, query_str="topic 2", top_k=5,
        )
        assert len(result) == 5

    def test_without_query(self):
        nodes = [_FakeNode(f"node {i}", score=float(i) / 10)
                 for i in range(15)]
        result = rerank_nodes_by_dharma_engine(
            nodes, query_str=None, top_k=5,
        )
        assert len(result) == 5
