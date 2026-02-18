"""Tests for lmm.integrations.langchain — no LangChain dependency needed."""

from __future__ import annotations

import numpy as np

from lmm.integrations.langchain import (
    DharmaDocumentCompressor,
    DharmaExampleSelector,
    rerank_by_dharma_engine,
    rerank_by_diversity,
    rerank_by_diversity_adaptive,
)


class TestRerankByDiversity:
    def test_basic(self):
        rng = np.random.RandomState(42)
        relevance = rng.rand(20)
        embeddings = rng.randn(20, 8)
        indices = rerank_by_diversity(relevance, embeddings, k=5)
        assert len(indices) == 5
        assert len(set(indices)) == 5

    def test_k_equals_n(self):
        rng = np.random.RandomState(42)
        relevance = rng.rand(5)
        embeddings = rng.randn(5, 4)
        indices = rerank_by_diversity(relevance, embeddings, k=5)
        assert len(indices) == 5


class TestRerankByDiversityAdaptive:
    def test_basic(self):
        rng = np.random.RandomState(42)
        relevance = rng.rand(20)
        embeddings = rng.randn(20, 8)
        query_emb = rng.randn(8)
        indices = rerank_by_diversity_adaptive(
            relevance, embeddings, query_emb, k=5,
        )
        assert len(indices) == 5


class TestRerankByDharmaEngine:
    def test_basic(self):
        rng = np.random.RandomState(42)
        relevance = rng.rand(15)
        embeddings = rng.randn(15, 8)
        query_emb = rng.randn(8)
        indices = rerank_by_dharma_engine(
            relevance, embeddings, query_emb, k=5,
        )
        assert len(indices) == 5


class _FakeDoc:
    def __init__(self, text: str):
        self.page_content = text


class TestDharmaDocumentCompressor:
    def test_compress(self):
        docs = [_FakeDoc(f"document {i} about topic {i % 3}") for i in range(20)]
        compressor = DharmaDocumentCompressor(k=5)
        result = compressor.compress_documents(docs, query="topic 1")
        assert len(result) == 5

    def test_empty(self):
        compressor = DharmaDocumentCompressor(k=5)
        result = compressor.compress_documents([], query="test")
        assert result == []

    def test_fewer_than_k(self):
        docs = [_FakeDoc("short doc")]
        compressor = DharmaDocumentCompressor(k=5)
        result = compressor.compress_documents(docs, query="test")
        assert len(result) == 1


class TestDharmaExampleSelector:
    def test_select(self):
        examples = [
            {"input": f"example {i}", "output": f"result {i}"}
            for i in range(10)
        ]
        selector = DharmaExampleSelector(examples=examples, k=3)
        selected = selector.select_examples({"input": "test query"})
        assert len(selected) == 3

    def test_add_example(self):
        selector = DharmaExampleSelector(k=2)
        selector.add_example({"input": "quantum physics research", "output": "science"})
        selector.add_example({"input": "classical music theory", "output": "arts"})
        selector.add_example({"input": "quantum computing hardware", "output": "tech"})
        selected = selector.select_examples({"input": "quantum physics"})
        assert 1 <= len(selected) <= 2

    def test_empty(self):
        selector = DharmaExampleSelector(k=3)
        selected = selector.select_examples({"input": "test"})
        assert selected == []
