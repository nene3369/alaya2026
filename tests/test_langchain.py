"""Tests for lmm.integrations.langchain — no LangChain dependency needed."""

from __future__ import annotations

import asyncio

import numpy as np

from lmm.integrations.langchain import (
    DharmaDocumentCompressor,
    DharmaExampleSelector,
    DharmaRunnable,
    langchain_tool_to_lmm,
    register_langchain_tools,
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


# ---------------------------------------------------------------------------
# Tool Bridge tests (no LangChain dependency)
# ---------------------------------------------------------------------------


class _FakeLCTool:
    """Mimics LangChain BaseTool for testing without LangChain installed."""

    name = "fake_search"
    description = "Fake search tool for testing"

    def invoke(self, params):
        return f"searched for {params}"

    async def ainvoke(self, params):
        return f"async searched for {params}"


class _FakeLCToolError:
    """A LangChain-like tool that always raises."""

    name = "error_tool"
    description = "Always errors"

    def invoke(self, params):
        raise RuntimeError("tool error")


class _FakeRunnable:
    """Mimics a LangChain Runnable for testing."""

    def invoke(self, params):
        return f"chain result: {params}"


class TestLangChainToolBridge:
    def test_wrap_lc_tool_has_protocol_attrs(self):
        lc_tool = _FakeLCTool()
        wrapper = langchain_tool_to_lmm(lc_tool)
        assert wrapper.name == "fake_search"
        assert wrapper.description == "Fake search tool for testing"
        assert wrapper.category == "langchain"

    def test_execute_success(self):
        lc_tool = _FakeLCTool()
        wrapper = langchain_tool_to_lmm(lc_tool)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                wrapper.execute({"query": "test"}),
            )
        finally:
            loop.close()
        assert result.status.value == "success"
        assert "searched" in str(result.output)

    def test_execute_error(self):
        lc_tool = _FakeLCToolError()
        wrapper = langchain_tool_to_lmm(lc_tool)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                wrapper.execute({"query": "test"}),
            )
        finally:
            loop.close()
        assert result.status.value == "error"
        assert "tool error" in result.error

    def test_custom_category(self):
        lc_tool = _FakeLCTool()
        wrapper = langchain_tool_to_lmm(lc_tool, category="search")
        assert wrapper.category == "search"


class TestRegisterLangChainTools:
    def test_registers_multiple_tools(self):
        from lmm.tools.base import ToolRegistry
        registry = ToolRegistry()
        tools = [_FakeLCTool(), _FakeLCTool()]
        count = register_langchain_tools(tools, registry)
        # Both have same name so only 1 slot in registry, but count=2
        assert count == 2

    def test_returns_count(self):
        from lmm.tools.base import ToolRegistry
        registry = ToolRegistry()
        count = register_langchain_tools([], registry)
        assert count == 0


class TestDharmaRunnable:
    def test_execute_with_dict_input(self):
        runnable = _FakeRunnable()
        tool = DharmaRunnable(runnable, name="test_chain")
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                tool.execute({"input": "hello"}),
            )
        finally:
            loop.close()
        assert result.status.value == "success"
        assert "chain result" in str(result.output)

    def test_satisfies_tool_protocol(self):
        runnable = _FakeRunnable()
        tool = DharmaRunnable(runnable, name="my_chain", description="a chain")
        assert tool.name == "my_chain"
        assert tool.description == "a chain"
        assert tool.category == "langchain"
