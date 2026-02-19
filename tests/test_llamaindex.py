"""Tests for lmm.integrations.llamaindex — no LlamaIndex dependency needed."""

from __future__ import annotations

import asyncio

from lmm.integrations.llamaindex import (
    DharmaQueryEngineTool,
    llamaindex_tool_to_lmm,
    register_llamaindex_tools,
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


# ---------------------------------------------------------------------------
# Tool Bridge tests (no LlamaIndex dependency)
# ---------------------------------------------------------------------------


class _FakeLIToolMetadata:
    name = "fake_li_tool"
    description = "Fake LlamaIndex tool"


class _FakeLITool:
    """Mimics a LlamaIndex tool for testing without dependency."""

    metadata = _FakeLIToolMetadata()

    def call(self, **kwargs):
        return f"called with {kwargs}"


class _FakeLIToolError:
    """A LlamaIndex-like tool that always raises."""

    metadata = type("M", (), {"name": "error_tool", "description": "Always errors"})()

    def call(self, **kwargs):
        raise RuntimeError("li tool error")


class _FakeQueryEngine:
    """Mimics a LlamaIndex QueryEngine for testing."""

    def query(self, query_str):
        return f"answer to {query_str}"


class TestLlamaIndexToolBridge:
    def test_wrap_li_tool_protocol_attrs(self):
        li_tool = _FakeLITool()
        wrapper = llamaindex_tool_to_lmm(li_tool)
        assert wrapper.name == "fake_li_tool"
        assert wrapper.description == "Fake LlamaIndex tool"
        assert wrapper.category == "llamaindex"

    def test_execute_success(self):
        li_tool = _FakeLITool()
        wrapper = llamaindex_tool_to_lmm(li_tool)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                wrapper.execute({"query": "test"}),
            )
        finally:
            loop.close()
        assert result.status.value == "success"
        assert "called" in str(result.output)

    def test_execute_error(self):
        li_tool = _FakeLIToolError()
        wrapper = llamaindex_tool_to_lmm(li_tool)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                wrapper.execute({"query": "test"}),
            )
        finally:
            loop.close()
        assert result.status.value == "error"
        assert "li tool error" in result.error

    def test_custom_category(self):
        li_tool = _FakeLITool()
        wrapper = llamaindex_tool_to_lmm(li_tool, category="search")
        assert wrapper.category == "search"


class TestRegisterLlamaIndexTools:
    def test_registers_multiple(self):
        from lmm.tools.base import ToolRegistry
        registry = ToolRegistry()
        tools = [_FakeLITool()]
        count = register_llamaindex_tools(tools, registry)
        assert count == 1
        assert registry.count == 1

    def test_returns_zero_for_empty(self):
        from lmm.tools.base import ToolRegistry
        registry = ToolRegistry()
        count = register_llamaindex_tools([], registry)
        assert count == 0


class TestDharmaQueryEngineTool:
    def test_execute_query(self):
        engine = _FakeQueryEngine()
        tool = DharmaQueryEngineTool(engine, name="test_engine")
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                tool.execute({"query": "what is life?"}),
            )
        finally:
            loop.close()
        assert result.status.value == "success"
        assert "answer to what is life?" in str(result.output)

    def test_missing_query_param(self):
        engine = _FakeQueryEngine()
        tool = DharmaQueryEngineTool(engine)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                tool.execute({}),
            )
        finally:
            loop.close()
        assert result.status.value == "error"
        assert "query" in result.error.lower()

    def test_satisfies_tool_protocol(self):
        engine = _FakeQueryEngine()
        tool = DharmaQueryEngineTool(
            engine, name="my_engine", description="A query engine",
        )
        assert tool.name == "my_engine"
        assert tool.description == "A query engine"
        assert tool.category == "llamaindex"
