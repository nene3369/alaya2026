"""Tests for lmm.tools — Tool Use / Actuation (身・Kaya)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from lmm.tools.base import (
    ToolRegistry,
    ToolResult,
    ToolSandbox,
    ToolStatus,
    SandboxPolicy,
)
from lmm.tools.executor import ToolExecutor


# ---------------------------------------------------------------------------
# Dummy tool for testing
# ---------------------------------------------------------------------------

class EchoTool:
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes input back."

    @property
    def category(self) -> str:
        return "test"

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=params.get("text", ""),
            relevance_score=0.8,
            node_indices=params.get("node_indices", []),
        )


class SlowTool:
    @property
    def name(self) -> str:
        return "slow"

    @property
    def description(self) -> str:
        return "Sleeps for a while."

    @property
    def category(self) -> str:
        return "test"

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        await asyncio.sleep(params.get("seconds", 10))
        return ToolResult(status=ToolStatus.SUCCESS, output="done")


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests: ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        assert reg.count == 1
        assert reg.get("echo") is not None
        assert reg.get("nonexistent") is None

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tools = reg.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"
        assert tools[0]["category"] == "test"

    def test_by_category(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        assert len(reg.by_category("test")) == 1
        assert len(reg.by_category("filesystem")) == 0


# ---------------------------------------------------------------------------
# Tests: ToolSandbox
# ---------------------------------------------------------------------------

class TestToolSandbox:
    def test_blocks_denied_commands(self):
        sandbox = ToolSandbox()
        result = sandbox.validate("shell", {"command": "rm -rf /"})
        assert result is not None
        assert "blocked" in result.lower()

    def test_allows_safe_commands(self):
        sandbox = ToolSandbox()
        result = sandbox.validate("shell", {"command": "ls -la"})
        assert result is None

    def test_blocks_denied_paths(self):
        sandbox = ToolSandbox()
        result = sandbox.validate("file_read", {"path": "/etc/shadow"})
        assert result is not None

    def test_truncate_output(self):
        sandbox = ToolSandbox(SandboxPolicy(max_output_bytes=10))
        result = sandbox.truncate_output("a" * 100)
        assert len(result) < 100


# ---------------------------------------------------------------------------
# Tests: ToolExecutor
# ---------------------------------------------------------------------------

class TestToolExecutor:
    def _make_executor(self) -> ToolExecutor:
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(SlowTool())
        return ToolExecutor(reg)

    def test_execute_success(self):
        ex = self._make_executor()
        result = _run(ex.execute("echo", {"text": "hello"}))
        assert result.status == ToolStatus.SUCCESS
        assert result.output == "hello"

    def test_execute_not_found(self):
        ex = self._make_executor()
        result = _run(ex.execute("nonexistent", {}))
        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error.lower()

    def test_execute_timeout(self):
        ex = self._make_executor()
        ex.sandbox.policy.max_execution_seconds = 0.1
        result = _run(ex.execute("slow", {"seconds": 10}))
        assert result.status == ToolStatus.TIMEOUT

    def test_execute_denied(self):
        ex = self._make_executor()
        result = _run(ex.execute("echo", {"command": "rm -rf /"}))
        assert result.status == ToolStatus.DENIED

    def test_to_action_results(self):
        ex = self._make_executor()
        results = [
            ToolResult(
                status=ToolStatus.SUCCESS,
                output="data",
                node_indices=[0, 1, 2],
                relevance_score=0.9,
            ),
            ToolResult(status=ToolStatus.ERROR, error="fail"),
        ]
        actions = ex.to_action_results(results, n_variables=8)
        assert len(actions) == 1
        assert actions[0].relevance_score == 0.9

    def test_execute_many(self):
        ex = self._make_executor()
        results = _run(ex.execute_many([
            {"tool": "echo", "params": {"text": "a"}},
            {"tool": "echo", "params": {"text": "b"}},
        ]))
        assert len(results) == 2
        assert all(r.status == ToolStatus.SUCCESS for r in results)

    def test_history_tracking(self):
        ex = self._make_executor()
        _run(ex.execute("echo", {"text": "test"}))
        assert len(ex.history) == 1
        assert ex.history[0]["tool"] == "echo"
