"""ToolExecutor — orchestrates sandboxed tool dispatch for Active Inference.

Bridges the gap between the ActiveInferenceEngine's action selection and
concrete external actions.  Each execution cycle:
  1. Validate via ToolSandbox (Upali precept check)
  2. Execute with timeout
  3. Convert ToolResult → ActionResult for belief injection
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Dict, List

import numpy as np

from lmm.reasoning.active import ActionResult
from lmm.tools.base import (
    ToolRegistry,
    ToolResult,
    ToolSandbox,
    ToolStatus,
)


class ToolExecutor:
    """Execute tools within sandbox constraints and return ActionResults.

    This is the 「手」(hand) of the system — the interface through which
    the Active Inference engine acts upon the external world.

    Parameters
    ----------
    registry : ToolRegistry
        Available tools.
    sandbox : ToolSandbox | None
        Sandbox policy enforcer.  Defaults to standard policy.
    max_concurrent : int
        Maximum number of tools that can run in parallel.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        sandbox: ToolSandbox | None = None,
        max_concurrent: int = 4,
    ) -> None:
        self.registry = registry
        self.sandbox = sandbox or ToolSandbox()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._history: List[Dict[str, Any]] = []

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool with sandbox validation and timeout."""
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Tool not found: '{tool_name}'",
            )

        # Sandbox validation (Upali gate)
        violation = self.sandbox.validate(tool_name, params)
        if violation is not None:
            result = ToolResult(
                status=ToolStatus.DENIED,
                error=violation,
            )
            self._record(tool_name, params, result)
            return result

        # Execute with timeout and concurrency limit
        t0 = time.monotonic()
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    tool.execute(params),
                    timeout=self.sandbox.policy.max_execution_seconds,
                )
            except asyncio.TimeoutError:
                result = ToolResult(
                    status=ToolStatus.TIMEOUT,
                    error=f"Tool '{tool_name}' timed out after "
                          f"{self.sandbox.policy.max_execution_seconds}s",
                )
            except Exception as exc:
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Tool '{tool_name}' raised: {exc}",
                )

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        result.output = self.sandbox.truncate_output(result.output)
        self._record(tool_name, params, result)
        return result

    async def execute_many(
        self,
        calls: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        """Execute multiple tool calls concurrently.

        Each item in *calls* should have ``"tool"`` and ``"params"`` keys.
        """
        coros = [
            self.execute(c["tool"], c.get("params", {}))
            for c in calls
        ]
        return list(await asyncio.gather(*coros))

    def to_action_results(
        self,
        results: List[ToolResult],
        n_variables: int,
    ) -> List[ActionResult]:
        """Convert ToolResults into ActionResults for belief injection.

        Only successful results are converted.  Node indices are taken from
        the ToolResult's ``node_indices`` field; if empty, a hash-based
        mapping distributes the effect across the belief vector.
        """
        action_results: List[ActionResult] = []
        for r in results:
            if r.status != ToolStatus.SUCCESS:
                continue
            if r.node_indices:
                indices = np.array(r.node_indices, dtype=int)
            else:
                # Hash-based distribution: spread effect across nodes
                content = str(r.output)
                h = int(hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest(), 16) % (2 ** 32)
                n_affected = max(1, n_variables // 8)
                rng = np.random.RandomState(h)
                indices = rng.choice(n_variables, size=n_affected, replace=False)
            action_results.append(
                ActionResult(
                    node_indices=indices,
                    relevance_score=r.relevance_score,
                )
            )
        return action_results

    def _record(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: ToolResult,
    ) -> None:
        self._history.append({
            "tool": tool_name,
            "params": {k: str(v)[:200] for k, v in params.items()},
            "status": result.status.value,
            "elapsed_ms": result.elapsed_ms,
            "timestamp": time.time(),
        })
        # Keep last 200 entries
        if len(self._history) > 200:
            self._history = self._history[-200:]

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)
