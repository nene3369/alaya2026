"""Built-in tools — file I/O, code execution, HTTP fetch.

These are the default 「身」(body) capabilities shipped with Digital Dharma OS.
Each tool satisfies the Tool protocol and can be registered in a ToolRegistry.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from lmm.tools.base import ToolResult, ToolStatus


# ---------------------------------------------------------------------------
# File Read
# ---------------------------------------------------------------------------

class FileReadTool:
    """Read a file and return its contents."""

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def category(self) -> str:
        return "filesystem"

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", "")
        if not path:
            return ToolResult(status=ToolStatus.ERROR, error="Missing 'path'")
        try:
            p = Path(path)
            content = await asyncio.to_thread(p.read_text, encoding="utf-8")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=content,
                relevance_score=0.7,
            )
        except Exception as exc:
            return ToolResult(status=ToolStatus.ERROR, error=str(exc))


# ---------------------------------------------------------------------------
# File Write
# ---------------------------------------------------------------------------

class FileWriteTool:
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path."

    @property
    def category(self) -> str:
        return "filesystem"

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return ToolResult(status=ToolStatus.ERROR, error="Missing 'path'")
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(p.write_text, content, encoding="utf-8")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Written {len(content)} bytes to {path}",
                relevance_score=0.8,
            )
        except Exception as exc:
            return ToolResult(status=ToolStatus.ERROR, error=str(exc))


# ---------------------------------------------------------------------------
# Code Execution (sandboxed subprocess)
# ---------------------------------------------------------------------------

class CodeExecutionTool:
    """Execute Python code in a sandboxed subprocess."""

    @property
    def name(self) -> str:
        return "code_execute"

    @property
    def description(self) -> str:
        return "Execute Python code in a sandboxed subprocess and return stdout/stderr."

    @property
    def category(self) -> str:
        return "compute"

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        code = params.get("code", "")
        timeout = min(params.get("timeout", 30), 60)
        if not code:
            return ToolResult(status=ToolStatus.ERROR, error="Missing 'code'")

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
            ) as f:
                f.write(code)
                tmp_path = f.name

            proc = await asyncio.create_subprocess_exec(
                "python3", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )

            os.unlink(tmp_path)

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=output,
                    relevance_score=0.9,
                    metadata={"stderr": errors} if errors else {},
                )
            return ToolResult(
                status=ToolStatus.ERROR,
                output=output,
                error=errors or f"Exit code {proc.returncode}",
            )

        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                error=f"Code execution timed out after {timeout}s",
            )
        except Exception as exc:
            return ToolResult(status=ToolStatus.ERROR, error=str(exc))


# ---------------------------------------------------------------------------
# HTTP Fetch
# ---------------------------------------------------------------------------

class HttpFetchTool:
    """Fetch a URL and return the response body."""

    @property
    def name(self) -> str:
        return "http_fetch"

    @property
    def description(self) -> str:
        return "Fetch content from a URL via HTTP GET."

    @property
    def category(self) -> str:
        return "network"

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        url = params.get("url", "")
        if not url:
            return ToolResult(status=ToolStatus.ERROR, error="Missing 'url'")
        try:
            import httpx
            timeout = min(params.get("timeout", 15), 30)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=resp.text[:500_000],
                    relevance_score=0.6,
                    metadata={
                        "status_code": resp.status_code,
                        "content_type": resp.headers.get("content-type", ""),
                    },
                )
        except Exception as exc:
            return ToolResult(status=ToolStatus.ERROR, error=str(exc))


# ---------------------------------------------------------------------------
# Shell Command
# ---------------------------------------------------------------------------

class ShellCommandTool:
    """Execute a shell command (pre-validated by sandbox)."""

    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def description(self) -> str:
        return "Run a shell command and return stdout/stderr."

    @property
    def category(self) -> str:
        return "system"

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        command = params.get("command", "")
        timeout = min(params.get("timeout", 30), 60)
        if not command:
            return ToolResult(status=ToolStatus.ERROR, error="Missing 'command'")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=output,
                    relevance_score=0.7,
                )
            return ToolResult(
                status=ToolStatus.ERROR,
                output=output,
                error=errors or f"Exit code {proc.returncode}",
            )
        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                error=f"Command timed out after {timeout}s",
            )
        except Exception as exc:
            return ToolResult(status=ToolStatus.ERROR, error=str(exc))


def register_builtins(registry) -> None:
    """Register all built-in tools."""
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(CodeExecutionTool())
    registry.register(HttpFetchTool())
    registry.register(ShellCommandTool())
